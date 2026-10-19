"""Pre-registered bounded diagnosis, never a replacement long-training campaign.

See docs/random_start_rl_diagnosis_v4_plan.md. All privileged logs remain outside
the policy-facing tensors; fitting uses only recorded legal observations/actions.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import gzip
import hashlib
import json
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import rl
import long_train as runner

OUT = rl.ROOT / "outputs/random_start_rl_diagnosis_v4"
SOURCE = rl.ROOT / "outputs/random_start_time_budget_v3_coverage256"
CHECKPOINTS = {"initial": "initial.pt", "degraded_100": "update_00100.pt",
               "late_4600": "update_04600.pt"}


def save(path, data):
    runner.atomic_json(path, data)


def load_policy(path):
    model = rl.JointPolicy(324)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "panels.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    panels, metadata = {}, {}
    for map_name in ("L0", "L1"):
        for split in ("train", "validation", "test"):
            key = f"{map_name}/{split}"
            panels[key], metadata[key] = rl.layout.build_split(map_name, split)
    audit = rl.layout.audit_split_content(panels)
    assert audit["ok"], audit
    legacy = rl.layout.build_split("L0", "validation",
                                    split_version=rl.layout.LEGACY_SPLIT_VERSION)[0]
    overlap = rl.layout.audit_split_content({"train": panels["L0/train"], "legacy": legacy})
    sources = [Path(__file__), Path(rl.__file__), Path(runner.__file__)]
    sources += list((rl.ROOT / "src/coimagine/random_start").glob("*.py"))
    sources += list((rl.ROOT / "src/coimagine/continuous2d").glob("*.py"))
    hashes = {}
    for source in sources:
        relative = source.relative_to(rl.ROOT)
        target = out / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    save(manifest, dict(version=rl.layout.SPLIT_VERSION, metadata=metadata,
                       instances={k: [x.as_dict() for x in v] for k, v in panels.items()},
                       content_audit=audit, legacy_L0_overlap=overlap,
                       test_policy_evaluation=False))
    save(out / "manifest.json", dict(environment=rl.E.EnvConfig().as_dict(),
         checkpoint_hashes={k: hashlib.sha256((SOURCE / "checkpoints" / v).read_bytes()).hexdigest()
                            for k, v in CHECKPOINTS.items()}, source_sha256=hashes,
         plan="docs/random_start_rl_diagnosis_v4_plan.md", torch_version=torch.__version__,
         numpy_version=np.__version__, optimizer="fresh AdamW, lr=3e-4, weight_decay=0"))
    print("split audit", audit["unique"], "unique; legacy duplicates", len(overlap["duplicates"]), flush=True)


def tracking():
    names = json.loads((SOURCE / "manifest.json").read_text())["config"]["tracking_instances"]
    panel = {x.name: x for x in rl.layout.build_split("L0", "train")[0]}
    return [panel[name] for name in names]


def replay_job(job):
    instance, weights, seed, deterministic, log_path = job
    model = rl.JointPolicy(324)
    model.load_state_dict({key: torch.from_numpy(value) for key, value in weights.items()})
    env = rl.E.RandomStartEnv(instance)
    episode = rl.rollout(env, model, stochastic=not deterministic,
                         generator=torch.Generator().manual_seed(seed))
    if episode["termination"] in rl.E.ANOMALY_TERMINATIONS:
        raise RuntimeError((instance.name, episode["termination"]))
    exits = Counter(end["exit_reason"] for b in env.boundaries for end in b["skill_ends"].values())
    durations = np.asarray([b["periods"] for b in env.boundaries])
    repeats = sum(a["action_index"] == b["action_index"]
                  for a, b in zip(env.boundaries, env.boundaries[1:]))
    episode.update(instance=instance.name, behaviour_seed=seed,
                   exits=dict(exits), one_tick_decisions=int(np.sum(durations == 1)),
                   short_decisions=int(np.sum(durations <= 2)),
                   short_time_fraction=float(durations[durations <= 2].sum() / durations.sum()),
                   repeated_action_pairs=repeats, median_periods=float(np.median(durations)),
                   reward_components=env.reward_components)
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as handle:
            json.dump(dict(instance=instance.as_dict(), boundaries=env.boundaries,
                           rows=env.rows), handle)
    for d in episode["decisions"]:
        d["observations"] = d["observations"].numpy()
        d["boundary_frame"] = d["boundary_frame"].numpy()
    return episode


def collect(executor, model, instances, seeds=(0,), deterministic=True, log_dir=None):
    weights = runner.snapshot(model)
    jobs = [(instance, weights, seed, deterministic,
             str(log_dir / f"{instance.name}_{seed}.json.gz") if log_dir else None)
            for seed in seeds for instance in instances]
    episodes = list(executor.map(replay_job, jobs))
    for ep in episodes:
        for d in ep["decisions"]:
            for key in ("observations", "boundary_frame"):
                d[key] = torch.from_numpy(d[key])
    return episodes


def summary(episodes):
    return dict(success=float(np.mean([e["success"] for e in episodes])),
                episodes=len(episodes), decisions=sum(len(e["decisions"]) for e in episodes),
                rows=[{k: v for k, v in e.items() if k != "decisions"} for e in episodes])


def replay(out, executor):
    models = {k: load_policy(SOURCE / "checkpoints" / v) for k, v in CHECKPOINTS.items()}
    instances = tracking()
    report = {}
    for name, model in models.items():
        episodes = collect(executor, model, instances, log_dir=out / "logs" / name)
        report[name] = summary(episodes)
        torch.save(episodes, out / f"replay_{name}.pt")
        save(out / "replay.json", report)
        print("replay", name, report[name]["success"], flush=True)
    validation = rl.layout.build_split("L0", "validation")[0]
    for name in ("initial", "late_4600"):
        result = summary(collect(executor, models[name], validation))
        report[f"validation_v2_{name}"] = result
        save(out / "replay.json", report)
        print("new validation", name, result["success"], flush=True)
    episodes = collect(executor, models["initial"], instances, seeds=(0, 1, 2), deterministic=False)
    torch.save(episodes, out / "stochastic24.pt")
    save(out / "stochastic24.json", summary(episodes))
    print("stochastic24", summary(episodes)["success"], flush=True)


def flat_grad(parameters, gradients):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                      for p, g in zip(parameters, gradients)])


def cosine(a, b):
    den = a.norm() * b.norm()
    return float(torch.dot(a, b) / den) if den > 1e-15 else None


def gradient_audit(model, episodes):
    """Exact first-epoch loss gradients, all groups use the SAME batch denominator."""
    entries = rl.flatten(episodes)["entries"]
    parameters = list(model.parameters())
    size = sum(p.numel() for p in parameters)
    names = ["success", "failure_short", "failure_long"]
    vectors = {f"{kind}/{name}": torch.zeros(size) for kind in ("actor", "critic") for name in names}
    counts, losses = Counter(), defaultdict(float)
    offset = 0
    replay_error = 0.0
    for episode in episodes:
        grouped = defaultdict(list)
        for decision, hidden in rl.decision_states(model, episode):
            entry = entries[offset]
            offset += 1
            group = "success" if episode["success"] else (
                "failure_short" if decision["periods"] <= 2 else "failure_long")
            counts[group] += 1
            logp, _ = model.evaluate(hidden, *decision["action"])
            replay_error = max(replay_error, abs(float(logp.detach()) - decision["logp"]))
            ratio = torch.exp(logp - decision["logp"])
            adv = entry["advantage"]
            actor = -torch.minimum(ratio * adv, ratio.clamp(1-rl.CLIP_RANGE, 1+rl.CLIP_RANGE) * adv)
            critic = rl.VALUE_WEIGHT * (model.value(hidden) - entry["target"]) ** 2
            grouped[f"actor/{group}"].append(actor)
            grouped[f"critic/{group}"].append(critic)
        keys = list(grouped)
        for i, key in enumerate(keys):
            loss = torch.stack(grouped[key]).sum() / len(entries)
            grad = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=i < len(keys)-1)
            vectors[key] += flat_grad(parameters, grad)
            losses[key] += float(loss.detach())
    shared = torch.cat([torch.full((p.numel(),), name.startswith(("encoder.", "gru.")), dtype=torch.bool)
                        for name, p in model.named_parameters()])
    actor = sum(vectors[f"actor/{name}"] for name in names)
    critic = sum(vectors[f"critic/{name}"] for name in names)
    total = actor + critic
    report = dict(decisions=len(entries), counts=dict(counts), loss_contributions=dict(losses),
        logp_replay_max_abs=replay_error,
        gradient_norms={key: float(value.norm()) for key, value in vectors.items()},
        shared_gradient_norms={key: float(value[shared].norm()) for key, value in vectors.items()},
        actor_norm=float(actor.norm()), weighted_critic_norm=float(critic.norm()),
        shared_actor_norm=float(actor[shared].norm()), shared_critic_norm=float(critic[shared].norm()),
        shared_actor_critic_cosine=cosine(actor[shared], critic[shared]),
        actor_success_failure_cosine=cosine(vectors["actor/success"],
                   vectors["actor/failure_short"] + vectors["actor/failure_long"]),
        total_preclip_norm=float(total.norm()), global_clip_scale=min(1.0, 5.0/(float(total.norm())+1e-6)),
        advantage_by_group={name: {"raw_mean": float(np.mean([e["advantage_raw"] for e in entries
                    if ("success" if e["episode_success"] else ("failure_short" if e["periods"]<=2
                        else "failure_long")) == name])),
                "normalized_mean": float(np.mean([e["advantage"] for e in entries
                    if ("success" if e["episode_success"] else ("failure_short" if e["periods"]<=2
                        else "failure_long")) == name]))} for name in names if counts[name]})
    return report, total


@torch.no_grad()
def path_scores(model, episodes):
    scores = []
    for ep in episodes:
        logps, hits, joint_logs = [], [], []
        for d, hidden in rl.decision_states(model, ep):
            logp, _ = model.evaluate(hidden, *d["action"])
            logps.append(float(logp))
            hits.append(model.sample(hidden, deterministic=True)[:2] == tuple(d["action"]))
            first = torch.log_softmax(model.head_first(hidden), dim=-1)[0]
            second = model.head_second(torch.cat([hidden.expand(15, -1),
                                       model.embed(torch.arange(15))], dim=-1))
            joint_logs.append((first[:, None] + torch.log_softmax(second, dim=-1)).flatten())
        scores.append(dict(instance=ep["instance"], success=ep["success"],
            mean_logp=float(np.mean(logps)), accuracy=float(np.mean(hits)),
            joint_logs=torch.stack(joint_logs)))
    return scores


def path_change(before, after):
    rows = []
    for a, b in zip(before, after):
        kl = torch.sum(a["joint_logs"].exp() * (a["joint_logs"] - b["joint_logs"]), dim=-1)
        rows.append(dict(instance=a["instance"], success=a["success"],
                         mean_logp_change=b["mean_logp"]-a["mean_logp"],
                         action_agreement_before=a["accuracy"], action_agreement_after=b["accuracy"],
                         mean_exact_joint_KL=float(kl.mean()), max_exact_joint_KL=float(kl.max())))
    return rows


def audit(out, executor):
    model = load_policy(SOURCE / "checkpoints/initial.pt")
    episodes = torch.load(out / "stochastic24.pt", weights_only=False)
    batch = episodes[:8]  # seed 0, every tracking instance, fixed before outcomes
    fixed_paths = torch.load(out / "replay_initial.pt", weights_only=False)
    baseline = path_scores(model, fixed_paths)
    gradients, total = gradient_audit(model, batch)
    save(out / "gradients.json", gradients)
    print("gradients", gradients["counts"], "actor", gradients["shared_actor_norm"],
          "critic", gradients["shared_critic_norm"], flush=True)
    report = {}
    original_flatten, original_weight = rl.flatten, rl.VALUE_WEIGHT
    for mode in ("joint", "actor_only", "critic_only"):
        candidate = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=rl.LEARNING_RATE, weight_decay=0)
        captured = []
        original_step = optimizer.step
        def step(*args, **kwargs):
            if not captured:
                params = list(candidate.parameters())
                captured.append(flat_grad(params, [p.grad for p in params]))
            return original_step(*args, **kwargs)
        optimizer.step = step
        def zero_advantages(eps):
            batch_data = original_flatten(eps)
            for entry in batch_data["entries"]:
                entry["advantage"] = 0.0
            return batch_data
        try:
            rl.VALUE_WEIGHT = 0.0 if mode == "actor_only" else original_weight
            rl.flatten = zero_advantages if mode == "critic_only" else original_flatten
            metrics = rl.update(candidate, optimizer, batch)
        finally:
            rl.flatten, rl.VALUE_WEIGHT = original_flatten, original_weight
        result = dict(update=metrics, path_changes=path_change(baseline, path_scores(candidate, fixed_paths)))
        if mode == "joint":
            expected = total * min(1.0, 5.0 / (float(total.norm()) + 1e-6))
            error = float((expected-captured[0]).norm() / expected.norm().clamp_min(1e-12))
            result["gradient_sum_vs_real_update_relative_error"] = error
            assert error < 1e-4, error
        result["closed_loop"] = summary(collect(executor, candidate, tracking()))
        report[mode] = result
        torch.save(candidate.state_dict(), out / f"one_update_{mode}.pt")
        save(out / "one_update.json", report)
        print("one update", mode, result["closed_loop"]["success"], flush=True)


def fit(out, executor, balanced=False):
    episodes = [ep for ep in torch.load(out / "replay_initial.pt", weights_only=False) if ep["success"]]
    if not episodes:
        save(out / "fit.json", {"status": "no_successful_demonstrations"})
        return
    instances = [x for x in tracking() if x.name in {e["instance"] for e in episodes}]
    results = json.loads((out / "fit.json").read_text()) if balanced else {}
    modes = ("scratch_episode_mean",) if balanced else ("warm", "scratch")
    for mode in modes:
        torch.manual_seed(0)
        model = load_policy(SOURCE / "checkpoints/initial.pt") if mode == "warm" else rl.JointPolicy(324)
        optimizer = torch.optim.AdamW(model.parameters(), lr=rl.LEARNING_RATE, weight_decay=0)
        count = sum(len(e["decisions"]) for e in episodes)
        curve = []
        for epoch in range(31):
            if epoch in (0, 10, 20, 30):
                scores = path_scores(model, episodes)
                point = dict(epoch=epoch, paths=[{k: v for k, v in r.items() if k != "joint_logs"} for r in scores],
                             closed_loop=summary(collect(executor, model, instances)))
                curve.append(point)
                results[mode] = dict(decisions=count, curve=curve)
                save(out / "fit.json", results)
                print("fit", mode, epoch, [round(s["accuracy"], 3) for s in scores],
                      point["closed_loop"]["success"], flush=True)
            if epoch == 30:
                break
            optimizer.zero_grad()
            for ep in episodes:
                losses = [-model.evaluate(hidden, *d["action"])[0]
                          for d, hidden in rl.decision_states(model, ep)]
                denominator = len(episodes)*len(losses) if balanced else count
                (torch.stack(losses).sum()/denominator).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        torch.save(model.state_dict(), out / f"fit_{mode}.pt")


def main():
    global SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "replay", "audit", "fit", "fit_balanced"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--artifact-root", type=Path, default=rl.ROOT,
                        help="Use the original repository artifacts when executing frozen code")
    args = parser.parse_args()
    SOURCE = args.artifact_root / "outputs/random_start_time_budget_v3_coverage256"
    torch.set_num_threads(1)
    torch.manual_seed(0)
    started = time.time()
    if args.stage == "prepare":
        prepare(args.out)
    else:
        manifest = json.loads((args.out / "manifest.json").read_text())
        if manifest["environment"] != rl.E.EnvConfig().as_dict():
            raise ValueError("Diagnostic environment changed; run the frozen code with --artifact-root")
        with concurrent.futures.ProcessPoolExecutor(max_workers=8,
                mp_context=mp.get_context("spawn"), initializer=runner.worker_init) as executor:
            if args.stage == "fit_balanced":
                fit(args.out, executor, balanced=True)
            else:
                {"replay": replay, "audit": audit, "fit": fit}[args.stage](args.out, executor)
    save(args.out / f"timing_{args.stage}.json", dict(wall_seconds=time.time()-started))


if __name__ == "__main__":
    main()
