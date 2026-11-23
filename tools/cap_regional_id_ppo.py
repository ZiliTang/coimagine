"""On-policy clipped PPO from the fixed supervised ten-ID actor."""

import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from experiments.planar_tasks.authoring_v2 import cap_trace_value_ppo as prior
from tools import cap_regional_id_actor as base

study = base.study
OUT = base.OUT.parent / "cap_regional_id_ppo_20261120_01"
DOCS = base.DOCS.parent / "2026-11-20_regional_id_ppo"
POOL = base.OUT.parent / "cap_trace_value_full_20261118_02/panels/online_training.json"
ROUNDS = 40
PER_ROUND = 50
MONITOR_EVERY = 10
POLICY = None
TEMP = 1.0
PROPOSALS = study.dq.Proposals


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = base.ExpertActor()
        self.actor.load_state_dict(
            torch.load(base.OUT / "fixed.pt", map_location="cpu", weights_only=True)
        )
        self.critic = nn.Sequential(nn.Linear(2 * len(self.actor.mean), 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.constant_(self.critic[-1].bias, 0.5)

    def forward(self, x):
        z = self.actor.encode(x)
        legal = self.actor.legal(x)
        safe = legal.clone()
        safe[:, 0] |= ~legal.any(1)
        logits = (self.actor.net(z) / TEMP).masked_fill(~safe, -torch.inf)
        return Categorical(logits=logits), self.critic(z).squeeze(-1), legal


def environment(fixture):
    dq = study.dq
    return dq.fast_runtime.ScopedEnvironment(
        fixture,
        dq.study.CFG,
        fast_feedback=True,
        record_details=False,
        history_factory=dq.fast_runtime.AuditedCachedHistory,
    )


@torch.inference_mode()
def calibration(pool):
    logits = []
    alternative = []
    for fixture in pool.values():
        env = environment(fixture)
        try:
            x, _, _, _, _, p = PROPOSALS().panel(env.history)
            raw = POLICY.actor(torch.from_numpy(x[None]))[0]
            if not raw.isfinite().any():
                continue
            greedy = int(raw.argmax())
            different = torch.tensor([a != p[greedy] for a in p]) & raw.isfinite()
            if different.any():
                logits.append(raw)
                alternative.append(different)
        finally:
            env.close_reads()
            study.dq.fast_runtime.ACTIVE = None
    scores = torch.stack(logits)
    mask = torch.stack(alternative)

    def rate(t):
        return float(((scores / t).softmax(1) * mask).sum(1).mean())

    low, high = 1.0, 1.0
    if rate(1) < 0.15:
        while rate(high) < 0.15 and high < 128:
            high *= 2
        for _ in range(30):
            mid = (low + high) / 2
            if rate(mid) < 0.15:
                low = mid
            else:
                high = mid
    return dict(
        temperature=high,
        raw_alternative=rate(1),
        scaled_alternative=rate(high),
        multi_action_starts=len(logits),
    )


@torch.inference_mode()
def episode(job):
    cid, fixture, seed, directory, training = job
    env = environment(fixture)
    proposals = PROPOSALS()
    rng = np.random.default_rng(seed)
    states = []
    choices = []
    logps = []
    values = []
    entropies = []
    trajectory = []
    alternate = eligible = switches = distinct_switches = fallback = 0
    previous = first = None
    try:
        for step in range(300):
            x, _, _, valid, pool, raw = proposals.panel(env.history)
            dist, v, mask = POLICY(torch.from_numpy(x[None]))
            logits = dist.logits[0]
            greedy = int(logits.argmax())
            if training:
                p = dist.probs[0].double().numpy()
                i = int(rng.choice(len(p), p=p / p.sum()))
                states.append(x)
                choices.append(i)
                values.append(float(v))
                logps.append(float(logits[i]))
                entropies.append(float(dist.entropy()))
            else:
                i = greedy
            if mask.any():
                assert mask[0, i]
                cap = i
                action = raw[i]
                eligible += int(valid.sum() > 1)
                alternate += int(action != raw[greedy])
                changed = previous is not None and previous != cap
                switches += int(changed)
                distinct_switches += int(changed and (not mask[0, previous] or raw[previous] != action))
            else:
                cap = None
                action = pool[0]
                fallback += 1
            if step == 0:
                first = cap
            if not training:
                trajectory.append(cap)
            previous = cap
            env.step(action, remaining_ticks=1800 - env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods >= 1800:
                break
        success = int(env.env.success)
        n = step + 1
        row = dict(
            case_id=cid,
            success=success,
            ticks=env.env.total_periods,
            boundaries=n,
            history=env.history.sha256,
            initial_cap=first,
            switches=switches,
            distinct_switches=distinct_switches,
            fallback=fallback,
            alternate=alternate,
            eligible=eligible,
        )
        if training:
            v = np.asarray(values, np.float32)
            reward = np.zeros(n, np.float32)
            reward[-1] = success
            adv = prior.advantages(reward, v)
            shard = directory / (cid + ".npz")
            np.savez(
                shard,
                x=np.stack(states),
                chosen=np.asarray(choices),
                old_logp=np.asarray(logps, np.float32),
                adv=adv,
                returns=adv + v,
            )
            row.update(entropy=float(np.mean(entropies)), shard=str(shard))
        else:
            row["selected_caps"] = trajectory
        return row
    finally:
        env.close_reads()
        study.dq.fast_runtime.ACTIVE = None


def collect(fixtures, items, directory, training):
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [(cid, fixtures[cid], seed, directory, training) for cid, seed in items]
    with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context("fork")) as pool:
        rows = list(pool.map(episode, jobs))
    assert [r["case_id"] for r in rows] == [k for k, _ in items]
    return rows


def update(rows, optimizer, rng):
    arrays = []
    for r in rows:
        with np.load(r["shard"]) as d:
            arrays.append({k: d[k] for k in d.files})
    b = {k: torch.from_numpy(np.concatenate([d[k] for d in arrays])) for k in arrays[0]}
    del arrays
    return update_batch(POLICY, b, optimizer, rng)


def update_batch(policy, b, optimizer, rng, *, meter=None):
    adv = b["adv"]
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    chunks = torch.arange(len(adv)).split(128)

    def delta(ix):
        dist, _, _ = policy(b["x"][ix])
        return dist.log_prob(b["chosen"][ix]) - b["old_logp"][ix]

    with torch.no_grad():
        error = max(float(delta(ix).abs().max()) for ix in chunks)
    if error > 1e-3:
        raise ValueError(f"on-policy log probability mismatch {error}")
    stats = []
    for epoch in range(prior.EPOCHS):
        for ix in torch.from_numpy(rng.permutation(len(adv))).split(128):
            dist, v, legal = policy(b["x"][ix])
            ratio = (dist.log_prob(b["chosen"][ix]) - b["old_logp"][ix]).exp()
            eligible = legal.any(1)
            objective = torch.minimum(ratio * adv[ix], ratio.clamp(0.8, 1.2) * adv[ix])
            actor = -(objective * eligible).sum() / eligible.sum().clamp_min(1)
            if meter is not None:
                meter.before(ix, -objective * eligible / eligible.sum().clamp_min(1), policy)
            critic = (v - b["returns"][ix]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            (actor + 0.5 * critic).backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            if meter is not None:
                meter.after(policy)
            stats.append(
                (
                    float(actor.detach()),
                    float(critic.detach()),
                    float(((ratio - 1).abs() > 0.2).float().mean()),
                )
            )
        with torch.no_grad():
            d = torch.cat([delta(ix) for ix in chunks])
            kl = float((d.expm1() - d).mean())
        if kl > prior.TARGET_KL:
            break
    a, c, f = np.mean(stats, axis=0)
    return dict(
        actor_loss=a,
        critic_loss=c,
        clip_fraction=f,
        kl=kl,
        epochs=epoch + 1,
        updates=len(stats),
        old_logp_error=error,
    )


def save_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def prepare(pool):
    dev = study.read(study.OUT / study.ext.DEV)
    held = study.read(study.OUT / "held.json")
    frozen = study.read(study.OUT / "frozen.json")
    assert (
        study.ext.old.digest(dev) == frozen["development"]
        and study.ext.old.digest(held) == frozen["held"]
    )
    hashes = {study.ext.input_hash(f) for f in pool.values()}
    assert len(pool) == len(hashes) == 200 and not set(pool).intersection(dev | held)
    assert hashes.isdisjoint(study.ext.input_hash(f) for f in (dev | held).values())
    rng = np.random.default_rng(2026112017)
    keys = sorted(pool)
    schedule = [
        [(str(k), int(rng.integers(2**31))) for k in rng.choice(keys, PER_ROUND, replace=False)]
        for _ in range(ROUNDS)
    ]
    monitor = []
    for n in sorted({len(f["movable_objects"]) for f in pool.values()}):
        group = [k for k in keys if len(pool[k]["movable_objects"]) == n]
        monitor.extend(rng.choice(group, 10, replace=False).tolist())
    assert len(monitor) == 40
    study.save(OUT / "schedule.json", schedule)
    study.save(OUT / "monitor_ids.json", monitor)
    return schedule, monitor, dev, held


def run():
    global POLICY, TEMP, PROPOSALS
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    manifest = study.read(POOL.parent / "manifest.json")
    assert study.sha(POOL) == manifest["files"][POOL.name]
    cases = study.read(POOL)["cases"]
    assert all(study.ext.input_hash(c["fixture"]) == c["content_sha256"] for c in cases)
    pool = {c["case_id"]: c["fixture"] for c in cases}
    schedule, monitor, dev, held = prepare(pool)
    init = study.read(base.DOCS / "results.json")["sha256"]["fixed"]
    assert study.sha(base.OUT / "fixed.pt") == init
    backend = OUT / "cpp_backend.json"
    if backend.exists() and study.read(backend)["use_cpp"]:
        from tools import cap_regional_cpp as cpp

        _, manifest = cpp.load()
        audit = study.read(backend)
        if audit["key"] != manifest["key"] or audit["binary"] != manifest["binary"]:
            raise ValueError("CAP backend changed since validation")
        PROPOSALS = cpp.Proposals
    else:
        study.init_models(study.read(study.OUT / "frozen.json")["sids"])
        PROPOSALS = study.dq.Proposals
    POLICY = Policy()
    optimizer = torch.optim.Adam(
        [
            dict(params=POLICY.actor.parameters(), lr=3e-5),
            dict(params=POLICY.critic.parameters(), lr=3e-4),
        ]
    )
    protocol_file = OUT / "protocol.json"
    if not protocol_file.exists():
        cal = calibration(pool)
        study.save(
            protocol_file,
            dict(
                parent=init,
                actor="unchanged supervised fixed 24000 ten-ID model",
                seed=17,
                gamma=1,
                gae=0.95,
                clip=0.2,
                epochs=4,
                kl=0.015,
                batch=128,
                actor_lr=3e-5,
                critic_lr=3e-4,
                entropy_bonus=0,
                terminal_success_reward=1,
                other_reward=0,
                time_or_decision_limit_terminal=True,
                temperature_calibration=cal,
                online_pool=study.sha(POOL),
                schedule=study.sha(OUT / "schedule.json"),
                rounds=ROUNDS,
                episodes=ROUNDS * PER_ROUND,
                monitor="40 online-training rooms, greedy every 500, no selection",
                primary="fixed final greedy minus frozen initial on 200 natural held rooms, other panels secondary",
            ),
        )
    protocol = study.read(protocol_file)
    assert protocol["parent"] == init and protocol["schedule"] == study.sha(OUT / "schedule.json")
    TEMP = protocol["temperature_calibration"]["temperature"]
    print("PROTOCOL", protocol, flush=True)
    current = OUT / "latest.pt"
    hist = []
    if current.is_file():
        state = torch.load(current, map_location="cpu", weights_only=False)
        POLICY.load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        hist = state["history"]
        rng.bit_generator.state = state["rng"]

    def monitor_round(r):
        dest = OUT / "monitor" / f"r{r:02d}"
        if not (dest / "results.json").exists():
            study.save(OUT / "stage.json", dict(stage="monitor", round=r))
            study.save(dest / "results.json", collect(pool, [(k, 0) for k in monitor], dest, False))

    if not hist:
        monitor_round(0)
    elif len(hist) % MONITOR_EVERY == 0:
        monitor_round(len(hist))
    study.save(OUT / "history.json", hist)
    temp = OUT / "current_rollouts"
    if temp.exists():
        shutil.rmtree(temp)
    for r in range(len(hist) + 1, ROUNDS + 1):
        study.save(OUT / "stage.json", dict(stage="collect", round=r))
        rows = collect(pool, schedule[r - 1], temp, True)
        study.save(OUT / "stage.json", dict(stage="fit", round=r))
        metrics = update(rows, optimizer, rng)
        lines = [{k: v for k, v in x.items() if k != "shard"} for x in rows]
        study.save(OUT / f"round_{r:02d}.json", lines)
        n = sum(x["boundaries"] for x in rows)
        hist.append(
            dict(
                round=r,
                episodes=r * PER_ROUND,
                success=sum(x["success"] for x in rows),
                boundaries=n,
                entropy=sum(x["entropy"] * x["boundaries"] for x in rows) / n,
                alternate=sum(x["alternate"] for x in rows),
                eligible=sum(x["eligible"] for x in rows),
                **metrics,
            )
        )
        save_checkpoint(
            current,
            dict(
                policy=POLICY.state_dict(),
                optimizer=optimizer.state_dict(),
                history=hist,
                rng=rng.bit_generator.state,
            ),
        )
        study.save(OUT / "history.json", hist)
        print("PPO", hist[-1], flush=True)
        shutil.rmtree(temp)
        if r % MONITOR_EVERY == 0:
            monitor_round(r)
    save_checkpoint(OUT / "final_actor.pt", POLICY.actor.state_dict())
    for panel, n in base.PANELS.items():
        dest = OUT / "evaluation" / panel
        if (dest / "results.json").exists():
            continue
        fixtures = dev if panel in ("dev", "aug") else held
        ids = [(k, 0) for k in sorted(fixtures) if k.startswith(panel + "_")]
        assert len(ids) == n
        study.save(OUT / "stage.json", dict(stage="evaluate", panel=panel, round=ROUNDS))
        rows = collect(fixtures, ids, dest, False)
        study.save(dest / "results.json", rows)
        print("EVAL", panel, len(rows), sum(r["success"] for r in rows), flush=True)
    from tools.cap_regional_id_ppo_report import report

    report()


if __name__ == "__main__":
    run()
