"""S1 audit reconstruction: save one raw batch and the models before/after one PPO update.

The warm start, the seed-0 traversal and the first update are deterministic, so the first
pilot batch can be rebuilt exactly. This provides the task-book S1 evidence that was not
embedded in the pilot logs: the raw batch, update-before/after weights, a parallel-vs-learner
probability/value replay check, complete-failure period counts and an A/B update comparison.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENT = Path(__file__).resolve().parent
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
import long_train as lt  # noqa: E402
import rl  # noqa: E402


def recorded_update(pilot, index=1):
    rows = [json.loads(line) for line in (pilot / "updates.jsonl").read_text().splitlines()]
    return next(row for row in rows if row["update"] == index)


def build_jobs(panels, state, hidden, environment):
    rng = np.random.default_rng(np.random.SeedSequence([0, 1]))
    chosen = lt.balanced_indices(0, 1, len(panels["train"]), 8)
    jobs = [(panels["train"][index], state, hidden, int(rng.integers(2 ** 31)), False,
             environment, False) for index in chosen]
    return chosen, jobs, rng


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--root", type=Path, default=lt.STABILITY_ROOT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    panels = lt.build_stability_panels(verify=True)
    environment = rl.E.EnvConfig()
    out = args.root / "pilot_audit"
    out.mkdir(parents=True, exist_ok=True)
    weights = torch.load(lt.WARM_START, map_location="cpu", weights_only=True)
    observation_dim = len(rl.E.RandomStartEnv(panels["train"][0]).observation_fields())
    reference = rl.JointPolicy(observation_dim, hidden=rl.HIDDEN, critic_shared_grad=True)
    reference.load_state_dict(weights)
    chosen, jobs, _rng = build_jobs(panels, lt.snapshot(reference), reference.hidden,
                                    environment)
    report = {"update": 1, "chosen_train_indices": chosen,
              "instance_names": [job[0].name for job in jobs],
              "behaviour_seeds": [job[3] for job in jobs],
              "source": "reconstructed from the frozen warm start and the recorded seed-0 "
                        "traversal; the pilot update 1 used the same batch"}
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn"),
            initializer=lt.worker_init) as executor:
        episodes = lt.collect(executor, jobs)
        # parallel-vs-learner probability/value replay, before any update
        logp_error, value_error = 0.0, 0.0
        with torch.no_grad():
            for episode in episodes:
                for decision, hidden in rl.decision_states(reference, episode):
                    logp, _ = reference.evaluate(hidden, *decision["action"])
                    logp_error = max(logp_error, abs(logp.item() - decision["logp"]))
                    value_error = max(value_error, abs(reference.value(hidden).item()
                                                       - decision["value"]))
        report["parallel_replay"] = dict(logp_max_error=logp_error,
                                         value_max_error=value_error,
                                         ok=bool(max(logp_error, value_error) < 1e-5))
        failure_periods = sorted({int(episode["periods"]) for episode in episodes
                                  if not episode["success"]})
        report["failure_periods"] = failure_periods
        report["complete_failures_are_1800"] = all(periods == 1800
                                                   for periods in failure_periods)
        report["batch"] = dict(episodes=len(episodes),
                               decisions=sum(len(episode["decisions"]) for episode in episodes),
                               success=float(np.mean([episode["success"] for episode in episodes])),
                               return_mean=float(np.mean([episode["return"]
                                                          for episode in episodes])))
        torch.save({"episodes": [
            {**{key: episode[key] for key in
                ("instance", "behaviour_seed", "success", "termination", "periods",
                 "skills_used", "final_goal_distance", "return", "reward_components",
                 "exits", "short_decisions", "short_time_fraction")},
             "decisions": [{key: decision[key] for key in
                            ("skill_index", "action", "logp", "value", "reward", "periods",
                             "prefix_before", "return_to_go", "terminated", "success",
                             "truncated")} | {"boundary_frame": decision["boundary_frame"].numpy(),
                                              "observations": decision["observations"].numpy()}
                           for decision in episode["decisions"]]}
            for episode in episodes],
            "jobs": [(job[0].name, job[3]) for job in jobs]}, out / "batch_update_00001.pt")
        comparisons = {}
        for arm, flag in lt.STABILITY_ARMS.items():
            policy = rl.JointPolicy(observation_dim, hidden=rl.HIDDEN,
                                    critic_shared_grad=flag)
            policy.load_state_dict(weights)
            torch.save(policy.state_dict(), out / f"{arm}_before_update_00001.pt")
            optimizer = torch.optim.AdamW(policy.parameters(), lr=rl.LEARNING_RATE,
                                          weight_decay=0)
            metrics = rl.update(policy, optimizer, episodes)
            torch.save(policy.state_dict(), out / f"{arm}_after_update_00001.pt")
            recorded = recorded_update(args.root / ("pilot_A" if arm == "shared" else "pilot_B"))
            comparisons[arm] = {
                "recomputed": {key: metrics[key] for key in
                               ("decisions", "grad_norm_last", "ratio_mean", "entropy_mean",
                                "value_error_mean", "policy_change_mean")},
                "recorded": {key: recorded.get(key) for key in
                             ("decisions", "grad_norm_last", "ratio_mean", "entropy_mean",
                              "value_error_mean", "policy_change_mean")},
                "decisions_equal": int(metrics["decisions"]) == int(recorded["decisions"]),
                "weights_after_sha256": lt.snapshot_hash(lt.snapshot(policy)),
            }
            comparisons[arm]["max_abs_metric_difference"] = max(
                abs(float(metrics[key]) - float(recorded[key]))
                for key in ("grad_norm_last", "ratio_mean", "entropy_mean",
                            "value_error_mean", "policy_change_mean")
                if recorded.get(key) is not None)
        report["arms"] = comparisons
        before_a = torch.load(out / "shared_before_update_00001.pt", weights_only=True)
        before_b = torch.load(out / "stopped_before_update_00001.pt", weights_only=True)
        a = torch.load(out / "shared_after_update_00001.pt", weights_only=True)
        b = torch.load(out / "stopped_after_update_00001.pt", weights_only=True)
        report["before_update_identical"] = all(
            torch.equal(before_a[key], before_b[key]) for key in before_a)
        report["after_update_tensors_differing"] = [
            key for key in a if not torch.equal(a[key], b[key])]
        # The decisive intervention on the *same real batch*: zero every actor advantage and
        # take one real PPO update. Then the only surviving gradient is the value loss.
        original_flatten = rl.flatten

        def zero_advantages(episodes):
            batch = original_flatten(episodes)
            for entry in batch["entries"]:
                entry["advantage"] = 0.0
            return batch

        rl.flatten = zero_advantages
        zero_report = {}
        try:
            for arm, flag in lt.STABILITY_ARMS.items():
                policy = rl.JointPolicy(observation_dim, hidden=rl.HIDDEN,
                                        critic_shared_grad=flag)
                policy.load_state_dict(before_a)
                optimizer = torch.optim.AdamW(policy.parameters(), lr=rl.LEARNING_RATE,
                                              weight_decay=0)
                rl.update(policy, optimizer, episodes)
                after = policy.state_dict()
                changed = [key for key in after if not torch.equal(after[key], before_a[key])]
                zero_report[arm] = dict(
                    changed_tensors=changed,
                    shared_changed=any(key.startswith(("encoder.", "gru.")) for key in changed),
                    actor_heads_changed=any(
                        key.startswith(("head_first.", "head_second.", "embed."))
                        for key in changed),
                    value_head_changed=any(key.startswith("value_head.") for key in changed))
                torch.save(after, out / f"zero_advantage_{arm}_after_update_00001.pt")
        finally:
            rl.flatten = original_flatten
        report["zero_actor_advantage_one_step"] = zero_report
    lt.atomic_json(out / "summary.json", report)
    print(json.dumps(report, indent=1, default=str)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
