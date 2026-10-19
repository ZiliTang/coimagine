"""S0 verification for the bounded stability comparison.

Run once after the code is frozen and before the two pilot arms start. It checks, from actual
forward passes and actual closed-loop rollouts rather than assumptions:

* the independent-v2 train/validation/test panels are physically disjoint;
* the warm start is the checkpoint named by the diagnosis manifest;
* both arms have the same parameters and identical forward probabilities and values;
* the deterministic original-8 closed-loop results are identical before training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import rl
import long_train as lt


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else lt.STABILITY_ROOT
    panels, forward_check = lt.setup_stability_root(root)
    weights = torch.load(lt.WARM_START, map_location="cpu", weights_only=True)
    observation_dim = len(rl.E.RandomStartEnv(panels["train"][0]).observation_fields())
    arms = {}
    for name, flag in lt.STABILITY_ARMS.items():
        policy = rl.JointPolicy(observation_dim, hidden=rl.HIDDEN, critic_shared_grad=flag)
        policy.load_state_dict(weights)
        rows = []
        for instance in panels["tracking"]:
            episode = rl.rollout(rl.E.RandomStartEnv(instance), policy, stochastic=False)
            rows.append(dict(instance=instance.name, success=bool(episode["success"]),
                             periods=int(episode["periods"]),
                             decisions=len(episode["decisions"]),
                             logp=[float(decision["logp"]) for decision in episode["decisions"]],
                             value=[float(decision["value"]) for decision in episode["decisions"]]))
        arms[name] = dict(critic_mode=policy.contract()["critic_mode"], rows=rows)
    differences = []
    for a, b in zip(arms["shared"]["rows"], arms["stopped"]["rows"]):
        assert a["instance"] == b["instance"]
        assert a["success"] == b["success"]
        assert a["periods"] == b["periods"]
        assert len(a["logp"]) == len(b["logp"])
        for x, y, u, v in zip(a["logp"], b["logp"], a["value"], b["value"]):
            differences.append(max(abs(x - y), abs(u - v)))
    report = dict(
        forward_check=forward_check,
        tracking_instances=[instance.name for instance in panels["tracking"]],
        shared_success=[row["success"] for row in arms["shared"]["rows"]],
        stopped_success=[row["success"] for row in arms["stopped"]["rows"]],
        shared_periods=[row["periods"] for row in arms["shared"]["rows"]],
        stopped_periods=[row["periods"] for row in arms["stopped"]["rows"]],
        max_abs_logp_or_value_difference=float(max(differences)) if differences else 0.0,
        equal=bool(max(differences) < 1e-9) if differences else True,
        note="deterministic argmax rollouts; identical weights and identical forward reads, "
             "so any drift would indicate a bug in the arm switch or in the frozen panels")
    (root / "setup").mkdir(parents=True, exist_ok=True)
    lt.atomic_json(root / "setup" / "initial_arm_equality.json", report)
    print(json.dumps({key: report[key] for key in
                      ("shared_success", "stopped_success", "max_abs_logp_or_value_difference",
                       "equal")}, indent=1))
    if not report["equal"]:
        raise SystemExit("arm initial behaviour differs before training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
