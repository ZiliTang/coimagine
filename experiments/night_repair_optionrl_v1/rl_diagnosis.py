"""Why does this task reward random exploration as much as a trained policy? (task book 5.3)

Reports, per scene class and split, the success rate of the trained policy beside a uniform legal
policy under identical task rules, and the distribution of the outcome reasons. The pooled number is
never used on its own, because the scene mix dominates it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_repair_optionrl_v1.option_env import (  # noqa: E402
    MENU,
    OptionRLConfig,
    OptionRLEnv,
    default_goal,
)
from experiments.night_repair_optionrl_v1.ppo import legal_actions  # noqa: E402
from coimagine.local_contact_option.roots import build_roots  # noqa: E402


def run(roots, goal_distance: float, policy, *, seed: int, max_skills: int = 12,
        tolerance: float = 0.20) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    hidden = policy.initial_hidden(1) if policy is not None else None
    scenes: Dict[str, Dict[str, object]] = {}
    for root in roots:
        env = OptionRLEnv(OptionRLConfig(seed=int(rng.integers(0, 1 << 30)),
                                         goal_tolerance=tolerance, max_skills=max_skills))
        goal = default_goal(root, goal_distance)
        observation = env.reset(root, goal)
        allowed = legal_actions(root)
        hidden = policy.initial_hidden(1) if policy is not None else None
        success = False
        used = 0
        for _ in range(max_skills):
            if policy is not None:
                with torch.no_grad():
                    logits, _value, hidden = policy.step(torch.tensor(observation[None, :]), hidden)
                    mask = torch.full((len(MENU),), float("-inf"))
                    mask[allowed] = 0.0
                    action = int(torch.argmax(logits[0] + mask))
            else:
                action = int(allowed[rng.integers(0, len(allowed))])
            result = env.step(action)
            used += 1
            if result.terminated:
                success = True
                break
            if result.truncated:
                break
            observation = result.observation
        scene = scenes.setdefault(root.interaction_class,
                                  {"episodes": 0, "successes": 0, "closest_m": []})
        scene["episodes"] += 1
        scene["successes"] += int(success)
        scene["closest_m"].append(float(np.hypot(*(env.privileged_object()[:2] - goal))))
    for scene in scenes.values():
        scene["success_rate"] = scene["successes"] / max(1, scene["episodes"])
        scene["median_closest_m"] = float(np.median(scene["closest_m"]))
        scene.pop("closest_m")
    total = sum(s["episodes"] for s in scenes.values())
    succeeded = sum(s["successes"] for s in scenes.values())
    return {"episodes": total, "successes": succeeded,
            "success_rate": succeeded / max(1, total), "per_scene": scenes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(ROOT / "outputs/night_repair_optionrl_v1/option_rl/policy_seed0.pt"))
    parser.add_argument("--goal-distance", type=float, default=0.45)
    parser.add_argument("--tolerance", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "outputs/night_repair_optionrl_v1/option_rl"))
    args = parser.parse_args()
    pool = build_roots("test", 24)
    validation, test = pool[:64], pool[64:128]
    policy = None
    path = Path(args.policy)
    if path.exists():
        from experiments.night_repair_optionrl_v1.collect import load_policy
        policy = load_policy(path)
    report = {
        "goal_distance_m": args.goal_distance, "tolerance_m": args.tolerance,
        "policy_loaded": policy is not None, "policy": str(path),
        "validation": {"trained": run(validation, args.goal_distance, policy, seed=args.seed)
                       if policy is not None else None,
                       "random": run(validation, args.goal_distance, None, seed=args.seed)},
        "test": {"trained": run(test, args.goal_distance, policy, seed=args.seed + 1)
                 if policy is not None else None,
                 "random": run(test, args.goal_distance, None, seed=args.seed + 1)}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "rl_diagnosis.json").write_text(json.dumps(report, indent=1, default=float))
    for split in ("validation", "test"):
        for arm in ("trained", "random"):
            entry = report[split][arm]
            if entry is None:
                continue
            print(f"{split:10s} {arm:8s} success={entry['success_rate']:.4f} "
                  f"({entry['successes']}/{entry['episodes']})")
            for scene, values in sorted(entry["per_scene"].items()):
                print(f"    {scene:16s} n={values['episodes']:3d} "
                      f"success={values['success_rate']:.3f} "
                      f"median_closest={values['median_closest_m']:.3f} m")
    print("RL_DIAGNOSIS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
