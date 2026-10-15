"""Three-way comparison on the same frozen roots, with a paired test.

The trained and random arms see **the same initial states**, so the comparison is paired; two
independent proportions would discard that. This script also evaluates the **pre-update** policy
reconstructed from the same seed, which separates an initialisation effect from learning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_repair_optionrl_v1.option_env import MENU, ObservationSpec  # noqa: E402
from experiments.night_repair2_v1.collect import load_policy  # noqa: E402
from experiments.night_repair2_v1.ppo import collect, paired_comparison, per_root, summarise  # noqa: E402
from experiments.night_repair2_v1.rl import RecurrentPolicy  # noqa: E402
from coimagine.local_contact_option.roots import build_roots  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(
        ROOT / "outputs/night_repair2_v1/option_rl/policy_best_seed0.pt"))
    parser.add_argument("--goal-distance", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "outputs/night_repair2_v1/option_rl"))
    args = parser.parse_args()
    pool = build_roots("test", 24)
    validation, test = pool[:64], pool[64:128]
    trained = load_policy(Path(args.checkpoint))
    # the pre-update policy: the same construction the trainer used, before any gradient step
    torch.manual_seed(args.seed)
    initial = RecurrentPolicy(ObservationSpec().width, len(MENU), hidden=128)
    initial.eval()
    arms = {"random": None, "initial_policy": initial, "trained": trained}
    runs: Dict[str, Dict[str, List]] = {}
    for name, policy in arms.items():
        runs[name] = {
            "validation": collect(policy, validation, args.goal_distance, seed=args.seed + 7,
                                  deterministic=policy is not None),
            "test": collect(policy, test, args.goal_distance, seed=args.seed + 8,
                            deterministic=policy is not None)}
    report: Dict[str, object] = {
        "checkpoint": args.checkpoint, "goal_distance_m": args.goal_distance,
        "note": "all three arms run on the same frozen root sets; the comparison is paired by root",
        "summary": {name: {split: summarise(episodes) for split, episodes in splits.items()}
                    for name, splits in runs.items()},
        "paired": {
            "trained_vs_random": {split: paired_comparison(runs["trained"][split],
                                                           runs["random"][split])
                                  for split in ("validation", "test")},
            "initial_vs_random": {split: paired_comparison(runs["initial_policy"][split],
                                                           runs["random"][split])
                                  for split in ("validation", "test")},
            "trained_vs_initial": {split: paired_comparison(runs["trained"][split],
                                                            runs["initial_policy"][split])
                                   for split in ("validation", "test")}},
        "per_root": {f"{name}_{split}": per_root(episodes)
                     for name, splits in runs.items() for split, episodes in splits.items()},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "policy_comparison.json").write_text(json.dumps(report, indent=1, default=float))
    for name in ("random", "initial_policy", "trained"):
        for split in ("validation", "test"):
            entry = report["summary"][name][split]
            print(f"  {name:15s} {split:11s} succ={entry['success_rate']:.4f} "
                  f"progress_towards={entry['mean_progress_towards_goal_m']:+.4f} "
                  f"end={entry['mean_end_distance_m']:.4f}")
    for key, splits in report["paired"].items():
        for split, value in splits.items():
            print(f"  {key:20s} {split:11s} shared={value['shared_roots']} "
                  f"trained_only={value['trained_only_successes']} "
                  f"random_only={value['random_only_successes']} "
                  f"McNemar_p={value['mcnemar_exact_p']:.4f}")
    print("COMPARE_POLICIES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
