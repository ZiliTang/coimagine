"""Recompute the frozen phase-C payoffs with the CONTRACT score, from the saved numbers only.

The first run applied an extra ``max(0.0, ...)`` to ``progress``. CONTRACT section 3.3 defines

    progress = min(1, (d_start - d_end) / max(d_start, 1e-6))

with **no** lower clamp, so an action that pushes the object *away* from the goal must earn negative
progress. That clamp was an unauthorised change to the score, and it is what made the task-signal
look identically zero. This script re-derives everything from the distances, durations and action
costs already stored in ``frozen_c_result.json`` -- no re-simulation -- and reports, separately:

* whether **any** action produces positive progress in either world (a task-signal question);
* whether the two worlds **need different actions** (the near-optimal-set / regret question);
* the difference between the clamped and contract-conform scores.

    python experiments/option_prediction_mechanism_v1/phase_c_frozen/recompute_frozen_c.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/option_prediction_mechanism_v1/phase_c_frozen/frozen_c_result.json"


def progress_contract(start: float, end: float) -> float:
    """CONTRACT 3.3 verbatim: clamp at 1, never at 0."""

    return min(1.0, (start - end) / max(start, 1e-6))


def progress_clamped(start: float, end: float) -> float:
    """What the first run actually used: an extra lower clamp that was not in the contract."""

    return min(1.0, max(0.0, (start - end) / max(start, 1e-6)))


def score(completion: float, progress: float, duration: float, action_cost: float) -> float:
    return completion + 0.5 * progress - 0.05 * duration - 0.10 * action_cost


def analyse(runs: Dict[str, List[dict]], delta: float, clamp: bool) -> dict:
    progress = progress_clamped if clamp else progress_contract
    names = [run["command"] for run in runs["A"]]
    q = {"A": {}, "B": {}}
    task = {"A": {}, "B": {}}
    for world in ("A", "B"):
        for run in runs[world]:
            value = progress(run["start_distance"], run["end_distance"])
            completion = 1.0 if run["success"] else 0.0
            q[world][run["command"]] = score(completion, value, run["duration"],
                                             run["action_cost"])
            task[world][run["command"]] = completion + 0.5 * value
    best = {world: max(q[world].values()) for world in ("A", "B")}
    near = {world: sorted(name for name in names if best[world] - q[world][name] <= delta + 1e-12)
            for world in ("A", "B")}
    argmax = {world: max(names, key=lambda name: q[world][name]) for world in ("A", "B")}
    regret = max(best["A"] - q["A"][argmax["B"]], best["B"] - q["B"][argmax["A"]])
    all_knowing = 0.5 * (best["A"] + best["B"])
    common = max(0.5 * (q["A"][name] + q["B"][name]) for name in names)
    return {
        "q_A": q["A"], "q_B": q["B"],
        "task_outcome_A": task["A"], "task_outcome_B": task["B"],
        "near_optimal_A": near["A"], "near_optimal_B": near["B"],
        "near_optimal_disjoint": not (set(near["A"]) & set(near["B"])),
        "argmax_A": argmax["A"], "argmax_B": argmax["B"],
        "mutual_regret": regret, "delta": delta,
        "all_knowing_50_50": all_knowing, "common_action_50_50": common,
        "information_gap_50_50": all_knowing - common,
        "positive": bool(not (set(near["A"]) & set(near["B"])) and regret > delta),
        "any_positive_progress_A": any(value > 0 for value in task["A"].values()),
        "any_positive_progress_B": any(value > 0 for value in task["B"].values()),
        "best_task_outcome_A": max(task["A"].values()),
        "best_task_outcome_B": max(task["B"].values()),
        "worst_task_outcome_A": min(task["A"].values()),
        "worst_task_outcome_B": min(task["B"].values()),
        "needs_different_actions": bool(
            not (set(near["A"]) & set(near["B"])) and regret > delta),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", default=str(SOURCE.parent / "frozen_c_recomputed.json"))
    args = parser.parse_args()
    payload = json.loads(Path(args.source).read_text())
    delta = payload["delta"]
    result = {"note": "recomputed from the saved distances/durations/costs; no re-simulation",
              "delta": delta, "candidates": {}}
    for name, entry in payload["candidates"].items():
        contract = analyse(entry["runs"], delta, clamp=False)
        clamped = analyse(entry["runs"], delta, clamp=True)
        result["candidates"][name] = {
            "contract_score": contract,
            "previously_reported_clamped_score": clamped,
            "shift": entry["shift"],
            "observation_identity_max_abs_difference":
                entry["observation_identity"]["max_abs_difference"],
            "first_period_identity_max_abs_difference":
                entry["first_period_identity"]["max_abs_difference_across_all_commands"],
        }
        print(f"{name}: contract progress range A "
              f"[{min(contract['task_outcome_A'].values()):+.4f}, "
              f"{max(contract['task_outcome_A'].values()):+.4f}] B "
              f"[{min(contract['task_outcome_B'].values()):+.4f}, "
              f"{max(contract['task_outcome_B'].values()):+.4f}]")
        print(f"   positive progress? A={contract['any_positive_progress_A']} "
              f"B={contract['any_positive_progress_B']} | "
              f"different actions needed? {contract['needs_different_actions']} "
              f"(disjoint={contract['near_optimal_disjoint']}, regret={contract['mutual_regret']:.4f})")
        print(f"   contract positive={contract['positive']} vs clamped positive={clamped['positive']}")
        print(f"   task-only gap all-knowing-common = "
              f"{contract['information_gap_50_50']:+.4f} (contract) / "
              f"{clamped['information_gap_50_50']:+.4f} (clamped)")
    target = Path(args.out)
    target.write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    print(f"wrote {target}")
    print("RECOMPUTE_FROZEN_C_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
