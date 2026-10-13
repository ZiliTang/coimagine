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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract_score import analyse  # noqa: E402
SOURCE = ROOT / "outputs/option_prediction_mechanism_v1/phase_c_frozen/frozen_c_result.json"


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
        contract = analyse(entry["runs"], delta, clamp_progress=False)
        clamped = analyse(entry["runs"], delta, clamp_progress=True)
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
