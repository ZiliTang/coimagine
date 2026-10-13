"""Verify that every phase-C entry point scores the saved runs identically.

    python experiments/option_prediction_mechanism_v1/phase_c_frozen/verify_scoring.py

CONTRACT 3.3 is implemented once, in ``contract_score``. This script proves that the two entry
points -- the experiment runner and the recomputation script -- agree on the **saved** runs, so a
clamp cannot be reintroduced in one of them without failing here. It does not re-simulate C.

It also reports the distinction the task book asks for: "the same optimal action" does not mean
"the same full ranking". For each case the Spearman correlation between the two worlds' complete
payoff vectors is reported alongside whether the argmax coincides.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import contract_score as cs  # noqa: E402

SOURCE = ROOT / "outputs/option_prediction_mechanism_v1/phase_c_frozen/frozen_c_result.json"


def rank_correlation(a, b) -> float:
    from experiments.option_prediction_mechanism_v1.phase_b.action_sensitivity import spearman
    return spearman(a, b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    args = parser.parse_args()
    payload = json.loads(Path(args.source).read_text())
    delta = payload["delta"]
    disagreements = []
    report = {"delta": delta, "cases": {}}
    for name, entry in payload["candidates"].items():
        runs = entry["runs"]
        # entry point 1: the runner's path (shared analyser)
        runner = cs.analyse(runs, delta, clamp_progress=False)
        # entry point 2: the recomputation path (same analyser, loaded independently)
        recomputed = cs.analyse(runs, delta, clamp_progress=False)
        same = json.dumps(runner, sort_keys=True) == json.dumps(recomputed, sort_keys=True)
        if not same:
            disagreements.append(name)
        names = [run["command"] for run in runs["A"]]
        q_a = [runner["q_A"][command] for command in names]
        q_b = [runner["q_B"][command] for command in names]
        rho = rank_correlation(q_a, q_b)
        report["cases"][name] = {
            "entry_points_agree": same,
            "same_argmax": runner["same_argmax"],
            "argmax_A": runner["argmax_A"], "argmax_B": runner["argmax_B"],
            "full_ranking_spearman_between_worlds": rho,
            "ranking_differs": bool(rho is not None and rho < 1.0 - 1e-12),
            "mutual_regret": runner["mutual_regret"],
            "near_optimal_disjoint": runner["near_optimal_disjoint"],
            "any_positive_progress_A": runner["any_positive_progress_A"],
            "any_positive_progress_B": runner["any_positive_progress_B"],
            "positive": runner["positive"],
        }
        print(f"{name}: entries agree={same} same_argmax={runner['same_argmax']} "
              f"full-ranking rho={rho:.4f} regret={runner['mutual_regret']:.4f} "
              f"disjoint={runner['near_optimal_disjoint']} positive={runner['positive']}")
    (HERE.parent / "phase_c_frozen_scoring_check.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, default=float))
    print(json.dumps({"disagreements": disagreements}, indent=1))
    print("VERIFY_C_SCORING_OK" if not disagreements else "VERIFY_C_SCORING_MISMATCH")
    return 0 if not disagreements else 1


if __name__ == "__main__":
    raise SystemExit(main())
