#!/usr/bin/env python
"""Round-23: frozen network and planner, only the post-execution geometric rule changes.

Three rules: the encoder estimate (current behaviour), the scan as a *constraint* over the
compatible placements, and a privileged true-geometry upper bound.  For every shipped
failure the script records whether each rule recovered it, so "the geometric correction
explains the deep regressions" can be checked case by case rather than from one example.

Writes ``outputs/grid/diagnosis/fusion_correction.json``.  No retraining, no thresholds, no
public-feedback branch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.diagnostics.conditional_planner import run_correction_comparison  # noqa: E402
from coimagine.learning.belief_model import load_belief_model  # noqa: E402

OUT = ROOT / "outputs" / "grid" / "diagnosis" / "fusion_correction.json"
CHECKPOINT = ROOT / "outputs" / "grid" / "world_model" / "belief.npz"


def main() -> int:
    model = load_belief_model(str(CHECKPOINT))
    result = run_correction_comparison(model, n_scenes=100, depths=(1, 2, 3), beam=1)
    payload = {"checkpoint": str(CHECKPOINT), "comparison": result}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {OUT}\n")
    print(f"measured {result['n_measured']}, exact reachable {result['n_exact_reachable']}")
    print("| depth | rule | success | mean total cost | vs shipped: recovered / regressed |")
    for row in result["depths"]:
        shipped = row["shipped"]
        print(
            f"| {row['depth']} | shipped | {shipped['successes']}/{shipped['n']} | "
            f"{shipped['mean_total_cost_all']:.2f} | - |"
        )
        for variant, data in row["variants"].items():
            summary = data["summary"]
            print(
                f"| {row['depth']} | {variant} | {summary['successes']}/{summary['n']} | "
                f"{summary['mean_total_cost_all']:.2f} | "
                f"{data['shipped_failures_recovered']}/{data['shipped_failures_total']} recovered, "
                f"{data['flips_vs_shipped']['regressed']} regressed |"
            )
        print(
            "    shipped failures per rule: "
            + json.dumps(
                {variant: data["per_shipped_failure"] for variant, data in row["variants"].items()}
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
