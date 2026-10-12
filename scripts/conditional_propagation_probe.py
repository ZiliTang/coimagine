#!/usr/bin/env python
"""Round-22: carry per-parameter conditional states through planning and update, then measure.

Read-only w.r.t. training: the same checkpoint, the same scenes, the same depth and beam.
The only difference between the two conditions is what is carried through the imagination and
the feedback update.  Writes ``outputs/grid/diagnosis/conditional_propagation.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.diagnostics.conditional_planner import (  # noqa: E402
    boundary_case_study,
    trace_boundaries,
    compare_records,
    conditional_chain_check,
    run_closed_loop_comparison,
    summarise_records,
)
from coimagine.diagnostics.diagnosis import run_factored_policy  # noqa: E402
from coimagine.diagnostics.state_association import default_chains  # noqa: E402
from coimagine.learning.belief_model import load_belief_model  # noqa: E402

OUT = ROOT / "outputs" / "grid" / "diagnosis" / "conditional_propagation.json"
CHECKPOINT = ROOT / "outputs" / "grid" / "world_model" / "belief.npz"


def main() -> int:
    model = load_belief_model(str(CHECKPOINT))

    chains = [conditional_chain_check(model, chain) for chain in default_chains()]
    comparison = run_closed_loop_comparison(
        model, n_scenes=100, depths=(1, 2, 3), beam=1, seed=55_000
    )

    # The shipped driver here must reproduce the repo's own runner, otherwise the comparison
    # would be measuring the harness rather than the propagation.
    consistency = []
    for depth in (1, 2, 3):
        reference = run_factored_policy(
            "search", model=model, n_scenes=100, horizon=8, depth=depth, beam=1,
            oracle_probe=False,
        )
        summary = summarise_records(
            [
                {"success": r.success, "action_cost": r.action_cost, "steps": r.steps,
                 "actions": [], "first_action": None, "wait_actions": 0,
                 "reachable_steps": r.reachable_steps,
                 "exact_reachable": r.exact_reachable, "scene_id": r.scene_id}
                for r in reference
            ]
        )
        shipped = comparison["depths"][depth - 1]["old_all"]
        consistency.append(
            {
                "depth": depth,
                "repo_runner_successes": summary["successes"],
                "harness_shipped_successes": shipped["successes"],
                "matches": summary["successes"] == shipped["successes"],
            }
        )

    # one improved case and one regressed case, picked from the flips themselves; the
    # regression is traced to the *second* boundary where it actually happens
    depth_one = next(row for row in comparison["depths"] if row["depth"] == 1)
    depth_two = next(row for row in comparison["depths"] if row["depth"] == 2)
    improved = depth_one["flips"]["improved_failure_to_success"]
    regressed = depth_two["flips"]["regressed_success_to_failure"]
    cases = []
    if improved:
        cases.append(trace_boundaries(model, improved[0]["scene_id"], depth=1, boundaries=1))
    if regressed:
        cases.append(trace_boundaries(model, regressed[0]["scene_id"], depth=2, boundaries=2))

    payload = {
        "checkpoint": str(CHECKPOINT),
        "chains": chains,
        "closed_loop": comparison,
        "shipped_driver_matches_repo_runner": consistency,
        "case_studies": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {OUT}\n")

    print("short chains (fixed actions, uniform prior, no feedback conditioning)")
    for chain in chains:
        print(f"  {chain['chain']}")
        for step in chain["steps"]:
            print(
                f"    step {step['step']} {str(step['action']):<12} "
                f"position TV shipped {step['shipped_tv']:.3f} -> conditional "
                f"{step['conditional_tv']:.3f} | P(released|theta) gap "
                f"{step['shipped_released_gap']:.2f} -> {step['conditional_released_gap']:.2f}"
            )

    print("\nclosed loop (same checkpoint, same 100 scenes, beam 1)")
    print(f"  measured {comparison['n_measured']}, exact reachable {comparison['n_exact_reachable']}")
    print("  | depth | propagation | success | mean total cost | wait-first | wait actions |")
    for row in comparison["depths"]:
        for name in ("old_all", "new_all"):
            summary = row[name]
            print(
                f"  | {row['depth']} | {name} | {summary['successes']}/{summary['n']} = "
                f"{summary['success_rate']:.3f} | {summary['mean_total_cost_all']:.2f} | "
                f"{summary['episodes_with_wait_first']} | {summary['total_wait_actions']} |"
            )
        flips = row["flips"]
        print(
            f"    flips: {len(flips['improved_failure_to_success'])} up, "
            f"{len(flips['regressed_success_to_failure'])} down; "
            f"reachable subset old {row['old_exact_reachable']['successes']}/"
            f"{row['old_exact_reachable']['n']} new {row['new_exact_reachable']['successes']}/"
            f"{row['new_exact_reachable']['n']}"
        )
    print("\nharness consistency:", consistency)
    for case in cases:
        print(f"\ncase scene {case['scene_id']} depth {case['depth']} ({case['physics']})")
        for label in ("old", "new"):
            for boundary in case[label]:
                print(f"  {label} boundary {boundary['boundary']}: chose {boundary['chosen']} {boundary['ranking']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
