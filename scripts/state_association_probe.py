#!/usr/bin/env python
"""Round-21 verification: is the multi-step error the network, or the state mixing?

Read-only.  Writes ``outputs/grid/diagnosis/state_association.json`` with

* the two short chains (continuous push; release then continuous push) run under four
  propagation schemes against a per-world exact reference;
* the key single-step transition split into the populations where the push actually moves
  the object and where it does not, under the true parameter and under a uniform belief.

No training, no method change, no planner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.diagnostics.state_association import (  # noqa: E402
    default_chains,
    key_transition_probe,
    run_chain_probe,
)
from coimagine.learning.belief_model import load_belief_model  # noqa: E402

OUT = ROOT / "outputs" / "grid" / "diagnosis" / "state_association.json"
CHECKPOINT = ROOT / "outputs" / "grid" / "world_model" / "belief.npz"


def main() -> int:
    model = load_belief_model(str(CHECKPOINT))
    chains = [run_chain_probe(model, chain) for chain in default_chains()]
    transitions = key_transition_probe(model, n_episodes=400, horizon=8, seed=0)
    payload = {
        "checkpoint": str(CHECKPOINT),
        "chains": chains,
        "key_transition": transitions,
        "notes": [
            "All learned conditions start from the encoder's dynamic vector for the observed "
            "true state; a hard one-hot is off-distribution and changes the prediction.",
            "The per-world exact reference keeps (origin, robots, theta, released) and "
            "executes per world; it never decodes a mixture.",
            "`exact_mixed` factorises the joint into b(theta) x marginal world, the "
            "exact-dynamics analogue of one marginal vector repeated per parameter.",
            "`learned_per_theta` keeps 16 vectors, one per parameter; it preserves the "
            "theta <-> (cell, released) association but not the (cell, released) correlation "
            "inside one parameter.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {OUT}")

    for chain in chains:
        print(f"\n== {chain['chain']}  actions={chain['actions']}  true={chain['true_physics']}")
        for step in chain["steps"]:
            encoder = step["encoder"]
            print(
                f"  step {step['step']:>2} {str(step['action']):<12} "
                f"encoder cell {encoder['argmax_cell']} true {encoder['true_cell']} "
                f"p={encoder['probability_on_true_cell']:.3f} "
                f"released_p={encoder['released_probability']:.3f} (true {encoder['true_released']})"
            )
        print("  deviation from the per-world reference (step: TV / dGoal / dReleased|theta):")
        for name, rows in chain["deviations"].items():
            text = " ".join(
                f"s{row['step']}:{row['position_tv']:.3f}/{row['goal_probability_gap']:+.3f}"
                f"/{row['max_released_by_theta_gap']:.2f}"
                for row in rows
            )
            print(f"    {name:<32} {text}")
        print(
            "  first deviation above tolerance: "
            + json.dumps(
                {k: (v["step"] if v else None) for k, v in chain["first_deviation"].items()}
            )
        )

    print("\n== key single-step transition")
    for action, entry in transitions["conditions"].items():
        print(f"  {action}: n={entry['n']} encoder_cell_top1={entry['encoder_cell_top1_rate']:.3f}")
        for condition in ("true_theta", "uniform_belief"):
            bucket = entry[condition]
            print(
                f"    {condition:<15} true_moved={bucket['true_moved']:<4} "
                f"predicted_move={bucket['predicted_move_rate_on_true_move']:.3f} "
                f"cell_top1={bucket['cell_top1_rate_on_true_move']:.3f} "
                f"meanP(move)={bucket['mean_move_probability_on_true_move']:.3f} | "
                f"true_not_moved={bucket['true_not_moved']:<4} "
                f"false_move={bucket['predicted_move_rate_on_true_not_moved']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
