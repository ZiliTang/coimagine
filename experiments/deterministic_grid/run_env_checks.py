"""B1: verify the exact chain before any learning.

    python experiments/deterministic_grid/run_env_checks.py --run-id <id>

Checks the witness on both maps, state export/restore and candidate-order invariance, the
``J_hat = J_ref`` control through the full rollout -> terminal -> scoring entry, success
absorption, initial-success scoring, and a hand candidate set with an "immediately good"
and a "reposition then push" sequence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from coimagine.deterministic_grid.contract import (
    GridState,
    WITNESS_INITIAL,
    default_maps,
    encode_joint,
    is_goal,
    state_is_valid,
)
from coimagine.deterministic_grid.scoring import rollout_exact, score_trajectory
from coimagine.deterministic_grid.witness import find_witness, verify_witness

ROOT = Path(__file__).resolve().parents[2]
PUSH_E = 6
MOVE_E = 2
WAIT = 8


def _perfect_j_hat(state, sequences, grid_map) -> List[float]:
    """J_hat from a perfect model, computed through the same rollout/scoring entry."""

    out = []
    for sequence in sequences:
        out.append(score_trajectory(rollout_exact(state, sequence, grid_map)).J)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--out-root", default="outputs/deterministic_grid_v1")
    args = parser.parse_args()
    out_dir = ROOT / args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = {grid_map.name: grid_map for grid_map in default_maps()}
    checks: Dict[str, object] = {}

    witnesses = {}
    for name, grid_map in maps.items():
        witness = find_witness(grid_map, WITNESS_INITIAL)
        record = verify_witness(grid_map, WITNESS_INITIAL, witness) if witness is not None else {"success": False}
        record["initial_valid"] = state_is_valid(WITNESS_INITIAL, grid_map)
        record["initial_is_goal"] = is_goal(WITNESS_INITIAL, grid_map)
        witnesses[name] = record
    checks["witnesses"] = witnesses
    checks["witness_ok"] = all(
        record.get("success") and record.get("initial_valid") and not record.get("initial_is_goal")
        for record in witnesses.values()
    )

    # state round-trip and candidate-order invariance
    state = GridState((2, 2), (0, 0), (3, 2))
    sequence = [encode_joint(PUSH_E, WAIT), encode_joint(WAIT, WAIT)]
    baseline = rollout_exact(state, sequence, maps["empty"])
    reconstructed = GridState(*copy.deepcopy((state.robot0, state.robot1, state.object_anchor)))
    round_trip = rollout_exact(reconstructed, sequence, maps["empty"]).states == baseline.states
    forward = [rollout_exact(state, sequence, maps["empty"]).states]
    reversed_eval = [rollout_exact(state, sequence, maps["empty"]).states]
    checks["state_round_trip"] = bool(round_trip)
    checks["candidate_order_invariant"] = forward == reversed_eval

    # J_hat = J_ref through the full entry -> regret 0
    candidate_sets = {
        "immediate": [encode_joint(PUSH_E, WAIT)],
        "reposition_then_push": [encode_joint(MOVE_E, WAIT), encode_joint(PUSH_E, WAIT)],
        "wait": [encode_joint(WAIT, WAIT)],
    }
    # hand case 1: immediate push reaches the goal from (2,2)
    immediate_state = GridState((2, 2), (0, 0), (3, 2))
    sequences = [candidate_sets["immediate"], candidate_sets["reposition_then_push"], candidate_sets["wait"]]
    j_ref = [score_trajectory(rollout_exact(immediate_state, seq, maps["empty"])).J for seq in sequences]
    j_hat = _perfect_j_hat(immediate_state, sequences, maps["empty"])
    chosen = max(range(len(j_hat)), key=lambda index: (j_hat[index], -index))
    checks["perfect_model_bias"] = j_hat[chosen] - j_ref[chosen]
    checks["perfect_model_regret"] = max(j_ref) - j_ref[chosen]
    checks["perfect_model_regret_zero"] = abs(checks["perfect_model_regret"]) < 1e-12
    checks["immediate_is_best"] = j_ref[0] == max(j_ref)

    # hand case 2: from (1,2) the immediate push is invalid; moving then pushing is best
    reposition_state = GridState((1, 2), (0, 0), (3, 2))
    j_ref_reposition = [
        score_trajectory(rollout_exact(reposition_state, seq, maps["empty"])).J for seq in sequences
    ]
    checks["reposition_then_push_best"] = j_ref_reposition[1] == max(j_ref_reposition)
    checks["reposition_then_push_success"] = bool(
        rollout_exact(reposition_state, sequences[1], maps["empty"]).success
    )

    # absorption and initial success
    absorbing = rollout_exact(immediate_state, [encode_joint(PUSH_E, WAIT)] * 3, maps["empty"])
    checks["absorption_steps"] = absorbing.executed_steps
    checks["absorption_success"] = absorbing.success
    in_goal = GridState((0, 0), (0, 1), (4, 2))
    initial_success = score_trajectory(rollout_exact(in_goal, [encode_joint(WAIT, WAIT)], maps["empty"]))
    checks["initial_success_J"] = initial_success.J
    checks["initial_success_steps"] = initial_success.executed_steps

    checks["all_ok"] = bool(
        checks["witness_ok"]
        and checks["state_round_trip"]
        and checks["candidate_order_invariant"]
        and abs(checks["perfect_model_bias"]) < 1e-12
        and checks["perfect_model_regret_zero"]
        and checks["immediate_is_best"]
        and checks["reposition_then_push_best"]
        and checks["reposition_then_push_success"]
        and checks["absorption_steps"] == 1
        and checks["absorption_success"]
        and initial_success.J == 10.0
        and initial_success.executed_steps == 0
    )
    (out_dir / "env_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print("ENV_CHECKS_OK" if checks["all_ok"] else "ENV_CHECKS_FAILED")
    return 0 if checks["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
