"""Round 3, phase C tests: the recorded positives must actually satisfy the frozen criterion.

These read the C artifacts rather than re-simulating (re-simulation lives in
``experiments/option_prediction_mechanism_v1/verify_c_positive.py``), so they are cheap enough to
run in the normal suite while still failing if a later edit weakens the observation-identity check,
lets a non-finite number into the comparison, exceeds the pair/skill budget, or records a
"positive" whose near-optimal sets actually intersect.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RECORDS = ROOT / "outputs/option_prediction_mechanism_v1/phase_c/phase_c_records.json"
BELIEF = ROOT / "outputs/option_prediction_mechanism_v1/phase_c/phase_c_belief_reference.json"

pytestmark = pytest.mark.skipif(not RECORDS.exists(),
                                reason="phase-C artifacts not produced on this machine")


def _payload() -> dict:
    return json.loads(RECORDS.read_text())


def _walk_finite(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_finite(item, f"{path}[{index}]")
    elif isinstance(value, float):
        assert math.isfinite(value), f"non-finite value at {path}: {value}"


def test_budget_and_delta_are_the_frozen_ones():
    payload = _payload()
    assert payload["pairs_total"] == 64, "the contract allows at most 32 + 32 pairs"
    assert payload["pairs_per_template"] == {"T1": 32, "T2": 32}
    assert payload["delta"] == pytest.approx(0.02), "delta was frozen before the search"
    assert len(payload["records"]) == 64
    for record in payload["records"]:
        assert len(record["skills"]) <= 8, "at most eight deployable joint skills per pair"
        assert record["legal"] is True, f"{record['pair_id']} is not a legal pair"


def test_no_non_finite_number_enters_the_observation_comparison():
    _walk_finite(_payload())


def test_every_recorded_positive_meets_the_frozen_criterion():
    positives = [record for record in _payload()["records"]
                 if record["analysis"].get("positive")]
    assert positives, "C found no positive; if that is the real result, update the expectation"
    for record in positives:
        analysis = record["analysis"]
        identity = record["observation_identity"]["max_abs_difference"]
        assert identity <= 1e-12, f"{record['pair_id']} observations differ by {identity}"
        assert analysis["near_optimal_disjoint"] is True
        assert not set(analysis["near_optimal_A"]) & set(analysis["near_optimal_B"])
        assert analysis["mutual_regret"] > analysis["delta"], (
            f"{record['pair_id']} regret {analysis['mutual_regret']} <= delta")
        assert analysis["observation_identical"] is True


def test_rejections_are_accounted_for():
    payload = _payload()
    positives = sum(1 for record in payload["records"]
                    if record["analysis"].get("positive"))
    assert positives == len(payload.get("positives", [])), (
        "the summary's positives list and the per-record flags disagree")


def test_belief_reference_respects_the_legal_branching_constraint():
    if not BELIEF.exists():
        pytest.skip("belief reference not produced")
    payload = json.loads(BELIEF.read_text())
    for pair_id, reference in payload["references"].items():
        all_knowing = reference["all_knowing_two_skill_value"]
        common = reference["common_plan_value"]
        legal = reference["legal_observation_conditioned_plan_value"]
        assert legal >= common - 1e-9, (
            f"{pair_id}: a planner that may branch where the observations differ can always "
            f"imitate the common plan, so legal >= common must hold")
        assert all_knowing >= legal - 1e-9, f"{pair_id}: all-knowing must be an upper bound"


# ---------------------------------------------------------------------------
# the frozen-protocol redo: the low-level controller must not inject hidden truth
# ---------------------------------------------------------------------------

FROZEN = (ROOT / "outputs/option_prediction_mechanism_v1/phase_c_frozen/frozen_c_result.json")


def test_frozen_case_has_identical_observation_and_first_control_period():
    """The correction's core check: identical observations => identical first reference/force."""

    if not FROZEN.exists():
        pytest.skip("frozen phase-C case not run on this machine")
    payload = json.loads(FROZEN.read_text())
    assert payload["candidates"], "the declared case should have been run"
    for name, entry in payload["candidates"].items():
        assert entry["observation_identity"]["identical"] is True, (
            f"{name}: the two worlds' whitelist observations must be identical")
        assert entry["observation_identity"]["max_abs_difference"] == 0.0
        assert entry["first_period_identity"]["identical"] is True, (
            f"{name}: with identical observations and controller state the first control period's "
            "reference and applied force must be identical; a difference would mean the frozen "
            "controller is reading hidden truth")
        assert entry["first_period_identity"]["max_abs_difference_across_all_commands"] == 0.0
        # and the worlds really are different, otherwise the check would be vacuous
        truth_a = entry["truth_differs"]["object_pose_A"]
        truth_b = entry["truth_differs"]["object_pose_B"]
        assert truth_a != truth_b


def test_frozen_case_is_recorded_as_a_negative_or_meets_the_full_criterion():
    """Do not let a later edit turn a non-positive into a "positive" without both conditions."""

    if not FROZEN.exists():
        pytest.skip("frozen phase-C case not run on this machine")
    payload = json.loads(FROZEN.read_text())
    delta = payload["delta"]
    for name, entry in payload["candidates"].items():
        analysis = entry["analysis"]
        disjoint = analysis["near_optimal_disjoint"]
        regret_ok = analysis["mutual_regret"] > delta
        assert analysis["positive"] == (disjoint and regret_ok), (
            f"{name}: 'positive' must mean disjoint near-optimal sets AND regret > delta")
        if analysis["positive"]:
            assert disjoint and regret_ok
        else:
            assert (not disjoint) or (not regret_ok), (
                f"{name} is recorded negative but satisfies the criterion")

