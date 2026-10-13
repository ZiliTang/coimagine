"""Focused tests for the Phase A audit logic.

Every expected value is worked out by hand in the comment. The tests use tiny synthetic
data; they do not mirror the implementation and they never load the real checkpoints.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from coimagine.deterministic_grid.audit import (
    AUDIT_TAG,
    DECOMP_TERMS,
    EXACT_SEED,
    RUN_METRIC_KEYS,
    AuditConfig,
    EvalState,
    _summarize_a4,
    a4_chain,
    aggregate_by_state,
    build_traced_case,
    decompose_components,
    decomposition_total,
    eval_state_coverage,
    group_statistics,
    model_rollout_detail,
    select_eval_states_v2,
)
from coimagine.deterministic_grid.contract import (
    GridState,
    empty_map,
    encode_joint,
    is_goal,
)
from coimagine.deterministic_grid.data import Dataset, TransitionSample
from coimagine.deterministic_grid.dynamics import transition_exact
from coimagine.deterministic_grid.model import ModelConfig, PointModel
from coimagine.deterministic_grid.scoring import ScoreComponents, rollout_exact, score_trajectory

MAP = empty_map()
MAPS = {"empty": MAP}
MOVE_E = 2
MOVE_W = 3
PUSH_E = 6
PUSH_N = 4
WAIT = 8


def _state(x0: int, y0: int, x1: int, y1: int, ox: int, oy: int) -> GridState:
    return GridState((x0, y0), (x1, y1), (ox, oy))


# ---------------------------------------------------------------------------
# A2: the SHA256 ordering rule
# ---------------------------------------------------------------------------


def _dataset_with_test_states(states, split: str = "test") -> Dataset:
    return Dataset(samples=[], state_split={state.key("empty"): split for state in states})


def test_sha256_selection_is_deterministic_and_not_coordinate_ordered() -> None:
    # 16 hand-picked legal non-terminal states; the coordinate-first ten are not the
    # SHA256-first ten because the digest order is uncorrelated with the string order.
    states = [
        _state(0, 0, 0, 1, 2, 2), _state(0, 1, 0, 2, 3, 3), _state(0, 2, 0, 3, 4, 1),
        _state(0, 3, 0, 4, 1, 4), _state(0, 4, 0, 5, 2, 0), _state(1, 0, 1, 1, 3, 1),
        _state(1, 1, 1, 2, 4, 4), _state(1, 2, 1, 3, 0, 2), _state(1, 3, 1, 4, 1, 0),
        _state(1, 4, 1, 5, 2, 3), _state(2, 0, 2, 1, 3, 2), _state(2, 1, 2, 2, 0, 1),
        _state(2, 2, 2, 3, 4, 3), _state(2, 3, 2, 4, 0, 4), _state(2, 4, 2, 5, 1, 1),
        _state(3, 0, 3, 1, 0, 3),
    ]
    # two extra splits that must never be selected, plus one goal state on the test split
    states_with_noise = states + [_state(4, 4, 4, 5, 4, 2), _state(5, 0, 5, 1, 0, 0)]
    dataset = _dataset_with_test_states(states)
    dataset.state_split[states_with_noise[-2].key("empty")] = "train"
    dataset.state_split[states_with_noise[-1].key("empty")] = "val"
    dataset.state_split[_state(0, 0, 0, 1, 4, 2).key("empty")] = "test"  # object in the goal

    first = select_eval_states_v2(dataset, MAPS)
    second = select_eval_states_v2(dataset, MAPS)
    assert [e.as_dict() for e in first] == [e.as_dict() for e in second]
    assert len(first) == 10

    # hand rule: order by SHA256("grid-audit-v2|" + state_key), take ten per map
    expected = sorted(
        states,
        key=lambda s: hashlib.sha256(f"{AUDIT_TAG}|{s.key('empty')}".encode()).hexdigest(),
    )[:10]
    assert [e.state for e in first] == expected
    assert all(e.digest == hashlib.sha256(f"{AUDIT_TAG}|{e.state_key}".encode()).hexdigest() for e in first)
    assert all(e.map_name == "empty" and e.rank == i for i, e in enumerate(first))
    # the rule is deliberately not the legacy coordinate-string order
    coordinate_order = sorted(states, key=lambda s: s.key("empty"))[:10]
    assert [e.state for e in first] != coordinate_order
    # the goal state and the train/val states are excluded
    assert all(not is_goal(e.state, MAP) for e in first)
    assert all(e.state_key in {s.key("empty") for s in states} for e in first)


def test_coverage_counts_distinct_cells() -> None:
    entries = [
        EvalState("empty", _state(0, 0, 1, 1, 2, 2), "k0", "d0", 0),
        EvalState("empty", _state(0, 0, 3, 3, 2, 2), "k1", "d1", 1),
        EvalState("empty", _state(4, 4, 1, 1, 0, 0), "k2", "d2", 2),
    ]
    coverage = eval_state_coverage(entries)
    # robot0 cells {(0,0),(4,4)} -> 2; robot1 cells {(1,1),(3,3)} -> 2; anchors -> 2
    assert coverage["overall"]["n_distinct_robot0_cells"] == 2
    assert coverage["overall"]["n_distinct_robot1_cells"] == 2
    assert coverage["overall"]["n_distinct_object_anchors"] == 2
    assert coverage["per_map"]["empty"]["n_states"] == 3


# ---------------------------------------------------------------------------
# A1: hand-worked group definitions
# ---------------------------------------------------------------------------


def _sample(state: GridState, joint: int) -> TransitionSample:
    next_state, info = transition_exact(state, joint, MAP)
    return TransitionSample(
        map_name="empty",
        state=state,
        joint_action=joint,
        next_state=next_state,
        object_moved=info.object_moved,
        robot_moved=info.robots_moved > 0,
        rollback=info.rollback,
        invalid_actions=info.invalid_actions,
    )


def test_group_definitions_match_a_hand_worked_example() -> None:
    # s_object: robot (2,2) pushes east, object anchor (3,2) -> (4,2): object moves.
    s_object = _sample(_state(2, 2, 0, 0, 3, 2), encode_joint(PUSH_E, WAIT))
    # s_robot_only: robot0 moves east, object anchor does not change -> object_stationary.
    s_robot_only = _sample(_state(0, 0, 0, 1, 2, 2), encode_joint(MOVE_E, WAIT))
    # s_frozen: two robots swap targets -> rollback, the whole state is unchanged.
    s_frozen = _sample(_state(0, 0, 2, 0, 2, 2), encode_joint(MOVE_E, MOVE_W))
    # s_blocked_push: qualified push north but the object anchor (2,4) is against the
    # top boundary, so nothing translates and the whole state is unchanged.
    s_blocked = _sample(_state(2, 3, 0, 0, 2, 4), encode_joint(PUSH_N, WAIT))

    # hand-worked truth: object_moved {s_object}; object_stationary {s_robot_only,
    # s_frozen, s_blocked}; full_state_unchanged {s_frozen, s_blocked}.
    assert s_object.object_moved and s_object.next_state.object_anchor == (4, 2)
    assert not s_robot_only.object_moved and s_robot_only.next_state.robot0 == (1, 0)
    assert s_robot_only.next_state != s_robot_only.state
    assert not s_frozen.object_moved and s_frozen.next_state == s_frozen.state
    assert not s_blocked.object_moved and s_blocked.next_state == s_blocked.state

    samples = [s_object, s_robot_only, s_frozen, s_blocked]
    dataset = Dataset(
        samples=samples,
        state_split={s.state.key("empty"): "test" for s in samples},
    )
    model = PointModel(ModelConfig(hidden=8, layers=1))
    stats = group_statistics(dataset, MAPS, {7: model}, split="test")

    n_by_group = {row["group"]: row["n"] for row in stats["rows"] if row["map"] == "both" and row["model_seed"] == 7}
    assert n_by_group["object_moved"] == 1
    assert n_by_group["object_stationary"] == 3
    assert n_by_group["full_state_unchanged"] == 2  # s_frozen and s_blocked only
    assert n_by_group["all"] == 4
    assert stats["relations"]["full_state_unchanged_subset_of_object_stationary"] is True
    assert stats["relations"]["full_state_unchanged_implies_object_stationary"] is True
    assert stats["relations"]["object_moved_plus_object_stationary_equals_all"] is True
    # pooled row exists and averages over the single seed
    pooled = [row for row in stats["rows"] if row["map"] == "both" and row["group"] == "object_stationary" and row["model_seed"] == "mean"]
    assert len(pooled) == 1 and pooled[0]["n"] == 3


# ---------------------------------------------------------------------------
# A3: the four-term decomposition
# ---------------------------------------------------------------------------


def test_four_terms_sum_to_the_total_deviation_hand_worked() -> None:
    # hand-worked: hat scores 1 success in 2 steps with distance 0 (J = 10-2 = 8);
    # ref fails after 3 steps at distance 4 (J = -3 - 2 = -5). Difference = 13.
    hat = {"success": 1, "executed_steps": 2, "terminal_distance": 0.0, "invalid_prediction": 0, "J": 8.0}
    ref = {"success": 0, "executed_steps": 3, "terminal_distance": 4.0, "invalid_prediction": 0, "J": -5.0}
    terms = decompose_components(hat, ref)
    assert terms["success_reward"] == pytest.approx(10.0)
    assert terms["executed_steps"] == pytest.approx(1.0)
    assert terms["terminal_distance"] == pytest.approx(2.0)
    assert terms["invalid_penalty"] == pytest.approx(0.0)
    assert decomposition_total(terms) == pytest.approx(hat["J"] - ref["J"])
    assert set(terms) == set(DECOMP_TERMS)

    # hand-worked illegal prediction: hat terminates at step 2 with invalid=1
    # (J = -2 - 0.5 - 10 = -12.5), ref the same but legal (J = -2.5); difference = -10.
    hat2 = {"success": 0, "executed_steps": 2, "terminal_distance": 1.0, "invalid_prediction": 1, "J": -12.5}
    ref2 = {"success": 0, "executed_steps": 2, "terminal_distance": 1.0, "invalid_prediction": 0, "J": -2.5}
    terms2 = decompose_components(hat2, ref2)
    assert terms2["invalid_penalty"] == pytest.approx(-10.0)
    assert decomposition_total(terms2) == pytest.approx(-10.0)


def test_decomposition_of_a_real_scored_rollout_is_exact() -> None:
    state = _state(2, 2, 0, 0, 3, 2)
    sequence = [encode_joint(PUSH_E, WAIT), encode_joint(PUSH_E, WAIT)]
    ref_result = rollout_exact(state, sequence, MAP)
    ref = score_trajectory(ref_result).as_dict()

    # a fake model that predicts a legal state with the object displaced to anchor (0,4)
    def step_fn(current, joint):
        return _state(current.robot0[0], current.robot0[1], current.robot1[0], current.robot1[1], 0, 4), True

    from coimagine.deterministic_grid.scoring import rollout

    hat = score_trajectory(rollout(state, sequence, MAP, step_fn)).as_dict()
    terms = decompose_components(hat, ref)
    assert decomposition_total(terms) == pytest.approx(hat["J"] - ref["J"], abs=1e-12)


def test_model_rollout_detail_matches_rollout_and_never_reuses_the_previous_state() -> None:
    state = _state(0, 0, 0, 1, 2, 2)
    sequence = [encode_joint(MOVE_E, WAIT), encode_joint(MOVE_E, WAIT)]
    true_states = rollout_exact(state, sequence, MAP).states
    first_prediction = _state(1, 0, 0, 1, 2, 2)

    def step_fn(current, joint):
        if current == state:
            return first_prediction, True
        return None, False  # illegal prediction on step 2

    score, records = model_rollout_detail(MAP, state, sequence, true_states, step_fn)
    # protocol: terminate at the failing step, do not fabricate a state and do not reuse
    # the previous legal state
    assert len(records) == 2
    assert records[0]["predicted_state"] == first_prediction
    assert records[1]["step"] == 1 and records[1]["input_state"] == first_prediction
    assert records[1]["invalid"] is True
    assert records[1]["termination_reason"] == "invalid_prediction"
    assert score.invalid_prediction == 1 and score.executed_steps == 2
    assert score.J == pytest.approx(-2.0 - 0.5 * score.terminal_distance - 10.0)


# ---------------------------------------------------------------------------
# A4: recursive contrast, termination and all-chain denominators
# ---------------------------------------------------------------------------


def test_recursive_chain_uses_its_own_last_legal_state_and_terminates_on_illegal() -> None:
    initial = _state(0, 0, 0, 1, 2, 2)
    sequence = [
        encode_joint(MOVE_E, WAIT),
        encode_joint(MOVE_E, WAIT),
        encode_joint(MOVE_E, WAIT),
    ]
    # true trajectory used only as the comparison reference
    true_states = rollout_exact(initial, sequence, MAP).states
    # the model diverges immediately: robot0 goes to (1,1), not the true (1,0)
    first_prediction = _state(1, 1, 0, 1, 2, 2)
    second_prediction = _state(2, 1, 0, 1, 2, 2)
    predictions = {initial: first_prediction, first_prediction: second_prediction}

    def step_fn(current, joint):
        if current in predictions:
            return predictions[current], True
        return None, False

    chain = a4_chain(MAP, initial, sequence, true_states, step_fn, mode="recursive")
    assert len(chain) == 3  # never a fourth, fabricated step
    # step 2 input is the model's own step-1 prediction, not the true state
    assert chain[1]["input_state"] == first_prediction
    assert chain[1]["input_state"] != true_states[1]
    assert chain[1]["predicted_state"] == second_prediction
    assert chain[2]["input_state"] == second_prediction
    assert chain[2]["invalid"] is True and chain[2]["termination_reason"] == "invalid_prediction"


def test_teacher_chain_starts_from_the_true_state_each_step() -> None:
    initial = _state(0, 0, 0, 1, 2, 2)
    sequence = [encode_joint(MOVE_E, WAIT), encode_joint(MOVE_E, WAIT)]
    true_states = rollout_exact(initial, sequence, MAP).states
    constant = _state(5, 5, 5, 4, 0, 0)

    def step_fn(current, joint):
        return constant, True

    teacher = a4_chain(MAP, initial, sequence, true_states, step_fn, mode="teacher")
    recursive = a4_chain(MAP, initial, sequence, true_states, step_fn, mode="recursive")
    assert [record["input_state"] for record in teacher] == [true_states[0], true_states[1]]
    assert [record["predicted_state"] for record in teacher] == [constant, constant]
    # recursive feeds its own prediction back, so its step-2 input is not the true state
    assert [record["input_state"] for record in recursive] == [initial, constant]


def test_a4_summary_uses_all_chain_denominators_not_only_survivors() -> None:
    # three chains over a fixed true trajectory: A terminates illegally at step 1;
    # B and C survive both steps, but B mismatches at step 2 and C never does.
    terminating = [
        {"step": 0, "legal": False, "invalid": True, "predicted_success": False,
         "false_success": False, "full_state_match": None, "position_error": None,
         "termination_reason": "invalid_prediction"},
    ]
    mismatching = [
        {"step": 0, "legal": True, "invalid": False, "predicted_success": False,
         "false_success": False, "full_state_match": True, "position_error": 0.0,
         "termination_reason": None},
        {"step": 1, "legal": True, "invalid": False, "predicted_success": False,
         "false_success": False, "full_state_match": False, "position_error": 2.0,
         "termination_reason": None},
    ]
    matching = [
        {"step": 0, "legal": True, "invalid": False, "predicted_success": False,
         "false_success": False, "full_state_match": True, "position_error": 0.0,
         "termination_reason": None},
        {"step": 1, "legal": True, "invalid": False, "predicted_success": False,
         "false_success": False, "full_state_match": True, "position_error": 0.0,
         "termination_reason": None},
    ]
    summary = _summarize_a4([terminating, mismatching, matching], [False, False, False], depth=2, n_chains=3)
    first, second = summary["steps"]
    # step 1: all three chains predict; the termination rate uses all three chains
    assert first["predictions_made"] == 3
    assert first["termination_rate_cumulative"] == pytest.approx(1.0 / 3.0)
    assert first["invalid_rate_cumulative"] == pytest.approx(1.0 / 3.0)
    assert first["failure_or_error_rate_cumulative_all_chains"] == pytest.approx(1.0 / 3.0)
    # step 2: only the two survivors predict (denominator 2), but the all-chain rates still
    # carry the terminated chain
    assert second["survivor_denominator"] == 2
    assert second["full_state_error_on_survivors"] == pytest.approx(0.5)
    assert second["position_error_on_survivors"] == pytest.approx(1.0)
    assert second["failure_or_error_rate_cumulative_all_chains"] == pytest.approx(2.0 / 3.0)
    assert second["termination_rate_cumulative"] == pytest.approx(1.0 / 3.0)
    assert summary["chains_invalid"] == 1


def test_a4_counts_a_missed_true_termination_separately_from_state_error() -> None:
    # the fixed true trajectory reaches the goal on its only step, but the model keeps
    # predicting a non-goal state: a termination-decision error, not a state error.
    initial = _state(2, 2, 0, 0, 3, 2)
    sequence = [encode_joint(PUSH_E, WAIT)]
    true_states = rollout_exact(initial, sequence, MAP).states
    assert len(true_states) == 2 and is_goal(true_states[-1], MAP)

    def step_fn(current, joint):
        return _state(1, 1, 0, 0, 2, 2), True  # legal but not the goal

    chain = a4_chain(MAP, initial, sequence, true_states, step_fn, mode="recursive")
    summary = _summarize_a4([chain], [True], depth=1, n_chains=1)
    assert summary["chains_missed_true_success"] == 1
    assert summary["steps"][0]["missed_true_success_rate_cumulative"] == pytest.approx(1.0)
    assert summary["chains_false_success"] == 0


# ---------------------------------------------------------------------------
# A1: the exact model never enters the learned mean
# ---------------------------------------------------------------------------


def _run_row(
    eval_set: str,
    model_seed: int,
    state_key: str,
    value: float,
    *,
    H: int = 4,
    K: int = 256,
    map_name: str = "empty",
    candidate_seed: int = 0,
) -> dict:
    row = {
        "eval_set": eval_set,
        "H": H,
        "K": K,
        "model_seed": model_seed,
        "map": map_name,
        "state_key": state_key,
        "candidate_seed": candidate_seed,
    }
    row.update({key: float(value) for key in RUN_METRIC_KEYS})
    return row


def test_aggregate_by_state_excludes_the_exact_row_from_the_learned_mean() -> None:
    # Hand-worked: one exact row (bias identically 0) and two non-zero learned rows.
    # Learned-only mean = (3.0 + 5.0) / 2 = 4.0.
    # If the exact row were wrongly included the mean would be (3 + 5 + 0) / 3 = 8/3.
    rows = [
        _run_row("v2", EXACT_SEED, "empty|s0", 0.0),
        _run_row("v2", 0, "empty|s0", 3.0),
        _run_row("v2", 1, "empty|s0", 5.0),
    ]
    aggregated = aggregate_by_state(rows, eval_set="v2")
    mean_row = next(r for r in aggregated if r["map"] == "all" and r["model_seed"] == "mean")
    assert mean_row["selected_bias"] == pytest.approx(4.0)
    assert mean_row["selected_bias"] != pytest.approx(8.0 / 3.0)
    assert mean_row["exact_row_excluded"] is True
    assert mean_row["aggregation"] == "learned_seeds_only"
    assert mean_row["seed_set"] == [0, 1]

    # The exact model keeps its own row and its own (zero) bias.
    exact_row = next(
        r for r in aggregated if r["map"] == "all" and r["model_seed"] == EXACT_SEED
    )
    assert exact_row["selected_bias"] == pytest.approx(0.0)


def test_state_first_mean_equals_learned_row_mean_when_repeats_are_equal() -> None:
    # Two states, each with two candidate seeds and two learned model seeds, plus the exact
    # model. Equal repeat count per state means "mean within state, then across states" must
    # equal the plain mean over the learned rows; the exact rows must not enter either.
    rows = []
    for state_key, shift in (("empty|s0", 0.0), ("empty|s1", 2.0)):
        for candidate_seed in (0, 1):
            for seed, base in ((0, 10.0), (1, 20.0)):
                rows.append(
                    _run_row(
                        "v2", seed, state_key, base + shift + 0.5 * candidate_seed,
                        candidate_seed=candidate_seed,
                    )
                )
            rows.append(
                _run_row("v2", EXACT_SEED, state_key, 0.0, candidate_seed=candidate_seed)
            )
    aggregated = aggregate_by_state(rows, eval_set="v2")
    mean_row = next(r for r in aggregated if r["map"] == "all" and r["model_seed"] == "mean")
    learned_rows = [r for r in rows if r["model_seed"] != EXACT_SEED]
    for metric in RUN_METRIC_KEYS:
        learned_plain = float(np.mean([r[metric] for r in learned_rows]))
        assert mean_row[metric] == pytest.approx(learned_plain, abs=1e-12)
    # Sanity: including the exact rows would give a different number here.
    all_rows = rows
    assert mean_row["selected_bias"] != pytest.approx(
        float(np.mean([r["selected_bias"] for r in all_rows]))
    )


# ---------------------------------------------------------------------------
# A2: the single-candidate counterfactual replaces exactly one J_hat entry
# ---------------------------------------------------------------------------


def _traced_case(*, model_seed: int = 0, candidate_seed: int = 0, H: int = 4, K: int = 8) -> dict:
    import torch

    torch.manual_seed(0)
    state = _state(0, 0, 0, 1, 2, 2)
    model = PointModel(ModelConfig(hidden=8, layers=1))
    run = {
        "map": "empty",
        "state_key": state.key("empty"),
        "H": H,
        "K": K,
        "candidate_seed": candidate_seed,
        "model_seed": model_seed,
    }
    return build_traced_case(
        {"empty": MAP}, {model_seed: model}, "empty", state, run, AuditConfig()
    )


def test_single_candidate_counterfactual_changes_at_most_one_j_hat_entry() -> None:
    case = _traced_case()
    chosen = case["chosen_candidate_id"]
    for key, payload in case["counterfactuals"].items():
        assert payload["scope"] == "chosen_candidate_only"
        assert payload["repaired_candidate_id"] == chosen
        baseline = payload["baseline_j_hat"]
        counterfactual = payload["counterfactual_j_hat"]
        assert len(baseline) == len(counterfactual) == case["actual_K"]
        changed = [i for i, (a, b) in enumerate(zip(baseline, counterfactual)) if a != b]
        # exactly zero or one entry may differ, and only the originally chosen candidate
        assert payload["n_changed_entries"] == len(changed) <= 1
        assert payload["changed_entry_indices"] == changed
        assert all(index == chosen for index in changed)
        assert payload["only_chosen_entry_replaced"] is True
        for index, (a, b) in enumerate(zip(baseline, counterfactual)):
            if index != chosen:
                assert a == b  # bit-identical float equality, not approx
        assert payload["baseline_chosen_model_return"] == case["chosen_model_J"]

    # The historical attachment really is the old all-candidate intervention.
    historical = case["counterfactuals_all_candidates_historical"]
    assert set(historical) == {"full_state", "object_anchor", "false_success"}
    for key, payload in historical.items():
        assert payload["scope"] == "all_candidates_historical"
        assert payload["n_changed_entries"] >= 0


def test_false_success_intervention_is_labelled_as_a_full_state_intervention() -> None:
    case = _traced_case()
    payload = case["counterfactuals"]["false_success"]
    assert payload["repair"] == "false_success"
    assert payload["intervention"] == "首次虚假成功处的完整状态干预"
    assert payload["replaces_whole_true_state"] is True
    assert payload["success_classification_only"] is False
    # The same naming rule holds in the historical all-candidate attachment.
    historical = case["counterfactuals_all_candidates_historical"]["false_success"]
    assert historical["intervention"] == "首次虚假成功处的完整状态干预"
