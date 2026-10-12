"""Tests for the conditional (per-parameter) state propagation.

The mechanism is checked with a small synthetic model so nothing depends on a checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from coimagine.diagnostics.conditional_planner import (
    ConditionalState,
    beam_value_conditional,
    compare_records,
    condition,
    correct_observables,
    expand,
    expand_actions,
    per_theta_outcome,
    plan_with_conditional_states,
    run_conditional_episode,
    summarise_records,
)
from coimagine.learning.belief_model import DYNAMIC_DIM, N_PHYSICS
from coimagine.learning.data import probe_scene
from coimagine.grid.state import HiddenPhysics


class _StubModel:
    """Released follows the parameter one-hot; the cell advances by the action index.

    ``observe`` is deterministic per parameter, so a branch either rules a hypothesis out or
    keeps it, which is the sharpest way to see whether the association survives.
    """

    def dynamic_state(self, observation):
        return _vector(cell=16, released=0.0)

    def step(self, dynamic, action, belief):
        dynamic = np.atleast_2d(dynamic)
        belief = np.atleast_2d(belief)
        rows = []
        for index in range(N_PHYSICS):
            cell = int(np.argmax(dynamic[0, 1:37]))
            rows.append(_vector(cell=(cell + action) % 36, released=float(index % 2))[0])
        return np.einsum("nt,ntd->nd", belief, np.stack(rows)[None, :, :])

    def cost_actions(self, states, belief):
        return np.zeros(np.atleast_2d(states).shape[0])

    def observe(self, dynamic, action, belief):
        dynamic = np.atleast_2d(dynamic)
        table = np.zeros((dynamic.shape[0], N_PHYSICS, 6))
        for row in range(dynamic.shape[0]):
            for index in range(N_PHYSICS):
                table[row, index, index % 2] = 1.0
        return table


def _vector(cell: int, released: float) -> np.ndarray:
    vector = np.zeros((1, DYNAMIC_DIM))
    vector[0, 0] = released
    vector[0, 1 + cell] = 1.0
    return vector


def _state(belief: np.ndarray | None = None) -> ConditionalState:
    rows = np.repeat(_vector(cell=16, released=0.0), N_PHYSICS, axis=0)
    return ConditionalState(
        rows, np.full(N_PHYSICS, 1.0 / N_PHYSICS) if belief is None else belief
    )


def test_expand_advances_every_parameter_from_its_own_row():
    model = _StubModel()
    state = _state()
    rows = expand(model, state, action=2)
    assert rows.shape == (N_PHYSICS, DYNAMIC_DIM)
    # the released bit differs across parameters, so the association exists after one step
    released = {round(float(row[0]), 6) for row in rows}
    assert released == {0.0, 1.0}


def test_expand_actions_returns_all_six_and_a_belief_mixed_aggregate():
    model = _StubModel()
    rows, costs, aggregate = expand_actions(model, _state())
    assert rows.shape == (6, N_PHYSICS, DYNAMIC_DIM)
    assert aggregate.shape == (6, DYNAMIC_DIM)
    assert costs.shape == (6,)


def test_per_theta_outcome_reads_each_rows_own_distribution():
    model = _StubModel()
    rows = expand(model, _state(), action=0)
    table = per_theta_outcome(model, rows, action=0)
    assert table.shape == (N_PHYSICS, 6)
    for index in range(N_PHYSICS):
        assert table[index, index % 2] == pytest.approx(1.0)


def test_condition_normalises_and_can_rule_a_hypothesis_out():
    belief = np.full(N_PHYSICS, 1.0 / N_PHYSICS)
    likelihood = np.array([1.0, 0.0] * (N_PHYSICS // 2))
    posterior = condition(belief, likelihood)
    assert posterior.sum() == pytest.approx(1.0)
    assert all(posterior[index] == 0.0 for index in range(1, N_PHYSICS, 2))
    assert all(posterior[index] > 0.0 for index in range(0, N_PHYSICS, 2))
    # an impossible observation must not produce NaN: it keeps the prior
    zero = condition(belief, np.zeros(N_PHYSICS))
    assert zero == pytest.approx(belief)


def test_correct_observables_keeps_each_rows_hidden_released_value():
    rows = np.stack(
        [_vector(cell=16 + index % 5, released=float(index % 2))[0] for index in range(N_PHYSICS)]
    )
    observed = _vector(cell=30, released=0.5)
    corrected = correct_observables(rows, observed)
    assert corrected.shape == rows.shape
    # the cell is taken from the observation for every row ...
    assert all(int(np.argmax(corrected[index, 1:37])) == 30 for index in range(N_PHYSICS))
    # ... and the hidden released value stays with its own parameter
    for index in range(N_PHYSICS):
        assert corrected[index, 0] == pytest.approx(float(index % 2))


def test_conditional_planner_returns_a_valid_action_and_all_branches():
    model = _StubModel()
    goal = frozenset({(4, y) for y in range(6)} | {(5, y) for y in range(6)})
    chosen, branches = plan_with_conditional_states(model, goal, _state(), depth=2, beam=1)
    assert 0 <= chosen < 6
    assert len(branches) == 6
    assert all(np.isfinite(branch["score"]) for branch in branches)
    assert min(branch["score"] for branch in branches) == pytest.approx(
        branches[chosen]["score"]
    )


def test_beam_keeps_whole_conditional_states():
    model = _StubModel()
    goal = frozenset({(4, y) for y in range(6)} | {(5, y) for y in range(6)})
    from coimagine.diagnostics.conditional_planner import _distance_table

    value = beam_value_conditional(
        model, _state(), _distance_table(goal), steps=2, beam=1
    )
    assert np.isfinite(value)


def test_conditional_episode_runs_and_records_actions():
    model = _StubModel()
    scene = probe_scene((1, 2), HiddenPhysics(resistance=1, constraint="free", traction=("grip", "grip")))
    record = run_conditional_episode(model, scene, scene_id=0, horizon=3, depth=1, beam=1)
    assert set(record) >= {"scene_id", "success", "action_cost", "steps", "actions"}
    assert record["steps"] == len(record["actions"])


def test_compare_records_reports_both_directions_of_flip():
    old = [
        {"scene_id": 0, "success": 0, "action_cost": 8.0, "actions": ["wait"], "first_action": "wait", "wait_actions": 1},
        {"scene_id": 1, "success": 1, "action_cost": 2.0, "actions": ["co_push:E"], "first_action": "co_push:E", "wait_actions": 0},
    ]
    new = [
        {"scene_id": 0, "success": 1, "action_cost": 3.0, "actions": ["co_push:E"], "first_action": "co_push:E", "wait_actions": 0},
        {"scene_id": 1, "success": 0, "action_cost": 8.0, "actions": ["wait"], "first_action": "wait", "wait_actions": 1},
    ]
    flips = compare_records(old, new)
    assert [entry["scene_id"] for entry in flips["improved_failure_to_success"]] == [0]
    assert [entry["scene_id"] for entry in flips["regressed_success_to_failure"]] == [1]
    assert flips["wait_first_old"] == 1 and flips["wait_first_new"] == 1


def test_summarise_records_charges_the_failure_cost_once():
    records = [
        {"scene_id": 0, "success": 1, "action_cost": 3.0, "steps": 3, "actions": [], "first_action": None, "wait_actions": 0},
        {"scene_id": 1, "success": 0, "action_cost": 2.0, "steps": 2, "actions": [], "first_action": None, "wait_actions": 0},
    ]
    summary = summarise_records(records)
    assert summary["successes"] == 1
    assert summary["mean_total_cost_all"] == pytest.approx((3.0 + 22.0) / 2)
    assert summary["mean_total_cost_success"] == pytest.approx(3.0)

def test_distance_table_matches_the_repo_definition():
    """Regression: the table was inverted once (min over cells of max over goal)."""

    from coimagine.diagnostics.belief_probe import _cell_distance_distribution
    from coimagine.diagnostics.conditional_planner import _distance_table

    goal = frozenset({(4, y) for y in range(6)} | {(5, y) for y in range(6)})
    assert _distance_table(goal) == pytest.approx(_cell_distance_distribution(None, goal))
    # an object one push from the goal (origin (3, 0), index 3) is at distance 1, not 5-6
    assert _distance_table(goal)[3] == pytest.approx(1.0)


class _CostStubModel(_StubModel):
    """``cost_actions`` pairs row ``a`` with action ``a``, like the real model."""

    def cost_actions(self, states, belief):
        costs = np.asarray([1.0, 1.0, 1.0, 0.3, 1.0, 0.0])
        return costs[: np.atleast_2d(states).shape[0]]


def test_action_costs_are_read_per_action_not_always_action_zero():
    """Regression: a single row scored through ``cost_actions`` always used action 0."""

    model = _CostStubModel()
    rows, costs, aggregate = expand_actions(model, _state())
    assert costs.tolist() == pytest.approx([1.0, 1.0, 1.0, 0.3, 1.0, 0.0])
    goal = frozenset({(4, y) for y in range(6)} | {(5, y) for y in range(6)})
    _chosen, branches = plan_with_conditional_states(model, goal, _state(), depth=1, beam=1)
    by_action = {branch["action"]: branch for branch in branches}
    assert by_action["wait"]["first_cost"] == pytest.approx(0.0)
    assert by_action["push:0:E"]["first_cost"] == pytest.approx(1.0)
    assert by_action["probe:0:E"]["first_cost"] == pytest.approx(0.3)

def test_scan_correction_keeps_every_compatible_placement_when_ambiguous():
    """The scan is a constraint: an ambiguous scan must not be turned into one point."""

    from coimagine.diagnostics.conditional_planner import correct_observables_scan
    from coimagine.grid.sensing import OccupancyScan
    from coimagine.grid.state import StaticMap

    static = StaticMap(width=6, height=6, target=frozenset({(4, y) for y in range(6)} | {(5, y) for y in range(6)}))
    # no object seen; one observed robot cell rules out the placements containing it
    scan = OccupancyScan(0, (0, 0), 5, {(0, 0): "robot"})
    # every row is concentrated on (0, 0), which the observed robot cell rules out, so the
    # correction must fall back to the whole compatible set rather than invent a placement
    rows = np.repeat(_vector(cell=0, released=0.0), N_PHYSICS, axis=0)
    rows[:, 0] = np.array([index % 2 for index in range(N_PHYSICS)], dtype=float)
    observation = np.zeros(188)
    observation[180:182] = 0.0
    corrected = correct_observables_scan(rows, observation, scan, static)
    support = (corrected[0, 1:37] > 0).sum()
    assert support > 1, "an ambiguous scan must keep several compatible placements"
    assert abs(corrected[0, 1:37].sum() - 1.0) < 1e-9
    # a row that is already compatible keeps its own belief: the scan is a constraint
    rows2 = np.repeat(_vector(cell=16, released=0.0), N_PHYSICS, axis=0)
    rows2[:, 0] = np.array([index % 2 for index in range(N_PHYSICS)], dtype=float)
    corrected2 = correct_observables_scan(rows2, observation, scan, static)
    assert int(np.argmax(corrected2[0, 1:37])) == 16
    # the hidden released value stays with its own row in both cases
    assert corrected[:, 0].tolist() == pytest.approx(rows[:, 0].tolist())
    assert corrected2[:, 0].tolist() == pytest.approx(rows2[:, 0].tolist())


def test_scan_correction_uses_the_unique_placement_when_the_object_is_seen():
    from coimagine.diagnostics.conditional_planner import correct_observables_scan
    from coimagine.grid.sensing import OccupancyScan
    from coimagine.grid.state import StaticMap

    static = StaticMap(width=6, height=6, target=frozenset())
    observed = {(2, 2): "object", (3, 2): "object", (2, 3): "object", (3, 3): "object"}
    scan = OccupancyScan(0, (1, 2), 5, observed)
    rows = np.repeat(_vector(cell=16, released=0.0), N_PHYSICS, axis=0)
    observation = np.zeros(188)
    observation[180:182] = np.asarray([1.0, 2.0]) / 5.0
    corrected = correct_observables_scan(rows, observation, scan, static)
    assert int(np.argmax(corrected[0, 1:37])) == 2 + 6 * 2
    assert corrected[0, 1 + 2 + 6 * 2] == pytest.approx(1.0)


def test_true_geometry_correction_is_a_privileged_one_hot_and_keeps_released():
    from coimagine.diagnostics.conditional_planner import correct_observables_true_geometry
    from coimagine.grid.sensing import OccupancyScan

    scan = OccupancyScan(0, (0, 0), 5, {(0, 0): "robot"})
    rows = np.repeat(_vector(cell=16, released=0.0), N_PHYSICS, axis=0)
    rows[:, 0] = np.array([(index + 1) % 2 for index in range(N_PHYSICS)], dtype=float)
    observation = np.zeros(188)
    corrected = correct_observables_true_geometry(rows, (3, 1), observation, scan)
    assert int(np.argmax(corrected[0, 1:37])) == 3 + 6 * 1
    assert corrected[0, 1 + 3 + 6 * 1] == pytest.approx(1.0)
    assert corrected[:, 0].tolist() == pytest.approx(rows[:, 0].tolist())


def test_conditional_episode_accepts_each_correction_rule():
    model = _StubModel()
    scene = probe_scene((1, 2), HiddenPhysics(resistance=1, constraint="free", traction=("grip", "grip")))
    for correction in ("encoder", "scan", "true_geometry"):
        record = run_conditional_episode(
            model, scene, scene_id=0, horizon=2, depth=1, beam=1, correction=correction
        )
        assert record["steps"] == len(record["actions"])
