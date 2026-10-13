"""Tests for the data_upgrade_v1 fixes (A3 closed-loop feasibility, geometry bounds).

Scenarios are constructed by scripted model outputs and fixed candidate pools; no model is
trained and no scene is changed to manufacture a case.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from coimagine.continuous2d.backend import BodyState
from coimagine.continuous2d.config import default_config
from coimagine.continuous2d.environment import Continuous2DEnv
from coimagine.continuous2d.scenes import scene_by_name
from coimagine.continuous2d.skills import Hold, MoveTo, PushContact
from coimagine.continuous2d_learning import closed_loop as closed_loop_module
from coimagine.continuous2d_learning.closed_loop import (
    METHOD_LEARNED,
    ClosedLoopConfig,
    run_episode,
)
from coimagine.continuous2d_learning.contract import geometric_violation
from coimagine.continuous2d_learning.features import BoundaryState
from coimagine.continuous2d_learning.rollout import (
    EXIT_INVALID,
    EXIT_NON_FINITE,
    EXIT_PHYSICAL_FAILURE,
    EXIT_SUCCESS,
    EXIT_WINDOW_COMPLETE,
    evaluate_sequence_learned,
)

CONFIG = default_config()
SCENE_NAME = "S0_dev"
KIND_MOVE, KIND_PUSH, KIND_HOLD = 0, 1, 2


class ScriptedModel(nn.Module):
    """A model whose output depends only on the two skill kinds in the input vector."""

    def __init__(self, plan):
        super().__init__()
        # plan[(kind_a, kind_b)] = {"delta": float, "termination": int, "finite": bool}
        self.plan = plan
        self.calls = 0

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        state = torch.zeros(batch, 18, dtype=torch.float32)
        logits = torch.full((batch, 3), -5.0, dtype=torch.float32)
        for row in range(batch):
            kind_a = int(torch.argmax(x[row, 41:44]).item())
            kind_b = int(torch.argmax(x[row, 49:52]).item())
            plan = self.plan.get((kind_a, kind_b), {"delta": 0.0, "termination": 0, "finite": True})
            self.calls += 1
            if plan.get("finite", True):
                state[row, :] = float(plan["delta"])
            else:
                state[row, :] = float("nan")
            logits[row, int(plan["termination"])] = 5.0
        return {
            "state": state,
            "hold": torch.zeros(batch, 1),
            "duration": torch.zeros(batch, 1),
            "termination": logits,
        }


def _start():
    scene = scene_by_name(SCENE_NAME)
    env = Continuous2DEnv(CONFIG)
    env.reset(scene)
    observation = env.observation()
    return scene, env, BoundaryState(
        object=observation.object, robots=observation.robots, success_counter=0.0
    )


# ---------------------------------------------------------------------------
# A3: legality has priority, and feasibility is the selection filter
# ---------------------------------------------------------------------------


def test_invalid_and_success_conflict_is_still_invalid():
    scene, _env, start = _start()
    model = ScriptedModel(
        {(KIND_PUSH, KIND_HOLD): {"delta": 50.0, "termination": 1, "finite": True}}
    )
    evaluation = evaluate_sequence_learned(
        model, scene, start, [(PushContact(0, (0.5, 0.0), (1.0, 0.0), 15.0), Hold())],
        CONFIG, candidate_id=0,
    )
    assert evaluation.exit_class == EXIT_INVALID
    assert evaluation.feasible is False
    assert evaluation.success is False


def test_non_finite_prediction_is_not_feasible():
    scene, _env, start = _start()
    model = ScriptedModel({(KIND_MOVE, KIND_HOLD): {"delta": 0.0, "termination": 0, "finite": False}})
    evaluation = evaluate_sequence_learned(
        model, scene, start, [(MoveTo((1.0, 1.0)), Hold())], CONFIG, candidate_id=0
    )
    assert evaluation.exit_class == EXIT_NON_FINITE
    assert evaluation.feasible is False


def test_legal_zero_delta_stays_feasible():
    scene, _env, start = _start()
    model = ScriptedModel({})
    evaluation = evaluate_sequence_learned(
        model, scene, start, [(Hold(), Hold())], CONFIG, candidate_id=0
    )
    assert evaluation.exit_class == EXIT_WINDOW_COMPLETE
    assert evaluation.feasible is True


def test_already_successful_start_is_absorbed_without_a_model_call():
    scene, _env, start = _start()
    absorbed = BoundaryState(
        object=start.object, robots=start.robots, success_counter=CONFIG.physics.success_hold_time
    )
    model = ScriptedModel({})
    evaluation = evaluate_sequence_learned(
        model, scene, absorbed, [(Hold(), Hold())], CONFIG, candidate_id=0
    )
    assert evaluation.exit_class == EXIT_SUCCESS
    assert evaluation.components.distance == 0.0
    assert evaluation.components.time == 0.0
    assert model.calls == 0


def test_already_failed_start_is_absorbed():
    scene, _env, start = _start()
    model = ScriptedModel({(KIND_MOVE, KIND_HOLD): {"delta": 0.0, "termination": 2, "finite": True}})
    evaluation = evaluate_sequence_learned(
        model, scene, start, [(MoveTo((1.0, 1.0)), Hold())], CONFIG, candidate_id=0
    )
    assert evaluation.exit_class == EXIT_PHYSICAL_FAILURE
    assert evaluation.feasible is False


# ---------------------------------------------------------------------------
# A3: closed loop rejects infeasible pools and never steps the live world
# ---------------------------------------------------------------------------


def _fixed_pool(specs):
    """A one-window sequence per spec, so the pool matches ``candidate_pool``'s shape."""

    def _pool(observation, config, depth, seed, *, cap=128):
        return [(spec,) for spec in specs]

    return _pool


def test_all_infeasible_pool_stops_without_stepping(monkeypatch):
    scene, env, start = _start()
    model = ScriptedModel(
        {
            (KIND_MOVE, KIND_HOLD): {"delta": 50.0, "termination": 0, "finite": True},
            (KIND_PUSH, KIND_HOLD): {"delta": 50.0, "termination": 0, "finite": True},
        }
    )
    specs = [(MoveTo((1.0, 1.0)), Hold()), (PushContact(0, (0.5, 0.0), (1.0, 0.0), 15.0), Hold())]
    monkeypatch.setattr(closed_loop_module, "candidate_pool", _fixed_pool(specs))
    record = run_episode(
        METHOD_LEARNED, SCENE_NAME, 0, 1, CONFIG,
        closed=ClosedLoopConfig(horizon=12), model=model, model_seed=0,
    )
    assert record["no_feasible_plan"] == 1
    assert record["termination"] == "planner_no_feasible_plan"
    assert record["windows"] == 1, "no window may be executed after an all-infeasible pool"
    assert record["decisions"][0]["stop_tick"] == 0
    assert record["decisions"][0]["rejected_candidates"] == 2
    assert record["decisions"][0]["rejection_rate"] == pytest.approx(1.0)
    assert record["physical_failure"] == 0


def test_mixed_pool_selects_only_the_feasible_candidate(monkeypatch):
    _scene, _env, _start_state = _start()
    model = ScriptedModel(
        {
            (KIND_MOVE, KIND_HOLD): {"delta": 50.0, "termination": 1, "finite": True},
            (KIND_PUSH, KIND_HOLD): {"delta": 50.0, "termination": 0, "finite": True},
        }
    )
    specs = [
        (MoveTo((1.0, 1.0)), Hold()),
        (PushContact(0, (0.5, 0.0), (1.0, 0.0), 15.0), Hold()),
        (Hold(), Hold()),
    ]
    monkeypatch.setattr(closed_loop_module, "candidate_pool", _fixed_pool(specs))
    record = run_episode(
        METHOD_LEARNED, SCENE_NAME, 0, 1, CONFIG,
        closed=ClosedLoopConfig(horizon=1), model=model, model_seed=0,
    )
    decision = record["decisions"][0]
    assert decision["feasible_count"] == 1
    assert decision["rejected_candidates"] == 2
    assert decision["predicted_exit_class"] == EXIT_WINDOW_COMPLETE
    assert decision["chosen_candidate_id"] == 2, "only the legal hold-hold candidate is feasible"


# ---------------------------------------------------------------------------
# A3: y uses the arena height
# ---------------------------------------------------------------------------


def test_out_of_bounds_uses_arena_height_for_y():
    scene = scene_by_name(SCENE_NAME)
    width, height = scene.arena
    margin = 0.5
    assert height + margin + 0.1 < width + margin, "the test needs a y beyond height but below width"
    _scene, _env, start = _start()

    def with_robot0_y(y):
        robots = (BodyState(start.robots[0].x, y, 0.0, 0.0, 0.0, 0.0), start.robots[1])
        return start.object, robots

    # inside the y margin: not an out-of-bounds violation
    inside = geometric_violation(*with_robot0_y(height + margin - 0.05), scene, CONFIG.physics)
    assert "out_of_bounds" not in inside["reasons"]
    # beyond the y margin but still below width + margin: the old code compared y with width
    beyond = geometric_violation(*with_robot0_y(height + margin + 0.1), scene, CONFIG.physics)
    assert "out_of_bounds" in beyond["reasons"]


# ---------------------------------------------------------------------------
# A3/A4: feasibility filtering and tolerance-based chain accounting
# ---------------------------------------------------------------------------


def test_feasibility_metrics_reports_na_without_feasible_candidates():
    from coimagine.continuous2d_learning.diagnostic import _feasibility_metrics
    from coimagine.continuous2d_learning.rollout import SequenceEvaluation
    from coimagine.continuous2d_learning.contract import score_components

    def make(exit_class):
        return SequenceEvaluation(
            candidate_id=0, j=-1.0,
            components=score_components(success=False, time=1.0, distance=0.1, failure=False, diagonal=4.0),
            exit_class=exit_class, windows_executed=1, terminal_state=None,
            predicted_windows=[], trajectory=[],
        )

    evaluations = [make(EXIT_INVALID), make(EXIT_NON_FINITE), make(EXIT_PHYSICAL_FAILURE)]
    metrics = _feasibility_metrics([0, 1, 2], np.asarray([-20.0, -20.0, -20.0]), np.asarray([-0.1, -0.2, -0.3]), evaluations, evaluations)
    assert metrics["feasible_count"] == 0
    assert metrics["rejected_count"] == 3
    assert metrics["rejection_rate"] == 1.0
    assert metrics["conditional_regret"] is None, "NA must not be replaced by 0"
    assert metrics["feasible_chosen_candidate_id"] is None


def test_feasibility_metrics_selects_only_feasible():
    from coimagine.continuous2d_learning.diagnostic import _feasibility_metrics
    from coimagine.continuous2d_learning.rollout import SequenceEvaluation
    from coimagine.continuous2d_learning.contract import score_components

    def make(candidate_id, exit_class, j):
        return SequenceEvaluation(
            candidate_id=candidate_id, j=j,
            components=score_components(success=False, time=1.0, distance=0.1, failure=False, diagonal=4.0),
            exit_class=exit_class, windows_executed=1, terminal_state=None,
            predicted_windows=[], trajectory=[],
        )

    evaluations = [make(0, EXIT_INVALID, -20.0), make(1, EXIT_WINDOW_COMPLETE, -0.5), make(2, EXIT_SUCCESS, 9.0)]
    truth = [make(0, EXIT_WINDOW_COMPLETE, -0.4), make(1, EXIT_WINDOW_COMPLETE, -0.6), make(2, EXIT_SUCCESS, 9.5)]
    metrics = _feasibility_metrics([0, 1, 2], np.asarray([-20.0, -0.5, 9.0]), np.asarray([-0.4, -0.6, 9.5]), evaluations, truth)
    assert metrics["feasible_count"] == 2
    assert metrics["feasible_chosen_candidate_id"] == 2
    assert metrics["conditional_regret"] == pytest.approx(0.0)
    assert metrics["feasible_chosen_true_return"] == pytest.approx(9.5)


def _chain_record(state, next_state, termination_class, success, window_index):
    return {
        "schema_version": 2, "record_version": "test", "episode_id": "ep", "scene_id": SCENE_NAME,
        "scene_version": "closeout_v2", "scene_content_hash": "h", "split": "test", "seed": 0,
        "window_index": window_index, "t": 0,
        "state": {"boundary": "before_window", "robots": state[1], "object": state[0]},
        "next_state": {"boundary": "after_window", "robots": next_state[1], "object": next_state[0]},
        "joint_skill_specs": [{"kind": "hold"}, {"kind": "hold"}],
        "joint_skill_labels": ["hold", "hold"], "duration": 2.0, "duration_steps": 40,
        "termination": "success" if success else "window_complete",
        "termination_class": termination_class, "success": int(success),
        "success_counter_before": 0.0, "success_counter_after": 0.5 if success else 0.0,
        "contact": {"had_robot_object_contact": False, "max_penetration": 0.0, "penetration_by_pair": {}},
        "cost_components": {"time": 2.0, "steps": 40.0, "force_usage": 0.0},
        "config_hash": "c", "controller_version": "v", "code_version": "x", "source": {},
    }


def test_chain_metric_does_not_count_a_correct_success_absorption_as_error():
    from coimagine.data_upgrade.metrics import chain_tolerance_metrics

    _scene, env, start = _start()
    state = (list(start.object.as_tuple()), [list(s.as_tuple()) for s in start.robots])
    records = [
        _chain_record(state, state, "window_complete", False, 0),
        _chain_record(state, state, "success", True, 1),
    ]
    model = ScriptedModel({})  # legal zero delta, window_complete every window
    result = chain_tolerance_metrics(model, records, splits=["test"])
    # the model never predicts success, so the true success at window 2 is a missed termination,
    # but the accounting must remain finite and separated
    assert result["chains"] == 1
    assert len(result["all_chains"]["cumulative_termination_rate"]) == 2
    rates = result["all_chains"]["cumulative_termination_rate"]
    assert all(0.0 <= value <= 1.0 for value in rates)
    assert rates == sorted(rates), "cumulative rates must be non-decreasing"
    assert result["teacher_forced"]["n_predictions"] == [1, 1]
    assert result["recursive_survivors"]["alive_at_window_start"] == [1, 1]


def test_chain_metric_counts_wrong_termination_separately():
    from coimagine.data_upgrade.metrics import chain_tolerance_metrics

    _scene, env, start = _start()
    state = (list(start.object.as_tuple()), [list(s.as_tuple()) for s in start.robots])
    records = [_chain_record(state, state, "window_complete", False, index) for index in range(3)]
    model = ScriptedModel({(KIND_HOLD, KIND_HOLD): {"delta": 0.0, "termination": 1, "finite": True}})
    result = chain_tolerance_metrics(model, records, splits=["test"])
    assert result["all_chains"]["cumulative_wrong_termination_rate"][0] == 1.0
    assert result["teacher_forced"]["wrong_termination_rate"][0] == 1.0
    assert result["all_chains"]["cumulative_invalid_rate"][0] == 0.0
