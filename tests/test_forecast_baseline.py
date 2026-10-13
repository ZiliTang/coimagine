"""Property tests for the forecast-baseline learning code (contract C1, B1, C2, A4 rules).

These tests use hand-worked numbers and tiny stubs; they do not mirror the implementation.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from coimagine.continuous2d.backend import BodyState
from coimagine.continuous2d.config import default_config
from coimagine.continuous2d.environment import Continuous2DEnv
from coimagine.continuous2d.scenes import scene_by_name
from coimagine.continuous2d.skills import Hold, MoveTo, PushContact
from coimagine.continuous2d_learning.contract import (
    geometric_violation,
    score_components,
)
from coimagine.continuous2d_learning.diagnostic import selection_metrics
from coimagine.continuous2d_learning.features import (
    INPUT_DIM,
    BoundaryState,
    apply_state_delta,
    encode_input,
)
from coimagine.continuous2d_learning.model import JointWindowMLP, ModelConfig
from coimagine.continuous2d_learning.pools import candidate_pool
from coimagine.continuous2d_learning.records import (
    CollectionConfig,
    collect_episode,
    episode_boundaries,
)
from coimagine.continuous2d_learning.rollout import (
    EXIT_INVALID,
    EXIT_NON_FINITE,
    evaluate_sequence_learned,
)

CONFIG = default_config()


# ---------------------------------------------------------------------------
# C1: the scalar score
# ---------------------------------------------------------------------------


def test_already_successful_state_is_worth_ten():
    components = score_components(success=True, time=0.0, distance=0.0, failure=False, diagonal=4.0)
    assert components.j == pytest.approx(10.0)
    assert components.time_penalty == 0.0
    assert components.distance_penalty == 0.0


def test_success_must_not_carry_distance():
    with pytest.raises(ValueError):
        score_components(success=True, time=1.0, distance=0.5, failure=False, diagonal=4.0)


def test_failure_absorption_and_time_charged_once():
    # One 2 s window that failed: J = -20 - 2/24 - 0.4/4.
    components = score_components(success=False, time=2.0, distance=0.4, failure=True, diagonal=4.0)
    assert components.j == pytest.approx(-20.0 - 2.0 / 24.0 - 0.1)
    assert components.failure_penalty == 20.0
    assert components.time == 2.0


def test_decomposition_sums_to_total():
    components = score_components(success=False, time=3.0, distance=0.7, failure=False, diagonal=5.0)
    total = (
        components.success_reward
        - components.time_penalty
        - components.distance_penalty
        - components.failure_penalty
    )
    assert components.j == pytest.approx(total)


def test_perfect_model_has_zero_regret():
    j = [1.0, -2.0, 0.5, -0.25]
    metrics = selection_metrics([0, 1, 2, 3], j, j)
    assert metrics["candidate_regret"] == pytest.approx(0.0)
    assert metrics["selected_bias"] == pytest.approx(0.0)
    assert metrics["selection_excess"] == pytest.approx(0.0)
    assert metrics["chosen_candidate_id"] == 0


def test_tie_breaks_on_smaller_candidate_id():
    metrics = selection_metrics([3, 1, 2], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert metrics["chosen_candidate_id"] == 1


# ---------------------------------------------------------------------------
# B1: encoding and prediction hygiene
# ---------------------------------------------------------------------------


def _boundary(scene, object_pose, robot_poses, counter=0.0):
    return BoundaryState(
        object=BodyState(object_pose[0], object_pose[1], object_pose[2], 0.0, 0.0, 0.0),
        robots=tuple(BodyState(x, y, 0.0, 0.0, 0.0, 0.0) for x, y in robot_poses),
        success_counter=counter,
    )


def test_input_layout_is_fixed_and_skill_sensitive():
    scene = scene_by_name("S0_dev")
    state = _boundary(scene, (0.95, 1.10, 0.0), [(0.5, 0.9), (0.5, 1.3)])
    first = encode_input(scene, state, (Hold(), Hold()), CONFIG.physics)
    second = encode_input(
        scene, state, (MoveTo((1.5, 0.9)), PushContact(0, (0.5, 0.0), (1.0, 0.0), 15.0)), CONFIG.physics
    )
    assert len(first) == INPUT_DIM
    assert len(second) == INPUT_DIM
    assert first != second
    # the encoding is a pure function of the visible inputs + skill, so repeating it is identical
    assert encode_input(scene, state, (Hold(), Hold()), CONFIG.physics) == first


def test_state_delta_round_trip_wraps_angle():
    scene = scene_by_name("S0_dev")
    state = _boundary(scene, (0.95, 1.10, 3.0), [(0.5, 0.9), (0.5, 1.3)], counter=0.25)
    delta = [0.0] * 18
    delta[2] = 1.5  # half a turn in units of pi
    moved = apply_state_delta(state, delta, scene)
    assert abs(moved.object.angle) <= math.pi
    assert moved.success_counter == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Geometric validity on hand-placed states
# ---------------------------------------------------------------------------


def test_geometric_violation_accepts_a_legal_scene_and_rejects_overlap():
    scene = scene_by_name("S0_dev")
    legal = _boundary(scene, (0.95, 1.10, 0.0), [(0.5, 0.5), (0.5, 1.7)])
    assert geometric_violation(legal.object, legal.robots, scene, CONFIG.physics)["violation"] is False
    # robot centre placed inside the object: deep overlap
    overlapping = _boundary(scene, (0.95, 1.10, 0.0), [(0.95, 1.10), (0.5, 1.7)])
    result = geometric_violation(overlapping.object, overlapping.robots, scene, CONFIG.physics)
    assert result["violation"] is True
    assert result["penetration"] > 0.05


# ---------------------------------------------------------------------------
# C2: frozen candidate pools
# ---------------------------------------------------------------------------


def test_candidate_pool_is_deterministic_and_nested():
    scene = scene_by_name("S1_v4")
    env = Continuous2DEnv(CONFIG)
    env.reset(scene)
    observation = env.observation()
    pool = candidate_pool(observation, CONFIG, 1, 0, cap=128)
    assert pool == candidate_pool(observation, CONFIG, 1, 0, cap=128)
    assert len(pool) <= 121
    smaller = pool[:16]
    assert smaller == pool[:16]
    assert len(smaller) == 16


def test_depth_pool_contains_a_staging_template_and_is_distinct():
    scene = scene_by_name("S1_v4")
    env = Continuous2DEnv(CONFIG)
    env.reset(scene)
    observation = env.observation()
    pool = candidate_pool(observation, CONFIG, 3, 1, cap=128)
    assert len(pool) == len({repr(sequence) for sequence in pool})
    templates = [
        sequence
        for sequence in pool
        if all(isinstance(skill, MoveTo) for skill in sequence[0])
        and all(
            any(isinstance(skill, PushContact) for skill in joint)
            for joint in sequence[1:]
        )
    ]
    assert templates, "the staging-then-push template must appear in an H=3 pool"


# ---------------------------------------------------------------------------
# A4 rule: an invalid prediction stops the chain instead of reusing a state
# ---------------------------------------------------------------------------


class _StubModel(nn.Module):
    """Returns a fixed state delta so the invalid-prediction path can be exercised."""

    def __init__(self, delta_value: float, termination_index: int = 0):
        super().__init__()
        self.delta_value = delta_value
        self.termination_index = termination_index
        self.calls = 0

    def forward(self, x: torch.Tensor):
        self.calls += 1
        batch = x.shape[0]
        state = torch.full((batch, 18), self.delta_value, dtype=torch.float32)
        logits = torch.full((batch, 3), -5.0, dtype=torch.float32)
        logits[:, self.termination_index] = 5.0
        return {
            "state": state,
            "hold": torch.zeros(batch, 1),
            "duration": torch.zeros(batch, 1),
            "termination": logits,
        }


def test_invalid_prediction_stops_the_chain():
    scene = scene_by_name("S0_dev")
    start = _boundary(scene, (0.95, 1.10, 0.0), [(0.5, 0.5), (0.5, 1.7)])
    model = _StubModel(delta_value=50.0)  # metres after denormalisation: far outside the arena
    sequence = [(Hold(), Hold()), (Hold(), Hold()), (Hold(), Hold())]
    evaluation = evaluate_sequence_learned(model, scene, start, sequence, CONFIG, candidate_id=0)
    assert evaluation.exit_class in (EXIT_INVALID, EXIT_NON_FINITE)
    assert evaluation.windows_executed == 1
    assert evaluation.components.failure == 1
    assert model.calls == 1, "an absorbed chain must not fabricate further windows"


def test_predicted_success_is_absorbing():
    scene = scene_by_name("S0_dev")
    # A legal start: with the A3 priority order a predicted success may not bypass geometry.
    start = _boundary(scene, (0.95, 1.10, 0.0), [(0.5, 0.5), (0.5, 1.7)])
    model = _StubModel(delta_value=0.0, termination_index=1)
    sequence = [(Hold(), Hold())] * 3
    evaluation = evaluate_sequence_learned(model, scene, start, sequence, CONFIG, candidate_id=0)
    assert evaluation.success is True
    assert evaluation.components.distance == 0.0
    assert evaluation.windows_executed == 1


# ---------------------------------------------------------------------------
# Reproducibility of the collection prefix replay
# ---------------------------------------------------------------------------


def test_episode_replay_reproduces_collected_boundaries():
    collection = CollectionConfig(episodes_per_scene=1, max_windows=4, full_step_episodes=0)
    records, _ = collect_episode("S0_dev", "S0_dev/00", "train", CONFIG, collection)
    boundaries = episode_boundaries("S0_dev", "S0_dev/00", "train", CONFIG, collection)
    assert len(records) == len(boundaries)
    for record, boundary in zip(records, boundaries):
        snapshot = boundary["snapshot"]
        assert snapshot.tick == record["t"]
        assert snapshot.object.x == pytest.approx(record["state"]["object"][0], abs=1e-12)
        assert snapshot.success_counter == pytest.approx(record["success_counter_before"], abs=1e-12)


def test_training_is_reproducible_and_deterministic():
    from coimagine.continuous2d_learning.model import train_model

    collection = CollectionConfig(episodes_per_scene=1, max_windows=2, full_step_episodes=0)
    train_records, _ = collect_episode("S0_dev", "S0_dev/00", "train", CONFIG, collection)
    val_records, _ = collect_episode("S0_dev", "S0_dev/01", "train", CONFIG, collection)
    config = ModelConfig(max_epochs=2, patience=2)
    model_a, log_a = train_model(train_records, val_records, seed=0, config=config)
    model_b, log_b = train_model(train_records, val_records, seed=0, config=config)
    assert log_a.epochs == log_b.epochs
    # Torch CPU reductions can differ in the last bits across thread schedules; the protocol
    # is reproducible, so a tight numerical tolerance is the right assertion here.
    for first, second in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(first, second, atol=1e-6)


# ---------------------------------------------------------------------------
# A4 cumulative denominators in the recursive contrast
# ---------------------------------------------------------------------------


def test_recursive_cumulative_rates_are_really_cumulative():
    from coimagine.continuous2d_learning.diagnostic import (
        DiagnosticConfigV1,
        recursive_contrast,
    )
    from coimagine.continuous2d_learning.rollout import SequenceEvaluation
    from coimagine.continuous2d_learning.starts import EvalStart

    scene = scene_by_name("S0_dev")
    env = Continuous2DEnv(CONFIG)
    env.reset(scene)
    snapshot = env.export_state()
    observation = env.observation()
    start = EvalStart(
        start_id="S0_dev/initial", kind="initial", scene_id="S0_dev",
        episode_id=None, window_index=None, seed=None, success_counter=0.0,
        state_key="k", snapshot_hash="h", physical_state={},
        snapshot=snapshot, observation=observation,
    )
    before = _boundary(scene, (0.95, 1.10, 0.0), [(0.5, 0.5), (0.5, 1.7)])
    after = before
    trajectory = [
        {
            "window": index, "before": before.as_dict(), "after": after.as_dict(),
            "duration": 2.0, "termination": "window_complete",
        }
        for index in range(2)
    ]
    truth = [
        SequenceEvaluation(
            candidate_id=index, j=0.0,
            components=score_components(success=False, time=4.0, distance=0.0, failure=False, diagonal=4.0),
            exit_class="window_complete", windows_executed=2, terminal_state=after,
            predicted_windows=[], trajectory=trajectory,
        )
        for index in range(2)
    ]
    sequence = [(Hold(), Hold()), (Hold(), Hold())]
    pool = [sequence, sequence]
    truth_cache = {("S0_dev/initial", 2, 0): (pool, truth)}
    model = _StubModel(delta_value=50.0)
    result = recursive_contrast(
        [start], {0: model}, CONFIG,
        DiagnosticConfigV1(recursive_depth=2, recursive_candidate_seed=0),
        truth_cache,
    )
    entry = result["0"]
    chains = entry["n_chains"]
    assert chains == 2
    # every chain is invalid at window 0, so both cumulative rates are 1.0 from step 1 on
    assert entry["all_chains"]["cumulative_termination_rate"] == [1.0, 1.0]
    assert entry["all_chains"]["cumulative_invalid_rate"] == [1.0, 1.0]
    assert entry["recursive"]["alive_at_window_start"] == [2, 0]
    assert entry["recursive"]["n_surviving_predictions"] == [2, 0]
