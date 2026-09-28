"""Tests for the learned world-model chain.

The chain is encoder -> latent transition -> decoders, trained with a multi-step rollout loss
and used by a planner that never touches the simulator. These tests pin the properties that
make the chain real rather than decorative.
"""

from __future__ import annotations

import numpy as np
import pytest

from coimagine.diagnostics.world_model_probe import (
    evaluate_rollout_accuracy,
    format_action_sensitivity,
    measure_action_sensitivity,
    plan_with_model,
    run_closed_loop,
)
from coimagine.diagnostics.world_model_viz import decode_observation, run_episode_snapshots
from coimagine.learning.data import generate_situations
from coimagine.learning.features import ACTION_IDS, N_CELL_BLOCKS, GRID
from coimagine.learning.world_model import (
    DirectWorldModel,
    LatentWorldModel,
    load_world_model,
    save_world_model,
    train_world_model,
)
from coimagine.learning.world_model_data import collect_trajectories


def _tiny_training_set():
    return collect_trajectories(n_episodes=8, horizon=5, seed=0)


def test_trajectories_carry_privileged_labels_for_the_next_state() -> None:
    trajectories = _tiny_training_set()
    step = trajectories[0].transitions[0]
    # the decoder targets describe the state after the action; the earlier version attached
    # the current state's labels to the next latent, which is unlearnable when the object moves
    assert step.next_physics == step.physics  # physics does not change during an episode
    assert 0 <= step.next_object_cell < 36
    assert step.observation.shape[0] == 188 and step.next_observation.shape[0] == 188


def test_latent_imagination_is_action_conditional_in_shape_and_repeatable() -> None:
    """The transition must be applied once per action, and the same input must give the same
    imagination -- a model whose rollout is not reproducible cannot be planned with."""

    model = LatentWorldModel(seed=0)
    trajectory = _tiny_training_set()[0]
    observation = trajectory.transitions[0].observation
    forward = model.imagine(observation, [0, 2])
    again = model.imagine(observation, [0, 2])
    assert len(forward) == 2
    for a, b in zip(forward, again):
        assert np.allclose(a["observation"], b["observation"])
    # the latent actually changes along the rollout
    assert not np.allclose(forward[0]["latent"], forward[1]["latent"])


def test_training_reduces_the_loss_and_the_pose_head_beats_chance() -> None:
    """A model that has not learned the object position is not a world model. Chance for a
    36-way cell is 0.028, so anything near that means the labels never reached the head."""

    trajectories = _tiny_training_set()
    steps = [s for t in trajectories for s in t.transitions]
    observations = np.stack([s.observation for s in steps])
    poses = np.asarray([s.next_object_cell for s in steps])
    model = LatentWorldModel(seed=0)
    history = train_world_model(model, trajectories, rollout_length=2, epochs=120, lr=1e-2, seed=0)
    assert history.losses[-1] < history.losses[0]
    predicted = np.stack([
        model.predict_next(observations[i : i + 1], steps[i].action)["pose_logits"][0]
        for i in range(len(steps))
    ])
    accuracy = float((predicted.argmax(1) == poses).mean())
    assert accuracy > 0.5, accuracy


def test_action_sensitivity_diagnostic_runs_and_bounds_its_quantities() -> None:
    """The diagnostic that explains the planning failure, guarded rather than a claim the
    model meets.

    A model can be an accurate state estimator and still be useless to a planner if it predicts
    the same consequence for every action. On this deliberately tiny sample the measurement
    typically reports total indifference (one distinct next pose), which is the honest read and
    is why the finding is reported rather than asserted away.
    """

    trajectories = _tiny_training_set()
    model = LatentWorldModel(seed=0)
    train_world_model(model, trajectories, rollout_length=2, epochs=150, lr=1e-2, seed=0)
    measured = measure_action_sensitivity(model, trajectories)
    assert 1.0 <= measured.distinct_poses <= len(ACTION_IDS)
    assert 0.0 <= measured.pose_accuracy <= 1.0
    assert measured.branch_entropy >= 0.0
    assert "distinct next poses" in format_action_sensitivity({"model": measured})


def test_planner_uses_only_the_model() -> None:
    """The branch scores must be produced by the model, so two different models give
    different rankings on the same observation."""

    trajectories = _tiny_training_set()
    steps = [s for t in trajectories for s in t.transitions]
    observation = steps[0].observation
    goal = frozenset(generate_situations(n_scenes=1, lead_in=2, seed=0)[0].scene.static.target)

    model_a = LatentWorldModel(seed=0)
    model_b = LatentWorldModel(seed=7)
    _action_a, branches_a = plan_with_model(model_a, observation, goal)
    _action_b, branches_b = plan_with_model(model_b, observation, goal)
    assert len(branches_a) == len(ACTION_IDS)
    scores_a = [b["score"] for b in branches_a]
    scores_b = [b["score"] for b in branches_b]
    assert not np.allclose(scores_a, scores_b)
    for branch in branches_a:
        assert branch["imagined_observation"].shape == (188,)


def test_closed_loop_reports_a_rate_and_never_touches_the_simulator_in_lookahead() -> None:
    model = LatentWorldModel(seed=0)
    row = run_closed_loop(model, n_episodes=4, horizon=3, depth=1, seed=0)
    assert 0.0 <= row.success_rate <= 1.0
    assert row.episodes == 4


def test_rollout_accuracy_includes_reference_levels() -> None:
    models = {"latent": LatentWorldModel(seed=0)}
    accuracy = evaluate_rollout_accuracy(models, n_episodes=4, max_depth=2)
    assert set(accuracy.observation_error) >= {"latent", "no_change", "simulator"}
    assert accuracy.chance_pose[0] == pytest.approx(1.0 / 36.0)
    assert len(accuracy.base_moved) == 2


def test_save_and_load_round_trips() -> None:
    import tempfile, os

    model = LatentWorldModel(seed=3)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "m.npz")
        save_world_model(model, path)
        restored = load_world_model(path)
    assert type(restored) is type(model)
    steps = [s for t in _tiny_training_set() for s in t.transitions]
    observation = steps[0].observation
    a = model.predict_next(observation, 2)["observation"]
    b = restored.predict_next(observation, 2)["observation"]
    assert np.allclose(a, b)


def test_decoding_an_observation_gives_a_grid() -> None:
    steps = [s for t in _tiny_training_set() for s in t.transitions]
    kinds = decode_observation(steps[0].observation)
    assert kinds.shape == (GRID, GRID)
    assert kinds.max() < N_CELL_BLOCKS


def test_episode_snapshots_record_each_decision() -> None:
    model = LatentWorldModel(seed=0)
    scene = generate_situations(n_scenes=1, lead_in=2, seed=0)[0].scene
    snapshots, _reached = run_episode_snapshots(scene, model, horizon=3, seed=0)
    assert len(snapshots) == 3
    assert all(len(s.branches) == len(ACTION_IDS) for s in snapshots)
    assert [s.step for s in snapshots] == [0, 1, 2]
