"""Tests for the continuous contact parameterisation and its learning probe."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from coimagine.diagnostics.calibration import build_contact_scene
from coimagine.diagnostics.continuous_probe import (
    DECISION_THRESHOLD,
    format_continuous_markdown,
    run_continuous_probe,
)
from coimagine.grid.continuous import (
    FORCE_MAX,
    feasible_mask,
    force_threshold,
    nearest_axis,
    push_outcome,
)
from coimagine.grid.state import Direction, HiddenPhysics
from coimagine.learning.continuous_data import (
    CONTINUOUS_ACTION_DIM,
    DEFAULT_ANGLES,
    DEFAULT_FORCES,
    continuous_action_vector,
    continuous_feature_dim,
    generate_continuous_dataset,
    to_arrays,
)


def _scene(physics: HiddenPhysics, released: bool = False):
    return replace(build_contact_scene(), physics=physics, released=released)


def test_nearest_axis_picks_the_closest_direction() -> None:
    assert nearest_axis(0.0)[0] is Direction.EAST
    assert nearest_axis(30.0)[0] is Direction.EAST
    assert nearest_axis(90.0)[0] is Direction.NORTH
    assert nearest_axis(200.0)[0] is Direction.WEST
    # the alignment is the cosine to the *nearest* axis, so it never drops below cos(45)
    assert nearest_axis(0.0)[1] == pytest.approx(1.0)
    # at 60 degrees the nearest axis is north (cos 30), not east (cos 60)
    assert nearest_axis(60.0)[0] is Direction.NORTH
    assert nearest_axis(60.0)[1] == pytest.approx(0.8660254)
    # exactly between two axes the tie is broken by the fixed axis order (east first)
    assert nearest_axis(45.0)[0] is Direction.EAST


def test_force_threshold_is_set_by_the_hidden_physics() -> None:
    light = _scene(HiddenPhysics(1, "free", ("grip", "grip")))
    heavy = _scene(HiddenPhysics(2, "free", ("grip", "grip")))
    slipping = _scene(HiddenPhysics(1, "free", ("slip", "grip")))
    assert force_threshold(light, [0]) == pytest.approx(1.0)
    assert force_threshold(heavy, [0]) == pytest.approx(2.0)
    assert force_threshold(slipping, [0]) == float("inf")
    # two gripping robots halve the per-robot requirement
    assert force_threshold(heavy, [0, 1]) == pytest.approx(1.0)


def test_feasible_region_is_a_rectangle_in_angle_and_force() -> None:
    scene = _scene(HiddenPhysics(1, "free", ("grip", "grip")))
    angles = (-50.0, -40.0, 0.0, 40.0, 50.0)
    below = feasible_mask(scene, {0: 0.75}, angles)
    at = feasible_mask(scene, {0: 1.0}, angles)
    # force threshold: 0.75 is too weak everywhere
    assert not any(below)
    # at the threshold the angular window is feasible, the outside samples are not
    assert at == [False, True, True, True, False]


def test_latched_objects_are_never_feasible_until_released() -> None:
    latched = _scene(HiddenPhysics(1, "constrained", ("grip", "grip")))
    released = _scene(HiddenPhysics(1, "constrained", ("grip", "grip")), released=True)
    assert not push_outcome(latched, {0: 2.0}, 0.0).moved
    assert push_outcome(released, {0: 2.0}, 0.0).moved


def test_outcome_reports_the_reason() -> None:
    scene = _scene(HiddenPhysics(2, "free", ("grip", "grip")))
    weak = push_outcome(scene, {0: 1.0}, 0.0)
    assert not weak.moved and weak.reason == "insufficient_force"
    strong = push_outcome(scene, {0: 2.0}, 0.0)
    assert strong.moved and strong.reason == "moved"


def test_continuous_features_have_the_declared_dimension() -> None:
    from coimagine.learning.features import OBSERVATION_DIM

    assert continuous_feature_dim() == OBSERVATION_DIM + CONTINUOUS_ACTION_DIM
    assert len(continuous_action_vector(0.0, FORCE_MAX)) == CONTINUOUS_ACTION_DIM
    # the action encoding is bounded and direction-preserving
    s, c, f = continuous_action_vector(0.0, 1.0)
    assert (s, c) == pytest.approx((0.0, 1.0))
    assert f == pytest.approx(0.5)


def test_continuous_dataset_shape_and_labels() -> None:
    angles = (-10.0, 0.0, 10.0)
    forces = (0.5, 1.0)
    samples = generate_continuous_dataset(
        n_scenes=4, lead_in=1, seed=0, angles=angles, forces=forces
    )
    assert len(samples) == 4 * len(angles) * len(forces)
    obs, action, y, physics_index, pose_index = to_arrays(samples)
    assert obs.shape[1] + action.shape[1] == continuous_feature_dim()
    assert action.shape[1] == CONTINUOUS_ACTION_DIM
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert physics_index.min() >= 0 and pose_index.min() >= 0


@pytest.mark.slow
def test_continuous_probe_reports_region_metrics() -> None:
    result = run_continuous_probe(
        n_train_scenes=12, n_test_scenes=5, epochs=30, seed=0
    )
    for field_name in (
        "region_iou",
        "threshold_error",
        "decision_success",
        "balanced_accuracy",
    ):
        value = getattr(result.overall, field_name)
        assert not np.isnan(value), field_name
    assert 0.0 <= result.overall.region_iou <= 1.0
    assert 0.0 <= result.overall.decision_success <= 1.0
    assert 0.0 <= result.overall.balanced_accuracy <= 1.0
    assert "region IoU" in format_continuous_markdown(result)
    assert result.meta["angles"] and result.meta["forces"]
    assert DECISION_THRESHOLD == 0.5
