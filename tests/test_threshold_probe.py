"""Tests for the force-threshold probe.

The regret accounting is the part worth testing by hand: the probe's claim rests on it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from dataclasses import replace

from coimagine.diagnostics.calibration import build_contact_scene
from coimagine.diagnostics.threshold_probe import (
    FAIL_PENALTY,
    PUSH_BASE_COST,
    PROBE_ANGLE,
    format_threshold_markdown,
    run_threshold_probe,
)
from coimagine.grid.continuous import FORCE_MAX, push_outcome
from coimagine.grid.state import HiddenPhysics


def _scene(physics, released=False):
    return replace(build_contact_scene(), physics=physics, released=released)


def test_push_outcome_and_threshold_are_consistent() -> None:
    """The threshold really is the smallest force that works."""

    for physics in (
        HiddenPhysics(1, "free", ("grip", "grip")),
        HiddenPhysics(2, "free", ("grip", "grip")),
    ):
        scene = _scene(physics)
        for force in (0.5, 1.0, 1.5, 2.0):
            outcome = push_outcome(scene, {0: force}, PROBE_ANGLE)
            expected = force >= physics.resistance
            assert outcome.moved is expected, (physics.label, force)


def test_declared_costs_are_what_the_probe_uses() -> None:
    assert PUSH_BASE_COST == 1.0
    assert FAIL_PENALTY == 2.0


def test_regret_accounting_by_hand() -> None:
    """Reproduce the probe's regret for a scene whose threshold is known."""

    physics = HiddenPhysics(1, "free", ("grip", "grip"))
    scene = _scene(physics)
    threshold = float(physics.resistance)

    # oracle: push exactly at the threshold and succeed
    assert push_outcome(scene, {0: threshold}, PROBE_ANGLE).moved
    # always max: succeeds but wastes force
    assert push_outcome(scene, {0: FORCE_MAX}, PROBE_ANGLE).moved
    assert FORCE_MAX - threshold == pytest.approx(1.0)
    # always min: fails, so the regret is the penalty plus the force spent
    assert not push_outcome(scene, {0: 0.25}, PROBE_ANGLE).moved
    assert FAIL_PENALTY + 0.25 == pytest.approx(2.25)


def test_latched_scene_cannot_be_pushed_at_any_force() -> None:
    scene = _scene(HiddenPhysics(1, "constrained", ("grip", "grip")))
    for force in (0.25, 1.0, 2.0):
        assert not push_outcome(scene, {0: force}, PROBE_ANGLE).moved


def test_slipping_robot_has_an_infinite_threshold() -> None:
    from coimagine.grid.continuous import force_threshold

    scene = _scene(HiddenPhysics(1, "free", ("slip", "grip")))
    assert math.isinf(force_threshold(scene, [0]))
    assert not push_outcome(scene, {0: FORCE_MAX}, PROBE_ANGLE).moved


@pytest.mark.slow
def test_threshold_probe_runs_and_beats_the_baselines() -> None:
    result = run_threshold_probe(
        n_train_situations=80, n_test_situations=40, epochs=150, seed=0
    )
    by_name = {s.name: s for s in result.summaries}
    assert by_name["oracle"].mean_regret == pytest.approx(0.0)
    assert 0.0 <= result.can_push_balanced_accuracy <= 1.0
    assert result.threshold_mae >= 0.0
    # The learned threshold must beat both fixed-force baselines on regret.
    assert by_name["learned"].mean_regret < by_name["always_max"].mean_regret
    assert by_name["learned"].mean_regret < by_name["always_min"].mean_regret
    assert "threshold MAE" in format_threshold_markdown(result)
