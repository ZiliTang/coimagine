"""Calibration: is the non-privileged predictor measurably worse than the ceiling?

These tests lock in two things: the harness runs and produces interpretable numbers,
and the gap is strictly positive in the calibrated configuration. If the gap ever
collapses to zero, the testbed has become too easy and the learned-model programme
loses its justification, so the failure should be loud.
"""

from __future__ import annotations

import json

import pytest

from coimagine.diagnostics import (
    CALIBRATION_CANDIDATES,
    CALIBRATION_TARGET,
    build_calibration_scene,
    format_calibration_markdown,
    run_calibration,
    write_calibration_report,
)
from coimagine.grid.planner import ContactContext, contact_candidates

TOL = 1e-9


def test_calibration_scene_is_built_the_right_way_up() -> None:
    """The object and robots must be where the builder says they are."""

    scene = build_calibration_scene((2, 1))
    assert scene.obj.origin == (2, 1)
    assert scene.obj.cells == frozenset({(2, 1), (3, 1), (2, 2), (3, 2)})
    assert {r.robot_id: r.pose for r in scene.robots} == {0: (0, 2), 1: (0, 3)}
    assert scene.static.target == CALIBRATION_TARGET


def test_calibration_candidates_are_pose_agnostic_and_present() -> None:
    scene = build_calibration_scene((2, 2))
    ctx = ContactContext(scene.static, scene.obj.origin, scene.robots)
    available = set(contact_candidates(ctx, public=True).ids())
    assert set(CALIBRATION_CANDIDATES) <= available


def test_non_privileged_predictor_loses_to_the_ceiling() -> None:
    """Locked finding: the gap is strictly positive in the calibrated setup."""

    records, _summary = run_calibration(
        radii=(2, 6), origins=((2, 1), (2, 2)), seeds=(0,)
    )
    assert len(records) == 4
    for record in records:
        assert record.gap > TOL, (record.true_origin, record.radius)
        assert record.ceiling_cost <= record.non_privileged_cost + TOL
    # With a wide enough scan the pose is known exactly, so any remaining gap is
    # attributable to the unknown contact physics rather than to localisation.
    wide = [r for r in records if r.radius == 6]
    assert all(r.p_correct_origin == pytest.approx(1.0) for r in wide)
    assert all(r.gap > TOL for r in wide)
    # The scan says nothing about the contact physics; that has to come from contact.
    assert all(r.p_correct_physics < 0.1 for r in records)


def test_calibration_report_writes_files(tmp_path) -> None:
    records, summary = run_calibration(radii=(6,), origins=((2, 2),), seeds=(0,))
    paths = write_calibration_report(records, summary, str(tmp_path))
    payload = json.loads((tmp_path / "calibration.json").read_text())
    assert payload["summary"]["git_revision"]
    assert "radius" in format_calibration_markdown(records)
    for key in ("csv", "json", "md"):
        assert key in paths
