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
    build_contact_scene,
    build_staged_scene,
    format_calibration_markdown,
    run_calibration,
    run_factor_sensitivity,
    run_model_gap,
    run_representability_gap,
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
        radii=(2, 6), origins=((2, 1), (2, 2)), seeds=(0,), frames=(1,)
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


def test_multi_frame_estimation_resolves_an_ambiguous_pose() -> None:
    """More viewpoints must not make the robot's estimate worse."""

    single, _ = run_calibration(radii=(2,), origins=((2, 2),), seeds=(0,), frames=(1,))
    multi, _ = run_calibration(radii=(2,), origins=((2, 2),), seeds=(0,), frames=(3,))
    assert single[0].belief_origins > 1
    assert multi[0].belief_origins <= single[0].belief_origins
    assert multi[0].p_correct_origin >= single[0].p_correct_origin


def test_calibration_report_writes_files(tmp_path) -> None:
    records, summary = run_calibration(radii=(6,), origins=((2, 2),), seeds=(0,))
    paths = write_calibration_report(records, summary, str(tmp_path))
    payload = json.loads((tmp_path / "calibration.json").read_text())
    assert payload["summary"]["git_revision"]
    assert "radius" in format_calibration_markdown(records)
    for key in ("csv", "json", "md"):
        assert key in paths


def test_staged_scene_puts_both_robots_on_the_west_face() -> None:
    scene = build_staged_scene((2, 1))
    assert scene.obj.origin == (2, 1)
    assert {r.robot_id: r.pose for r in scene.robots} == {0: (1, 1), 1: (1, 2)}


def test_model_error_gap_is_dominated_by_the_contact_mode() -> None:
    """Locked finding: at equal information, the contact/constraint mode is what a
    wrong model must get right; resistance and traction errors are tolerated here."""

    records = {r.model: r for r in run_model_gap()}
    assert records["true_model"].gap_vs_true_model == pytest.approx(0.0)
    # A wrong constraint mode is catastrophic: the planner waits for a release that
    # the environment will never grant.
    assert records["thinks_latched"].gap_vs_true_model > 1.0
    assert records["thinks_latched"].success == pytest.approx(0.0)
    # Wrong resistance or wrong teammate traction does not change this decision.
    assert records["thinks_heavy"].gap_vs_true_model == pytest.approx(0.0)
    assert records["thinks_b_slips"].gap_vs_true_model == pytest.approx(0.0)


def test_every_physics_factor_changes_the_decision() -> None:
    """Widened testbed: resistance, traction and the constraint mode all bite.

    This is what the earlier, staged-only testbed failed to show (resistance and
    traction errors cost nothing there), so it is the regression guard for the
    testbed being wide enough to study a model at all.
    """

    records = run_factor_sensitivity()
    assert {r.factor for r in records} == {"resistance", "traction", "constraint"}
    for record in records:
        assert record.gap > 0.0, record


def test_unrepresentable_contact_modes_are_catastrophic() -> None:
    """A model class that cannot express the truth fails outright for the two
    contact-mode factors; resistance is recoverable because the robust plan happens
    to bring the teammate anyway."""

    records = {r.factor: r for r in run_representability_gap()}
    for factor in ("traction", "constraint"):
        record = records[factor]
        assert record.gap > 10.0, record
        assert record.missing_success == pytest.approx(0.0)
    assert records["resistance"].gap == pytest.approx(0.0)


def test_contact_scene_staging_makes_teamwork_a_real_choice() -> None:
    scene = build_contact_scene()
    assert {r.robot_id: r.pose for r in scene.robots} == {0: (1, 1), 1: (1, 3)}
