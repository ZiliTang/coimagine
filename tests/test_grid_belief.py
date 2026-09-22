"""Stage 2 candidate generation and the per-robot belief filter."""

from __future__ import annotations

from dataclasses import replace

import pytest

from coimagine.grid import (
    Direction,
    HiddenPhysics,
    Hypothesis,
    JointSkill,
    LocalSkill,
    SKILL_HOLD,
    SKILL_PUSH,
    advance,
    execute,
    generate_candidates_for_scene,
    localization_marginal,
    localization_metrics,
    most_likely_origin,
    physics_metrics,
    pillar_occlusion,
    public_anchor_cells,
    ray_cast,
    scene_from_ascii,
    scan_log_likelihood,
    uniform_prior,
    update_with_scan,
    valid_object_origins,
)
from coimagine.grid.sensing import KIND_FREE, KIND_OBJECT, SensorNoise

# A and B are both on the west face of the object.
PUSHABLE = (
    "....",
    "AOO.",
    "BOO.",
    "....",
)


def _pushable(physics: HiddenPhysics):
    scene = scene_from_ascii(PUSHABLE, name="pushable")
    return replace(scene, physics=physics)


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


def test_candidates_depend_only_on_public_information() -> None:
    """Neither the object pose nor the hidden physics may change the ID space."""

    scene = _pushable(HiddenPhysics(1, "free", ("grip", "grip")))
    base = generate_candidates_for_scene(scene).ids()

    moved = replace(scene, obj=scene.obj.translated((1, 0)))
    hidden = replace(
        scene, physics=HiddenPhysics(2, "constrained", ("slip", "slip"))
    )
    released = replace(scene, released=True)

    assert generate_candidates_for_scene(moved).ids() == base
    assert generate_candidates_for_scene(hidden).ids() == base
    assert generate_candidates_for_scene(released).ids() == base


def test_candidates_have_unique_stable_ids_and_cover_every_kind() -> None:
    scene = _pushable(HiddenPhysics(1, "free", ("grip", "grip")))
    candidates = generate_candidates_for_scene(scene)
    ids = candidates.ids()
    assert len(ids) == len(set(ids))
    assert list(ids) == sorted(ids)
    kinds = {candidate.kind for candidate in candidates.candidates}
    assert kinds == {"co_push", "push", "probe", "release", "reposition", "wait"}


def test_anchor_cells_are_public_and_avoid_robots() -> None:
    scene = _pushable(HiddenPhysics(1, "free", ("grip", "grip")))
    poses = tuple(r.pose for r in scene.robots)
    anchors = public_anchor_cells(scene.static, poses, count=3)
    assert anchors == public_anchor_cells(scene.static, poses, count=3)
    assert not (set(anchors) & set(poses))
    assert all(not scene.static.is_wall(c) for c in anchors)


def test_valid_origins_exclude_wall_placements() -> None:
    scene = scene_from_ascii(
        (
            "....",
            ".##.",
            "AOO.",
            "BOO.",
        ),
        name="walls",
    )
    origins = valid_object_origins(scene.static)
    assert scene.obj.origin in origins
    for origin in origins:
        cells = {(origin[0] + dx, origin[1] + dy) for dx in (0, 1) for dy in (0, 1)}
        assert not (cells & scene.static.walls)


# --------------------------------------------------------------------------
# Belief filter
# --------------------------------------------------------------------------


def test_uniform_prior_is_normalised_and_contains_the_truth() -> None:
    scene = _pushable(HiddenPhysics(1, "free", ("grip", "grip")))
    belief = uniform_prior(scene.static, robot_poses=tuple(r.pose for r in scene.robots))
    assert sum(belief.values()) == pytest.approx(1.0)
    assert belief[Hypothesis(scene.obj.origin, scene.physics, scene.released)] > 0.0
    assert len(belief) == len(
        valid_object_origins(scene.static, blocked=tuple(r.pose for r in scene.robots))
    ) * 16


def test_scan_resolves_localization_but_not_physics() -> None:
    """The two errors must move independently, which is why they are reported apart."""

    scene = scene_from_ascii(PUSHABLE, name="pushable")
    belief = uniform_prior(scene.static, robot_poses=tuple(r.pose for r in scene.robots))
    before = physics_metrics(belief, scene.physics)["p_correct_physics"]

    belief = update_with_scan(belief, ray_cast(scene, 0, radius=6), scene.static)

    localization = localization_metrics(belief, scene.obj.origin)
    physics = physics_metrics(belief, scene.physics)
    assert localization["p_correct_origin"] > 0.9
    assert most_likely_origin(belief) is not None
    # Seeing the object says nothing about its resistance or traction.
    assert physics["p_correct_physics"] == pytest.approx(before, abs=1e-9)


def test_push_feedback_updates_the_physics_belief() -> None:
    scene = _pushable(HiddenPhysics(1, "free", ("grip", "grip")))
    belief = update_with_scan(
        uniform_prior(scene.static, robot_poses=tuple(r.pose for r in scene.robots)), ray_cast(scene, 0, radius=6), scene.static
    )
    joint = JointSkill(
        (LocalSkill(SKILL_PUSH, Direction.EAST), LocalSkill(SKILL_HOLD))
    )
    successor, result = execute(scene, joint)
    assert result.y_public == "moved"

    updated = advance(
        belief,
        scene.static,
        scene.robots,
        joint,
        robot_id=0,
        observed_feedback=result.feedback[0],
        y_public=result.y_public,
    )
    assert updated
    surviving = {hypothesis.physics for hypothesis in updated}
    assert all(p.resistance == 1 for p in surviving)
    assert all(p.constraint == "free" for p in surviving)
    assert all(p.traction[0] == "grip" for p in surviving)
    assert physics_metrics(updated, scene.physics)["p_correct_physics"] > 1.0 / 16.0
    # The object moved, so the localization belief must move with it.
    assert most_likely_origin(updated) == successor.obj.origin


def test_impossible_public_event_empties_the_belief() -> None:
    scene = _pushable(HiddenPhysics(1, "free", ("grip", "grip")))
    belief = uniform_prior(scene.static, robot_poses=tuple(r.pose for r in scene.robots))
    joint = JointSkill(
        (LocalSkill(SKILL_PUSH, Direction.EAST), LocalSkill(SKILL_HOLD))
    )
    updated = advance(
        belief,
        scene.static,
        scene.robots,
        joint,
        robot_id=0,
        observed_feedback="drove",
        y_public="stalled",  # 'drove' with no movement is impossible here
    )
    assert updated == {}


def test_scan_likelihood_rejects_object_on_an_observed_free_cell() -> None:
    scene = scene_from_ascii(PUSHABLE, name="pushable")
    scan = ray_cast(scene, 0, radius=6)
    assert KIND_OBJECT in scan.observed.values()
    assert KIND_FREE in scan.observed.values()
    free_cell = next(c for c, k in scan.observed.items() if k == KIND_FREE)
    # A hypothesis that puts an object cell on an observed-free cell is impossible.
    assert scan_log_likelihood(scan, free_cell, scene.static, SensorNoise()) == -float("inf")


def test_private_scans_give_the_two_robots_different_beliefs() -> None:
    """The sensing constraint must produce genuinely different private evidence."""

    scene = pillar_occlusion()
    prior = uniform_prior(scene.static, robot_poses=tuple(r.pose for r in scene.robots))
    belief_a = update_with_scan(prior, ray_cast(scene, 0, radius=6), scene.static)
    belief_b = update_with_scan(prior, ray_cast(scene, 1, radius=6), scene.static)

    support_a = set(localization_marginal(belief_a))
    support_b = set(localization_marginal(belief_b))
    assert support_b <= support_a
    assert len(support_a) > len(support_b)
    assert (
        localization_metrics(belief_b, scene.obj.origin)["p_correct_origin"]
        > localization_metrics(belief_a, scene.obj.origin)["p_correct_origin"]
    )


def test_release_transition_is_inferred_from_feedback() -> None:
    scene = _pushable(HiddenPhysics(1, "constrained", ("grip", "grip")))
    belief = update_with_scan(
        uniform_prior(scene.static, robot_poses=tuple(r.pose for r in scene.robots)), ray_cast(scene, 0, radius=6), scene.static
    )
    release = JointSkill(
        (LocalSkill("release", Direction.EAST), LocalSkill(SKILL_HOLD))
    )
    successor, result = execute(scene, release)
    assert result.released
    updated = advance(
        belief,
        scene.static,
        scene.robots,
        release,
        robot_id=0,
        observed_feedback=result.feedback[0],
        y_public=result.y_public,
    )
    surviving = {hypothesis.physics.constraint for hypothesis in updated}
    assert surviving == {"constrained"}
    assert all(hypothesis.released for hypothesis in updated)
