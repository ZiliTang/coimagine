"""Stage 2 dynamics: contact geometry, coalition capacity, preparation and cost."""

from __future__ import annotations

import random

import pytest

from coimagine.grid import (
    DURATION_PROBE,
    FEEDBACK_SYMBOLS,
    FeedbackNoise,
    HiddenPhysics,
    JointSkill,
    LocalSkill,
    Direction,
    OBS_DROVE,
    OBS_IDLE,
    OBS_SLIPPED,
    OBS_STRAINED,
    SKILL_HOLD,
    SKILL_PROBE,
    SKILL_PUSH,
    SKILL_RELEASE,
    SKILL_REPOSITION,
    SKILL_WAIT,
    Y_MOVED,
    Y_STALLED,
    execute,
    feedback_likelihood,
    push_contact_ok,
    sample_feedback,
    scene_from_ascii,
    shortest_path,
)

# A and B are both on the west face, so both can push EAST.
PUSHABLE = (
    "....",
    "AOO.",
    "BOO.",
    "....",
)

# The object's east face is against a wall.
BLOCKED = (
    "..##",
    "AOO#",
    "BOO#",
    "....",
)

# Walls force a detour: straight-line distance is 4, the real path is longer.
DETOUR = (
    ".....",
    ".##..",
    "A.#..",
    ".OO..",
    ".OO.B",
)


def _push(direction: Direction = Direction.EAST) -> LocalSkill:
    return LocalSkill(SKILL_PUSH, direction=direction)


def _scene(rows, physics: HiddenPhysics):
    scene = scene_from_ascii(rows, name="test")
    from dataclasses import replace

    return replace(scene, physics=physics)


GRIP2 = HiddenPhysics(resistance=2, constraint="free", traction=("grip", "grip"))
GRIP1 = HiddenPhysics(resistance=1, constraint="free", traction=("grip", "grip"))
SLIP_B = HiddenPhysics(resistance=2, constraint="free", traction=("grip", "slip"))
CONSTRAINED = HiddenPhysics(resistance=1, constraint="constrained", traction=("grip", "grip"))


def test_cooperative_push_moves_the_object() -> None:
    scene = _scene(PUSHABLE, GRIP2)
    joint = JointSkill((_push(), _push()))
    successor, result = execute(scene, joint)
    assert result.y_public == Y_MOVED
    assert successor.obj.origin == (2, 1)
    assert result.cost == pytest.approx(1.0)
    assert result.feedback[0] == OBS_DROVE and result.feedback[1] == OBS_DROVE
    assert not result.failures


def test_single_push_fails_against_resistance_two() -> None:
    scene = _scene(PUSHABLE, GRIP2)
    joint = JointSkill((_push(), LocalSkill(SKILL_HOLD)))
    successor, result = execute(scene, joint)
    assert result.y_public == Y_STALLED
    assert successor.obj.origin == scene.obj.origin
    assert "insufficient_capacity" in result.failures
    assert result.feedback[0] == OBS_STRAINED
    assert result.feedback[1] == OBS_IDLE


def test_single_push_succeeds_against_resistance_one() -> None:
    scene = _scene(PUSHABLE, GRIP1)
    joint = JointSkill((_push(), LocalSkill(SKILL_WAIT)))
    successor, result = execute(scene, joint)
    assert result.y_public == Y_MOVED
    assert successor.obj.origin == (2, 1)


def test_slipping_robot_does_not_contribute_capacity() -> None:
    scene = _scene(PUSHABLE, SLIP_B)
    joint = JointSkill((_push(), _push()))
    successor, result = execute(scene, joint)
    assert result.y_public == Y_STALLED
    assert "insufficient_capacity" in result.failures
    assert result.feedback[1] == OBS_SLIPPED
    assert result.feedback[0] == OBS_STRAINED
    assert result.detail["capacity"] == 1


def test_constrained_object_requires_a_release() -> None:
    scene = _scene(PUSHABLE, CONSTRAINED)

    # Pushing a latched object does nothing.
    _, stalled = execute(scene, JointSkill((_push(), _push())))
    assert stalled.y_public == Y_STALLED
    assert "constrained_contact" in stalled.failures

    # A real release action performs the contact-mode transition.
    release = JointSkill((LocalSkill(SKILL_RELEASE, Direction.EAST), LocalSkill(SKILL_HOLD)))
    released_scene, released = execute(scene, release)
    assert released.released and released_scene.released
    assert released.success
    assert released.feedback[0] == OBS_DROVE

    # Now the same push succeeds.
    moved_scene, moved = execute(released_scene, JointSkill((_push(), _push())))
    assert moved.y_public == Y_MOVED
    assert moved_scene.obj.origin == (2, 1)


def test_release_on_a_free_object_is_a_wasted_action() -> None:
    scene = _scene(PUSHABLE, GRIP1)
    joint = JointSkill((LocalSkill(SKILL_RELEASE, Direction.EAST), LocalSkill(SKILL_HOLD)))
    successor, result = execute(scene, joint)
    assert not successor.released
    assert any("no_effect" in f for f in result.failures)
    assert result.cost == pytest.approx(1.0)
    assert result.y_public == Y_STALLED


def test_probe_does_not_move_the_object_but_reveals_capacity() -> None:
    # Against resistance 2 a probe by one robot feels strain.
    scene = _scene(PUSHABLE, GRIP2)
    _, weak = execute(
        scene, JointSkill((LocalSkill(SKILL_PROBE, Direction.EAST), LocalSkill(SKILL_HOLD)))
    )
    assert weak.y_public == Y_STALLED
    assert weak.cost == pytest.approx(DURATION_PROBE)
    assert weak.feedback[0] == OBS_STRAINED

    # Against resistance 1 the same probe drives.
    scene1 = _scene(PUSHABLE, GRIP1)
    _, strong = execute(
        scene1, JointSkill((LocalSkill(SKILL_PROBE, Direction.EAST), LocalSkill(SKILL_HOLD)))
    )
    assert strong.feedback[0] == OBS_DROVE


def test_mixed_push_directions_are_rejected() -> None:
    scene = _scene(PUSHABLE, GRIP1)
    joint = JointSkill((_push(Direction.EAST), _push(Direction.NORTH)))
    successor, result = execute(scene, joint)
    assert result.y_public == Y_STALLED
    assert "mixed_push_directions" in result.failures
    assert successor.obj.origin == scene.obj.origin


def test_blocked_destination_stops_the_push() -> None:
    scene = _scene(BLOCKED, GRIP1)
    joint = JointSkill((_push(), LocalSkill(SKILL_HOLD)))
    successor, result = execute(scene, joint)
    assert result.y_public == Y_STALLED
    assert result.failures == ()
    # The push was physically legal but the destination was walled off, so the
    # feedback is a strain rather than an error code.
    assert result.feedback[0] == OBS_STRAINED
    assert successor.obj.origin == scene.obj.origin


def test_reposition_is_charged_its_full_path() -> None:
    scene = _scene(DETOUR, GRIP1)
    joint = JointSkill(
        (LocalSkill(SKILL_REPOSITION, target=(4, 2)), LocalSkill(SKILL_WAIT))
    )
    successor, result = execute(scene, joint)
    assert successor.robot_pose(0) == (4, 2)
    # Straight-line Manhattan distance is 4; the walls force a longer path.
    assert result.cost > 4
    assert result.cost == pytest.approx(8.0)
    assert result.feedback[0] == OBS_IDLE


def test_reposition_to_an_illegal_target_fails_without_teleporting() -> None:
    scene = _scene(DETOUR, GRIP1)
    target = next(iter(scene.obj.cells))  # inside the object
    joint = JointSkill((LocalSkill(SKILL_REPOSITION, target=target), LocalSkill(SKILL_WAIT)))
    successor, result = execute(scene, joint)
    assert successor.robot_pose(0) == scene.robot_pose(0)
    assert any("destination_blocked" in f for f in result.failures)
    assert result.cost == pytest.approx(0.0)


def test_shortest_path_respects_walls_and_blocked_cells() -> None:
    scene = _scene(DETOUR, GRIP1)
    path = shortest_path(scene, (0, 2), (4, 2), blocked=scene.obj.cells)
    assert path is not None
    assert path[0] == (0, 2) and path[-1] == (4, 2)
    assert all(not scene.static.is_wall(c) for c in path)
    assert not (set(path) & scene.obj.cells)
    assert shortest_path(scene, (0, 2), (2, 2)) is None  # inside a wall


def test_push_contact_geometry() -> None:
    scene = _scene(PUSHABLE, GRIP1)
    assert push_contact_ok(scene, 0, Direction.EAST)
    assert push_contact_ok(scene, 1, Direction.EAST)
    assert not push_contact_ok(scene, 0, Direction.WEST)
    assert not push_contact_ok(scene, 0, Direction.NORTH)


def test_feedback_channel_is_a_valid_distribution() -> None:
    noisy = FeedbackNoise(flip_prob=0.25)
    for nominal in FEEDBACK_SYMBOLS:
        total = sum(
            feedback_likelihood(observed, nominal, noisy)
            for observed in FEEDBACK_SYMBOLS
        )
        assert total == pytest.approx(1.0)

    rng = random.Random(0)
    always = FeedbackNoise(flip_prob=0.999999)
    assert all(
        sample_feedback(OBS_DROVE, always, rng) != OBS_DROVE for _ in range(5)
    )
    assert sample_feedback(OBS_DROVE, FeedbackNoise(), rng) == OBS_DROVE


# Two cooperative pushes east: the object ends inside a 2x2 goal on the far side.
TWO_PUSH = (
    ".....",
    "AOOTT",
    "BOOTT",
    ".....",
)


def test_pushers_walk_with_the_object_across_two_boundaries() -> None:
    """A successful push advances the pushers, so no reposition is needed."""

    scene = _scene(TWO_PUSH, GRIP2)
    assert not scene.is_goal()
    joint = JointSkill((_push(), _push()))

    first_scene, first = execute(scene, joint)
    assert first.y_public == Y_MOVED
    assert first_scene.obj.origin == (2, 1)
    assert first_scene.robot_pose(0) == (1, 2)
    assert first_scene.robot_pose(1) == (1, 1)
    assert not first_scene.is_goal()

    second_scene, second = execute(first_scene, joint)
    assert second.y_public == Y_MOVED
    assert second_scene.obj.origin == (3, 1)
    assert second_scene.is_goal()


def test_pusher_does_not_advance_into_a_blocked_cell() -> None:
    """If the forward cell is a wall the pusher stays put but the object still moves."""

    rows = (
        ".....",
        "AOO.#",
        "BOO.#",
        ".....",
    )
    scene = _scene(rows, GRIP1)
    joint = JointSkill((_push(), LocalSkill(SKILL_HOLD)))
    successor, result = execute(scene, joint)
    assert result.y_public == Y_MOVED
    # Robot A would step to (1, 2), which is free, so it advances.
    assert successor.robot_pose(0) == (1, 2)
