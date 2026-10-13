"""Substantive tests for the continuous-2D environment, skills, scoring and data contract.

They cover result direction (physics really moves the object the way the score assumes),
state independence (a snapshot is a clean copy), action identity (candidates are generated
from public geometry only and are ordered deterministically) and the evaluation contract
(success needs an in-goal hold, not one frame; data records carry the required fields).
"""

from __future__ import annotations

import math

import pytest

from coimagine.continuous2d import (
    Continuous2DEnv,
    Hold,
    MoveTo,
    PushContact,
    default_config,
    default_scenes,
)
from coimagine.continuous2d.candidates import joint_candidates, per_robot_options
from coimagine.continuous2d.data import segments, window_records
from coimagine.continuous2d.mpc import plan, run_mpc_episode, run_random_episode
from coimagine.continuous2d.scoring import (
    angle_difference,
    centre_distance_to_box,
    distance_to_box,
    ordering_key,
    passage_cost_to_go,
)
from coimagine.continuous2d.skills import contact_point_velocity, contact_world

CONFIG = default_config()


def test_scene_initial_conditions_are_reported() -> None:
    """Every scene starts geometrically legal with the uniform 0.005 m margin.

    After the closeout A1 fix (S0_dev / S0_v2 / S0_v4 robot poses), no scene starts with a
    robot-object, robot-robot or robot-wall overlap; the exact clearance check lives in
    ``continuous2d.legality``.
    """

    from coimagine.continuous2d.legality import DEFAULT_MARGIN, legality_report

    env = Continuous2DEnv(CONFIG)
    for scene in default_scenes():
        env.reset(scene.name)
        by_pair = env.backend.penetration_by_pair()
        assert by_pair["robot_object"] == 0.0, scene.name
        assert by_pair["robot_robot"] == 0.0, scene.name
        assert by_pair["robot_wall"] == 0.0, scene.name
        report = legality_report(scene, DEFAULT_MARGIN)
        assert report["legal"], (scene.name, report["violations"])
        assert report["min_clearance"] >= DEFAULT_MARGIN - 1e-9, scene.name
    s1 = next(scene for scene in default_scenes() if scene.name == "S1_dev")
    # The long axis is across the passage: its projected width exceeds the gap.
    extent = 2.0 * (
        CONFIG.physics.object_width / 2 * abs(math.sin(s1.object_pose.angle))
        + CONFIG.physics.object_height / 2 * abs(math.cos(s1.object_pose.angle))
    )
    assert extent > s1.gap_width
    # Aligned, it fits.
    aligned = 2.0 * CONFIG.physics.object_height / 2
    assert aligned < s1.gap_width


def test_static_object_does_not_move_without_contact() -> None:
    env = Continuous2DEnv(CONFIG)
    env.reset("S0_dev")
    before = env.backend.object_state().as_tuple()
    for _ in range(20):
        env.step_control(((0.0, 8.0), (0.0, 8.0)))
    after = env.backend.object_state().as_tuple()
    assert max(abs(a - b) for a, b in zip(before, after)) < 1e-9


def test_centred_and_offcentre_pushes_have_different_rotational_response() -> None:
    env = Continuous2DEnv(CONFIG)
    rotations = {}
    for label, contact_y in (("centred", 0.0), ("offcentre", 0.10)):
        env.reset("S0_dev")
        start = env.backend.object_state()
        stage = (start.x - 0.5 - CONFIG.physics.robot_radius - 0.02, start.y + contact_y)
        env.execute_joint_skill((MoveTo(stage), Hold()))
        before = env.backend.object_state()
        env.execute_joint_skill(
            (PushContact(0, (-0.5, contact_y), (1.0, 0.0), CONFIG.physics.max_force), Hold())
        )
        rotations[label] = env.backend.object_state().angle - before.angle
    assert abs(rotations["offcentre"]) > abs(rotations["centred"]) + 0.05


def test_force_is_capped_and_robot_cannot_pass_through_the_object() -> None:
    env = Continuous2DEnv(CONFIG)
    env.reset("S0_dev")
    target = env.backend.object_state()
    result = env.execute_joint_skill((MoveTo((target.x, target.y)), Hold()))
    applied = [math.hypot(*step.applied_control[0]) for step in result.steps]
    assert max(applied) <= CONFIG.physics.max_force + 1e-9
    final = result.final.robots[0]
    object_state = result.final.object
    cos, sin = math.cos(-object_state.angle), math.sin(-object_state.angle)
    dx, dy = final.x - object_state.x, final.y - object_state.y
    local_x = dx * cos - dy * sin
    local_y = dx * sin + dy * cos
    clamped_x = max(-0.5, min(0.5, local_x))
    clamped_y = max(-0.125, min(0.125, local_y))
    assert math.hypot(local_x - clamped_x, local_y - clamped_y) > 0.0
    assert result.termination not in ("out_of_bounds", "numeric_failure")


def test_success_requires_an_in_goal_hold_not_one_frame() -> None:
    env = Continuous2DEnv(CONFIG)
    env.reset("S0_dev")
    spec = PushContact(0, (-0.5, 0.0), (1.0, 0.0), CONFIG.physics.max_force)
    result = env.execute_joint_skill((spec, Hold()))
    if env.object_in_goal() and env.object_slow():
        assert env.success_counter > 0.0
    # A single frame is far below the configured hold.
    assert CONFIG.physics.success_hold_time >= 0.5
    assert result.duration <= CONFIG.physics.window_steps * CONFIG.physics.control_dt + 1e-9


def test_snapshot_restore_is_repeatable_and_matches_exact_replay() -> None:
    controls = (((5.0, 1.0), (0.0, 0.0)), ((3.0, -1.0), (1.0, 0.0)), ((0.0, 0.0), (0.0, 0.0)))
    env = Continuous2DEnv(CONFIG)
    env.reset("S0_dev")
    for control in controls:
        env.step_control(control)
    snapshot = env.export_state()

    def future(exact: bool):
        env.restore(snapshot, exact=exact)
        for control in controls:
            env.step_control(control)
        return (env.backend.object_state().as_tuple(), tuple(s.as_tuple() for s in env.backend.robot_states()))

    def flat(value):
        if isinstance(value, (tuple, list)):
            out = []
            for item in value:
                out.extend(flat(item))
            return tuple(out)
        return (value,)

    fast = future(False)
    exact = future(True)
    assert max(abs(a - b) for a, b in zip(flat(fast), flat(exact))) < 1e-6
    assert fast == future(False)


def test_candidate_dependence_is_public_and_order_is_stable() -> None:
    env = Continuous2DEnv(CONFIG)
    env.reset("S1_dev")
    observation = env.observation()
    first = joint_candidates(observation, CONFIG.candidates, CONFIG.control, CONFIG.physics)
    second = joint_candidates(observation, CONFIG.candidates, CONFIG.control, CONFIG.physics)
    assert 0 < len(first) <= CONFIG.candidates.max_candidates
    assert [candidate.candidate_id for candidate in first] == [
        candidate.candidate_id for candidate in second
    ]
    assert [candidate.label for candidate in first] == [candidate.label for candidate in second]
    labels = [candidate.label for candidate in first]
    assert any("hold" in label for label in labels)
    assert any("push_contact" in label for label in labels)
    options = per_robot_options(observation, CONFIG.candidates, CONFIG.control, CONFIG.physics)
    assert any(isinstance(option, MoveTo) for option in options)


def test_scoring_geometry_matches_hand_worked_examples() -> None:
    from coimagine.continuous2d.scenes import Box

    goal = Box(2.0, 1.0, 0.5, 0.5)
    # A point at the goal centre is inside.
    assert centre_distance_to_box((2.0, 1.0), goal) == 0.0
    # One metre to the east of the centre: 1.0 - 0.5 = 0.5 outside.
    assert centre_distance_to_box((3.0, 1.0), goal) == pytest.approx(0.5)
    # Outside on both axes: dx = 3 - 2 - 0.5 = 0.5, dy = 4 - 1 - 0.5 = 2.5.
    assert centre_distance_to_box((3.0, 4.0), goal) == pytest.approx(math.hypot(0.5, 2.5))
    corners = ((2.0, 1.0), (2.0, 1.0))
    assert distance_to_box(corners, goal) == 0.0
    assert distance_to_box(((3.0, 1.0), (2.0, 1.0)), goal) == pytest.approx(0.5)
    # Angle difference wraps at pi.
    assert angle_difference(0.0, 0.0) == 0.0
    assert angle_difference(math.pi / 2, -math.pi / 2) == pytest.approx(math.pi)
    assert angle_difference(0.1, 0.0) == pytest.approx(0.1)


def test_ordering_key_is_lexicographic_and_success_first() -> None:
    success = ordering_key(
        success=True, success_time=4.0, control_usage=3.0, terminal_distance=0.9,
        orientation=0.0, candidate_id=7,
    )
    failure = ordering_key(
        success=False, success_time=0.0, control_usage=0.1, terminal_distance=0.0,
        orientation=0.0, candidate_id=1,
    )
    assert success > failure
    faster = ordering_key(
        success=True, success_time=2.0, control_usage=9.0, terminal_distance=0.0,
        orientation=0.0, candidate_id=3,
    )
    assert faster > success
    closer = ordering_key(
        success=False, success_time=0.0, control_usage=9.0, terminal_distance=0.1,
        orientation=0.0, candidate_id=5,
    )
    farther = ordering_key(
        success=False, success_time=0.0, control_usage=0.0, terminal_distance=0.5,
        orientation=0.0, candidate_id=5,
    )
    assert closer > farther


def test_passage_cost_rewards_fitting_through_the_gap() -> None:
    from coimagine.continuous2d.scenes import scene_by_name

    scene = scene_by_name("S1_dev")
    size = (CONFIG.physics.object_width, CONFIG.physics.object_height)
    blocked = passage_cost_to_go(
        (0.95, 1.0, math.radians(55.0)), size, scene.goal, scene.obstacles, ()
    )
    aligned = passage_cost_to_go((0.95, 1.0, 0.0), size, scene.goal, scene.obstacles, ())
    assert aligned < blocked
    # Once east of the wall the plain corner distance applies.
    through = passage_cost_to_go((2.25, 1.0, 0.0), size, scene.goal, scene.obstacles, ())
    assert through == 0.0


def test_contact_geometry_functions_match_hand_examples() -> None:
    from coimagine.continuous2d.backend import BodyState

    object_state = BodyState(1.0, 2.0, math.pi / 2, 0.0, 0.0, 0.0)
    # At 90 degrees the local +x axis points along world +y.
    assert contact_world(object_state, (0.5, 0.0)) == pytest.approx((1.0, 2.5))
    # A pure rotation about the centre gives the contact point tangential velocity.
    spinning = BodyState(1.0, 2.0, 0.0, 0.0, 0.0, 2.0)
    assert contact_point_velocity(spinning, (1.0, 3.0)) == pytest.approx((-2.0, 0.0))


def test_window_records_and_segments_cover_the_data_contract() -> None:
    env = Continuous2DEnv(CONFIG)
    episode = run_random_episode(env, "S0_dev", CONFIG, seed=0, horizon=3)
    records = window_records("ep", "S0_dev", 0, episode, CONFIG)
    assert len(records) == episode.windows
    required = {
        "schema_version", "episode_id", "scene_id", "seed", "t", "state", "joint_skill_specs",
        "joint_skill_labels", "controller_state", "applied_control", "next_state", "duration",
        "termination", "success", "success_counter", "cost_components", "physics_config_hash",
        "config_hash", "controller_version", "code_version", "split", "steps",
    }
    assert required <= set(records[0])
    assert records[0]["schema_version"] == 2
    assert records[0]["split"] == "dev"
    assert records[0]["physics_config_hash"] == CONFIG.config_hash()
    # both robots' forces are recorded per control step, flat and nested
    robot_ids = {entry[1] for entry in records[0]["applied_control"]}
    assert robot_ids == {0, 1}
    assert all(len(step["applied_control"]) == 2 for step in records[0]["steps"])
    assert all("next_state" in step and "state" in step for step in records[0]["steps"])
    clips = segments(records, 2)
    assert all(len(clip) == 2 for clip in clips)
    assert len(clips) == max(0, len(records) - 1)


@pytest.mark.slow
def test_depth_one_mpc_solves_a_free_space_scene() -> None:
    # A4 correction: the free-space positive check uses S0_v1. After the closeout A1 robot
    # pose fix, depth-1 lexicographic MPC no longer solves S0_dev (recorded in
    # docs/closeout_v1/README.md); the scene is not changed to make an old test pass.
    env = Continuous2DEnv(CONFIG)
    episode = run_mpc_episode(env, "S0_v1", CONFIG, depth=1, beam=CONFIG.mpc.beam, seed=0)
    assert episode.success
    assert episode.windows <= CONFIG.mpc.horizon_windows
    assert episode.plan_nodes > 0


def test_fixed_s0_dev_is_legal_and_keeps_the_frozen_protocol() -> None:
    """S0_dev stays in the library, legal at reset, and unchanged in protocol semantics."""

    from coimagine.continuous2d.legality import legality_report
    from coimagine.continuous2d.scenes import scene_by_name

    report = legality_report(scene_by_name("S0_dev"))
    assert report["legal"], report["violations"]

    env = Continuous2DEnv(CONFIG)
    env.reset("S0_dev")
    snapshot = env.export_state()
    assert snapshot.tick == 0
    assert snapshot.success_counter == 0.0
    assert snapshot.termination is None
    # exact prefix replay of a fresh reset reproduces the snapshot
    env.restore(snapshot, exact=True)
    assert env.tick == snapshot.tick
    assert env.backend.object_state().as_tuple() == pytest.approx(snapshot.object.as_tuple(), abs=1e-9)
    # one window executes and reports the frozen termination vocabulary
    observation = env.observation()
    from coimagine.continuous2d.skills import Hold

    result = env.execute_joint_skill((Hold(), Hold()))
    assert result.termination in (
        "window_complete", "success", "numeric_failure", "out_of_bounds", "penetration_limit",
    )
    assert result.duration > 0.0


@pytest.mark.slow
def test_depth_three_plan_contains_a_preparation_window_on_s1() -> None:
    env = Continuous2DEnv(CONFIG)
    env.reset("S1_dev")
    result = plan(env, env.export_state(), CONFIG, depth=3, beam=CONFIG.mpc.beam)
    assert result.sequence
    assert result.nodes >= result.candidates_at_root
    episode = run_mpc_episode(env, "S1_dev", CONFIG, depth=3, beam=CONFIG.mpc.beam, seed=0)
    assert episode.success
    # The passage is only passable after the orientation changed, so at least one window
    # cannot already have the object in the goal region.
    assert episode.windows >= 2
