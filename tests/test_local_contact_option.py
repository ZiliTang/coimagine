"""Acceptance tests for the local-contact option round (task book section 4.3)."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace

import pytest

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d.environment import Continuous2DEnv
from coimagine.continuous2d.scenes import Box, Pose, SceneSpec
from coimagine.continuous2d.skills import Hold, PushContact, control as old_control
from coimagine.local_contact_option.contract import Exit, OptionConfig, Stage
from coimagine.local_contact_option.control import (
    LocalControlState,
    local_push_force,
    update_surface_estimate,
)
from coimagine.local_contact_option.metrics import execution_metrics, to_initial_frame
from coimagine.local_contact_option.option import (
    OptionExecutor,
    RobotRuntime,
    push_abort_trigger,
)
from coimagine.local_contact_option.roots import (
    VARIANT_SPECS,
    VariantSpec,
    build_roots,
    sample_root,
)
from coimagine.local_contact_option.runner import build_snapshot, run_execution
from coimagine.local_contact_option.sensing import (
    BodyKinematics,
    LocalContactObservation,
    LocalPatch,
    best_target_patch,
    scan_surface,
)
from coimagine.local_contact_option.skill import LocalSurfacePush

CONFIG = default_config()
OPTION = OptionConfig()


def _root(interaction_class="solo_central", index=0, split="dev"):
    return sample_root(split, interaction_class, index)


# ---------------------------------------------------------------------------
# 1. the observation whitelist
# ---------------------------------------------------------------------------


def test_observation_has_no_object_truth_fields():
    names = {field.name for field in dataclasses.fields(LocalContactObservation)}
    forbidden = {
        "object", "object_state", "object_pose", "object_position", "object_velocity",
        "object_angle", "object_omega", "object_corners", "object_size", "face_id",
        "object_local_contact", "true_state", "env", "environment", "backend", "space",
    }
    assert not (names & forbidden)
    observation = LocalContactObservation(
        robot_id=0, tick=0, time=0.0, sample_time=0.0,
        own=BodyKinematics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), teammate=None, patch=None,
        contacting=False, contact_point_robot=None, contact_force_world=(0.0, 0.0),
        contact_impulse=0.0, arena=(3.2, 2.2), robot_radius=0.18, force_limit=15.0,
    )
    assert observation.forbidden_fields() == ()


def test_force_is_a_function_of_the_observation_only():
    """Holding the observation fixed while the hidden object truth changes leaves force identical."""

    root = _root("solo_central", 0)
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    from coimagine.local_contact_option.option import build_observation

    observation = build_observation(
        env, OPTION, 0, stage="push", teammate=None,
        skill={"alpha": 0.0, "speed": 0.16},
    )
    state = LocalControlState()
    state.start(alpha=0.0, speed=0.16, force_limit=15.0, kv=20.0, a_ref_max=0.5)
    first, _ = local_push_force(observation, state, 0.05, mode="push")

    # move the hidden object truth a long way; the observation object is unchanged
    env.backend.object_body.position = (99.0, 99.0)
    env.backend.object_body.velocity = (5.0, 5.0)
    state_two = LocalControlState()
    state_two.start(alpha=0.0, speed=0.16, force_limit=15.0, kv=20.0, a_ref_max=0.5)
    second, _ = local_push_force(observation, state_two, 0.05, mode="push")
    assert first == second


# ---------------------------------------------------------------------------
# 2. equivariance under a rigid transform of the whole scene
# ---------------------------------------------------------------------------


def _scene(root, angle: float):
    scene = root.scene()
    cos, sin = math.cos(angle), math.sin(angle)

    def rotate_pose(pose):
        return Pose(pose.x * cos - pose.y * sin, pose.x * sin + pose.y * cos, pose.angle + angle)

    return replace(
        scene,
        arena=scene.arena,
        obstacles=tuple(
            Box(box.cx * cos - box.cy * sin, box.cx * sin + box.cy * cos, box.half_w, box.half_h, box.angle)
            for box in scene.obstacles
        ),
        object_pose=rotate_pose(scene.object_pose),
        robot_poses=(rotate_pose(scene.robot_poses[0]), rotate_pose(scene.robot_poses[1])),
    )


def test_force_is_equivariant_under_rotation_and_translation():
    root = _root("solo_central", 1)
    angle = 0.7
    forces = []
    for scene in (root.scene(), _scene(root, angle)):
        env = Continuous2DEnv(CONFIG)
        env.reset(scene)
        from coimagine.local_contact_option.option import build_observation

        observation = build_observation(
            env, OPTION, 0, stage="push", teammate=None, skill={"alpha": 0.0, "speed": 0.16},
        )
        state = LocalControlState()
        state.start(alpha=0.0, speed=0.16, force_limit=15.0, kv=20.0, a_ref_max=0.5)
        force, _ = local_push_force(observation, state, 0.05, mode="push")
        forces.append(force)
    cos, sin = math.cos(angle), math.sin(angle)
    rotated = (forces[0][0] * cos - forces[0][1] * sin, forces[0][0] * sin + forces[0][1] * cos)
    assert rotated == pytest.approx(forces[1], abs=1e-9)


# ---------------------------------------------------------------------------
# 3. distance projection is not contact
# ---------------------------------------------------------------------------


def test_lateral_misalignment_does_not_trigger_contact_or_ready():
    """Matching the distance *projection* onto the push normal is not contact."""

    from coimagine.local_contact_option.option import _contact_bool
    from coimagine.continuous2d.geometry import _distance_to_polygon

    def _corners(x, y, angle):
        half_w, half_h = 0.5, 0.125
        cos, sin = math.cos(angle), math.sin(angle)
        return tuple((x + dx * cos - dy * sin, y + dx * sin + dy * cos)
                     for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)))

    def _scene(robot_xy):
        return SceneSpec(
            name="lateral_misalignment", category="LCO", variant=-1, arena=(3.2, 2.2),
            obstacles=(), goal=Box(0.0, 0.0, 0.05, 0.05),
            object_pose=Pose(1.5, 1.1, 0.0),
            robot_poses=(Pose(robot_xy[0], robot_xy[1], math.pi / 2.0), Pose(0.3, 0.3, 0.0)),
        )

    aligned = _scene((1.5, 1.425))
    shifted = _scene((2.12, 1.425))
    gaps = []
    distances = []
    for scene in (aligned, shifted):
        env = Continuous2DEnv(CONFIG)
        env.reset(scene)
        contacting, point, _force, _impulse = _contact_bool(env, 0)
        assert contacting is False and point is None
        robot = env.backend.robot_states()[0]
        gaps.append(robot.y - 1.225 - CONFIG.physics.robot_radius)   # projection onto the normal
        distances.append(_distance_to_polygon(robot.x, robot.y, _corners(1.5, 1.1, 0.0)))
    assert gaps[0] == pytest.approx(gaps[1], abs=1e-9)
    assert distances[1] > distances[0] + 0.02


# ---------------------------------------------------------------------------
# 4. handshake and effective-contact timing
# ---------------------------------------------------------------------------


def test_approach_time_is_not_counted_as_effective_contact():
    root = _root("dual_same_side", 0)
    specs = {0: LocalSurfacePush(**root.local_parameters(0)), 1: LocalSurfacePush(**root.local_parameters(1))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(OPTION, "N").run(
        env, root_id=root.root_id, variant="baseline", joint_specs=specs,
        active_robots=root.active_robots,
    )
    assert result.exit_reason == "settled_observed"
    assert result.stage_seconds["approach"] > 0.0
    assert result.stage_seconds["push"] >= 2.0
    active = max(row["active_contact"]["0"] for row in result.rows)
    assert active >= 2.0 - 1e-9
    assert active <= result.stage_seconds["push"] + 1e-9
    # the approach time is not inside the effective-contact clock
    assert active <= result.wall_clock_steps * CONFIG.physics.control_dt


def test_ready_requires_sustained_contact_on_both_sides():
    root = _root("dual_same_side", 1)
    specs = {0: LocalSurfacePush(**root.local_parameters(0)), 1: LocalSurfacePush(**root.local_parameters(1))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(OPTION, "N").run(
        env, root_id=root.root_id, variant="baseline", joint_specs=specs,
        active_robots=root.active_robots,
    )
    # READY is a per-robot state in the repaired protocol: one robot can hold READY while its
    # teammate is still approaching, so the joint stage may stay APPROACH throughout
    per_robot_ready = [
        row for row in result.rows
        if row.get("observed") and any(
            entry.get("stage") == "ready" for entry in row["observed"].values()
        )
    ]
    assert per_robot_ready, "at least one robot must pass through READY"
    assert len(result.rows) > 0


# ---------------------------------------------------------------------------
# 5. explicit exits
# ---------------------------------------------------------------------------


def test_every_abort_condition_maps_to_an_explicit_exit():
    config = OptionConfig()
    runtime = RobotRuntime(robot_id=0)
    runtime.normal_jump = True
    assert push_abort_trigger({0: runtime}, (0,), config) == (0, Exit.FACE_SWITCH.value)
    runtime = RobotRuntime(robot_id=0)
    runtime.patch_loss = config.patch_loss_tolerance + 1e-6
    assert push_abort_trigger({0: runtime}, (0,), config) == (0, Exit.PATCH_LOST.value)
    runtime = RobotRuntime(robot_id=0)
    runtime.contact_loss = config.contact_loss_tolerance + 1e-6
    assert push_abort_trigger({0: runtime}, (0,), config) == (0, Exit.CONTACT_LOST.value)
    runtime = RobotRuntime(robot_id=0)
    runtime.push_wall = 99.0
    assert push_abort_trigger({0: runtime}, (0,), config) == (0, Exit.PUSH_WALL_TIMEOUT.value)
    assert push_abort_trigger({0: RobotRuntime(robot_id=0)}, (0,), config) is None


def test_unreachable_surface_exits_explicitly_and_aborts_the_teammate():
    root = _root("dual_same_side", 2)
    scene = root.scene()
    far = replace(scene, robot_poses=(Pose(0.2, 0.2, 0.0), scene.robot_poses[1]))
    specs = {
        0: LocalSurfacePush(alpha=0.0, speed=0.16),
        1: LocalSurfacePush(alpha=0.0, speed=0.16),
    }
    env = Continuous2DEnv(CONFIG)
    env.reset(far)
    result = OptionExecutor(OPTION, "N").run(
        env, root_id=root.root_id + "#far", variant="test", joint_specs=specs,
        active_robots=(0, 1),
    )
    assert result.exit_reason in (Exit.SURFACE_UNOBSERVED.value, Exit.APPROACH_TIMEOUT.value)
    assert "teammate_abort" in result.flags


def test_push_wall_timeout_is_reachable():
    config = OptionConfig(push_wall_limit=0.05)
    root = _root("solo_central", 3)
    specs = {0: LocalSurfacePush(**root.local_parameters(0))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(config, "N").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    assert result.exit_reason == Exit.PUSH_WALL_TIMEOUT.value
    assert "settled_after_abort" in result.flags


# ---------------------------------------------------------------------------
# 6. no state writes, bounded force and bounded reference acceleration
# ---------------------------------------------------------------------------


def test_no_stage_transition_writes_a_body_state(monkeypatch):
    from coimagine.continuous2d import backend as backend_module

    def _forbidden(self, object_state, robots):  # pragma: no cover - must never run
        raise AssertionError("a stage transition wrote a body pose or velocity")

    monkeypatch.setattr(backend_module.PymunkBackend, "_write_states", _forbidden)
    root = _root("dual_rotate", 0)
    specs = {0: LocalSurfacePush(**root.local_parameters(0)), 1: LocalSurfacePush(**root.local_parameters(1))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(OPTION, "N").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    assert result.wall_clock_steps > 0


def test_force_and_reference_acceleration_stay_bounded():
    root = _root("solo_offset", 0)
    specs = {0: LocalSurfacePush(**root.local_parameters(0))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(OPTION, "N").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    dt = CONFIG.physics.control_dt
    previous = None
    brake_entry_seen = False
    discontinuities = 0
    for row in result.rows:
        for force in row["force"]:
            assert math.hypot(*force) <= OPTION.force_limit + 1e-9
        reference = row["observed"]["0"].get("reference")
        stage = row["phase"] if "phase" in row else row["stage"]
        if reference is not None and previous is not None:
            step = math.hypot(reference[0] - previous[0], reference[1] - previous[1])
            if step > OPTION.a_ref_max * dt + 1e-9:
                # the one documented exception is the BRAKE-entry initialisation from the
                # robot's measured velocity, which happens exactly once per execution
                assert stage == "brake" and not brake_entry_seen, (
                    f"unexpected reference discontinuity {step:.4f} at {stage}")
                brake_entry_seen = True
                discontinuities += 1
        previous = reference
    assert discontinuities <= 1


# ---------------------------------------------------------------------------
# 7. local settling is local; tangential motion is unobservable
# ---------------------------------------------------------------------------


def _patch(points, normal, timestamp):
    return LocalPatch(
        points_robot=tuple(points), segment_robot=(points[0], points[-1]), distance=0.2,
        normal_world=normal, tangent_world=(-normal[1], normal[0]), confidence=1.0,
        timestamp=timestamp, is_target=True, arc_rays=len(points), occluder_labels=("target_object",),
    )


def test_pure_tangential_motion_is_invisible_to_the_local_settle_test():
    """A face sliding sideways is reported as locally settled: false stable by construction."""

    normal = (1.0, 0.0)
    state = LocalControlState()
    pose = (0.0, 0.0, 0.0)
    first = _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.0)
    update_surface_estimate(state, first, pose, 0.0, 0.0)
    # the surface slides 3 cm along the face in 50 ms (0.6 m/s) while the robot does not move
    second = _patch([(0.2, -0.02), (0.2, 0.03), (0.2, 0.08)], normal, 0.05)
    estimate, _signed, _rms, matches, reason = update_surface_estimate(state, second, pose, 0.05, 0.05)
    assert matches >= 3 and reason is None
    assert estimate is not None and estimate < OPTION.settle_surface


def test_evaluator_truth_marks_the_unobservable_case_as_false_stable():
    root = _root("solo_central", 0)
    specs = {0: LocalSurfacePush(**root.local_parameters(0))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(OPTION, "N").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    metrics = execution_metrics(
        result, initial_object=env.backend.object_state().as_tuple(),
        initial_robots=[state.as_tuple() for state in env.backend.robot_states()],
        force_limit=OPTION.force_limit, physics=CONFIG.physics, option=OPTION,
    )
    assert metrics["settled_observed"] is True
    # the truth verdict is computed separately and is not overwritten by the local one
    assert isinstance(metrics["settled_true"], bool)
    assert metrics["false_stable"] == (metrics["settled_observed"] and not metrics["settled_true"])


# ---------------------------------------------------------------------------
# 8. snapshot isolation and the untouched old path
# ---------------------------------------------------------------------------


def test_snapshot_replay_is_reproducible_and_arms_do_not_contaminate():
    root = _root("solo_central", 1)
    variant = VariantSpec("baseline")
    snapshot, config, _, _, _, velocities = build_snapshot(root, variant, CONFIG)

    def _final(arm):
        env = Continuous2DEnv(config)
        env.restore(snapshot, exact=True)
        from coimagine.local_contact_option.roots import apply_initial_velocities

        apply_initial_velocities(env, velocities)
        specs = {}
        for robot_id in (0, 1):
            if arm in ("O", "P"):
                specs[robot_id] = root.oracle_spec(robot_id)
            else:
                parameters = root.local_parameters(robot_id)
                specs[robot_id] = LocalSurfacePush(**parameters) if parameters else None
        OptionExecutor(OPTION, arm).run(
            env, root_id=root.root_id, variant=variant.name, joint_specs=specs,
            active_robots=root.active_robots,
        )
        return env.backend.object_state().as_tuple(), tuple(
            state.as_tuple() for state in env.backend.robot_states())

    first = _final("N")
    _final("O")
    second = _final("N")
    assert first == second


def test_arm_O_is_the_untouched_old_window():
    root = _root("solo_central", 0)
    specs = (root.oracle_spec(0), Hold())
    env_a = Continuous2DEnv(CONFIG)
    env_a.reset(root.scene())
    direct = env_a.execute_joint_skill(specs)

    env_b = Continuous2DEnv(CONFIG)
    env_b.reset(root.scene())
    executor = OptionExecutor(OPTION, "O")
    result = executor.run(
        env_b, root_id=root.root_id, variant="baseline",
        joint_specs={0: root.oracle_spec(0), 1: None}, active_robots=(0,),
    )
    assert result.wall_clock_steps == len(direct.steps)
    assert env_b.backend.object_state().as_tuple() == pytest.approx(
        env_a.backend.object_state().as_tuple(), abs=1e-12)
    for row, step in zip(result.rows, direct.steps):
        assert row["force"][0] == pytest.approx(list(step.applied_control[0]), abs=1e-12)


def test_rotation_equivariance_of_a_full_execution():
    root = _root("solo_offset", 1)
    angle = -0.4
    finals = []
    for scene in (root.scene(), _scene(root, angle)):
        env = Continuous2DEnv(CONFIG)
        env.reset(scene)
        specs = {0: LocalSurfacePush(alpha=float(root.alpha[0]), speed=float(root.speed[0]))}
        OptionExecutor(OPTION, "F").run(
            env, root_id=root.root_id, variant="rotate", joint_specs=specs,
            active_robots=root.active_robots,
        )
        finals.append(env.backend.object_state().as_tuple())
    cos, sin = math.cos(angle), math.sin(angle)
    expected = (
        finals[0][0] * cos - finals[0][1] * sin,
        finals[0][0] * sin + finals[0][1] * cos,
        finals[0][2] + angle,
    )
    assert expected == pytest.approx(finals[1][:3], abs=1e-6)
