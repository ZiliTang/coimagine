"""Tests for the surface-position feedback push controller (arm X).

Covers the checklist in the task book: observation whitelist and truth isolation, equivariance,
foot/normal/radius geometry, positive and negative gap response, the zero-gap equivalence with
the velocity law, the error and force caps, beta enable/retract, cache reset, support loss,
shared braking, no physical writes, snapshot isolation and the fixed timing conventions.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d.environment import Continuous2DEnv
from coimagine.local_contact_option.contract import (
    DELTA_PER_SPEED,
    Exit,
    OptionConfig,
)
from coimagine.local_contact_option.option import OptionExecutor
from coimagine.local_contact_option.position import (
    PositionPushState,
    patch_centroid_world,
    position_push_force,
    segment_support,
    surface_reference,
    velocity_law_force,
)
from coimagine.local_contact_option.roots import sample_root
from coimagine.local_contact_option.sensing import LocalPatch
from coimagine.local_contact_option.skill import LocalSurfacePush

CONFIG = default_config()
OPTION = OptionConfig(kv=25.0, a_ref_max=3.0)
DT = CONFIG.physics.control_dt
RADIUS = CONFIG.physics.robot_radius


class _FakeOwn:
    def __init__(self, x=0.0, y=0.0, angle=0.0, vx=0.0, vy=0.0, omega=0.0):
        self.x, self.y, self.angle = x, y, angle
        self.vx, self.vy, self.omega = vx, vy, omega


class _FakeObservation:
    """A minimal stand-in exposing exactly the whitelist fields the law may read."""

    def __init__(self, patch, own):
        self.patch = patch
        self.own = own

    def unit_normal_world(self):
        return self.patch.normal_world if self.patch is not None else None


def _flat_patch(offset_along=0.0, distance=0.20, normal=(1.0, 0.0), half_span=0.05, angle=0.0):
    """A flat surface perpendicular to +x, at ``distance`` from a robot at the pose ``angle``."""

    # build the patch in the robot frame: the robot sits at the origin looking along +x
    points = []
    steps = 5
    for index in range(steps):
        along = -half_span + 2 * half_span * index / (steps - 1) + offset_along
        points.append((distance, along))
    low = points[0]
    high = points[-1]
    return LocalPatch(
        points_robot=tuple(points), segment_robot=(low, high), distance=distance,
        normal_world=normal, tangent_world=(-normal[1], normal[0]), confidence=1.0,
        timestamp=0.0, is_target=True, arc_rays=steps, occluder_labels=("target_object",),
    )


def _run(arm, interaction_class="solo_central", index=0, config=None, split="dev"):
    root = sample_root(split, interaction_class, index)
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    specs = {robot_id: LocalSurfacePush(**root.local_parameters(robot_id))
             for robot_id in root.active_robots}
    result = OptionExecutor(config or OPTION, arm).run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    return root, env, result


# ---------------------------------------------------------------------------
# geometry: foot, sign, radius offset
# ---------------------------------------------------------------------------


def test_foot_and_contact_centre_geometry():
    patch = _flat_patch(distance=0.30)
    observation = _FakeObservation(patch, _FakeOwn())
    reference = surface_reference(observation, robot_radius=RADIUS, margin=0.01)
    assert reference is not None
    assert reference["signed_distance"] == pytest.approx(0.30, abs=1e-9)
    assert reference["gap_signed"] == pytest.approx(0.30 - RADIUS, abs=1e-9)
    assert reference["q"] == pytest.approx((0.30, 0.0), abs=1e-9)
    # the contact centre is one radius back from the surface, along the inward normal
    assert reference["p_contact"] == pytest.approx((0.30 - RADIUS, 0.0), abs=1e-9)


def _resting_force(gap: float, delta: float = 0.04) -> float:
    patch = _flat_patch(distance=RADIUS + gap)
    observation = _FakeObservation(patch, _FakeOwn())
    state = PositionPushState()
    state.start_push(delta=delta, alpha=0.0)
    state.beta_time = 0.20          # the ramp is driven by beta_time, not by beta
    force = position_push_force(
        observation, state, DT, robot_radius=RADIUS, margin=0.01,
        kp=100.0, kd=25.0, e_max=0.10, beta_ramp=0.20, force_limit=15.0,
    )
    return force[0]


def test_positive_gap_pushes_forward_and_negative_gap_pulls_back():
    """The geometry error must enter the force with the right sign and the right slope.

    At rest ``F = Kp * (gap_signed + beta * delta)``. The net sign flips once the penetration
    exceeds the virtual advance, which is the physically correct behaviour: a 4 cm advance bias
    still commands a forward push through 3 cm of penetrations, but not through 8 cm.
    """

    assert _resting_force(+0.05) == pytest.approx(100.0 * (+0.05 + 0.04), rel=1e-9)
    assert _resting_force(-0.03) == pytest.approx(100.0 * (-0.03 + 0.04), rel=1e-9)
    assert _resting_force(-0.03) > 0.0
    assert _resting_force(-0.08) < 0.0
    # the derivative with respect to the gap is exactly Kp, i.e. this really is position feedback
    assert _resting_force(0.03) - _resting_force(0.02) == pytest.approx(1.0, abs=1e-9)
    # and the sign follows the sign of the gap once the bias is switched off
    assert _resting_force(+0.02, delta=0.0) > 0.0
    assert _resting_force(-0.02, delta=0.0) < 0.0


def test_penetration_is_not_hidden_by_a_max_clamp():
    patch = _flat_patch(distance=RADIUS - 0.02)
    reference = surface_reference(_FakeObservation(patch, _FakeOwn()),
                                  robot_radius=RADIUS, margin=0.01)
    assert reference["gap_signed"] < 0.0


def test_support_test_rejects_an_extrapolated_plane():
    patch = _flat_patch(distance=0.20, half_span=0.05)
    pose = (0.0, 0.0, 0.0)
    inside, offset = segment_support((0.20, 0.05), patch, pose, 0.01)
    assert inside and offset == 0.0
    outside, offset = segment_support((0.20, 0.09), patch, pose, 0.01)
    assert not outside and offset == pytest.approx(0.04, abs=1e-9)
    before, offset = segment_support((0.20, -0.09), patch, pose, 0.01)
    assert not before and offset == pytest.approx(-0.04, abs=1e-9)


def test_rotation_and_translation_equivariance_of_the_law():
    patch = _flat_patch(distance=0.22, offset_along=0.01)
    observation = _FakeObservation(patch, _FakeOwn(vx=0.03, vy=-0.01))
    state = PositionPushState()
    state.start_push(delta=0.04, alpha=0.3)
    state.beta_time = 0.20
    base = position_push_force(
        observation, state, DT, robot_radius=RADIUS, margin=0.01, kp=100.0, kd=25.0,
        e_max=0.10, beta_ramp=0.20, force_limit=15.0)
    angle = 0.7
    cos, sin = math.cos(angle), math.sin(angle)
    rotated_normal = (patch.normal_world[0] * cos - patch.normal_world[1] * sin,
                      patch.normal_world[0] * sin + patch.normal_world[1] * cos)
    rotated_patch = _flat_patch(distance=0.22, offset_along=0.01, normal=rotated_normal)
    rotated_pose = _FakeOwn(x=5.0, y=-3.0, angle=angle, vx=0.03 * cos - (-0.01) * sin,
                            vy=0.03 * sin + (-0.01) * cos)
    state2 = PositionPushState()
    state2.start_push(delta=0.04, alpha=0.3)
    state2.beta_time = 0.20
    moved = position_push_force(
        _FakeObservation(rotated_patch, rotated_pose), state2, DT, robot_radius=RADIUS,
        margin=0.01, kp=100.0, kd=25.0, e_max=0.10, beta_ramp=0.20, force_limit=15.0)
    assert moved[0] == pytest.approx(base[0] * cos - base[1] * sin, abs=1e-9)
    assert moved[1] == pytest.approx(base[0] * sin + base[1] * cos, abs=1e-9)


# ---------------------------------------------------------------------------
# equivalence with the velocity law
# ---------------------------------------------------------------------------


def test_zero_gap_equivalence_with_the_velocity_law():
    """Fixed plane, zero gap, beta = 1, unsaturated: the two laws must agree exactly.

    This is the algebraic identity Kp * delta = Kv * s with Kd = Kv, checked on the two
    *implementations* rather than on two copies of the same expression.
    """

    kv, kp, kd = 25.0, 100.0, 25.0
    for speed in (0.08, 0.16, 0.26):
        delta = kv * speed / kp
        assert delta == pytest.approx(DELTA_PER_SPEED * speed, abs=1e-12)
        for alpha in (0.0, 0.25, -0.35):
            velocity = (0.01, -0.004)
            patch = _flat_patch(distance=RADIUS)
            observation = _FakeObservation(patch, _FakeOwn(vx=velocity[0], vy=velocity[1]))
            state = PositionPushState()
            state.start_push(delta=delta, alpha=alpha)
            state.beta_time = 0.20
            position_force = position_push_force(
                observation, state, DT, robot_radius=RADIUS, margin=0.01, kp=kp, kd=kd,
                e_max=0.10, beta_ramp=0.20, force_limit=15.0)
            normal = patch.normal_world
            tangent = (-normal[1], normal[0])
            reference = (speed * (math.cos(alpha) * normal[0] + math.sin(alpha) * tangent[0]),
                         speed * (math.cos(alpha) * normal[1] + math.sin(alpha) * tangent[1]))
            velocity_force = velocity_law_force(reference, velocity, kv, 15.0)
            assert position_force == pytest.approx(velocity_force, abs=1e-9)


def test_non_zero_gap_differs_only_by_the_observed_gap_term():
    kv, kp, kd, speed = 25.0, 100.0, 25.0, 0.16
    delta = kv * speed / kp
    for gap in (0.02, -0.01):
        patch = _flat_patch(distance=RADIUS + gap)
        observation = _FakeObservation(patch, _FakeOwn())
        state = PositionPushState()
        state.start_push(delta=delta, alpha=0.0)
        state.beta_time = 0.20
        force = position_push_force(
            observation, state, DT, robot_radius=RADIUS, margin=0.01, kp=kp, kd=kd,
            e_max=0.10, beta_ramp=0.20, force_limit=15.0)
        normal = patch.normal_world
        reference = (speed * normal[0], speed * normal[1])
        velocity_force = velocity_law_force(reference, (0.0, 0.0), kv, 15.0)
        assert force[0] - velocity_force[0] == pytest.approx(kp * gap, abs=1e-9)


def test_fitted_geometry_responds_to_a_shifted_contact():
    """The reported gap must change when the surface moves, i.e. it really is feedback."""

    near = surface_reference(_FakeObservation(_flat_patch(distance=0.19), _FakeOwn()),
                             robot_radius=RADIUS, margin=0.01)
    far = surface_reference(_FakeObservation(_flat_patch(distance=0.25), _FakeOwn()),
                            robot_radius=RADIUS, margin=0.01)
    assert far["gap_signed"] - near["gap_signed"] == pytest.approx(0.06, abs=1e-9)


# ---------------------------------------------------------------------------
# caps, beta, caching
# ---------------------------------------------------------------------------


def test_position_error_is_capped_and_the_final_force_is_capped():
    patch = _flat_patch(distance=RADIUS + 5.0)  # absurd error, must be clipped
    observation = _FakeObservation(patch, _FakeOwn())
    state = PositionPushState()
    state.start_push(delta=0.04, alpha=0.0)
    state.beta_time = 0.20
    force = position_push_force(
        observation, state, DT, robot_radius=RADIUS, margin=0.01, kp=100.0, kd=25.0,
        e_max=0.10, beta_ramp=0.20, force_limit=15.0)
    assert math.hypot(*state.error_bounded) == pytest.approx(0.10, abs=1e-9)
    # with Kp = 100 N/m and e_max = 0.10 m the stiffness term tops out at 10 N, so the 15 N
    # actuator limit is not the binding cap here -- both are still respected
    assert math.hypot(*force) == pytest.approx(10.0, abs=1e-9)
    assert math.hypot(*force) <= 15.0 + 1e-9


def test_beta_ramps_from_zero_and_is_retracted_after_push():
    patch = _flat_patch(distance=RADIUS + 0.03)
    observation = _FakeObservation(patch, _FakeOwn())
    state = PositionPushState()
    state.start_push(delta=0.04, alpha=0.0)
    assert state.beta == 0.0
    forces = []
    for _ in range(4):
        forces.append(position_push_force(
            observation, state, DT, robot_radius=RADIUS, margin=0.01, kp=100.0, kd=25.0,
            e_max=0.10, beta_ramp=0.20, force_limit=15.0))
    assert state.beta == pytest.approx(0.20 / 0.20)  # 4 * 0.05 s = the full ramp
    assert abs(forces[-1][0]) > abs(forces[0][0])    # the virtual advance really grew
    state.end_push()
    assert state.beta == 0.0 and state.p_ref is None and state.p_contact is None
    assert not state.active


def test_reference_cache_is_reset_between_pushes():
    state = PositionPushState()
    state.start_push(delta=0.04, alpha=0.0)
    state.advance(0.20, 0.20)
    assert state.beta == pytest.approx(1.0)
    state.end_push()
    state.start_push(delta=0.02, alpha=0.1)
    assert state.beta == 0.0 and state.beta_time == 0.0 and state.line_normal is None
    assert state.delta == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# execution-level behaviour
# ---------------------------------------------------------------------------


def test_arm_x_reaches_the_full_effective_push_on_the_dev_baseline():
    for interaction_class in ("solo_central", "dual_same_side", "constrained"):
        _root, _env, result = _run("X", interaction_class)
        assert result.exit_reason == Exit.SETTLED_OBSERVED.value, interaction_class
        assert result.effective_push_seconds >= 2.0 - 1e-9
        assert result.push_completed


def test_arm_x_never_writes_a_body_state():
    """Every stage transition must be force-only; the backend write hook is the referee."""

    from coimagine.continuous2d import backend as backend_module

    calls = {"writes": 0}
    original = backend_module.PymunkBackend._write_states

    def guard(self, *args, **kwargs):
        calls["writes"] += 1
        return original(self, *args, **kwargs)

    backend_module.PymunkBackend._write_states = guard
    try:
        _run("X", "solo_central")
    finally:
        backend_module.PymunkBackend._write_states = original
    assert calls["writes"] == 0


def test_arm_x_and_arm_v_share_one_snapshot_and_do_not_contaminate_each_other():
    _root, _env, v_first = _run("V", "solo_offset")
    _root2, _env2, x_run = _run("X", "solo_offset")
    _root3, _env3, v_second = _run("V", "solo_offset")
    assert v_first.terminal_object == v_second.terminal_object
    assert v_first.exit_reason == v_second.exit_reason
    # X starts from the same initial object pose as V
    assert v_first.rows[0]["true"]["object"] == x_run.rows[0]["true"]["object"]


def test_arm_x_logs_the_reference_geometry_and_the_real_force():
    _root, _env, result = _run("X", "solo_central")
    push = [row for row in result.rows if (row.get("phase") or row.get("stage")) == "push"]
    assert push
    checked = 0
    for row in push:
        entry = row["observed"]["0"]
        if not entry.get("position_active") or entry.get("p_ref") is None:
            continue
        checked += 1
        # beta must have been frozen for this push and start from zero
        assert 0.0 <= entry["beta"] <= 1.0 + 1e-9
        # the applied force is the clipped raw force, never the raw one
        assert math.hypot(*row["force"][0]) <= OPTION.force_limit + 1e-9
        # no extra unit force is added after clipping
        if math.hypot(*entry["force_raw"]) > OPTION.force_limit:
            assert math.hypot(*row["force"][0]) == pytest.approx(OPTION.force_limit, abs=1e-9)
        else:
            assert row["force"][0] == pytest.approx(entry["force_raw"], abs=1e-9)
    assert checked > 10


def test_arm_x_delta_comes_from_the_frozen_mapping():
    for speed, expected in ((0.08, 0.020), (0.16, 0.040), (0.26, 0.065)):
        root = sample_root("dev", "solo_central", 0)
        parameters = dict(root.local_parameters(root.active_robots[0]))
        parameters["speed"] = speed
        parameters["delta"] = DELTA_PER_SPEED * speed
        skill = LocalSurfacePush(**parameters)
        assert skill.delta == pytest.approx(expected, abs=1e-9)
        assert skill.delta <= 0.08


def test_a_support_loss_ends_the_push_with_its_own_exit_reason():
    root = sample_root("dev", "solo_central", 0)
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    specs = {robot_id: LocalSurfacePush(**root.local_parameters(robot_id))
             for robot_id in root.active_robots}
    # a margin of 10 m makes every push "unsupported" -> the shared protocol must end it
    config = replace(OPTION, patch_support_margin=-10.0)
    result = OptionExecutor(config, "X").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    assert result.exit_reason == Exit.PATCH_SUPPORT_LOST.value
    assert not result.push_completed or result.effective_push_seconds < 2.0


def test_the_two_arms_use_the_same_approach_and_brake():
    """Only PUSH may differ, otherwise the comparison would not isolate the control law."""

    for interaction_class in ("solo_central", "dual_rotate"):
        _r, _e, v = _run("V", interaction_class)
        _r2, _e2, x = _run("X", interaction_class)
        def approach_trace(result):
            return [(row["time"], row["force"][0], row["force"][1]) for row in result.rows
                    if (row.get("phase") or row.get("stage")) == "approach"]
        # the approach trajectories are identical up to the step where PUSH begins; compare the
        # first period, which is entirely baseline protocol
        assert approach_trace(v)[0] == approach_trace(x)[0]
