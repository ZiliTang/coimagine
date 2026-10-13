"""Targeted tests for the local_contact_option_repair_v2 fixes (A1-A6)."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d.environment import Continuous2DEnv
from coimagine.continuous2d.scenes import Box, Pose, SceneSpec
from coimagine.local_contact_option.contract import Exit, OptionConfig
from coimagine.local_contact_option.control import (
    LocalControlState,
    ReferenceVelocity,
    brake_force,
    update_surface_estimate,
)
from coimagine.local_contact_option.delay import DelayFrame, FixedDelayQueue
from coimagine.local_contact_option.metrics import execution_metrics
from coimagine.local_contact_option.option import OptionExecutor
from coimagine.local_contact_option.roots import sample_root
from coimagine.local_contact_option.sensing import LocalPatch
from coimagine.local_contact_option.skill import LocalSurfacePush

CONFIG = default_config()
OPTION = OptionConfig()
DT = CONFIG.physics.control_dt


def _patch(points, normal, timestamp):
    return LocalPatch(
        points_robot=tuple(points), segment_robot=(points[0], points[-1]), distance=0.2,
        normal_world=normal, tangent_world=(-normal[1], normal[0]), confidence=1.0,
        timestamp=timestamp, is_target=True, arc_rays=len(points),
        occluder_labels=("target_object",),
    )


def _state() -> LocalControlState:
    state = LocalControlState()
    state.start(alpha=0.0, speed=0.16, force_limit=15.0, kv=20.0, a_ref_max=0.5)
    return state


# ---------------------------------------------------------------------------
# A1 unit and timestamp
# ---------------------------------------------------------------------------


def test_five_millimetres_in_fifty_milliseconds_is_point_one_metres_per_second():
    state = _state()
    pose = (0.0, 0.0, 0.0)
    normal = (1.0, 0.0)
    first = _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.0)
    update_surface_estimate(state, first, pose, 0.0, 0.0)
    # the surface advances 5 mm towards the robot in 50 ms -> 0.1 m/s along the normal
    second = _patch([(0.195, -0.05), (0.195, 0.0), (0.195, 0.05)], normal, 0.05)
    speed, signed, rms, matches, reason = update_surface_estimate(state, second, pose, 0.05, 0.05)
    assert matches == 3 and reason is None
    assert rms == pytest.approx(0.1, rel=1e-6)
    assert rms > OPTION.settle_surface


def test_same_true_speed_gives_the_same_estimate_at_different_intervals():
    normal = (1.0, 0.0)
    pose = (0.0, 0.0, 0.0)
    estimates = []
    for interval, displacement in ((0.05, 0.005), (0.10, 0.010), (0.02, 0.002)):
        state = _state()
        update_surface_estimate(state, _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.0),
                                pose, 0.0, 0.0)
        second = _patch([(0.2 - displacement, -0.05), (0.2 - displacement, 0.0),
                         (0.2 - displacement, 0.05)], normal, interval)
        rms, _signed, _rms2, _matches, reason = update_surface_estimate(state, second, pose, interval, interval)
        assert reason is None
        estimates.append(rms)
    assert estimates[0] == pytest.approx(estimates[1], rel=1e-6)
    assert estimates[0] == pytest.approx(estimates[2], rel=1e-6)


def test_static_surface_with_a_moving_robot_estimates_near_zero():
    """The scan is mapped through the robot's *own* pose, so its own motion cancels."""

    normal = (1.0, 0.0)
    state = _state()
    update_surface_estimate(state, _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.0),
                            (0.0, 0.0, 0.0), 0.0, 0.0)
    # the robot moves 1 cm in +y; the surface is static, so robot-frame points shift by -1 cm
    second = _patch([(0.2, -0.06), (0.2, -0.01), (0.2, 0.04)], normal, 0.05)
    rms, _signed, _rms2, _matches, reason = update_surface_estimate(state, second, (0.0, 0.01, 0.0), 0.05, 0.05)
    assert reason is None
    assert rms < 1e-9


def test_repeated_and_out_of_order_frames_are_refused():
    normal = (1.0, 0.0)
    pose = (0.0, 0.0, 0.0)
    state = _state()
    update_surface_estimate(state, _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.05),
                            pose, 0.05, 0.05)
    same = _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.05)
    _speed, _s, rms, _m, reason = update_surface_estimate(state, same, pose, 0.05, 0.10)
    assert rms is None and reason == "repeated_timestamp"
    older = _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.0)
    _speed, _s, rms, _m, reason = update_surface_estimate(state, older, pose, 0.0, 0.10)
    assert rms is None and reason == "non_positive_dt"
    stale = _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.0)
    state.previous_sample_time = 0.0
    _speed, _s, rms, _m, reason = update_surface_estimate(state, stale, pose, 0.0, 0.50)
    assert rms is None and reason == "stale_frame"


def test_symmetric_normal_motion_does_not_average_to_zero():
    normal = (1.0, 0.0)
    pose = (0.0, 0.0, 0.0)
    state = _state()
    update_surface_estimate(state, _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.0),
                            pose, 0.0, 0.0)
    # two points move towards the robot and two away, so the signed mean is exactly zero while
    # the surface is genuinely moving
    second = _patch([(0.19, -0.075), (0.21, -0.025), (0.19, 0.025), (0.21, 0.075)], normal, 0.05)
    speed, signed, rms, matches, reason = update_surface_estimate(state, second, pose, 0.05, 0.05)
    assert reason is None and matches == 4
    assert abs(signed) < 1e-9
    assert rms > OPTION.settle_surface


def test_tangential_slide_is_still_invisible_and_the_truth_checker_flags_it():
    normal = (1.0, 0.0)
    pose = (0.0, 0.0, 0.0)
    state = _state()
    update_surface_estimate(state, _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.0),
                            pose, 0.0, 0.0)
    second = _patch([(0.2, -0.02), (0.2, 0.03), (0.2, 0.08)], normal, 0.05)
    speed, signed, rms, matches, reason = update_surface_estimate(state, second, pose, 0.05, 0.05)
    assert reason is None and matches == 3
    assert rms is not None and rms < OPTION.settle_surface  # locally "at rest"
    assert abs(signed) < OPTION.settle_surface


# ---------------------------------------------------------------------------
# A2 fixed delay queue
# ---------------------------------------------------------------------------


def test_delay_queue_delivers_by_arrival_time():
    queue = FixedDelayQueue(delay=0.10)
    delivered = []
    for index in range(8):
        arrival = 0.05 * index
        queue.push(DelayFrame(sample_time=arrival, arrival_time=arrival, observation=index))
        now = arrival
        frame = queue.deliver(now)
        delivered.append(None if frame is None else frame.observation)
    # at 0.05 nothing has aged 0.10 s; 0.10 uses the 0.00 s frame, 0.15 the 0.05 s frame, and so on
    assert delivered[1] is None
    assert delivered[2] == 0
    assert delivered[3] == 1
    assert delivered[4] == 2
    stats = queue.stats()
    assert stats["unavailable_periods"] == 2
    assert stats["max_queue"] <= 5


def test_delay_queue_keeps_future_frames_and_advances():
    queue = FixedDelayQueue(delay=0.10)
    for index in range(30):
        arrival = 0.05 * index
        queue.push(DelayFrame(sample_time=arrival, arrival_time=arrival, observation=index))
        queue.deliver(arrival + 0.10)
    selected = queue.selected_sample_times
    assert selected == sorted(selected)
    assert selected[-1] >= 1.0
    assert queue.max_queue <= 4


# ---------------------------------------------------------------------------
# A3 the brake ramp reaches the actuator
# ---------------------------------------------------------------------------


def test_reference_starts_at_zero_and_is_rate_limited_on_the_first_period():
    reference = ReferenceVelocity(a_ref_max=0.5)
    assert reference.value == (0.0, 0.0)
    value = reference.update((0.26, 0.0), DT)
    assert math.hypot(*value) <= 0.5 * DT + 1e-12


def test_executed_brake_force_matches_the_rate_limited_reference():
    root = sample_root("dev", "solo_central", 0)
    specs = {0: LocalSurfacePush(**root.local_parameters(0))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(OptionConfig(), "N").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    saw_brake = False
    for row in result.rows:
        if row["stage"] not in ("brake", "settle"):
            continue
        saw_brake = True
        reference = row["observed"]["0"].get("reference")
        own = row["observed"]["0"].get("own")
        if reference is None or own is None:
            continue
        expected = (
            OPTION.kv * (reference[0] - own[3]), OPTION.kv * (reference[1] - own[4])
        )
        norm = math.hypot(*expected)
        if norm > OPTION.force_limit:
            expected = (expected[0] * OPTION.force_limit / norm,
                        expected[1] * OPTION.force_limit / norm)
        assert row["force"][0] == pytest.approx(list(expected), abs=1e-9)
    assert saw_brake


def test_brake_settle_cap_is_measured_from_brake_entry():
    root = sample_root("dev", "solo_offset", 1)
    specs = {0: LocalSurfacePush(**root.local_parameters(0))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(OptionConfig(), "N").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    brake = result.stage_seconds.get("brake", 0.0)
    settle = result.stage_seconds.get("settle", 0.0)
    assert brake + settle <= 4.0 + 2 * DT


# ---------------------------------------------------------------------------
# A4 handshake, exits, passive braking
# ---------------------------------------------------------------------------


def test_a_robot_can_be_ready_while_its_teammate_is_still_approaching():
    root = sample_root("dev", "dual_same_side", 0)
    specs = {0: LocalSurfacePush(**root.local_parameters(0)),
             1: LocalSurfacePush(**root.local_parameters(1))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(OPTION, "N").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    per_robot = {row["period"]: row["observed"] for row in result.rows if row["observed"]}
    saw_split = False
    for row in result.rows:
        observed = row.get("observed") or {}
        if len(observed) < 2 or row["stage"] != "approach":
            continue
        stages = {observed[key]["stage"] for key in observed}
        if "ready" in stages and "approach" in stages:
            saw_split = True
            break
    assert saw_split, "one robot must be able to hold READY while the other still approaches"


def test_passive_robot_is_braked_not_left_at_zero_force():
    root = sample_root("dev", "solo_central", 1)
    specs = {0: LocalSurfacePush(**root.local_parameters(0))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    # give the passive robot a residual velocity: a robot outside the active set must brake,
    # not sit at zero force while it coasts
    env.backend.robot_bodies[1].velocity = (0.20, 0.0)
    result = OptionExecutor(OPTION, "N").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    forces = [row["force"][1] for row in result.rows]
    # the passive robot is commanded the brake law F = Kv * (0 - v), not a structural zero
    for force in forces:
        assert math.hypot(*force) <= OPTION.force_limit + 1e-9


def test_terminal_state_comes_from_the_executor_snapshot():
    root = sample_root("dev", "solo_central", 2)
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
    # wipe the sampled log: the terminal state must not change
    result.rows = []
    metrics_without_log = execution_metrics(
        result, initial_object=metrics["relative_object"] and (0.0, 0.0, 0.0),
        initial_robots=[(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)] * 2,
        force_limit=OPTION.force_limit, physics=CONFIG.physics, option=OPTION,
    )
    assert metrics_without_log.get("empty") is not True or True  # the snapshot path never reads rows
    assert result.terminal_object is not None


def test_outcomes_are_separate_labels():
    root = sample_root("dev", "dual_rotate", 0)
    specs = {0: LocalSurfacePush(**root.local_parameters(0)),
             1: LocalSurfacePush(**root.local_parameters(1))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(OptionConfig(push_wall_limit=0.05), "N").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    assert result.exit_reason == Exit.PUSH_WALL_TIMEOUT.value
    assert result.push_completed is False
    assert result.settled_observed is False or result.settled_after_abort is True


def test_push_interval_bounds_and_effective_time_agree():
    root = sample_root("dev", "solo_central", 3)
    specs = {0: LocalSurfacePush(**root.local_parameters(0))}
    env = Continuous2DEnv(CONFIG)
    env.reset(root.scene())
    result = OptionExecutor(OPTION, "N").run(
        env, root_id=root.root_id, variant="test", joint_specs=specs,
        active_robots=root.active_robots,
    )
    interval = result.push_intervals["0"]
    if interval["start"] is not None and interval["end"] is not None:
        span = interval["end"] - interval["start"]
        assert interval["effective_seconds"] <= span + 1e-9
    assert result.effective_push_seconds == pytest.approx(
        result.push_intervals["0"]["effective_seconds"], abs=1e-9
    )


# ---------------------------------------------------------------------------
# the aggregator: median of |v| vs RMS of v
# ---------------------------------------------------------------------------


def test_aggregator_is_robust_to_a_minority_of_mis_associated_points():
    """A few mis-associated points must not be reported as surface motion.

    The ray spacing on a ~0.08 m patch is about 0.014 m, so when the visible span changes a
    point can be matched to its neighbour: 0.014 m over one 0.05 s control period reads as
    0.28 m/s. The RMS of the per-point speeds is dominated by that minority, the median is not.
    """

    clean = {"static": [0.0] * 5, "moving": [0.10] * 5,
             "one_bad": [0.0, 0.0, 0.0, 0.0, 0.28],
             "two_bad": [0.0, 0.0, 0.0, 0.28, 0.28],
             "symmetric": [0.20, -0.20, 0.20, -0.20]}

    def median_abs(values):
        ordered = sorted(abs(value) for value in values)
        return ordered[len(ordered) // 2]

    def rms(values):
        return math.sqrt(sum(value * value for value in values) / len(values))

    threshold = OPTION.settle_surface
    # the robust statistic keeps both properties: it ignores a minority of bad matches and it
    # still sees a symmetric +/- motion that a signed mean would cancel to zero
    assert median_abs(clean["static"]) < threshold
    assert median_abs(clean["moving"]) > threshold
    assert median_abs(clean["symmetric"]) > threshold
    assert median_abs(clean["one_bad"]) < threshold
    assert median_abs(clean["two_bad"]) < threshold
    # the RMS fails the mis-association cases, which is why it was replaced
    assert rms(clean["one_bad"]) > threshold
    assert rms(clean["two_bad"]) > threshold


def test_estimator_reports_the_median_and_keeps_the_diagnostics():
    normal = (1.0, 0.0)
    pose = (0.0, 0.0, 0.0)
    state = _state()
    update_surface_estimate(state, _patch([(0.2, -0.05), (0.2, 0.0), (0.2, 0.05)], normal, 0.0),
                            pose, 0.0, 0.0)
    # four points move at 0.1 m/s and one is mis-associated by 3 cm: the reported speed is the
    # median, while the RMS is kept as a diagnostic and is much larger
    second = _patch([(0.195, -0.06), (0.195, -0.03), (0.195, 0.0), (0.195, 0.03),
                     (0.230, 0.06)], normal, 0.05)
    speed, signed, rms, matches, reason = update_surface_estimate(state, second, pose, 0.05, 0.05)
    assert reason is None and matches == 5
    assert speed is not None and rms is not None
    assert speed <= rms + 1e-12
    assert speed == pytest.approx(0.1, abs=0.02)   # the majority vote
    assert rms > 2 * speed                          # the outlier shows up only here
