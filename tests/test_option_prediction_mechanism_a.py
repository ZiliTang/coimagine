"""Round 3, phase A tests: timestamp discipline, motion misjudgement, abort accounting.

These cover exactly the risks the round-3 contract names: sensor sample time vs control-period
time, one frame = one piece of evidence, stale/gapped evidence, the local blind spot for purely
tangential motion, rest-then-motion window lag, and the reachability of ``settled_after_abort``.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.local_contact_option.control import (  # noqa: E402
    LocalControlState,
    update_settle_evidence,
)
from coimagine.local_contact_option.contract import (  # noqa: E402
    SETTLE_SURFACE,
    SURFACE_FRESHNESS,
    SURFACE_MAX_SAMPLE_GAP,
)
from coimagine.local_contact_option.sensing import LocalPatch  # noqa: E402

WINDOW = 0.20
MIN_BASELINE = 0.10
WALL_XS = tuple(0.10 + 0.02 * index for index in range(12))


def make_patch(points, distance=0.30, normal=(0.0, 1.0), timestamp=0.0):
    return LocalPatch(
        points_robot=tuple(points), segment_robot=(points[0], points[-1]), distance=distance,
        normal_world=normal, tangent_world=(-normal[1], normal[0]), confidence=1.0,
        timestamp=timestamp, is_target=True, arc_rays=len(points), occluder_labels=())


def evidence(state, patch, sample_time, now, window=WINDOW):
    return update_settle_evidence(
        state, patch, (0.0, 0.20, 0.0), sample_time, now,
        window=window, min_baseline=MIN_BASELINE)


def feed(state, points_at, span=0.20, window=WINDOW, dt=0.05, pose_at=None):
    """Deliver one frame per control period and return the last period's evidence.

    ``pose_at(t)`` is the robot pose that scanned the frame; a robot moving towards a static wall
    has a changing pose *and* changing robot-frame returns, which is exactly what must cancel.
    """

    value = None
    time = 0.0
    while time <= span + 1e-9:
        patch = make_patch(points_at(time), timestamp=time)
        pose = pose_at(time) if pose_at else (0.0, 0.20, 0.0)
        value = update_settle_evidence(state, patch, pose, time, time,
                                       window=window, min_baseline=MIN_BASELINE)
        time = round(time + dt, 6)
    return value


# ---------------------------------------------------------------------------
# 1. static object, robot moves: self-motion must cancel
# ---------------------------------------------------------------------------


def test_static_surface_with_robot_self_motion_reads_zero():
    """The robot closes 5 mm per period; the world points of a static wall are identical."""

    state = LocalControlState()
    # the robot advances 0.05 m/s towards a wall fixed at world y = 0.50, so the *robot-frame*
    # returns shrink by exactly as much as the pose moves
    value = feed(state, lambda t: [(x, 0.30 - 0.05 * t) for x in WALL_XS],
                 pose_at=lambda t: (0.0, 0.20 + 0.05 * t, 0.0))
    assert value.reason is None
    assert value.speed == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. known constant normal velocity, below and above the threshold
# ---------------------------------------------------------------------------


def _constant_velocity_estimate(speed):
    state = LocalControlState()
    return feed(state, lambda t: [(x, 0.30 + speed * t) for x in WALL_XS])


def test_constant_normal_velocity_below_and_above_threshold():
    slow = _constant_velocity_estimate(0.015)
    fast = _constant_velocity_estimate(0.030)
    assert slow.speed == pytest.approx(0.015, rel=1e-6)
    assert slow.speed < SETTLE_SURFACE
    assert fast.speed == pytest.approx(0.030, rel=1e-6)
    assert fast.speed > SETTLE_SURFACE


# ---------------------------------------------------------------------------
# 3. rotation: opposite normal motion must not average-cancel
# ---------------------------------------------------------------------------


def test_rotation_is_not_cancelled_by_spatial_averaging():
    """Half the points move +v, half -v, the signed mean is 0 -- the statistic must not be."""

    state = LocalControlState()
    # a rotation about the contact centre: displacement grows linearly with (x - centre)
    centre = 0.5 * (WALL_XS[0] + WALL_XS[-1])
    rate = 0.10  # m/s at the extreme points

    def points(t):
        return [(x, 0.30 + rate * t * (x - centre) / (WALL_XS[-1] - centre)) for x in WALL_XS]

    value = feed(state, points)
    assert value.signed == pytest.approx(0.0, abs=1e-9), "the signed spatial mean cancels"
    assert value.speed is not None and value.speed > SETTLE_SURFACE, (
        "the primary statistic is the median of the per-point signed magnitudes, so a rotation "
        "cannot hide behind a cancelling mean")


# ---------------------------------------------------------------------------
# 4. noise, delay, repeated frames, long gaps
# ---------------------------------------------------------------------------


def test_two_millimetre_noise_is_resolved_by_the_window():
    rng = random.Random(3)
    state = LocalControlState()
    readings = []
    for index in range(8):
        now = index * 0.05
        patch = make_patch([(x, 0.30 + rng.uniform(-0.002, 0.002)) for x in WALL_XS],
                           timestamp=now)
        value = evidence(state, patch, now, now)
        if value.speed is not None:
            readings.append(value.speed)
    assert readings
    assert max(readings) < SETTLE_SURFACE


def test_repeated_frame_is_one_piece_of_evidence():
    state = LocalControlState()
    state_marker = []
    for index in range(5):
        now = index * 0.05
        value = evidence(state, make_patch([(x, 0.30) for x in WALL_XS], timestamp=now), now, now)
        state_marker.append(value.new_sample)
    assert state_marker == [True, True, True, True, True]
    # the same frame delivered again carries no new evidence
    repeated = evidence(state, make_patch([(x, 0.30) for x in WALL_XS], timestamp=0.20), 0.20, 0.25)
    assert repeated.new_sample is False
    assert repeated.reason is None, "a repeat does not make the window unusable"
    assert repeated.speed is not None


def test_stale_frame_and_long_gap_are_unavailable():
    state = LocalControlState()
    evidence(state, make_patch([(x, 0.30) for x in WALL_XS], timestamp=0.0), 0.0, 0.0)
    stale = evidence(state, make_patch([(x, 0.30) for x in WALL_XS], timestamp=0.0),
                     0.0, SURFACE_FRESHNESS + 0.05)
    assert stale.reason == "stale_frame" and stale.speed is None

    fresh = LocalControlState()
    evidence(fresh, make_patch([(x, 0.30) for x in WALL_XS], timestamp=0.0), 0.0, 0.0)
    gapped = evidence(fresh, make_patch([(x, 0.30) for x in WALL_XS],
                                        timestamp=SURFACE_MAX_SAMPLE_GAP + 0.05),
                      SURFACE_MAX_SAMPLE_GAP + 0.05, SURFACE_MAX_SAMPLE_GAP + 0.05)
    assert gapped.reason == "sample_gap_too_large"


def test_missing_patch_and_repeat_do_not_pretend_to_be_evidence():
    state = LocalControlState()
    assert evidence(state, None, 0.0, 0.0).reason == "no_patch"
    assert update_settle_evidence(state, make_patch([(x, 0.30) for x in WALL_XS]),
                                  (0.0, 0.2, 0.0), None, 0.0,
                                  window=WINDOW, min_baseline=MIN_BASELINE).reason == "no_sample_time"


# ---------------------------------------------------------------------------
# 5. rest-then-motion: the window lag must not confirm a motion that just started
# ---------------------------------------------------------------------------


def test_longer_window_lags_more_when_motion_restarts():
    """Quantify validation 5: how many periods after motion restarts does each window notice?"""

    def periods_to_notice(window: float, speed: float = 0.05) -> int:
        state = LocalControlState()
        now = 0.0
        for index in range(int(window / 0.05) + 2):
            evidence(state, make_patch([(x, 0.30) for x in WALL_XS], timestamp=now), now, now,
                     window=window)
            now += 0.05
        for step in range(1, 60):
            value = evidence(
                state, make_patch([(x, 0.30 + speed * now) for x in WALL_XS], timestamp=now),
                now, now, window=window)
            if value.speed is not None and value.speed >= SETTLE_SURFACE:
                return step
            now += 0.05
        return 10 ** 6

    short = periods_to_notice(0.20)
    long = periods_to_notice(0.30)
    assert short <= long, "the longer window carries more stale quiet evidence"


# ---------------------------------------------------------------------------
# 6. the declared blind spot: purely tangential motion
# ---------------------------------------------------------------------------


def test_pure_tangential_motion_is_a_declared_blind_spot():
    """A genuinely moving object can read quiet when the motion is along the surface.

    This is asserted, not fixed: the local normal-scan estimator cannot see tangential slip, and
    the write-up must state that the settle fix does not certify the whole object state.
    """

    state = LocalControlState()
    # 0.10 m/s sideways: the object really moves, the normal displacement stays zero
    value = feed(state, lambda t: [(x + 0.10 * t, 0.30) for x in WALL_XS])
    assert value.speed is not None and value.speed < SETTLE_SURFACE
    assert value.speed == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# abort accounting: settled_after_abort must be reachable
# ---------------------------------------------------------------------------


def test_settled_after_abort_is_reachable_on_a_real_abort_then_settle_trajectory():
    """``dev/dual_rotate/002 baseline`` loses contact during PUSH but comes to rest later."""

    from coimagine.local_contact_option.contract import OptionConfig
    from coimagine.local_contact_option.roots import build_roots
    from coimagine.local_contact_option.runner import run_execution
    from coimagine.local_contact_option.roots import VARIANT_SPECS

    roots = {root.root_id: root for root in build_roots("dev", 4)}
    root = roots["dev/dual_rotate/002"]
    variant = next(spec for spec in VARIANT_SPECS if spec.name == "baseline")
    from coimagine.continuous2d.config import default_config

    record, _rows, _report = run_execution(
        root, variant, "V", OptionConfig(kv=25.0, a_ref_max=3.0, approach_max_speed=0.30,
                                         approach_contact_speed=0.06, approach_gain=4.0,
                                         ready_precontact_speed=0.02,
                                         settle_window=0.20),
        default_config())
    outcome = record["arm_outcome"]
    assert outcome["primary_exit_reason"], "this trajectory is expected to abort"
    assert outcome["locally_settled"], "and to reach the quiet criterion during SETTLE"
    assert outcome["settled_after_abort"] is True
    assert outcome["settled_observed"] is False, "an abort must never be a completion"
    assert outcome["exit_reason"] == outcome["primary_exit_reason"]
    metrics = record["metrics"]
    assert metrics["locally_settled"] and not metrics["settled_observed"]
    assert metrics["primary_exit_reason"] == "contact_lost"
