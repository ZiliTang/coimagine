"""Round 2 tests: the settle-estimator measurements and the round-2 analysis guarantees.

These pin the properties the round-2 audit *depends on*, so that a later change which breaks the
measurement fails here rather than silently producing a different number:

* the surface-speed estimator is world-frame (robot ego-motion cancels) and rotation-sensitive;
* a longer matching baseline really does lower the noise floor on a noisy static surface, which is
  the whole reason the ``noise`` pressure test could not settle;
* a repeated sample timestamp does not destroy the verdict, and a missing observation never counts
  as evidence of stillness;
* the optional active-brake preload is off by zero, bounded, and reads only the local normal;
* the B target carries no object global pose component (invariant under a global rigid transform);
* the leave-one-root-out split never lets a variant of a held-out root into the training fold.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.local_contact_option.control import (  # noqa: E402
    LocalControlState,
    brake_force,
    brake_force_with_hold,
    update_surface_estimate,
    update_windowed_surface_estimate,
)
from coimagine.local_contact_option.sensing import (  # noqa: E402
    BodyKinematics,
    LocalContactObservation,
    LocalPatch,
)

from experiments.local_position_push_v2.analyse_b import (  # noqa: E402
    leave_one_root_out_folds,
    local_geometry,
)

WINDOW = 0.30
MIN_BASELINE = 0.05


def make_patch(points, distance, normal=(0.0, 1.0), timestamp=0.0):
    return LocalPatch(
        points_robot=tuple(points), segment_robot=(points[0], points[-1]), distance=distance,
        normal_world=normal, tangent_world=(-normal[1], normal[0]), confidence=1.0,
        timestamp=timestamp, is_target=True, arc_rays=len(points), occluder_labels=())


WALL_XS = tuple(0.10 + 0.02 * index for index in range(12))


def observation(x, y, angle, patch):
    return LocalContactObservation(
        robot_id=0, tick=0, time=patch.timestamp if patch else 0.0,
        sample_time=patch.timestamp if patch else None,
        own=BodyKinematics(x, y, angle, 0.0, 0.0, 0.0), teammate=None, patch=patch,
        contacting=patch is not None, contact_point_robot=None, contact_force_world=(0.0, 0.0),
        contact_impulse=0.0, arena=(3.2, 2.2), robot_radius=0.075, force_limit=15.0)


# ---------------------------------------------------------------------------
# frame
# ---------------------------------------------------------------------------


def test_estimator_is_world_frame_not_robot_relative():
    """A static surface scanned while the robot moves 1 cm must read zero."""

    state = LocalControlState()
    first = make_patch([(x, 0.30) for x in WALL_XS], 0.30, timestamp=0.0)
    second = make_patch([(x, 0.29) for x in WALL_XS], 0.29, timestamp=0.05)
    assert update_surface_estimate(state, first, (0.0, 0.20, 0.0), 0.0, 0.0)[0] is None
    speed, _signed, _rms, matches, reason = update_surface_estimate(
        state, second, (0.0, 0.21, 0.0), 0.05, 0.05)
    assert reason is None and matches == len(WALL_XS)
    assert speed == pytest.approx(0.0, abs=1e-9)


def test_estimator_sees_a_translating_surface():
    state = LocalControlState()
    first = make_patch([(x, 0.30) for x in WALL_XS], 0.30, timestamp=0.0)
    shifted = make_patch([(x, 0.304) for x in WALL_XS], 0.304, timestamp=0.05)
    update_surface_estimate(state, first, (0.0, 0.20, 0.0), 0.0, 0.0)
    speed, _s, _r, _m, _why = update_surface_estimate(
        state, shifted, (0.0, 0.20, 0.0), 0.05, 0.05)
    assert speed == pytest.approx(0.004 / 0.05, rel=1e-6)


# ---------------------------------------------------------------------------
# statistical power of the windowed estimator
# ---------------------------------------------------------------------------


def _noisy_static_sequence(rng, count, sigma, baseline_step=0.05):
    """A static surface observed ``count`` times with per-ray range noise."""

    patches = []
    for index in range(count):
        timestamp = index * baseline_step
        points = [(x, 0.30 + rng.uniform(-sigma, sigma)) for x in WALL_XS]
        patches.append(make_patch(points, 0.30, timestamp=timestamp))
    return patches


def test_windowed_estimator_lowers_the_noise_floor_on_a_noisy_static_surface():
    """2 mm per-ray noise: the 0.05 s baseline straddles the threshold, the 0.30 s one does not.

    This is the measured reason the ``noise`` pressure-test variant never settled: the deployed
    estimator divides two-scan noise by 0.05 s, which is ~0.057 m/s, while the deployed threshold
    is 0.02 m/s. No threshold change could have fixed it; the baseline had to widen.
    """

    import random
    import statistics
    patches = _noisy_static_sequence(random.Random(20261013), 12, sigma=0.002)
    instant_state, window_state = LocalControlState(), LocalControlState()
    instant_readings, window_readings = [], []
    for index, patch in enumerate(patches):
        now = index * 0.05
        value = update_surface_estimate(instant_state, patch, (0.0, 0.20, 0.0), now, now)[0]
        if value is not None:
            instant_readings.append(value)
        value = _windowed_value(window_state, patch, now, now)
        if value is not None:
            window_readings.append(value)
    assert instant_readings and window_readings
    # the 0.05 s baseline reads ~0.025 m/s on a surface that is exactly static, i.e. it straddles
    # the deployed 0.02 m/s threshold; the 0.30 s window stays clear of it
    assert statistics.median(instant_readings) > 0.02, "the short baseline is noise-dominated"
    assert max(window_readings) < 0.02, "the 0.30 s baseline should resolve a static surface"
    assert statistics.median(window_readings) < 0.5 * statistics.median(instant_readings)


def test_windowed_estimator_tolerates_repeated_timestamps():
    """The 100 ms delay pressure test delivers the same frame twice; the verdict must survive."""

    state = LocalControlState()
    patches = _noisy_static_sequence(__import__("random").Random(7), 8, sigma=0.0)
    readings = []
    for index, patch in enumerate(patches):
        now = index * 0.05
        readings.append(_windowed_value(state, patch, now, now))
        # the same frame is delivered again on the next control period
        readings.append(_windowed_value(state, patch, now, now + 0.05))
    available = [value for value in readings if value is not None]
    assert available, "a repeated timestamp must not wipe the window"
    assert max(available) < 0.02


def _windowed_value(state, patch, sample_time, now):
    return update_windowed_surface_estimate(
        state, patch, (0.0, 0.20, 0.0), sample_time, now,
        window=WINDOW, min_baseline=MIN_BASELINE)[0]


def test_missing_observation_is_never_evidence_of_stillness():
    state = LocalControlState()
    assert _windowed_value(state, None, None, 0.0) is None
    first = make_patch([(x, 0.30) for x in WALL_XS], 0.30, timestamp=0.0)
    assert _windowed_value(state, first, 0.0, 0.0) is None       # no baseline yet
    stale = make_patch([(x, 0.29) for x in WALL_XS], 0.29, timestamp=0.5)
    assert _windowed_value(state, stale, 0.5, 1.0) is None       # stale frame


# ---------------------------------------------------------------------------
# optional active brake (off by default in this round)
# ---------------------------------------------------------------------------


def test_active_brake_is_inert_at_zero_and_bounded_otherwise():
    patch = make_patch([(x, 0.30) for x in WALL_XS], 0.30)
    obs = observation(0.0, 0.20, 0.0, patch)
    state_plain = LocalControlState()
    state_zero = LocalControlState()
    plain = brake_force(obs, state_plain, 0.05)
    zero = brake_force_with_hold(obs, state_zero, 0.05, 0.0)
    assert zero == pytest.approx(plain)

    state = LocalControlState()
    state.reference = None
    held = brake_force_with_hold(obs, state, 0.05, 100.0)
    assert math.hypot(*held) <= state.force_limit + 1e-9
    # the preload points into the surface, i.e. gets a positive normal component
    assert held[1] > 0.0


def test_active_brake_without_a_normal_is_the_plain_brake():
    obs = observation(0.0, 0.20, 0.0, None)
    state_a, state_b = LocalControlState(), LocalControlState()
    assert brake_force_with_hold(obs, state_a, 0.05, 5.0) == pytest.approx(
        brake_force(obs, state_b, 0.05))


# ---------------------------------------------------------------------------
# B analysis guarantees
# ---------------------------------------------------------------------------


def _row(x, y, angle, segment, normal=(0.0, 1.0)):
    return {"observed": {"0": {"own": [x, y, angle, 0.0, 0.0, 0.0],
                               "segment_world": [list(segment[0]), list(segment[1])],
                               "patch_normal": [normal[0], normal[1]]}}}


def test_b_target_has_no_global_pose_component():
    """Rigidly moving the whole world must not change the observable terminal target.

    The global rigid transform moves the robot, the observed surface *and* its normal together,
    which is exactly what the world frame does to a local measurement. The B target is built from
    the robot's own pose, so it must come back identical.
    """

    segment = ((0.30, 0.50), (0.30, 0.70))
    normal = (0.0, 1.0)
    base = local_geometry(_row(0.0, 0.20, 0.3, segment, normal), 0)
    for angle in (0.0, 0.7, -2.1):
        cos, sin = math.cos(angle), math.sin(angle)
        for dx, dy in ((0.0, 0.0), (1.3, -0.4)):

            def transform(point):
                return (dx + point[0] * cos - point[1] * sin,
                        dy + point[0] * sin + point[1] * cos)

            def rotate(direction):
                return (direction[0] * cos - direction[1] * sin,
                        direction[0] * sin + direction[1] * cos)

            origin = transform((0.0, 0.20))
            moved = _row(origin[0], origin[1], 0.3 + angle,
                         (transform(segment[0]), transform(segment[1])),
                         rotate(normal))
            other = local_geometry(moved, 0)
            assert other[0] == pytest.approx(base[0], abs=1e-9)
            assert other[1] == pytest.approx(base[1], abs=1e-9)
            # the normal angle is measured against the robot's own start heading, so it is
            # invariant too; only the raw world-frame normal rotates
            start_heading = moved["observed"]["0"]["own"][2]
            assert (math.atan2(other[2][1], other[2][0]) - start_heading) == pytest.approx(
                math.atan2(base[2][1], base[2][0]) - 0.3, abs=1e-9)


def test_leave_one_root_out_never_splits_a_root():
    groups = ["rootA"] * 8 + ["rootB"] * 8 + ["rootC"] * 8
    seen = 0
    for root, train, test in leave_one_root_out_folds(groups):
        seen += 1
        for index, group in enumerate(groups):
            if group == root:
                assert test[index] and not train[index]
            else:
                assert train[index] and not test[index]
    assert seen == 3


def test_round2_analysis_scripts_are_run_read_only(tmp_path):
    """The A1/B scripts must not need a controller rerun to produce their numbers."""

    for name in ("audit_a1.py", "analyse_a2_counterfactual.py", "analyse_b.py"):
        text = (ROOT / "experiments" / "local_position_push_v2" / name).read_text()
        assert "run_split" not in text
        assert "Continuous2DEnv" not in text
