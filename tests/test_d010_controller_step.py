"""D010 M3 controller-step fixtures; no env/Pymunk/scan_lidar."""
import math
from dataclasses import replace
from coimagine.random_start.lidar import LidarScan, SurfacePatch
from coimagine.random_start.skills import RobotKinematics, SkillObservation
from experiments.option_rl_v13 import controller as C
from experiments.option_rl_v19.unknown_shape import ContactCandidate, UnknownShapePush


def _patch(x=1.0, y0=-.5, y1=.5):
    pts = tuple((x, y0 + (y1-y0)*i/6) for i in range(7))
    return SurfacePatch(
        points_robot=tuple((p[0]-0.0, p[1]-0.0) for p in pts),
        points_world=pts,
        distance=x, centroid_distance=x, nearest_distance=x,
        normal_world=(-1.0, 0.0) if x < 0 else (1.0, 0.0),
        tangent_world=(0.0, 1.0) if x >= 0 else (0.0, -1.0),
        confidence=1.0, arc_rays=7, span_m=y1-y0, rms_m=.001, valid=True)


def _obs(patch, contact=False, vx=0., vy=0.):
    own = RobotKinematics(0., 0., 0., vx, vy, 0.)
    mate = RobotKinematics(3., 3., 0., 0., 0., 0.)
    scan = LidarScan(0, 0, 0., (0., 0.), 0., (), (), (), ())
    return SkillObservation(
        robot_id=0, tick=0, time=0., own=own, teammate=mate, scan=scan,
        patches=(patch,), patch=patch, contacting_object=contact,
        contact_point_robot=(.18, 0.) if contact else None,
        contacting_static=False, contacting_teammate=False,
        goal_xy=(2., 2.), goal_radius=.2, arena=(5., 4.),
        force_limit=40., remaining_seconds=10., elapsed_seconds=0.,
        static_nearest=None, teammate_nearest=None)


def _skill():
    action = C.ACTIONS[2]  # PUSH_E, direction +x
    return UnknownShapePush(action)


def test_missing_candidate_cannot_shift():
    s = _skill()
    s.contact_phase = 'shift'; s.shift_sign = 1.0
    s.retreat_until = -1.; s.shift_until = 99.; s.candidate = None
    f, info = s.step(_obs(_patch()), .05)
    assert s.contact_phase in ('wait', 'approach')
    assert s.shift_sign == 0.0
    assert math.hypot(*info['desired_velocity']) < 1e-9


def test_contact_time_counted_once_per_tick():
    s = _skill()
    s.step(_obs(_patch(), contact=True), .05)
    a = s.effective_contact
    s.step(_obs(_patch(), contact=True), .05)
    b = s.effective_contact
    assert abs(a - .05) < 1e-12
    assert abs((b-a) - .05) < 1e-12


def test_surface_moved_two_cm_revalidates_candidate():
    s = _skill()
    s.step(_obs(_patch(1.0)), .05)
    c0 = s.candidate
    assert c0 is not None
    s.step(_obs(_patch(1.03)), .05)
    assert s.candidate is not None
    assert s.candidate.id != c0.id
    assert s.candidate.reason == 'support_invalid'


def test_small_support_change_keeps_candidate_identity():
    s = _skill()
    s.step(_obs(_patch(1.0)), .05)
    c0 = s.candidate; v0 = c0.verified_time
    s.step(_obs(_patch(1.01)), .05)
    assert s.candidate is not None
    assert s.candidate.id == c0.id
    assert s.candidate.verified_time > v0


def test_align_position_ok_but_speed_high_does_not_touch():
    s = _skill()
    f, info = s.step(_obs(_patch(.2), vy=.10), .05)
    assert s.contact_phase == 'align'
    assert s.stage == 'align'
    assert s.align_settle == 0.0


def test_align_speed_ok_but_position_high_does_not_touch():
    s = _skill()
    f, info = s.step(_obs(_patch(.2, .0, 1.0)), .05)
    assert s.contact_phase == 'align'
    assert s.align_settle == 0.0


def test_align_continuous_conditions_reach_touch_after_settle():
    s = _skill()
    rows = [_obs(_patch(.2), vy=0.) for _ in range(3)]
    for o in rows:
        s.step(o, .05)
    assert s.contact_phase == 'touch'
    assert s.align_settle >= .15 - 1e-12


def test_align_interruption_resets_settle():
    s = _skill()
    s.step(_obs(_patch(.2)), .05)
    s.step(_obs(_patch(.2)), .05)
    assert s.align_settle > 0
    s.step(_obs(_patch(.2, .2, 1.2)), .05)
    assert s.align_settle == 0.0
    assert s.contact_phase == 'align'


def test_align_timeout_waits_instead_of_forcing_touch():
    s = _skill()
    for _ in range(14):
        s.step(_obs(_patch(.2, .0, 1.0)), .05)
    assert s.contact_phase == 'wait'
    assert s.align_settle == 0.0
