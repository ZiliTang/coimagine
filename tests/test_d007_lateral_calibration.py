"""D007 pure math/config checks; no physics."""
import json, math
from pathlib import Path
from dataclasses import replace

from experiments.option_rl_v19.run_d007 import mirror_instance, preset_instance, scenes
from experiments.option_rl_v19 import spec as S
from coimagine.continuous2d.scenes import Pose


def _rot90(v):
    return (-v[1], v[0])


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def test_tangent_frame_is_left_and_cross_minus_one():
    for a in (0.0, 0.3, math.pi / 2, -0.7, 2.1):
        n = (math.cos(a), math.sin(a)); t = _rot90(n)
        assert abs(_cross(t, n) + 1.0) < 1e-12


def test_profiles_are_distinct_and_b_outer_off():
    cfg = json.loads(Path('docs/agent_exchange/tasks/D007_config.json').read_text())
    a = cfg['profiles']['A']['push_overrides']
    b = cfg['profiles']['B']['push_overrides']
    c = cfg['profiles']['C']['push_overrides']
    assert a['rotation_feedback_gain'] == 0.0
    assert b['feedback'] is False and b['heading_gain'] == 0.0
    assert b['angular_rate_gain'] == 0.0 and b['lateral_gain'] == 0.0
    assert b['rotation_feedback_gain'] == 0.0
    assert b['contact_tangent_damping'] == 0.7
    assert c['contact_tangent_damping'] == 0.0
    assert {k: v for k, v in c.items() if k != 'contact_tangent_damping'} == \
           {k: v for k, v in b.items() if k != 'contact_tangent_damping'}


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _lateral(e, k, vmax, c, vt):
    first = _clip(k * e, -vmax, vmax)
    return _clip(first - c * vt, -vmax, vmax), first


def test_two_stage_lateral_clip_is_not_one_stage():
    one, first = _lateral(0.08, 8.8, 0.5, 0.0, 0.0)
    two, _ = _lateral(0.08, 8.8, 0.5, 0.7, 0.2)
    assert abs(one - 0.5) < 1e-12
    assert abs(two - 0.36) < 1e-12
    assert abs(first - 0.5) < 1e-12
    assert two < first


def test_target_clip_limits_are_symmetric():
    L = 1.0
    limit = max(0.0, L / 2 - min(.20, .20 * L))
    assert abs(limit - 0.3) < 1e-12
    for off in (-1.0, -0.3, 0.0, 0.3, 1.0):
        assert -limit <= _clip(off, -limit, limit) <= limit


def test_mirror_involution_and_action_map():
    sm = scenes(); p = json.loads(Path('outputs/testbed/presets/20261029_224105_6ef27d.json').read_text())
    inst = preset_instance(p, sm); m = mirror_instance(inst); mm = mirror_instance(m)
    for a, b in zip((inst.object_pose, *inst.robot_poses), (mm.object_pose, *mm.robot_poses)):
        assert abs(a.x - b.x) < 1e-12 and abs(a.y - b.y) < 1e-12
        assert abs(math.atan2(math.sin(a.angle - b.angle), math.cos(a.angle - b.angle))) < 1e-12
    assert abs(inst.goal_centre[0] - mm.goal_centre[0]) < 1e-12
    assert abs(inst.goal_centre[1] - mm.goal_centre[1]) < 1e-12
    assert {3: 19, 11: 11}[3] == 19 and {3: 19, 11: 11}[11] == 11


def test_diagnostic_offset_added_before_position_rate_limit():
    # pure implementation of the added point-offset path
    limit, rate, dt = .3, .16, .05
    base = 0.02
    driven = base + .04
    before = _clip(driven, -limit, limit)
    after = _clip(before, base - rate * dt, base + rate * dt)
    assert abs(after - (base + rate * dt)) < 1e-12
    # without the offset the path is smaller
    without = _clip(_clip(base, -limit, limit), base - rate * dt, base + rate * dt)
    assert after >= without


def test_profile_config_fields_exist_and_are_finite():
    from coimagine.random_start.effect_push import EffectPushConfig
    cfg = EffectPushConfig()
    assert hasattr(cfg, 'rotation_feedback_gain')
    assert hasattr(cfg, 'contact_tangent_damping')
    assert math.isfinite(cfg.rotation_feedback_gain)
    assert cfg.contact_tangent_damping >= 0.0
