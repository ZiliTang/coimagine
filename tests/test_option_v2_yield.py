"""V2's observable contracts, real slot switching, and event/probability wiring."""
from dataclasses import replace
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch

from coimagine.random_start.lidar import LidarScan, SurfacePatch
from coimagine.random_start.skills import RobotKinematics
from experiments.option_rl_v19.unknown_shape import (
    select_simple_point, select_simple_pair, make_unknown_skill,
)
from experiments.option_rl_v21 import spec as S
from experiments.option_rl_v21.event_env import EventEnv, legal_pair
from experiments.option_rl_v21.interface import Policy, metadata, validate_metadata
from experiments.option_rl_v21.system import OptionSystemEnv
from experiments.option_rl_v21.yield_skill import YieldSkill
from experiments.option_rl_v21.review_cases import review_scenes
from experiments.option_rl_v21.train import save_checkpoint, load_checkpoint
from experiments.option_rl_v45.train import RunningMeanStd


def observation(normal=(1., 0.), x=.18):
    obs = OptionSystemEnv(review_scenes()['v2_idle']).boundary_observations()[0]
    points = tuple((x, -.3+i*.1) for i in range(7))
    p = SurfacePatch(points_robot=points, points_world=points, distance=abs(x),
        centroid_distance=abs(x), nearest_distance=abs(x), normal_world=normal,
        tangent_world=(-normal[1], normal[0]), confidence=1., arc_rays=7, span_m=.6,
        rms_m=0., valid=True)
    return replace(obs, own=RobotKinematics(0., 0., 0., 0., 0., 0.), patch=p, patches=(p,))


def test_selector_and_mask_respect_normal_sign_and_preserve_opposed_intents():
    east = observation(); west = observation((-1., 0.), -.18)
    assert select_simple_point(west, S.ACTIONS[2]) is None
    assert select_simple_point(east, S.ACTIONS[2]) is not None
    assert select_simple_pair(east, S.ACTIONS[2], west, S.ACTIONS[2]) is None
    assert select_simple_pair(observation(x=.4), S.ACTIONS[2],
                              observation((-1., 0.), -.4), S.ACTIONS[18]) is not None
    assert S.mask(west)[2] == False and S.mask(west)[18] == True
    assert legal_pair(S.ACTIONS[2], S.ACTIONS[18])


def test_externally_allocated_opposite_face_cannot_bypass_direction_guard():
    obs = observation((-1., 0.))
    skill = make_unknown_skill(S.ACTIONS[2])
    skill.allocation_managed = True
    skill.allocated_face = obs.patch
    _, info = skill.step(obs, .05)
    assert not info['direction_valid'] and info['direction_unavailable']
    assert info['direction_alignment'] == pytest.approx(-1.)
    assert info['desired_velocity'] == pytest.approx((0., 0.))
    assert not skill.contact_ready and not skill.joint_active


def test_v2_mask_does_not_call_dimension_prior(monkeypatch):
    from experiments.option_rl_v13 import controller as old
    monkeypatch.setattr(old, 'action_mask', lambda *a, **k: pytest.fail('dimension prior called'))
    m = S.mask(observation())
    assert len(m) == 47 and m[S.YIELD] and m[S.STOP] and m[2]
    assert not any(m[i] for i in S.ACTION_ALIASES)


def test_yield_commits_while_robot_is_moving_without_brake_handshake():
    env = OptionSystemEnv(review_scenes()['v2_front_blocker'])
    slot = env.slots[1]
    slot.skill.reference.value = (.20, 0.)
    slot.select(S.ACTIONS[S.YIELD])
    assert slot.action.kind == 'yield' and slot.pending is None and slot.brake is None
    assert slot.skill.reference.value == (.20, 0.)
    assert 'yield_immediate_commit' in slot.events


def test_yield_uses_observations_not_goal_and_bounds_force_and_reference_acceleration():
    env = OptionSystemEnv(review_scenes()['v2_front_blocker'])
    obs = env.boundary_observations()[1]
    a, b = YieldSkill(S.ACTIONS[46]), YieldSkill(S.ACTIONS[46])
    fa, ia = a.step(obs, .05)
    fb, ib = b.step(replace(obs, goal_xy=(.2, .2)), .05)
    assert ia['yield_debug']['active']
    assert fa == pytest.approx(fb)
    assert ia['desired_velocity'] == pytest.approx(ib['desired_velocity'])
    assert math.hypot(*ia['desired_velocity']) <= .4+1e-12
    assert math.hypot(*fa) <= obs.force_limit+1e-12
    assert math.hypot(*ia['reference_velocity']) <= .5*.05+1e-12


def test_yield_in_empty_space_stays_stationary():
    obs = OptionSystemEnv(review_scenes()['v2_idle']).boundary_observations()[0]
    force, info = YieldSkill(S.ACTIONS[46]).step(obs, .05)
    assert info['stage'] == 'yield_idle'
    assert info['desired_velocity'] == [0., 0.] and force == (0., 0.)


def test_enclosed_yield_reports_blocked_instead_of_inventing_clearance():
    obs = OptionSystemEnv(review_scenes()['v2_idle']).boundary_observations()[0]
    scan = LidarScan(robot_id=0, tick=0, sample_time=0., origin=(2., 2.), heading=0.,
        ray_angles=tuple(i*math.pi/36 for i in range(72)), distances=(.19,)*72,
        valid=(True,)*72, category=(2,)*72)
    obs = replace(obs, own=RobotKinematics(2., 2., 0., 0., 0., 0.), scan=scan)
    _, info = YieldSkill(S.ACTIONS[46]).step(obs, .05)
    assert info['yield_debug']['blocked'] and info['stage'] == 'yield_blocked'


def test_push_yield_events_ignore_yield_travel_and_keep_progress_clock():
    world = EventEnv(review_scenes()['v2_front_blocker'])
    first = world.step(2, 46)
    assert first['reason'] == 'contact' and first['contact'][0]
    progress = world.step(2, 46)
    assert progress['reason'] == 'progress' and progress['progress'][0] >= .2
    assert progress['progress'][1] == 0.
    assert progress['periods']*.05 == pytest.approx(progress['elapsed_s'])


def test_unavailable_push_returns_early_and_idle_yield_gets_time_boundary():
    world = EventEnv(review_scenes()['v2_front_blocker'])
    event = world.step(2, 2)
    assert event['reason'] == 'direction_unavailable' and event['elapsed_s'] <= .20
    idle = EventEnv(review_scenes()['v2_idle']).step(46, 44)
    assert idle['reason'] == 'yield_review' and idle['elapsed_s'] == pytest.approx(1.)
    with pytest.raises(ValueError):
        world.step(2, 43)


def test_v2_policy_can_sample_yield_and_checkpoint_rejects_v1(tmp_path):
    torch.set_num_threads(1)
    world = EventEnv(review_scenes()['v2_front_blocker'])
    obs, m0, m1 = world.observe()
    from experiments.option_rl_v21.event_env import OBS_EXTRA
    assert len(obs) == len(world.env.observation_fields())+len(OBS_EXTRA) == 973
    model = Policy(len(obs), 32)
    assert int(model.legal_pairs.sum()) == 1168
    x = torch.as_tensor(obs)[None, :]
    for _ in range(12):
        actions, logp, value = model.act(x, (m0, m1))
        a, b = [int(v.item()) for v in actions]
        assert m0[a] and m1[b] and legal_pair(S.ACTIONS[a], S.ACTIONS[b])
        ds, _ = model.distributions(x, (m0, m1), a0=actions[0])
        assert sum(d.log_prob(v) for d, v in zip(ds, actions)).item() == pytest.approx(logp.item(), abs=1e-6)
    for side in (0, 1):
        masks = [np.zeros(47, bool), np.zeros(47, bool)]
        masks[side][46] = True; masks[1-side][2] = True
        with torch.no_grad():
            actions, logp, value = model.act(x, masks)
            ds, _ = model.distributions(x, masks, a0=actions[0])
            assert actions[side].item() == 46 and torch.isfinite(logp) and torch.isfinite(value)
            assert sum(d.log_prob(a) for d, a in zip(ds, actions)).item() == pytest.approx(logp.item())
    norm = RunningMeanStd(len(obs)); norm.update(obs[None, :])
    optim = dict(actor=torch.optim.Adam(model.actor_parameters()), critic=torch.optim.Adam(model.critic_parameters()))
    path = tmp_path/'v2.pt'; save_checkpoint(path, model, norm, optim, 0, 1)
    restored, _, _ = load_checkpoint(path)
    assert restored.obs_dim == 973
    from experiments.option_rl_v20.interface import metadata as v1_metadata
    with pytest.raises(ValueError, match='incompatible'):
        validate_metadata(v1_metadata())


def test_speed_only_changes_preserve_material_q_generation():
    env = OptionSystemEnv(review_scenes()['v2_dual_north'])
    env.decision_ticks = 1
    env.joint_step(10, 10)
    generation = env._q_generation
    skills = [s.skill for s in env.slots]
    env.joint_step(11, 10)
    assert env._q_generation == generation
    assert all(s.skill is old for s, old in zip(env.slots, skills))


def test_v2_session_budget_clock_is_control_time_and_does_not_leak_to_next_segment():
    from experiments.option_rl_v21.session import TestbedSession
    s = TestbedSession(review_scenes()['v2_idle'])
    s.add(46, 44, 20, time_budget_s=.30).add(44, 44, 2)
    r = s.run()
    assert r['periods'] == 6+8
    assert all(f['time_budget_s'] is None for f in r['frames'][-1]['progress'])
    assert r['frames'][0]['mb'][0]['yield_debug'] is not None


def test_testbed_has_tracked_scene_fallback_without_historical_outputs(tmp_path, monkeypatch):
    from experiments.option_rl_v19 import testbed_server as server
    monkeypatch.setattr(server, 'PANEL', tmp_path/'missing_panel.json')
    scenes = server._load_scenes()
    assert 'contact_0' in scenes and 'poly_octagon' in scenes
