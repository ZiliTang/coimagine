"""Efficient options: meaningful progress and exact observation-free stepping."""
from dataclasses import asdict,replace
from functools import partial
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
from coimagine.random_start import env as E,efficient_profile as EP
from coimagine.random_start.progress_guard import ProgressGuard
from coimagine.random_start.effect_push import make_effect_skill,ACTIONS_PUSH_BOUND_FOLLOW,joint_action_mask
from experiments.option_testbench import option_efficiency_v2_eval as R
from experiments.option_testbench.follow_peer_eval import instance_from_dict


def case(name='push_follow'):return next(c for c in R.cases() if c['name']==name)


def test_progress_guard_does_not_count_oscillation_or_wrong_direction_as_improvement():
    g=ProgressGuard();assert not g.update('rotate',0.,.05,.01)
    for i in range(59):assert not g.update('rotate',.004 if i%2 else -.004,.05,.01)
    assert g.update('rotate',-.005,.05,.01)
    assert g.info['marker']==0.


def test_progress_guard_allows_cumulative_meaningful_forward_progress():
    g=ProgressGuard();stalled=False
    for i in range(200):stalled|=g.update('move',i*.0004,.05,.01)
    assert not stalled
    assert not g.update('align',-.2,.05,.005)  # phases do not inherit unrelated plateaus.
    assert g.elapsed==0


def test_record_false_skips_unused_frame_and_history_feature_materialization(monkeypatch):
    c=R.translate(case(),'efficient');w=EP.make_env(instance_from_dict(c['instance']))
    def forbidden(*a,**k):raise AssertionError('unused recording work in rollout fast path')
    monkeypatch.setattr(w,'_row',forbidden);monkeypatch.setattr(w,'observation_vector',forbidden)
    b=w.joint_step(*c['actions'],record=False)
    assert b['skill_ends'][0]['exit_reason']=='normal_travel_complete'
    assert not w.rows and not w.history


def test_record_true_false_preserve_physics_reward_boundaries_and_future_inputs():
    c=R.translate(case(),'efficient');inst=instance_from_dict(c['instance']);a,b=EP.make_env(inst),EP.make_env(inst)
    for _ in range(2):
        ra=a.joint_step(*c['actions'],record=True);rb=b.joint_step(*c['actions'],record=False)
        assert ra==rb
        assert a.env.backend.prefix==b.env.backend.prefix
        assert a.heading==b.heading
        np.testing.assert_array_equal(a.boundary_vector(),b.boundary_vector())


def test_minimal_native_step_returns_identical_essential_safety_data_and_states():
    inst=instance_from_dict(case()['instance']);a,b=EP.make_env(inst).env,EP.make_env(inst).env
    for _ in range(12):
        oa,da=a.step_control(((1.,2.),(-.3,1.)))
        ob,db=b.step_control(((1.,2.),(-.3,1.)),minimal=True)
        assert oa is not None and ob is None
        for k in ('penetration','finite','in_bounds','success_hold','already_terminated'):assert da[k]==db[k]
        assert a.backend.object_state()==b.backend.object_state()
        assert a.backend.robot_states()==b.backend.robot_states()


def test_fast_path_retains_numeric_anomaly_check(monkeypatch):
    c=R.translate(case(),'efficient');w=EP.make_env(instance_from_dict(c['instance']))
    monkeypatch.setattr(w.env.backend,'finite',lambda:False)
    b=w.joint_step(*c['actions'],record=False)
    assert b['anomaly']=='numeric_failure' and b['periods']==1


def test_new_menu_removes_only_long_wait_and_keeps_all_physical_action_parameters():
    a=EP.ACTIONS;assert len(a)==62 and a[:60]==ACTIONS_PUSH_BOUND_FOLLOW[:60]
    assert [x.label for x in a[-2:]]==['WAIT_BRAKE[2s]','SUPPORT_FOLLOW_PUSH']
    assert all(x.index==i for i,x in enumerate(a))
    w=EP.make_env(instance_from_dict(case()['instance']));mask=EP.mask(w)
    p=next(x.index for x in a if x.label=='PUSH_N[0.30m]');rp=next(x.index for x in a if x.kind=='reposition')
    assert mask[p,61] and mask[61,p] and mask[p,rp] and not mask[60,61]


def test_short_wait_holds_identically_to_long_wait_until_active_peer_finishes():
    c=case('push_wait24');inst=instance_from_dict(c['instance'])
    long=E.RandomStartEnv(inst,E.EnvConfig(lidar_backend='numpy_batch',peer_clearance_recovery='boundary_v1'),
        skill_factory=EP.skill_factory,action_table=ACTIONS_PUSH_BOUND_FOLLOW)
    short=EP.make_env(inst);ca=R.translate(c,'efficient')
    long.joint_step(*c['actions'],record=True);short.joint_step(*ca['actions'],record=True)
    assert short.total_periods<long.total_periods
    assert [r['true'] for r in short.rows]==[r['true'] for r in long.rows[:len(short.rows)]]
    assert [r['forces'] for r in short.rows]==[r['forces'] for r in long.rows[:len(short.rows)]]


def test_reposition_guard_handles_find_to_align_without_stale_none_gap():
    c=R.translate(case('reposition_0.8_wait'),'efficient');w=EP.make_env(instance_from_dict(c['instance']))
    b=w.joint_step(*c['actions'],record=True)
    assert b['skill_ends'][0]['exit_reason']=='reposition_complete'


def test_preparation_close_conditions_and_actuator_limits_are_not_relaxed():
    from coimagine.random_start import spec
    assert EP.CONFIG.speed==.12 and EP.CONFIG.solo_approach_gap==spec.COOP_READY_GAP
    assert EP.CONFIG.contact_tangent_damping==.5
    assert EP.ROTATION_CONFIG.preparation_policy=='legacy'
    assert EP.ROTATION_CONFIG.contact_tangent_damping==0
    assert EP.CONFIG.follow_completion=='peer_push'


def test_disabled_optional_fields_preserve_archived_rl_controller_descriptor():
    from experiments.option_rl_v10.interface import metadata,validate_metadata
    data=metadata();validate_metadata(data)
    for name in ('preparation_policy','solo_approach_gap','progress_guard','contact_tangent_damping'):
        assert name not in data['controller_config']
