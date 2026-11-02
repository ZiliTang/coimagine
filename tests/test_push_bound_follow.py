"""PUSH-bound FOLLOW: public pair legality, causal handoff, failures and clocks."""
from dataclasses import replace
from functools import partial
import math
import pytest
from coimagine.random_start import env as E,spec
from coimagine.random_start.effect_push import (ACTIONS_PUSH_BOUND_FOLLOW as A,ACTIONS_REPOSITION_V2 as OLD,
    PUSH_BOUND_FOLLOW_VERSION,action_mask,joint_action_mask,action_mask_given_peer,type_scale_mask,make_effect_skill)
from experiments.option_testbench import push_bound_follow_eval as P
from experiments.option_testbench.follow_peer_eval import instance_from_dict


def main_case():return next(c for c in P.cases() if c['name']=='push_follow')


def test_single_follow_slot_and_unchanged_other_actions():
    assert len(A)==65 and len(OLD)==68
    assert A[:64]==OLD[:64]
    follow=[a for a in A if a.kind=='follow'];assert len(follow)==1
    assert follow[0].duration==0 and follow[0].scale_kind=='peer_push' and follow[0].scale_index==0
    assert PUSH_BOUND_FOLLOW_VERSION!='effect-options-v9-reposition-scales'


def test_conditional_masks_require_push_and_do_not_forbid_push_reposition():
    world=P.make_world(instance_from_dict(main_case()['instance']),'bound');obs=world.boundary_observations();cfg=P.CONFIGS['bound']
    f=64;p=next(a.index for a in A if a.label=='PUSH_N[0.30m]');r=next(a.index for a in A if a.kind=='reposition');w=next(a.index for a in A if a.kind=='wait')
    joint=joint_action_mask(obs,cfg,actions=A)
    assert joint[p,f] and joint[f,p] and joint[p,p]
    assert not joint[w,f] and not joint[f,w] and not joint[r,f] and not joint[f,f]
    assert joint[p,r] and joint[r,p]
    assert not action_mask_given_peer(obs[0],A[w],cfg,actions=A)[f]
    assert action_mask_given_peer(obs[0],A[p],cfg,actions=A)[f]
    conditioned=action_mask_given_peer(obs[0],A[f],cfg,actions=A)
    assert all(not v or a.kind=='push' for v,a in zip(conditioned,A))
    assert type_scale_mask(obs[0],cfg,actions=A)[13].tolist()==[True,False,False,False,False]
    assert not action_mask(obs[0],P.CONFIGS['timed'],actions=A)[f]
    assert not any(action_mask(obs[0],cfg,actions=OLD)[64:])


@pytest.mark.parametrize('other_kind',['wait','follow','reposition'])
def test_invalid_joint_follow_has_no_physical_side_effect(other_kind):
    world=P.make_world(instance_from_dict(main_case()['instance']),'bound');j=next(a.index for a in A if a.kind==other_kind)
    with pytest.raises(ValueError,match='requires a PUSH partner'):world.joint_step(64,j)
    assert world.total_periods==0 and not world.rows and not world.boundaries


def test_codebook_cannot_silently_ignore_legacy_follow_duration():
    with pytest.raises(ValueError,match='codebook'):make_effect_skill(OLD[65],config=P.CONFIGS['bound'])
    with pytest.raises(ValueError,match='codebook'):make_effect_skill(A[64],config=P.CONFIGS['timed'])
    skill=make_effect_skill(A[64],config=P.CONFIGS['bound']);obs=P.make_world(instance_from_dict(main_case()['instance']),'bound').boundary_observations()[1]
    with pytest.raises(RuntimeError,match='bound before'):skill.step(obs,.05)


@pytest.fixture(scope='module')
def records():
    panel={c['name']:c for c in P.cases()}
    return {name:P.run(panel[name],'bound') for name in ('push_follow','push_follow_swapped','former_follow_2s','former_follow_24s','push_unavailable','follower_unobserved')}


@pytest.mark.parametrize('name',['push_follow','push_follow_swapped'])
def test_follow_remains_until_push_done_and_causal_handoff_is_order_independent(records,name):
    r=records[name];m=r['metric'];bound=r['boundary']['execution']['bound_follow'];p,f=bound['pusher'],bound['follower']
    assert m['exits'][p]=='normal_travel_complete' and m['exits'][f]=='follow_push_complete'
    assert not m['follower_held_before_push_end'] and 0<=m['follow_minus_push_seconds']<=.85+1e-9
    assert not r['boundary']['skill_ends'][f]['residual_motion']
    stop=next(e for e in r['boundary']['execution']['events'] if e['event']=='follow_stop')
    assert stop['time']>=m['push_end_time']+.05-1e-9
    for row in r['rows']:
        info=row[f'r{f}'];binding=info.get('follow_binding')
        if binding and binding['stop_reason']:assert row['time']>=stop['time']+.05-1e-9
        if row['time']<=m['push_end_time']+.05-1e-9:assert info['kind']=='follow'
    assert m['bound_tracking_seconds']>0 and m['reward_error']<1e-8
    assert all(math.hypot(*force)<=spec.MAX_FORCE+1e-8 for row in r['rows'] for force in row['forces'])
    previous=(0.,0.)
    for row in r['rows']:
        info=row[f'r{f}']
        if info['kind']=='follow':
            reference=info['reference_velocity']
            assert math.dist(previous,reference)<=spec.A_REF_MAX*spec.CONTROL_DT+1e-8
            previous=reference


def test_former_two_and_twentyfour_second_follow_requests_map_to_identical_behavior(records):
    a,b=records['former_follow_2s'],records['former_follow_24s']
    assert a['case']['actions']==b['case']['actions']
    assert [row['true'] for row in a['rows']]==[row['true'] for row in b['rows']]
    assert [row['forces'] for row in a['rows']]==[row['forces'] for row in b['rows']]
    assert a['metric']['time_budgets']==b['metric']['time_budgets']


def test_push_failure_releases_follower_as_failure_not_success(records):
    r=records['push_unavailable'];m=r['metric']
    assert m['exits'][0]!='normal_travel_complete' and m['exits'][1]=='peer_push_failed'
    assert any(e['event']=='follow_stop' and e['reason']=='peer_push_failed' for e in r['boundary']['execution']['events'])
    assert m['seconds']<1.


def test_follower_failure_cancels_push_on_next_reported_boundary(records):
    r=records['follower_unobserved'];m=r['metric']
    assert m['exits'][1]=='patch_lost' and m['exits'][0]=='peer_follow_failed'
    event=next(e for e in r['boundary']['execution']['events'] if e['event']=='follow_failed')
    ends=r['boundary']['skill_ends']
    assert event['time']>=ends[1]['time']+.05-1e-9
    assert ends[0]['time']==pytest.approx(event['time'])


def test_task_timeout_is_not_extended_by_binding_or_braking():
    c=P.translated(main_case(),'bound');world=E.RandomStartEnv(instance_from_dict(c['instance']),E.EnvConfig(time_budget=.5),
        skill_factory=partial(make_effect_skill,config=P.CONFIGS['bound']),action_table=A,action_version=PUSH_BOUND_FOLLOW_VERSION)
    b=world.joint_step(*c['actions'],record=True)
    assert b['termination']=='task_budget_exhausted' and b['periods']==10
    assert b['reward']['failure_penalty']==spec.FAILURE_PENALTY


def test_follower_window_does_not_inflate_original_push_timeout():
    c=main_case();bound=P.translated(c,'bound');w=P.make_world(instance_from_dict(c['instance']),'bound')
    e=w.joint_step(*bound['actions']);budgets=e['execution']['requested_time_budgets']
    assert budgets[0]==14. and budgets[1]==pytest.approx(14.85)
    assert e['execution']['effective_window_seconds']==pytest.approx(14.85)
