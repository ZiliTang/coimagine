"""Lifecycle regressions; injected local failures test executor semantics only."""
import math
from dataclasses import replace
import numpy as np
import pytest
import torch
from experiments.option_rl_v13 import controller as C, interface as I
from experiments.option_rl_v13.mode_env import ContinuousModeEnv, Slot
from experiments.option_testbench.effect_push_eval import make_case, instance_from_dict, reposition_peer_cases

def action(kind='push', typ=2, speed=.16, direction=None):
    return next(a.index for a in C.ACTIONS if a.kind==kind and
                (kind!='push' or a.type_index==typ) and
                (kind not in ('push','reposition') or a.speed==speed) and
                (direction is None or a.direction==direction))

def world(coop=False):
    return ContinuousModeEnv(instance_from_dict(make_case('lifecycle', cooperative=coop, offset=0. if coop else .18)['instance']))

def fail(w, i, reason='blocked_no_progress'):
    w.slots[i].skill._finish(reason, w.boundary_observations()[i])

def test_repeated_failed_intent_and_speed_change_remain_latched():
    w=world();p=action();w.joint_step(p,C.STOP);s=w.slots[0].skill;fail(w,0)
    for cmd in (p, action(speed=.32), p):
        w.joint_step(cmd,C.STOP)
        assert w.slots[0].skill is s and s.done and s.exit_reason=='blocked_no_progress'
    assert w.slots[0].pending is None

def test_changed_failed_intent_requests_fresh_checks_after_braking():
    w=world();cw=action('reposition',speed=.12,direction='CW');ccw=action('reposition',speed=.12,direction='CCW')
    w.joint_step(cw,C.STOP);old=w.slots[0].skill;fail(w,0,'blocked_static_route')
    w.slots[0].select(C.ACTIONS[ccw])
    assert w.slots[0].skill is old and old.done and w.slots[0].pending is not None
    assert 'changed_intent_retry_requested' in w.slots[0].events
    for _ in range(15):w.joint_step(ccw,C.STOP)
    assert w.slots[0].skill is not old and w.slots[0].action.direction=='CCW'

def test_peer_failure_pauses_without_poisoning_healthy_controller_and_releases():
    w=world(True);p=action();w.joint_step(p,p);healthy=w.slots[0].skill;fail(w,1,'effect_face_lost')
    no_progress=healthy.no_progress_seconds
    for _ in range(6):
        b=w.joint_step(p,p)
        assert not healthy.done and b['skill_ends'][0]['stage']=='peer_wait'
    assert healthy.no_progress_seconds==no_progress
    assert w.slots[1].skill.done  # do not magically recover the failed side
    rp=action('reposition',speed=.12,direction='CCW')
    for _ in range(8):w.joint_step(p,rp)
    assert w.slots[0].skill is healthy and not healthy.done and w.slots[0].pause_reason is None
    assert healthy.pending_exit_reason is None

def test_follow_does_not_inherit_failure_from_push_peers_previous_mode():
    w=world(True);rp=action('reposition',speed=.12,direction='CW');p=action()
    w.joint_step(rp,C.STOP);fail(w,0,'blocked_static_route')
    w.slots[0].select(C.ACTIONS[p]);w.slots[0].brake.reference.value=(.2,0.)
    b=w.joint_step(p,C.FOLLOW)
    assert w.slots[0].pending is not None
    follow=w.slots[1].skill
    assert not follow.done and follow.bound_push_stop is None
    assert b['skill_ends'][1]['stage']=='peer_wait'
    for _ in range(10):w.joint_step(p,C.FOLLOW)
    assert follow.bound_push_stop is None and not follow.done

def test_new_pair_cannot_reuse_previous_release_and_recovery_flags():
    w=world(True);p=action();w.joint_step(p,p);a=w.slots[0].skill
    a.release=a.ready=a.contact_ready=a.joint_active=True;a.stage='recover';a.recovery_resume_stage='push'
    a.recovery_requested=True;w.recovery_started=0.;w.recovery_attempts=3
    w.slots[1].skill=C.make_skill(C.ACTIONS[p])
    w._coordinate(.2)
    assert a.stage=='approach' and not a.release and not a.contact_ready and not a.joint_active
    assert not a.recovery_requested and w.recovery_started is None and w.recovery_attempts==0

def test_new_switch_uses_live_brake_reference_instead_of_failed_skills_stale_reference():
    w=world();w.joint_step(action(),C.STOP);slot=w.slots[0];fail(w,0)
    slot.skill.reference.value=(.2,0.)
    for _ in range(10):slot.step(w.boundary_observations()[0])
    assert math.hypot(*slot.brake.reference.value)<1e-9
    slot.select(C.ACTIONS[action('reposition',speed=.12,direction='CW')])
    assert slot.brake.reference.value==(0.,0.)

def test_peer_wait_keeps_own_scan_safety_and_push_switch_waits_for_peer():
    w=world(True);p=action();w.joint_step(p,p)
    w.slots[1].select(C.ACTIONS[action(typ=8,speed=.04)])
    w._coordinate(.2)
    assert w.slots[0].pause_reason=='peer_push_switching'
    obs=w.boundary_observations()[0]
    danger=replace(obs,static_nearest=(C.spec.ROBOT_RADIUS+.001,(obs.own.x,obs.own.y+.181)))
    _,info=w.slots[0].step(danger)
    assert w.slots[0].skill.exit_reason=='safety_abort' and info['controller_failed']

def test_one_missing_scan_brakes_and_reacquires_but_prolonged_loss_latches():
    w=world();w.joint_step(action(),C.STOP);slot=w.slots[0];skill=slot.skill
    good=w.boundary_observations()[0];blind=replace(good,patch=None,patches=())
    _,info=slot.step(blind)
    assert not skill.done and info['stage']=='perception_wait' and info['desired_velocity']==(0.,0.)
    slot.pause_reason=None;slot.step(good)
    assert slot.skill is skill and not skill.done and slot.sensor_missing_seconds==0.
    for _ in range(5):slot.step(blind)
    assert skill.done and skill.exit_reason=='effect_face_lost'

def test_retry_does_not_bypass_static_corridor_guard():
    c=next(c for c in reposition_peer_cases() if c['name']=='detour_rp_CCW_static_gap_0.26')
    w=ContinuousModeEnv(instance_from_dict(c['instance']));rp=action('reposition',speed=.18,direction='CCW')
    for _ in range(100):
        w.joint_step(rp,C.STOP,record=True)
        if w.slots[0].skill.done:break
    assert w.slots[0].skill.exit_reason=='blocked_static_route'
    old=w.slots[0].skill
    for _ in range(5):w.joint_step(rp,C.STOP,record=True)
    assert w.slots[0].skill is old
    assert max(r['penetration'] for r in w.rows)<.002

def test_failed_controller_has_no_persistence_bonus_and_ppo_probability_matches():
    torch.set_num_threads(1);torch.manual_seed(1)
    p=I.Policy(822,32);old=I.Policy(822,32,failure_aware_persistence=False);old.load_state_dict(p.state_dict())
    x=torch.zeros(2,822);x[:,762]=1;x[1,789]=1
    m=[np.ones(46,bool)]*2
    a,lp,_=p.act(x,m,generator=torch.Generator().manual_seed(2))
    d,_=p.distributions(x,m,a0=a[0]);torch.testing.assert_close(lp,d[0].log_prob(a[0])+d[1].log_prob(a[1]))
    normal,_=p.distributions(x,m);prior,_=old.distributions(x,m)
    torch.testing.assert_close(normal[0].probs[0],prior[0].probs[0])
    assert float(normal[0].probs[1,:32].sum().detach())<float(prior[0].probs[1,:32].sum().detach())
    (-lp.mean()-.02*sum(z.entropy().mean() for z in d)).backward()
    assert all(torch.isfinite(q.grad).all() for q in p.parameters() if q.grad is not None)
    with pytest.raises(ValueError):
        from experiments.option_rl_v12.interface import metadata
        I.validate_metadata(metadata())

def test_metrics_count_waiting_separately_from_own_failures():
    from experiments.option_rl_v13.train import lifecycle_rates
    samples=[dict(actions=[dict(exits=['blocked_static_route','continuing'],
        phases=['blocked','peer_wait'],events=[]),dict(exits=['continuing','continuing'],
        phases=['perception_wait','push'],events=['changed_intent_retry_requested'])])]
    r=lifecycle_rates(samples)
    assert r['controller_failed_rate']==.25 and r['peer_wait_rate']==.25
    assert r['perception_wait_rate']==.25 and r['failed_or_wait_rate']==.75
    assert r['changed_intent_retry_rate']==.25
