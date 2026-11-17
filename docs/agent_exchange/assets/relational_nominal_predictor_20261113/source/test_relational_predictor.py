"""Nominal model boundary, branch work accounting, and actual search integration."""
from copy import deepcopy
import pytest
from experiments.planar_tasks.tasks import task_catalog
from experiments.planar_tasks.relational_search.environment import MultiObjectOptionEnv,MovableBox
from experiments.planar_tasks.relational_search.contract import History,digest
from experiments.planar_tasks.relational_search.estimator import estimate,fit_box
from experiments.planar_tasks.relational_search.predictor import NominalPredictor
from experiments.planar_tasks.relational_search.search import choose,Budget


def real(mass=3.):
    return MultiObjectOptionEnv(task_catalog()['planar_open_anchor'].instance,
                              [MovableBox('b',1.55,2.95,width=.6,height=.3,mass=mass)])


def history(env):
    h=History();h.append(env.public());return h


def push(ticks=10):return dict(target_id='entity-a',a0=2,a1=2,ticks=ticks)


def test_same_public_history_yields_same_hypothesis_despite_hidden_mass():
    a,b=real(),real(100.)
    assert a.public()==b.public()
    assert estimate(history(a))==estimate(history(b))
    assert all('mass' not in e for e in estimate(history(a))['entities'])
    with pytest.raises(TypeError):NominalPredictor(a)


def test_visible_geometry_hypothesis_and_explicit_missing_depth():
    box=fit_box([[1.,0.],[1.,.2],[1.,.4]],[[0.,0.]])
    assert box['x']>1. and box['height']==.25
    assert 'prior' in box['uncertainty']
    assert fit_box([[1.,0.]],[[0.,0.]]) is None
    h=history(real());h.events[0]['observation']['entities']=[]
    with pytest.raises(ValueError):estimate(h)


def test_stale_entity_not_reconstructed_from_hidden_truth():
    h=history(real());o=deepcopy(h.events[0]['observation'])
    o['tick']=10;o['entities']=[e for e in o['entities'] if e['id']!='b']
    for e in o['entities']:e['last_seen_tick']=10
    o['entity_legal_pairs'].pop('b')
    h.append(o,push())
    b=next(e for e in estimate(h)['entities'] if e['id']=='b')
    assert b['stale_ticks']==10 and b['velocity_assumption'].startswith('zero')


def test_model_runs_independently_and_replays_consumed_prefix():
    env=real();before=env.offline_audit();h=history(env)
    m=NominalPredictor(h,seconds=10.)
    first=m.predict(m.root,push())
    second=m.predict(first['handle'],push())
    assert m.cost_ticks(first['handle'],push())==20
    repeated=m.predict(first['handle'],push())
    assert second['observation']==repeated['observation']
    assert second['cost_seconds']==repeated['cost_seconds']
    assert [e['reserved_ticks'] for e in m.events]==[10,10,20]
    assert m.record()['usage']['used']==dict(ticks=40,calls=3)
    assert m.record()['models_created']==2
    assert env.offline_audit()==before
    assert h.sha256==history(env).sha256
    assert first['observation']['source']=='nominal_prediction'
    m.close();assert not m.live


def test_absolute_timestamps_after_real_prefix():
    env=real();h=history(env)
    a=push();env.step_command(a);h.append(env.public(),a)
    m=NominalPredictor(h,seconds=10.)
    r=m.predict(m.root,dict(target_id=None,a0=44,a1=44,ticks=10))
    assert r['observation']['tick']==20
    assert all(e['last_seen_tick']==20 for e in r['observation']['entities'])
    assert all(robot['scan']['tick']==20 for robot in r['observation']['robots'])
    assert r['observation']['decision']==2


def test_public_target_ownership_preserved_but_controller_cold_started():
    env=real();h=history(env)
    a=dict(target_id='b',a0=44,a1=44,ticks=10)
    for _ in range(2):
        env.step_command(a);h.append(env.public(),a)
    assert h.events[-1]['observation']['active_target']=='b'
    m=NominalPredictor(h,seconds=10.)
    result=m.predict(m.root,dict(target_id='b',a0=44,a1=10,ticks=10))
    assert result['observation']['active_target']=='b'
    assert result['observation']['tick']==30
    assert 'cold-start' in m.record()['reconstruction']


def test_budget_reservation_failure_and_hypothesis_mutation():
    m=NominalPredictor(history(real()),ticks=10,calls=1,seconds=10.)
    m.predict(m.root,push())
    with pytest.raises(RuntimeError):m.predict(m.root,push())
    assert m.record()['usage']['used']==dict(ticks=10,calls=1)
    other=NominalPredictor(history(real()),seconds=10.)
    other.hypothesis['goal']['radius']=.9
    with pytest.raises(ValueError):other.predict(other.root,push())


def test_invalid_nominal_action_preserves_conservative_work_record():
    m=NominalPredictor(history(real()),ticks=40,calls=2,seconds=10.)
    with pytest.raises(ValueError):m.predict(m.root,dict(target_id='unseen',a0=2,a1=2,ticks=10))
    assert m.record()['usage']['used']['ticks']==10
    assert m.events[-1]['status']=='error' and m.events[-1]['forward_ticks']==0


def test_search_uses_nominal_model_and_replay_costs():
    env=real();h=history(env);seal=env.offline_audit()
    m=NominalPredictor(h,ticks=200,calls=8,seconds=10.)
    result=choose(h,m.root,m,lambda h,a:float(a['a0']==2),lambda h:10.,
                  Budget(prediction_ticks=200,calls=4,seconds=5.,batch=8))
    assert result['status']=='predicted'
    assert result['action'] is not None
    assert result['prediction_ticks']<=200
    assert result['prediction_ticks']==m.record()['usage']['used']['ticks']
    assert env.offline_audit()==seal
    assert all(e['actual_ticks']==e['requested_ticks'] for e in result['trace'] if e['status']=='predicted')


def test_search_refuses_reconstruction_that_exceeds_budget():
    class Model:
        def cost_ticks(self,handle,action):return action['ticks']+1000
        def predict(self,*args):raise AssertionError('must not call an unaffordable branch')
    result=choose(history(real()),0,Model(),lambda h,a:0.,lambda h:0.,Budget(prediction_ticks=10,calls=1))
    assert result['calls']==0 and result['prediction_ticks']==0


def test_search_error_charge_cannot_be_made_negative():
    class Model:
        def predict(self,handle,a):
            o=deepcopy(history(real()).events[0]['observation']);o['tick']=a['ticks']
            for e in o['entities']:e['last_seen_tick']=a['ticks']
            return dict(handle=1,observation=o,ticks=a['ticks'],cost_seconds=0.,terminal=False,
                        success=False,charged_ticks=-100)
    result=choose(history(real()),0,Model(),lambda h,a:0.,lambda h:1.,Budget(prediction_ticks=40,calls=1))
    assert result['prediction_ticks']==40 and result['action'] is None
