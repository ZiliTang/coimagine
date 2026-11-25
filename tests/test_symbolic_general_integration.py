"""Public-only model inputs, actual physics metering, and canonical CAP binding."""
from copy import deepcopy
import pytest

from experiments.planar_tasks.authoring_v2.symbolic_general import scaffold, frontend, simulation
from experiments.planar_tasks.authoring_v2.symbolic_general.prediction import EffectModel
from experiments.planar_tasks.authoring_v2.symbolic_general.online import OnlineRollout
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture


def test_illegal_or_permission_bypassing_authored_rule_is_rejected():
    p=scaffold.starter();p['rules'][0]['when']=['lit',True,'bool']
    with pytest.raises(ValueError,match='epistemic permission'): scaffold.source(p)


def test_carrier_cannot_diverge_from_literal_strategy():
    text=scaffold.source(scaffold.starter())
    assert scaffold.verify_source(text).program==scaffold.starter()
    changed=text.replace('name=max(candidates)[2]', 'name="brake"')
    with pytest.raises(ValueError,match='carrier changed'):scaffold.verify_source(changed)


def test_semantics_ignore_extra_privileged_annotations_and_reset_permission():
    fixture=seed_fixture();w,m,_=simulation.replay_boundary(fixture,[])
    o=simulation.observation(w);z,cmd=frontend.semantic_packet(m,o)
    altered=deepcopy(o);altered['hidden_friction']=900;altered['true_object_pose']=[0,0,0]
    assert frontend.semantic_packet(m,altered)==(z,cmd)
    m['evidence']={'brake':dict(permit=False)}
    n=deepcopy(o);n['tick']+=1
    assert frontend.update_memory(m,dict(observation=n))['evidence']=={}


def test_fitting_room_ids_do_not_become_inputs_and_calibration_is_disjoint():
    w,m,_=simulation.replay_boundary(seed_fixture(),[])
    z,cmd=frontend.semantic_packet(m,simulation.observation(w))
    row=dict(room='a',action='brake',semantic=z,next_semantic=z,audit={'object_static':False})
    with pytest.raises(ValueError,match='overlap'):EffectModel.fit([row],[row])
    first=EffectModel.fit([row]);row=deepcopy(row);row['room']='b';second=EffectModel.fit([row])
    a=first.predict('brake',z,scaffold.host_schema());b=second.predict('brake',z,scaffold.host_schema())
    assert a['samples'][0]['values']==b['samples'][0]['values']
    assert a['samples'][0]['distance']==b['samples'][0]['distance']
    assert not a['supported'] and not a['calibrated']


def test_online_budget_no_free_physics_and_no_world_input():
    w,m,_=simulation.replay_boundary(seed_fixture(),[]);o=simulation.observation(w)
    z,commands=frontend.semantic_packet(m,o)
    online=OnlineRollout(max_branches=0,max_ticks=0)
    r=online.run(m,o,'brake',commands['brake'])
    assert not r['samples']
    assert online.ledger['ticks']==online.ledger['branches']==0
    assert online.ledger['budget_exhausted']==1
    assert w.env.total_periods==0


def test_path_safety_includes_initial_state_for_zero_tick_rejection(monkeypatch):
    from types import SimpleNamespace
    world=SimpleNamespace(env=SimpleNamespace(joint_step=lambda:None),
                          step=lambda command,remaining_ticks=None:{'periods':0})
    snapshot=dict(xy=[0.,0.],angle=0.,object_static=True,robot_static=False,robot_peer=False)
    monkeypatch.setattr(simulation,'truth',lambda world:deepcopy(snapshot))
    monkeypatch.setattr(simulation,'observation',lambda world:{'goal':{'xy':[1.,1.]}})
    measured=simulation.measured_step(world,{'ticks':6})
    assert measured['object_static'] is True
    assert measured['object_static_ticks']==0 and measured['event']['periods']==0


def test_public_context_distance_does_not_average_away_large_pose_difference():
    w,m,_=simulation.replay_boundary(seed_fixture(),[])
    z,_=frontend.semantic_packet(m,simulation.observation(w))
    shifted=deepcopy(z);shifted['distance']+=.9;shifted['gx']+=.9
    row=dict(room='development',action='push_both_1_6',semantic=shifted,next_semantic=shifted,
             audit=dict(object_static=True))
    model=EffectModel.fit([row])
    assert not model.predict('push_both_1_6',z,scaffold.host_schema())['samples']
    assert model.predict('push_both_1_6',shifted,scaffold.host_schema())['samples'][0]['distance']==0.


def test_vectorized_neighborhood_matches_scalar_public_distance():
    from experiments.planar_tasks.authoring_v2.symbolic_general.prediction import distance
    w,m,_=simulation.replay_boundary(seed_fixture(),[])
    z,_=frontend.semantic_packet(m,simulation.observation(w));rows=[];action='push_both_1_6'
    for i,delta in enumerate((0.,.01,.06,.15,.30,.9)):
        before=deepcopy(z);before['distance']+=delta;before['gx']+=delta
        if i==1:before['contact0']=not before['contact0']
        rows.append(dict(room=str(i),action=action,semantic=before,next_semantic=before,audit={'object_static':False}))
    model=EffectModel.fit(rows)
    expected={r['room']:distance(r['semantic'],z,action,model.feature_profile)for r in rows}
    expected={room:d for room,d in expected.items()if d<=model.packet['max_distance']}
    observed={s['room']:s['distance']for s in model.predict(action,z,scaffold.host_schema())['samples']}
    assert observed==pytest.approx(expected)
