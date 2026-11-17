from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import pytest
from experiments.planar_tasks.relational_search import pipeline_execution as X,transfer_tasks as T
from experiments.planar_tasks.semantic_rl import relational_transfer_features as G
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.environment import EventConfig


def observation():
    scene,boxes=X.fixture(T.cases()[0]['fixture']);return X.RecordedEnv(scene,boxes).public()


def frame(o,event_action=None):
    h=SimpleNamespace(episode='a',index=0,sha256='h',event=dict(observation=o,action=event_action));return h


def rotation90(o):
    o=deepcopy(o);height=o['robots'][0]['arena'][1]
    point=lambda p:[height-p[1],p[0]];vec=lambda p:[-p[1],p[0]]
    o['goal']['xy']=point(o['goal']['xy'])
    for e in o['entities']:e['points']=[point(p) for p in e['points']]
    for r in o['robots']:
        own=r['own'];own['x'],own['y']=point([own['x'],own['y']]);own['vx'],own['vy']=vec([own['vx'],own['vy']]);own['angle']+=math.pi/2
        r['arena']=list(reversed(r['arena']))
        for k in ['desired_velocity','reference_velocity','force']:
            if r['execution'].get(k) is not None:r['execution'][k]=vec(r['execution'][k])
        for p in r.get('patches',[]):p['normal_world']=vec(p['normal_world'])
        scan=r['scan'];scan['origin']=point(scan['origin']);scan['heading']+=math.pi/2;scan['ray_angles']=[a+math.pi/2 for a in scan['ray_angles']]
    return o


def test_geometry_and_action_rotation_equivariance(monkeypatch):
    monkeypatch.setattr(G,'current_event',lambda h:h.event)
    o=observation();a=G.RelativeGeometry();b=G.RelativeGeometry();a.advance(frame(o));b.advance(frame(rotation90(o)))
    np.testing.assert_allclose(a.vector,b.vector,atol=2e-10,rtol=0,equal_nan=True)
    for index in range(32):np.testing.assert_allclose(G.action_vector(index,a.theta),G.action_vector((index+8)%32,b.theta),atol=1e-12)
    for index in range(47,79):np.testing.assert_allclose(G.action_vector(index,a.theta),G.action_vector(47+(index-47+8)%32,b.theta),atol=1e-12)


def test_entity_label_renaming_and_causal_history(monkeypatch):
    monkeypatch.setattr(G,'current_event',lambda h:h.event)
    o=observation();other=deepcopy(o);rename={'entity-a':'opaque-z','entity-b':'opaque-q'}
    other['goal']['target_id']=rename[o['goal']['target_id']]
    other['active_target']=rename.get(o['active_target'],o['active_target'])
    for e in other['entities']:e['id']=rename[e['id']]
    for r in other['robots']:r['scan']['entity_ids']=[rename.get(x,x) for x in r['scan']['entity_ids']]
    a=G.RelativeGeometry();b=G.RelativeGeometry();h=frame(o);a.advance(h);b.advance(frame(other))
    np.testing.assert_array_equal(a.vector,b.vector)
    np.testing.assert_array_equal(a.candidate_geometry({'target_id':'entity-b'}),b.candidate_geometry({'target_id':'opaque-q'}))
    before=a.vector.copy();a.advance(h);np.testing.assert_array_equal(before,a.vector)
    h.index=2
    with pytest.raises(ValueError,match='skipped'):a.advance(h)


def test_common_scale_features_keep_all_forty_distinct_durations():
    rows=np.array([G.action_encoding(dict(target_id='a',a0=2,a1=2,ticks=t),'a',.7) for t in range(1,41)])
    np.testing.assert_array_equal(rows[:,-40:],np.eye(40));assert len(np.unique(rows,axis=0))==40
    assert not G.action_encoding(dict(target_id='a',a0=2),'a',.7)[-41:].any()


def test_live_feature_group_layout_and_candidate_geometry():
    source=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text();groups=dict(candidate={'duration_match':'A'},state={'contact_fraction':'D','blocked_fraction':'D','observed_entities':'D'})
    f=G.TransferFeatures(source,groups=groups,fast_embedding=True,action_protocol=protocol(EventConfig(mode='duration')))
    try:
        h=f.new_history();o=observation();h.append(o);tree=ActionTree(o);prefix={}
        for key in ['target_id','a0','a1','ticks']:
            p=f.panel(h,tree,prefix)
            assert len(p['attribution_context'])==len(f.layout['context_groups'])
            assert p['attribution_candidates'].shape==(len(p['choices']),len(f.layout['candidate_groups']))
            prefix=p['choices'][0]
        assert f.layout['candidate_groups'].count('common')==69
    finally:f.close()
