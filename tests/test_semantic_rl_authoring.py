from copy import deepcopy
import torch
import pytest
from experiments.planar_tasks.authoring_v2.semantic_rl import SemanticRLCampaign,SemanticRLSession,development_fixtures
from experiments.planar_tasks.relational_search.pipeline import read
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.environment import Environment,EventConfig
from experiments.planar_tasks.semantic_rl.learning import warm_start
from experiments.planar_tasks.relational_search import pipeline_execution as X

SOURCE='''NAMES=['preferred','duration']
def initial_memory():return {}
def update_memory(m,e):return m
def propose_actions(m,o):return [dict(target_id=None,a0=44,a1=44,ticks=4)]
def semantic_features(m,o,a,p):return dict(preferred=float(a in p),duration=a['ticks'])
def prefix_semantic_features(m,o,a,p):return dict(preferred=float(all(p[0][k]==v for k,v in a.items())),duration=a.get('ticks'))
'''


def test_versioned_wall_layout_and_legal_observation_boundary(tmp_path):
    old = development_fixtures()
    new = development_fixtures('stage1-wall-anchor-v2')
    assert old['open_transport'] == new['open_transport']
    assert old['clear_transport'] == new['clear_transport']
    assert 'scene_id' not in old['wall_release']
    scene, boxes = X.fixture(new['wall_release'])
    anchor = X.T.task_catalog()['planar_wall_anchor'].instance
    assert scene == anchor and boxes == ()
    assert X.fixture(old['wall_release'])[0].object_polygon is not None
    public = X.RecordedEnv(scene, boxes).public()
    assert not {'scene_id','object_polygon','obstacles','fixture_profile'} & public.keys()
    assert public['entities'] and public['entity_legal_pairs']
    polygon = X.scene_metadata(scene, boxes)['object_polygon']
    assert max(p[0] for p in polygon)-min(p[0] for p in polygon) == 1.
    assert max(p[1] for p in polygon)-min(p[1] for p in polygon) == .25
    c = SemanticRLCampaign.create(tmp_path/'v2', fixture_profile='stage1-wall-anchor-v2')
    assert read(c.path/'tasks/wall_release/fixture.json') == new['wall_release']
    assert read(c.path/'campaign.json')['fixture_profile'] == 'stage1-wall-anchor-v2'
    assert c.freeze()['status'] == 'verified'


def test_fixture_layout_rejects_unknown_and_success_radius_changes(tmp_path):
    f = development_fixtures('stage1-wall-anchor-v2')['wall_release']
    f['scene_id'] = '../hidden_scene'
    with pytest.raises(ValueError, match='scene_id'): X.fixture(f)
    f['scene_id'] = 'planar_wall_anchor'; f['goal']['radius'] = 1.
    with pytest.raises(ValueError, match='success radius'): X.fixture(f)
    with pytest.raises(ValueError, match='profile'):
        SemanticRLCampaign.create(tmp_path/'invalid', fixture_profile='typo')
    assert not (tmp_path/'invalid').exists()

def test_stage1_session_metering_gate_and_freeze(tmp_path):
    c=SemanticRLCampaign.create(tmp_path/'c',feature_runtime='cached_fork');s=SemanticRLSession(c.path)
    sid=s.tool('submit',dict(source=SOURCE,summary='schema and metering test'))['id']
    r=s.tool('test',dict(sid=sid,task='open_transport',max_ticks=8,max_decisions=2))
    assert r['ticks']==8 and r['status']=='incomplete'
    state=read(c.path/'campaign.json')
    assert state['used']['execution_ticks']==8 and state['used']['feature_batches']==3 and not state['reservations']
    with pytest.raises(ValueError,match='gate'):c.run_rl(sid,'f000')
    with pytest.raises(ValueError,match='successful'):s.tool('fit',dict(sid=sid,run_ids=[r['id']]))
    assert c.freeze()['status']=='verified'
    with pytest.raises(ValueError,match='frozen'):c.test(sid,'open_transport',max_ticks=1)


def test_batched_full_imitation_reduces_loss():
    torch.set_num_threads(1)
    h=Environment(X.seed_fixture(),EventConfig(mode='time')).history
    f=Features(SOURCE);tree=ActionTree(h.events[-1]['observation']);prefix={};panels=[];indices=[]
    teacher=dict(target_id=None,a0=44,a1=44,ticks=4)
    for key in ('target_id','a0','a1','ticks'):
        p=f.panel(h,tree,prefix);prefix=dict(prefix,**{key:teacher[key]})
        panels.append(p);indices.append(p['choices'].index(prefix))
    traces=[dict(panels=panels,indices=indices)]*34 # crosses minibatch boundary
    model,curve=warm_start(traces,f,epochs=15)
    assert curve[-1]['imitation']<curve[0]['imitation']
    assert model.semantic_sha256==f.sha256
