"""Action-scale units, interrupted execution and cross-protocol rejection."""
from copy import deepcopy
from dataclasses import asdict
import pytest
import torch
from experiments.planar_tasks.semantic_rl.environment import Environment, EventConfig
from experiments.planar_tasks.semantic_rl.scale import protocol, check_trace
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.learning import demonstration_traces, warm_start
from experiments.planar_tasks.semantic_rl.runner import rollout
from experiments.planar_tasks.semantic_rl.batched_ppo import prepare
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
from experiments.planar_tasks.relational_search.action_scale import duration_seconds

SOURCE='''NAMES=['duration']
def initial_memory():return {}
def update_memory(m,e):return m
def propose_actions(m,o):return [dict(target_id=None,a0=44,a1=44,ticks=20)]
def semantic_features(m,o,a,p):return dict(duration=a['ticks'])
'''


def test_duration_ignores_normal_contact_and_progress_but_honors_faults():
    e=Environment(seed_fixture(),EventConfig(mode='duration'),fast_feedback=True)
    a=dict(target_id=None,a0=44,a1=44,ticks=20)
    e.detector.update=lambda o:'contact'
    first=e.step(a)
    assert first['periods']==20 and first['reason']=='duration'
    e.detector.update=lambda o:'progress'
    assert e.step(a)['periods']==20
    e.detector.update=lambda o:'blocked'
    blocked=e.step(a)
    assert blocked['periods']==1 and blocked['reason']=='blocked'
    assert blocked['requested']['ticks']==20 and blocked['scale']['requested_ticks']==20
    assert blocked['scale']['executed_ticks']==1
    e.detector.update=lambda o:None
    cut=e.step(a,remaining_ticks=3)
    assert cut['reason']=='collector_cut' and cut['periods']==3


def test_target_handoff_is_internal_and_all_time_is_charged():
    e=Environment(seed_fixture(),EventConfig(mode='duration'),fast_feedback=True)
    e.detector.update=lambda o:None
    a=dict(target_id='entity-b',a0=44,a1=10,ticks=20)
    event=e.step(a)
    assert event['periods']==20 and event['reason']=='duration'
    assert event['handoffs'][0]['phase']=='switch_brake'
    assert any(x['phase']=='commit' for x in event['handoffs'])
    assert e.env.active_target=='entity-b'
    assert len(e.history.events)==2
    assert event['scale']['executed_ticks']*.05==1.
    assert e.history.events[-1]['action']==a


def test_scale_is_time_not_distance_and_declared_in_model_and_trace(tmp_path):
    torch.set_num_threads(1)
    assert duration_seconds(20)==1.
    for n in (0,41,True,.2):
        with pytest.raises(ValueError):duration_seconds(n)
    h=Environment(seed_fixture()).history
    expected=protocol(EventConfig(mode='duration'))
    f=Features(SOURCE,fast_embedding=True,runtime='cached_fork',action_protocol=expected)
    try:
        root=f.panel(h,ActionTree(h.events[-1]['observation']),{})
        model=Policy([root],f.sha256)
        tr=model.decide(h,f,teacher=dict(target_id=None,a0=44,a1=44,ticks=20))
        assert tr['action_protocol']==expected
        tr.update(periods=3,reward=-.15,terminal=False)
        check_trace(model,tr);prepare(model,[tr])
        altered=deepcopy(tr);altered['action_protocol']=protocol(EventConfig(mode='progress'))
        with pytest.raises(ValueError,match='protocol'):prepare(model,[altered])
        altered=deepcopy(tr);altered['periods']=21
        with pytest.raises(ValueError,match='periods'):prepare(model,[altered])
        model.save(tmp_path/'p.pt',{})
        loaded,_=Policy.load(tmp_path/'p.pt',f.sha256)
        assert loaded.action_protocol==expected
        with pytest.raises(ValueError,match='protocol'):
            rollout(seed_fixture(),SOURCE,model=loaded,features=f,config=EventConfig(),max_ticks=5)
    finally:f.close()


def test_legacy_checkpoint_cannot_silently_enter_new_duration_protocol():
    h=Environment(seed_fixture()).history;f=Features(SOURCE)
    m=Policy([f.panel(h,ActionTree(h.events[-1]['observation']),{})],f.sha256)
    with pytest.raises(ValueError,match='protocol'):
        rollout(seed_fixture(),SOURCE,model=m,features=f,config=EventConfig(mode='duration'),max_ticks=1)


def test_demo_label_is_requested_scale_and_protocol_must_match():
    expected=protocol(EventConfig(mode='duration'))
    f=Features(SOURCE,runtime='cached_fork',fast_embedding=True,action_protocol=expected)
    e=Environment(seed_fixture(),EventConfig(mode='duration'),fast_feedback=True)
    a=dict(target_id=None,a0=44,a1=44,ticks=20)
    h=e.history;first=dict(action=None,observation=deepcopy(h.events[0]['observation']),history_hash=h.sha256)
    e.detector.update=lambda o:'blocked';e.step(a)
    last=dict(action=a,observation=h.events[-1]['observation'],history_hash=h.sha256)
    # Synthetic successful label only for interface unit test; not task success evidence.
    demo=dict(summary=dict(status='success'),split='train',execution_mode='duration',
              event_config=asdict(EventConfig(mode='duration')),action_protocol=expected,public_rows=[first,last])
    try:
        traces=demonstration_traces(demo,f,split='train',mode='duration')
        assert traces[0]['action']['ticks']==20
        assert traces[0]['action_protocol']==expected
        bad=deepcopy(demo);bad['action_protocol']['unit']='metres'
        with pytest.raises(ValueError,match='protocol'):demonstration_traces(bad,f,split='train',mode='duration')
        mixed=deepcopy(traces);mixed.append(dict(traces[0],action_protocol=protocol(EventConfig())))
        with pytest.raises(ValueError,match='protocol'):warm_start(mixed,f,epochs=1)
    finally:f.close()


def test_author_campaign_freezes_and_uses_requested_scale_protocol(tmp_path):
    from experiments.planar_tasks.authoring_v2.semantic_rl import SemanticRLCampaign
    from experiments.planar_tasks.relational_search.pipeline import unpack,read
    c=SemanticRLCampaign.create(tmp_path/'new',execution_mode='duration',feature_runtime='cached_fork')
    source=SOURCE+'''\ndef prefix_semantic_features(m,o,p,recs):return dict(duration=p.get('ticks',0))\n'''
    sid=c.submit(source,'bounded interface check')['id']
    r=c.test(sid,'open_transport',max_ticks=3,max_decisions=2)
    result=unpack(c.path/'runs'/r['id']/'result.json.gz')
    assert result['action_protocol']==protocol(EventConfig(mode='duration'))
    assert result['summary']['ticks']==3
    assert result['events'][0]['scale']['requested_ticks']==20
    assert result['events'][0]['scale']['executed_ticks']==3
    assert any('ACTION_SCALE' in p.name for p in (c.path/'knowledge').iterdir())
    assert not read(c.path/'campaign.json')['reservations']
