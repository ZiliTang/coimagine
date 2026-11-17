"""In-memory training parity, ownership, storage opt-in and update integrity."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from experiments.planar_tasks.semantic_rl.memory_flow import MemoryHistory,current_event,compact_step
from experiments.planar_tasks.semantic_rl.stream_history import StreamHistory
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.environment import Environment,EventConfig
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.runner import rollout
from experiments.planar_tasks.semantic_rl.batched_ppo import prepare,evaluate_prepared,ppo_batch_update
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.relational_search.pipeline import save,unpack
from experiments.planar_tasks.authoring_v2.semantic_rl_parallel import collect,Config,record_trial,sha
from experiments.planar_tasks.semantic_rl.console import replay

SOURCE=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text()

@pytest.fixture
def setup():
    torch.set_num_threads(1);torch.manual_seed(41)
    cfg=EventConfig(mode='duration')
    f=Features(SOURCE,action_protocol=protocol(cfg),execution_backend='trusted_inprocess',trusted_source_sha256=digest(SOURCE))
    h=f.new_history();h.append(Environment(seed_fixture()).history.events[0]['observation'])
    model=Policy([f.panel(h,ActionTree(h.events[-1]['observation']),{})],f.sha256)
    try:yield f,model,cfg
    finally:f.close()


def test_memory_cursor_ownership_causality_and_no_json_chain(setup):
    f,model,cfg=setup
    observation=Environment(seed_fixture()).history.events[0]['observation']
    observation['privileged_secret']='hidden'
    h=MemoryHistory();h.append(observation);expected=deepcopy(current_event(h))
    observation['robots'].clear();h.events[0]['observation']['robots'].clear()
    assert current_event(h)==expected and 'privileged_secret' not in expected['observation']
    with pytest.raises(ValueError,match='snapshot'):h.snapshot()
    f.recommendation(h);count=f.worker.update_count
    f.recommendation(h);assert f.worker.update_count==count
    o=deepcopy(expected['observation']);a=dict(target_id=None,a0=44,a1=44,ticks=1)
    with pytest.raises(ValueError):h.append(o,a)
    o['tick']+=1;h.append(o,a);f.recommendation(h)
    o['tick']+=1;h.append(o,a);o['tick']+=1;h.append(o,a)
    with pytest.raises(ValueError,match='skipped'):f.recommendation(h)


def test_recording_and_compact_buffer_preserve_ppo(setup):
    f,model,cfg=setup
    kwargs=dict(model=model,features=f,config=cfg,max_ticks=60,max_decisions=2,seed=23,fast_feedback=True)
    old,full=rollout(seed_fixture(),SOURCE,**kwargs)
    diag,dfull=rollout(seed_fixture(),SOURCE,training_fastpath=True,**kwargs)
    lean,compact=rollout(seed_fixture(),SOURCE,training_fastpath=True,record_details=False,**kwargs)
    assert full and old['summary']['error'] is None
    assert diag['events']==old['events'] and diag['replay']==old['replay']
    for key in ('ticks','decisions','bootstrap','success','stop','error'):
        assert old['summary'][key]==diag['summary'][key]==lean['summary'][key]
    assert lean['replay'] is None and lean['events']==lean['public_rows']==lean['decisions']==[]
    a,b=prepare(model,full),prepare(model,compact)
    for x,y in zip(evaluate_prepared(model,a),evaluate_prepared(model,b)):assert torch.equal(x,y)
    for i,s in enumerate(compact):
        assert s['action']==full[i]['action']==dfull[i]['action']
        assert 'history_hash' not in s and 'choices' not in s['panels'][0]
        assert s['panels'][0]['semantic'].dtype==np.float32
        for field in ('reward','periods','logp','value','terminal'):assert s[field]==full[i][field]
    models=[deepcopy(model),deepcopy(model)];updates=[]
    for m,batch in zip(models,(full,compact)):
        updates.append(ppo_batch_update(m,torch.optim.Adam(m.parameters(),lr=3e-4),
            [dict(steps=batch,bootstrap=old['summary']['bootstrap'])],epochs=2,minibatch_size=2,target_kl=0))
    assert updates[0]['curve']==updates[1]['curve']
    assert all(torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items())
    bad=deepcopy(compact);bad[0]['panels'][0]['semantic_sha256']='wrong'
    with pytest.raises(ValueError,match='version'):prepare(model,bad)
    bad=deepcopy(compact);bad[0]['logp']+=1
    with pytest.raises(ValueError,match='mismatch'):ppo_batch_update(model,torch.optim.Adam(model.parameters()),[dict(steps=bad,bootstrap=0.)])


def test_collect_memory_storage_optin_logs_and_replay(setup,tmp_path,capsys):
    f,model,cfg=setup
    (tmp_path/'source.py').write_text(SOURCE);model.save(tmp_path/'model.pt',{})
    base=dict(id='a00000',source=str(tmp_path/'source.py'),checkpoint=str(tmp_path/'model.pt'),
        semantic_sha256=digest(SOURCE),checkpoint_sha256=sha(tmp_path/'model.pt'),policy_version=0,
        case=dict(id='case',family='clear',fixture=seed_fixture()),max_ticks=2,max_decisions=2,episode_seconds=60,
        seed=17,temperature=1.,feature_runtime='typed_stream',execution_mode='duration',
        execution_backend='trusted_inprocess',trusted_source_sha256=digest(SOURCE),batch_transport='memory')
    for details in (False,True):
        p=tmp_path/str(details)/'runs'/'a00000';p.mkdir(parents=True)
        result=collect(dict(base,path=str(p),record_details=details));s=result['summary']
        assert result['episode']['steps'] and s['save_seconds']>=0
        assert (p/'summary.json').exists()
        assert (p/'result.json.gz').exists()==details
        assert (p/'batch.json.gz').exists()==details
        response=replay(tmp_path,str(details),'a00000')
        assert bool(response['frames'])==details
        if details:
            raw=unpack(p/'batch.json.gz')
            assert raw[0]['action']==result['episode']['steps'][0]['action']
    log=capsys.readouterr().out
    assert 'details=False' in log and 'details=True' in log and 'save_seconds=' in log and 'batch_transport=memory' in log


def test_recording_schedule_is_preselected_and_off_by_default():
    assert not any(record_trial(Config(),i) for i in range(100))
    assert [i for i in range(8) if record_trial(Config(detail_trials=2,detail_every=5),i)]==[0,1,5]
    assert all(record_trial(Config(detail_every=1),i) for i in range(8))
    with pytest.raises(ValueError):Config(detail_trials=-1)
