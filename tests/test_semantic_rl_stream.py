"""Incremental protocol tests; synthetic causal streams, not success experiments."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
import torch
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.stream_history import StreamHistory
from experiments.planar_tasks.semantic_rl.actions import ActionTree,FIELDS
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.environment import Environment,EventConfig
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.runner import rollout
from experiments.planar_tasks.semantic_rl.learning import demonstration_traces,warm_start
from experiments.planar_tasks.semantic_rl.batched_ppo import prepare,evaluate_prepared,ppo_batch_update
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
from experiments.planar_tasks.relational_search.contract import canonical,digest,History

SOURCE=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text()
COUNT=SOURCE.replace("return {'ticks': 0, 'contact': 0, 'blocked': 0}","return {'ticks': 0, 'contact': 0, 'blocked': 0, 'count': 0}").replace("    dt = event", "    memory['count'] += 1\n    dt = event").replace("state = [memory['contact'] / denom, memory['blocked'] / denom, float(len(observation['entities']))]", "state = [float(memory['count']), float(observation['tick']), float('secret' in observation)]")
ACTION=dict(target_id=None,a0=44,a1=44,ticks=1)

@pytest.fixture
def observation():
    o=Environment(seed_fixture()).history.events[0]['observation']
    o=deepcopy(o);o['entities']=[];o['entity_legal_pairs']={}
    o['legal_pairs']=o['global_legal_pairs']=[[44,44]]
    return o


def push(h,o,tick):
    o=deepcopy(o);o['tick']=tick;o['secret']='not public'
    h.append(o,None if h.index<0 else ACTION)


def test_incremental_once_window_and_worker_restart(observation):
    f=Features(COUNT);h=f.new_history()
    try:
        for k in range(70):
            push(h,observation,k)
            f.recommendation(h)
            assert f.worker.update_count==k+1
            assert f.worker.frames[-1]['state']==[k+1.,float(k),0.]
            before=f.calls;f.recommendation(h);assert f.calls==before
            assert len(f.worker.frames)==min(k+1,8) and len(h.events)==1
            if k==35:f.close() # loss of broker is harmless: explicit state survives
        assert f.worker.memory['count']==70
        assert [x['state'][0] for x in f.worker.frames]==list(range(63,71))
        with f.decision(h):
            prefix={}
            for field in FIELDS:
                p=f.panel(h,ActionTree(h.events[-1]['observation']),prefix)
                assert p['semantic_protocol']['version']=='typed-stream-v1'
                assert p['semantic'][0][64]==63
                assert len(p['semantic'][0])==64+8*34+166
                prefix=p['choices'][0]
        assert f.worker.update_count==70
    finally:f.close()


def test_immutable_cursor_skips_and_reset(observation):
    h=StreamHistory();push(h,observation,0);old=h.sha256
    h.events[0]['observation']['tick']=500
    assert h.sha256==old and h.events[0]['observation']['tick']==0
    with pytest.raises(ValueError):h.append(observation,ACTION)
    f=Features(COUNT)
    try:
        f.recommendation(h);push(h,observation,1);push(h,observation,2)
        with pytest.raises(ValueError,match='skipped'):f.recommendation(h)
        fresh=f.new_history();push(fresh,observation,0);f.recommendation(fresh)
        assert f.worker.memory['count']==1
        with pytest.raises(ValueError,match='StreamHistory'):f.recommendation(History())
    finally:f.close()


def test_snapshot_restore_fork_and_source_version(observation):
    f=Features(COUNT);g=Features(COUNT);changed=Features(COUNT+'\n# new version\n')
    h=f.new_history()
    try:
        for k in range(4):push(h,observation,k);f.recommendation(h)
        snap=json.loads(json.dumps(f.snapshot()));h2=StreamHistory.restore(json.loads(json.dumps(h.snapshot())))
        g.restore(snap,h2)
        tree=ActionTree(h.events[-1]['observation'])
        assert canonical(f.panel(h,tree,{}))==canonical(g.panel(h2,tree,{}))
        push(h,observation,4);push(h2,observation,4)
        assert canonical(f.panel(h,tree,{}))==canonical(g.panel(h2,tree,{}))
        with pytest.raises(ValueError,match='version mismatch'):changed.restore(snap,StreamHistory.restore(h.snapshot()))
        with pytest.raises(ValueError,match='cursor mismatch'):g.restore(snap,h.fork())
        bad=deepcopy(snap);bad['value']['memory']['count']=999
        with pytest.raises(ValueError,match='checksum'):g.restore(bad,h2)
        truncated=StreamHistory.restore(h.snapshot())
        with pytest.raises(ValueError,match='prefix'):changed.recommendation(truncated)
    finally:f.close();g.close();changed.close()


def test_stream_temporal_mask_and_actual_elapsed(observation):
    f=Features(COUNT);h=f.new_history()
    try:
        push(h,observation,0);f.recommendation(h)
        o=deepcopy(observation);o['tick']=3;h.append(o,dict(ACTION,ticks=20));f.recommendation(h)
        rows=[f.worker.vector_window()[i:i+34] for i in range(0,8*34,34)]
        assert rows[0][-2:]==[0.,None]
        assert rows[-2][-2:]==[1.,.15] or rows[-2][-2:]==[1.,3*.05]
        assert rows[-1][-2:]==[1.,0.] and f.worker.memory['ticks']==3
    finally:f.close()


def test_panels_policy_checkpoint_and_ppo_protocol(observation,tmp_path):
    f=Features(COUNT);h=f.new_history()
    try:
        push(h,observation,0);p=f.panel(h,ActionTree(h.events[-1]['observation']),{})
        model=Policy([p],f.sha256)
        trace=model.decide(h,f,teacher=ACTION)
        trace.update(periods=1,reward=-.05,terminal=True)
        before=f.calls
        lp=model.evaluate(trace)[0]
        assert abs(float(lp.detach())-trace['logp'])<2e-5 and f.calls==before
        rows=prepare(model,[trace]);assert abs(float(evaluate_prepared(model,rows)[0][0].detach())-trace['logp'])<2e-5
        # One tiny optimizer step tests saved-panel reuse, not a research training run.
        update=ppo_batch_update(model,torch.optim.Adam(model.parameters(),lr=1e-5),[dict(steps=[trace],bootstrap=0.)],epochs=1)
        assert update['updates']==1 and f.calls==before
        model.save(tmp_path/'p.pt',{});loaded,_=Policy.load(tmp_path/'p.pt',f.sha256)
        assert loaded.semantic_protocol==model.semantic_protocol
        assert torch.equal(loaded.distribution(p)[0].logits,model.distribution(p)[0].logits)
        bad=deepcopy(trace);bad['panels'][0]['semantic_protocol']['history_length']=7
        with pytest.raises(ValueError,match='history protocol'):prepare(model,[bad])
        with pytest.raises(ValueError,match='history protocol'):model.evaluate(bad)
    finally:f.close()


def test_old_recording_offline_reextract_from_start(observation):
    old=History();rows=[]
    for k in range(4):
        o=deepcopy(observation);o['tick']=k;old.append(o,None if k==0 else ACTION)
        rows.append(dict(observation=o,action=None if k==0 else ACTION,history_hash=old.sha256))
    synthetic=dict(summary=dict(status='success'),execution_mode='time',public_rows=rows,split='train')
    f=Features(COUNT)
    try:
        traces=demonstration_traces(synthetic,f,split='train',mode='time')
        assert len(traces)==3 and f.worker.update_count==3 and f.calls==15
        assert [t['panels'][0]['semantic_window'][-1]['state'][0] for t in traces]==[1.,2.,3.]
        model,curve=warm_start(traces,f,epochs=1)
        assert len(curve)==1 and model.semantic_protocol==f.semantic_protocol
    finally:f.close()


def test_default_campaign_stream_meter_and_small_rollout(tmp_path):
    from experiments.planar_tasks.authoring_v2.semantic_rl import SemanticRLCampaign
    from experiments.planar_tasks.relational_search.pipeline import read
    c=SemanticRLCampaign.create(tmp_path/'campaign',execution_mode='duration')
    assert read(c.path/'campaign.json')['stage1']['feature_runtime']=='typed_stream'
    assert any('STREAM_SEMANTICS' in p.name for p in (c.path/'knowledge').iterdir())
    with pytest.raises(ValueError,match='SEMANTIC_API'):c.submit(Path('experiments/planar_tasks/semantic_rl/typed_example.py').read_text(),'wrong protocol')
    submission=c.submit(SOURCE,'neutral stream plumbing test')
    result=c.test(submission['id'],'open_transport',max_ticks=3,max_decisions=2)
    assert result['ticks']==3 and result['error'] is None
    assert c.state['used']['feature_batches']==5 # 2 rejected + 2 schema + 1 incremental CAP update
    assert not c.state['reservations']
    assert c.freeze()['status']=='verified'


def test_stream_full_menu_policy_execution_and_snapshot_bootstrap(tmp_path):
    cfg=EventConfig(mode='duration');f=Features(SOURCE,action_protocol=protocol(cfg))
    h=f.new_history();h.append(Environment(seed_fixture()).history.events[0]['observation'])
    try:
        p=f.panel(h,ActionTree(h.events[-1]['observation']),{})
        model=Policy([p],f.sha256)
        r,steps=rollout(seed_fixture(),SOURCE,model=model,features=f,config=cfg,max_ticks=2,max_decisions=2)
        assert r['summary']['error'] is None and steps and r['history_protocol']=='semantic-event-stream-v1'
        assert all(abs(float(model.evaluate(s)[0].detach())-s['logp'])<2e-5 for s in steps)
        from experiments.planar_tasks.semantic_rl.schema import migrate
        migrated,_=migrate(model,f.names,f.sha256)
        assert torch.equal(migrated.distribution(p)[0].logits,model.distribution(p)[0].logits)
    finally:f.close()


def test_snapshot_causal_time_and_independent_branch(observation):
    f=Features(COUNT);g=Features(COUNT);h=f.new_history()
    try:
        push(h,observation,0);f.recommendation(h)
        push(h,observation,1);f.recommendation(h)
        bad=h.snapshot();bad['previous_tick']=-1
        with pytest.raises(ValueError,match='timing'):StreamHistory.restore(bad)
        branch=h.fork();g.restore(f.snapshot(),branch)
        push(branch,observation,2);g.recommendation(branch)
        assert h.index==1 and f.worker.memory['count']==2 and g.worker.memory['count']==3
    finally:f.close();g.close()


def test_update_mutation_isolated_and_scoring_failure_does_not_double_update(observation):
    mutant=COUNT.replace("    return memory", "    event['observation']['tick'] = 999\n    return memory")
    f=Features(mutant);h=f.new_history()
    try:
        push(h,observation,0);f.recommendation(h)
        assert f.worker.frames[-1]['state']==[1.,0.,0.]
        before=f.worker.update_count
        # A rejected candidate does not commit another update or corrupt cached state.
        with pytest.raises(ValueError):f.worker.evaluate(h,[dict(ACTION,a0=999)],{})
        f.recommendation(h);assert f.worker.update_count==before
        assert h.events[-1]['observation']['tick']==0
    finally:f.close()


def test_replay_from_start_matches_incremental_snapshot(observation):
    f=Features(COUNT);g=Features(COUNT);h=f.new_history();replayed=g.new_history()
    try:
        for k in range(12):
            push(h,observation,k);f.recommendation(h)
            push(replayed,observation,k);g.recommendation(replayed)
        # Prefix score order/repetitions have no effect on subsequent state updates.
        for _ in range(3):f.panel(h,ActionTree(h.events[-1]['observation']),{})
        push(h,observation,12);push(replayed,observation,12)
        tree=ActionTree(h.events[-1]['observation'])
        assert canonical(f.panel(h,tree,{}))==canonical(g.panel(replayed,tree,{}))
        assert f.worker.update_count==g.worker.update_count==13
    finally:f.close();g.close()


def test_reserved_channel_capacity_is_available_to_future_sources():
    from experiments.planar_tasks.semantic_rl.typed_compiler import stream_config
    names=['a'+str(i) for i in range(64)];state=['s'+str(i) for i in range(32)]
    source="SEMANTIC_API='typed-stream-v1'\nNAMES="+repr(names)+"\nSTATE_NAMES="+repr(state)
    config=stream_config(source)
    assert config['state_capacity']==32 and config['state_names']==state
    with pytest.raises(ValueError,match='STATE_NAMES'):
        stream_config(source.replace(repr(state),repr(state+['too_many'])))
