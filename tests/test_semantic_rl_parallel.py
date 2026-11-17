from copy import deepcopy
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.request import urlopen
import pytest
import torch
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.relational_search.pipeline import save,pack
from experiments.planar_tasks.relational_search.pipeline_semantics import embedding
from experiments.planar_tasks.semantic_rl.environment import Environment
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.features import Features,embedding_batch
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.schema import Schema,migrate
from experiments.planar_tasks.semantic_rl.batched_ppo import prepare,evaluate_prepared,ppo_batch_update
from experiments.planar_tasks.relational_search.pipeline_server import handler_for
from experiments.planar_tasks.authoring_v2.semantic_rl_parallel import Config

SOURCE='''NAMES=['duration','recommended']
def initial_memory():return {}
def update_memory(m,e):return m
def propose_actions(m,o):return [dict(target_id=None,a0=44,a1=44,ticks=4)]
def semantic_features(m,o,a,p):return dict(duration=a['ticks'],recommended=float(a in p))
'''


def fixture():
    torch.set_num_threads(1);torch.manual_seed(3)
    env=Environment(X.seed_fixture());h=env.history;f=Features(SOURCE,fast_embedding=True)
    root=f.panel(h,ActionTree(h.events[-1]['observation']),{})
    return env,f,Policy([root],f.sha256)


def test_batch_embedding_preserves_all_candidate_values_and_old_features():
    env,f,m=fixture();env.step(dict(target_id=None,a0=44,a1=44,ticks=4));h=env.history
    actions=[dict(target_id=t,a0=i,a1=78-i,ticks=i%40+1) for t in [None,'entity-a','entity-b'] for i in range(79)]
    assert embedding_batch(h,actions)==[embedding(h,a) for a in actions]
    tree=ActionTree(h.events[-1]['observation'])
    assert Features(SOURCE).panel(h,tree,{})==f.panel(h,tree,{})


def test_reserved_migration_preserves_scores_and_expands_named_channels(tmp_path):
    env,f,old=fixture();h=env.history;tree=ActionTree(h.events[-1]['observation']);p=f.panel(h,tree,{})
    model,record=migrate(old,['duration','recommended'],f.sha256,capacity=64)
    schema=Schema(tuple(model.schema['slots']));newf=Features(SOURCE,schema=schema,fast_embedding=True)
    q=newf.panel(h,tree,{})
    assert len(q['semantic'][0])==230 and q['semantic'][0][2:64]==[0.]*62
    with torch.no_grad():
        assert torch.allclose(old.distribution(p)[0].logits,model.distribution(q)[0].logits,atol=1e-6)
        assert torch.allclose(old.value(p),model.value(q),atol=1e-6)
    expanded=SOURCE.replace("'recommended']","'recommended','extra']")+'''
_old_features=semantic_features
def semantic_features(m,o,a,p):return dict(_old_features(m,o,a,p),extra=999.)
'''
    newer,meta=migrate(model,['duration','recommended','extra'],digest(expanded))
    ns=Schema(tuple(newer.schema['slots']));panel=Features(expanded,schema=ns).panel(h,tree,{})
    assert ns.slots[:3]==('duration','recommended','extra')
    with torch.no_grad():
        assert torch.allclose(model.distribution(q)[0].logits,newer.distribution(panel)[0].logits,atol=1e-6)
        assert torch.allclose(model.value(q),newer.value(panel),atol=1e-6)
    newer.save(tmp_path/'model.pt',meta);restored,_=Policy.load(tmp_path/'model.pt',digest(expanded))
    assert restored.schema==newer.schema
    with pytest.raises(ValueError,match='semantic version'):newer.distribution(q)
    with pytest.raises(ValueError,match='removed'):ns.extend(['duration','extra'])
    with pytest.raises(ValueError,match='capacity'):Schema.create(['x'],1).extend(['x','y'])
    with pytest.raises(ValueError,match='train'):migrate(model,['duration','recommended','extra'],digest(expanded),calibration=dict(split='test',values={'extra':[1.]}))


def test_more_than_24_authored_semantics_require_explicit_schema():
    env,_,_=fixture();names=['s'+str(i) for i in range(25)]
    source=SOURCE.replace("NAMES=['duration','recommended']",'NAMES='+repr(names))+"\ndef semantic_features(m,o,a,p):return {n:0. for n in NAMES}\n"
    f=Features(source,schema=Schema.create(names));root=f.panel(env.history,ActionTree(env.history.events[-1]['observation']),{})
    assert len(root['semantic'][0])==230
    legacy=Features(source)
    with pytest.raises(ValueError,match='feature names'):legacy.panel(env.history,ActionTree(env.history.events[-1]['observation']),{})
    assert legacy.calls==1


def test_batched_probabilities_entropy_and_independent_episode_gae():
    env,f,model=fixture();trace=model.decide(env.history,f,temperature=2.,generator=torch.Generator().manual_seed(7))
    a=dict(deepcopy(trace),periods=4,reward=-.2,terminal=False)
    b=dict(deepcopy(trace),periods=9,reward=-3.,terminal=True)
    rows=prepare(model,[a,b])
    with torch.no_grad():
        lp,v,penalty,entropy=evaluate_prepared(model,rows)
        for i,s in enumerate([a,b]):
            expected=model.evaluate(s)
            assert float(lp[i])==pytest.approx(s['logp'],abs=2e-5)
            assert torch.allclose(v[i],expected[1],atol=1e-6)
            assert torch.allclose(penalty[i],expected[2],atol=1e-6)
            assert entropy[i]>=0
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    result=ppo_batch_update(model,opt,[dict(steps=[a],bootstrap=2.),dict(steps=[b],bootstrap=0.)],epochs=2,minibatch_size=1,target_kl=0)
    assert result['updates']==4 and result['logp_recompute_error']<=2e-5
    assert all(torch.isfinite(torch.tensor(r['loss'])) for r in result['curve'])
    # A bad stored likelihood must fail before any optimizer step.
    a['logp']+=1
    with pytest.raises(ValueError,match='mismatch'):ppo_batch_update(model,opt,[dict(steps=[a],bootstrap=0.)])


def test_rl_readonly_console_metrics_replay_and_unknown_fields(tmp_path):
    p=tmp_path/'semantic_rl'/'sample';p.mkdir(parents=True)
    save(p/'campaign.json',dict(runs=[],status='development',created=1,profile='semantic-ppo-parallel-v1'))
    save(p/'metrics.json',dict(rounds=[dict(round=0,success_rate=.5)],updates=[dict(optimizer_step=1,actor=-.2)]))
    (p/'runs/a00000').mkdir(parents=True)
    pack(p/'runs/a00000/result.json.gz',dict(replay=dict(frames=[dict(tick=0)],scene={})))
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(tmp_path/'relational_pipeline'));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    try:
        assert json.load(urlopen(base+'/api/rl'))[0]['id']=='sample'
        d=json.load(urlopen(base+'/api/rl/sample'));assert d['metrics']['updates'][0]['actor']==-.2
        assert json.load(urlopen(base+'/api/rl/sample/replay/a00000'))['frames'][0]['tick']==0
        assert 'PPO'.encode() in urlopen(base+'/rl').read()
    finally:server.shutdown();server.server_close();thread.join()


def test_parallel_defaults_and_invalid_limits():
    c=Config();assert c.workers==16 and c.rounds*c.workers==2048 and c.minibatch_size==64
    with pytest.raises(ValueError):Config(workers=0)
