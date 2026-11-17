"""Action semantics proposal identity, nonlinear choice and evidence boundaries."""
from copy import deepcopy
import pytest
import numpy as np
from experiments.planar_tasks.action_semantics.program import Program
from experiments.planar_tasks.action_semantics.network import train,MLP
from experiments.planar_tasks.action_semantics import execution,fitting
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.contract import History,digest
from experiments.planar_tasks.relational_search.pipeline import read,save,pack
from experiments.planar_tasks.authoring_v2.action_semantics import ActionCampaign,ActionSession

BRAKE=dict(target_id=None,a0=44,a1=44,ticks=4)
SOURCE='''NAMES=['duration','preferred']
def initial_memory():return {}
def update_memory(m,e):return m
def propose_actions(m,o):return [dict(target_id=None,a0=44,a1=44,ticks=4 if o['tick']%8==0 else 8)]
def semantic_features(m,o,a,p):return dict(duration=a['ticks'],preferred=float(a==p[0]))
'''

def history():
    v=X.seed_fixture();s,b=X.fixture(v);h=History();h.append(X.RecordedEnv(s,b,overrides=v['overrides']).public());return h


def test_proposal_is_candidate_and_hidden_extensions_are_stripped():
    h=history();p=Program(SOURCE);a=p.evaluate(h)
    assert a['proposals'][0]==BRAKE and BRAKE in a['actions']
    i=a['actions'].index(BRAKE);assert a['semantic'][i][2]==1  # appended first-proposal match
    h.events[-1]['observation']['truth_mass']=900
    hidden=SOURCE.replace("a['ticks'],preferred", "o.get('truth_mass',0),preferred")
    assert all(v[0]==0 for v in Program(hidden).evaluate(h)['values'])


def test_features_candidate_order_independent():
    h=history();extras=[dict(BRAKE,ticks=7),dict(BRAKE,ticks=11)];p=Program(SOURCE)
    a=p.evaluate(h,extras,generic=False);b=p.evaluate(h,extras[::-1],generic=False)
    assert a==b


def test_cap_identity_preserves_real_boundary_histories():
    # Independent CAP action function; online state-dependent durations, not seed success evidence.
    commands=[];tick=0
    for _ in range(3):
        a=dict(BRAKE,ticks=4 if tick%8==0 else 8);commands.append(a);tick+=a['ticks']
    expected=X.execute(X.seed_fixture(),commands)
    actual=execution.run(X.seed_fixture(),SOURCE,max_ticks=40,max_decisions=3)
    assert actual['commands']==commands
    assert [r['history_hash'] for r in actual['public_rows']]==[r['history_hash'] for r in expected['public_rows']]
    assert actual['summary']['prediction_calls']==0


def test_illegal_first_proposal_is_explicit_failure():
    source=SOURCE.replace("return [dict(target_id=None,a0=44,a1=44,ticks=4 if o['tick']%8==0 else 8)]", "return [dict(target_id=None,a0=44,a1=44,ticks=41),dict(target_id=None,a0=44,a1=44,ticks=4)]")
    r=execution.run(X.seed_fixture(),source,max_decisions=1)
    assert r['summary']['error'] and r['summary']['ticks']==0 and not r['commands']


def test_mlp_can_fit_interior_duration_optimum_and_restore():
    # Synthetic representational test, not a measured physical preference.
    items=[dict(kind='demonstration',semantic=[[.2],[.5],[1.],[2.]],positive=[1])]
    m,curve=train(items,'semantic',epochs=250)
    scores=m.score(items[0]['semantic']);assert int(np.argmax(scores))==1
    assert MLP.restore(m.record()).score(items[0]['semantic'])==pytest.approx(scores)
    assert curve[-1]['loss']<curve[0]['loss']


def test_campaign_mode_budgets_and_frozen_seal(tmp_path):
    c=ActionCampaign.create(tmp_path/'campaign',sequence=[BRAKE]);s=ActionSession(c.path)
    s.tool('new_round',{});t=s.tool('brief',dict(rid='r000',turn=dict(actor='engineering',summary='接口验收',output={})))['turn_id']
    sid=s.tool('submit_action_semantics',dict(rid='r000',source=SOURCE,turn_id=t))['id']
    a=s.tool('test_action_policy',dict(rid='r000',semantic_id=sid,turn_id=t,max_ticks=20,max_decisions=2))
    assert a['ticks']==12 and not a['success']
    state=read(c.path/'campaign.json');assert state['used']['execution_ticks']==12 and state['used']['feature_batches']==3
    assert not state['reservations']
    with pytest.raises(ValueError,match='use action'):s.tool('fit',{})
    assert 'action_guide.md' in s.tool('knowledge',{})
    assert c.freeze()['status']=='verified'
    with pytest.raises(ValueError,match='frozen'):s.tool('inspect_action',dict(rid='r000',semantic_id=sid))


def test_imitation_rejects_failed_and_heldout_attempts(tmp_path):
    c=ActionCampaign.create(tmp_path/'campaign',sequence=[BRAKE]);c.new_round();a=c.replay('r000')
    with pytest.raises(ValueError,match='successful'):fitting.build(c,SOURCE,[('r000',a['id'])],[])
    r=c._round('r000');r['split']='test';save(c._rpath('r000')/'round.json',r)
    with pytest.raises(ValueError,match='training group'):fitting.build(c,SOURCE,[('r000',a['id'])],[])


def test_learned_exec_uses_mlp_without_nominal_search(monkeypatch):
    def fail(*a,**kw):raise AssertionError('should not call predictor')
    monkeypatch.setattr(execution,'NominalPredictor',fail)
    monkeypatch.setattr(execution.Scorer,'__init__',lambda self,*a,**kw:None)
    monkeypatch.setattr(execution.Scorer,'packet_scores',lambda self,h,p:[100 if a==dict(BRAKE,ticks=40) else 0 for a in p['actions']])
    source=SOURCE.replace("else 8)]","else 8),dict(target_id=None,a0=44,a1=44,ticks=40)]")
    r=execution.run(X.seed_fixture(),source,mode='learned',fitted={},max_ticks=80,max_decisions=2)
    assert r['commands']==[dict(BRAKE,ticks=40)]*2
    assert [d['real_tick'] for d in r['decisions']]==[0,40]


def test_two_hidden_layer_head_can_exactly_embed_cap_policy():
    """Constructive expressivity, not an optimizer success claim."""
    import torch
    h=history();packet=Program(SOURCE).evaluate(h);dimension=len(packet['semantic'][0])
    m=MLP(np.zeros(dimension),np.ones(dimension))
    with torch.no_grad():
        for p in m.net.parameters():p.zero_()
        m.net[0].weight[0,len(packet['names'])]=1. # exact-first-proposal indicator
        m.net[2].weight[0,0]=1.
        m.net[4].weight[0,0]=1.
    fitted=dict(semantic_sha256=digest(SOURCE),models={'S':dict(checkpoint=m.record())})
    proposal=execution.run(X.seed_fixture(),SOURCE,max_ticks=40,max_decisions=3)
    ranked=execution.run(X.seed_fixture(),SOURCE,mode='learned',fitted=fitted,condition='S',max_ticks=40,max_decisions=3)
    assert proposal['commands']==ranked['commands']
    assert [r['history_hash'] for r in proposal['public_rows']]==[r['history_hash'] for r in ranked['public_rows']]
