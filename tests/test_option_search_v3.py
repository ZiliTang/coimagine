"""Local candidate semantics, temporal lookahead, lazy support and authoring gates."""
from pathlib import Path
from copy import deepcopy
import json
import sys
import time
import pytest
from experiments.planar_tasks.option_search_v3.search import Search,Budget,action_space,validate
from experiments.planar_tasks.option_search_v3.model import PrefixModel
from experiments.planar_tasks.option_search_v3 import build
from experiments.planar_tasks.option_search_v3.session import Session,verify_frozen
from experiments.planar_tasks.option_search_v3.runtime_policy import JsonlPolicy
from experiments.planar_tasks.interface import validate_action

OBS=dict(decision=0,legal_pairs=[[3,3],[34,34],[40,46],[44,44]],robots=[dict(goal_xy=[0,0])])
CTX=dict(observation=OBS,hypothesis=None,previous_action=None)
VALID='def evaluate_state(node,memory): return 0.\ndef rank_actions(parent,candidates,memory): return dict(costs=[0.]*len(candidates),proposals=[])\n'

class Scorer:
    def __init__(self,rank=None,value=None):
        self.rankfn=rank or (lambda p,c:dict(costs=[0.]*len(c)))
        self.valuefn=value or (lambda p:0.)
        self.nodes={};self.seen=[]
    def begin(self,context,deadline):pass
    def register(self,i,node,deadline):self.nodes[i]=deepcopy(node);return self.valuefn(node)
    def rank(self,i,candidates,deadline):
        self.seen.append((deepcopy(self.nodes[i]),deepcopy(candidates)))
        return self.rankfn(self.nodes[i],candidates)

class Model:
    last_reused=False
    def cost(self,parent,action):return action['ticks'],2
    def expand(self,parent,ident,action,deadline):
        self.last_reused=bool(parent['depth'])
        return dict(state=dict(object=dict(x=0,y=0),robots=[]),observation=parent['observation'],ticks=parent['elapsed_ticks']+action['ticks'],termination=None)


def test_lazy_stream_exhausts_original_language_without_duration_quantization():
    actions=list(action_space(OBS));assert len(actions)==160 and len(set(actions))==160
    assert [a.ticks for a in actions[:4]]==[40]*4
    for a in actions:validate_action(a.packet(),OBS)
    assert set(a.ticks for a in actions)==set(range(1,41))
    assert list(action_space(OBS,dict(a0=34,a1=34,ticks=7)))[0].ticks==7


@pytest.mark.parametrize('expansions',[0,1])
def test_every_command_selectable_via_nomination_despite_tiny_batch_and_model_rejection(expansions):
    class Reject(Model):
        def expand(self,*a):raise ValueError('approximate model rejects real action')
    for a in action_space(OBS):
        target=a.packet()
        rank=lambda p,c:dict(costs=[10000.]*len(c),proposals=[dict(action=target,cost=-10000.)])
        out,t=Search(Scorer(rank),Budget(expansions=expansions,candidate_batch=1,widen_batch=1)).choose(OBS,CTX,Reject())
        assert out==target;assert t['root_action_support']==160
        assert t['generated_candidates']<=3


def test_same_command_support_with_successful_predictor_and_feedback():
    previous=None
    for target in [dict(a0=3,a1=3,ticks=1),dict(a0=3,a1=3,ticks=7),dict(a0=34,a1=34,ticks=40)]:
        rank=lambda p,c:dict(costs=[10000.]*len(c),proposals=[dict(action=target,cost=-10000.)])
        scorer=Scorer(rank,value=lambda n:-10000. if n['incoming_action']==target else 10000.)
        out,t=Search(scorer,Budget(expansions=2,candidate_batch=1,widen_batch=1)).choose(OBS,dict(CTX,previous_action=previous),Model())
        assert out==target;previous=out


def test_neutral_ranking_does_not_add_short_duration_bias_and_reaches_six_seconds():
    scorer=Scorer()
    action,t=Search(scorer).choose(OBS,CTX,Model())
    assert t['maximum_forecast_ticks']==120
    assert t['suffix_ticks'] and set(t['suffix_ticks'])=={40}
    assert 'all_forecast_suffixes_one_tick' not in t['diagnostics']
    assert t['generated_candidates']<=6*24+3*8
    assert all(n['source'] in ('predicted','real_observation_estimate') for n in scorer.nodes.values())
    assert any(n['depth']==2 for n,c in scorer.seen)


def test_local_rank_changes_with_predicted_parent_and_values_back_up_beyond_retreat():
    obs=dict(OBS,legal_pairs=[[3,3],[44,44]])
    def rank(parent,candidates):
        want=3 if parent['depth']==0 else 44
        return dict(costs=[20. if a==dict(a0=want,a1=want,ticks=40) else 100. for a in candidates])
    class Retreat(Model):
        def expand(self,parent,ident,action,deadline):
            r=super().expand(parent,ident,action,deadline)
            r['state']['object']['x']=10 if parent['depth']==0 else 0
            if parent['depth']>0 and action['a0']==44:r.update(termination='success',success=True)
            return r
    scorer=Scorer(rank,value=lambda n:n['state']['object']['x'] if n['state'] else 2.)
    action,t=Search(scorer,Budget(expansions=2)).choose(obs,dict(CTX,observation=obs),Retreat())
    assert action==dict(a0=3,a1=3,ticks=40)
    assert [a['a0'] for a in t['selected_plan']]==[3,44]
    assert t['selected_cost_seconds']==4.
    assert scorer.seen[1][0]['state']['object']['x']==10


def test_deliberate_tiny_actions_remain_allowed_but_are_diagnosed():
    def rank(p,c):return dict(costs=[10. if a['ticks']==1 else 100. for a in c])
    _,t=Search(Scorer(rank),Budget(candidate_batch=160)).choose(OBS,CTX,Model())
    assert set(t['suffix_ticks'])=={1}
    assert 'all_forecast_suffixes_one_tick' in t['diagnostics']
    assert 'lookahead_time_target_not_reached' in t['diagnostics']


@pytest.mark.parametrize('reply',[dict(costs=[float('nan')]),dict(costs=[0.],proposals=[dict(action=dict(a0=999,a1=999,ticks=40),cost=-99.)])])
def test_bad_scores_or_nominations_cannot_modify_legality(reply):
    a,t=Search(Scorer(lambda p,c:reply),Budget(expansions=0,candidate_batch=1)).choose(OBS,CTX,Model())
    validate_action(a,OBS);assert t['hook_errors']


def test_prediction_limits_do_not_remove_requested_root():
    wanted=dict(a0=34,a1=34,ticks=40)
    scorer=Scorer(lambda p,c:dict(costs=[100.]*len(c),proposals=[dict(action=wanted,cost=-100.)]))
    a,t=Search(scorer,Budget(prediction_ticks=0,prediction_calls=0)).choose(OBS,CTX,Model())
    assert a==wanted;assert t['expansions']==0


class Client:
    def __init__(self):self.worlds={};self.log=[];self.serial=0
    def create(self,h):
        self.serial+=1;k=str(self.serial);self.worlds[k]=0;self.log.append(('create',k));return dict(handle=k,termination=None)
    def step(self,k,a):
        self.worlds[k]+=a['ticks'];self.log.append(('step',k,a))
        return dict(handle=k,ticks=self.worlds[k],termination=None,state={})
    def close(self,k):self.log.append(('close',k));del self.worlds[k]


def test_prefix_reuse_saves_simulation_and_sibling_reconstruction_is_fully_charged():
    client=Client();model=PrefixModel(client,{'legal_estimate':'only'})
    a=dict(a0=3,a1=3,ticks=40);root=dict(id=0,path_context=[])
    model.expand(root,1,a,time.monotonic()+1)
    parent=dict(id=1,path_context=[a]);assert model.cost(parent,a)==(40,2)
    model.expand(parent,2,a,time.monotonic()+1);assert model.last_reused and model.requested_ticks==80
    assert model.cost(parent,a)==(80,4)
    model.expand(parent,3,a,time.monotonic()+1);assert not model.last_reused and model.requested_ticks==160
    model.close();assert not client.worlds and model.calls==8


@pytest.fixture
def session(tmp_path,monkeypatch):
    lock=tmp_path/'release.json';lock.write_text('{}')
    monkeypatch.setattr(build,'LOCK',lock);monkeypatch.setattr(build,'verify',lambda:build.seal.verify_release())
    config=json.loads((build.seal.BASE/'development_config.json').read_text())
    config['model'].update(provider='synthetic',name='unused',settings={})
    config['prediction']=dict(version='nominal-v1',runtime=True)
    config['budget'].update(wall_seconds=60,response_seconds=3,episode_wall_seconds=10,control_ticks=4,policy_tests=2,resets=3)
    config.pop('finalization',None);config.pop('submission_settings',None)
    s=Session.create(tmp_path/'session',config,planner_budget=Budget(expansions=3,prediction_ticks=160,prediction_calls=12))
    yield s;s.close()


def test_old_hooks_rejected_then_new_atomic_submission_validates_without_physics(session):
    bad='def evaluate_nodes(c,n,m): return []\ndef score_actions(c,n,m): return []\n'
    r=session.tool('submit',dict(files=['heuristic.py'],contents={'heuristic.py':bad}));assert not r['ok']
    r=session.tool('submit',dict(files=['heuristic.py'],contents={'heuristic.py':VALID}));assert r['ok'],r
    verify_frozen(session.path);assert session.world is None
    assert session.state['usage'].get('control_ticks',0)==0


def test_author_cannot_replace_core_or_budget_and_test_uses_frozen_package(session):
    assert not session.tool('write_file',dict(path='search.py',text='bad'))['ok']
    (session.workspace/'search.py').write_text('bad')
    r=session.tool('test_policy',dict(files=['heuristic.py'],task=session.config['development_tasks'][0],max_ticks=1))
    assert r['ok'],r
    s=r['result']['summary'];assert s['error'] is None,s
    assert s['executed_control_ticks']==1
    assert 'process_exit' in s and s['process_exit']['harness_termination']
    assert session.state['prediction_usage']['ticks']>0
    assert session.tool('submit',dict(files=['heuristic.py']))['ok'];verify_frozen(session.path)
    p=session.path/'frozen/package';assert (p/'search.py').read_bytes()==(build.HERE/'search.py').read_bytes()
    (p/'search.py').write_text('tamper')
    with pytest.raises(ValueError):verify_frozen(session.path)


@pytest.mark.parametrize('statement,expected', [('sys.exit(7)',7),('os.kill(os.getpid(),9)',-9)])
def test_exit_code_signal_recorded_before_harness_cleanup(tmp_path,statement,expected):
    policy=JsonlPolicy([sys.executable,'-c','import os,sys;sys.stdin.readline();'+statement],tmp_path/'err')
    try:
        with pytest.raises(RuntimeError):policy.request({},3)
    finally:policy.close()
    assert policy.exit_info['returncode']==expected
    assert not policy.exit_info['harness_termination']


def test_submission_phase_and_mock_model_use_new_hook_contract(session,monkeypatch):
    from experiments.planar_tasks.option_search_v3 import driver
    from test_baseline_budget import enable_reserve
    enable_reserve(session);session.finalize('interface check')
    requests=[]
    def fake(unused,payload,timeout):
        requests.append(payload)
        return dict(ok=True,response=dict(model=session.config['model']['name'],choices=[dict(finish_reason='tool_calls',message=dict(content=None,tool_calls=[dict(id='submit',type='function',function=dict(name='submit',arguments=json.dumps(dict(files=['heuristic.py'],contents={'heuristic.py':VALID}))))]))]))
    monkeypatch.setattr(driver,'bounded_completion',fake)
    r=driver.drive(session,None)
    assert r['status']=='submitted',r
    assert 'rank_actions(parent' in requests[0]['messages'][0]['content']
    assert [t['function']['name'] for t in requests[0]['tools']]==['submit']
    assert session.state['usage'].get('control_ticks',0)==0
    verify_frozen(session.path)


def test_hung_hook_is_bounded_and_reported_without_rewriting_action_space(tmp_path,monkeypatch):
    import shutil
    from experiments.planar_tasks.option_search_v3.hook_client import HookClient
    shutil.copyfile(build.HERE/'hook_worker.py',tmp_path/'hook_worker.py')
    (tmp_path/'heuristic.py').write_text('def evaluate_state(n,m):\n while True: pass\ndef rank_actions(p,c,m): return {}\n')
    monkeypatch.chdir(tmp_path);hook=HookClient();start=time.monotonic()
    try:
        a,t=Search(hook,Budget(search_seconds=.2)).choose(OBS,CTX,Model())
        validate_action(a,OBS);assert t['hook_errors'];assert time.monotonic()-start<2
    finally:hook.close()


def test_failed_model_step_closes_its_synthetic_handle():
    class Failed(Client):
        def step(self,k,a):raise RuntimeError('synthetic step error')
    client=Failed();model=PrefixModel(client,{})
    with pytest.raises(RuntimeError):model.expand(dict(id=0,path_context=[]),1,dict(a0=3,a1=3,ticks=40),time.monotonic()+1)
    assert not client.worlds and model.calls==3 and model.requested_ticks==40


def test_nonfinite_state_value_is_diagnostic_and_output_stays_legal():
    a,t=Search(Scorer(value=lambda p:float('nan')),Budget(expansions=1)).choose(OBS,CTX,Model())
    validate_action(a,OBS);assert t['hook_errors'][0]['stage']=='evaluate_state'
