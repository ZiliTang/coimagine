"""Action-language support, multi-step search, and isolated hook contracts."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import time
import pytest
from experiments.planar_tasks.option_search.search import Search, Budget, action_space, validate
from experiments.planar_tasks.option_search.model import ReplayModel
from experiments.planar_tasks.option_search.hook_client import HookClient
from experiments.planar_tasks.interface import validate_action


OBS=dict(decision=0,legal_pairs=[[3,3],[34,34],[40,46],[44,44]],robots=[dict(goal_xy=[0,0])])


class Scorer:
    def __init__(self,fn):self.fn=fn;self.predictions={};self.context=None
    def begin(self,context,deadline):self.context=context;self.predictions={}
    def register(self,i,p,deadline):self.predictions[i]=p
    def score(self,records,deadline):
        return [self.fn(dict(n,prediction=self.predictions.get(n['id']),parent_prediction=self.predictions.get(n['parent_id'])),self.context) for n in records]


class Reject:
    def evaluate(self,plan,deadline):raise ValueError('model rejects a real-legal action')


def target_scorer(target):
    return Scorer(lambda n,c:-10000 if n['plan'][0]==target else 10000)


def test_action_language_equals_existing_validator_all_pairs_and_durations():
    actions=list(action_space(OBS));assert len(actions)==160
    assert len(set(actions))==160
    for a in actions:
        validate_action(a.packet(),OBS);validate(a.packet(),OBS)
    for a in (dict(a0=3,a1=3,ticks=0),dict(a0=3,a1=3,ticks=41),dict(a0=3,a1=3,ticks=True),dict(a0=999,a1=3,ticks=1)):
        with pytest.raises(ValueError):validate(a,OBS)
        with pytest.raises(ValueError):validate_action(a,OBS)


@pytest.mark.parametrize('expansions',[0,1])
def test_every_real_legal_command_can_actually_be_selected_even_if_model_rejects(expansions):
    for action in action_space(OBS):
        chosen,trace=Search(target_scorer(action.packet()),Budget(expansions=expansions)).choose(OBS,{},Reject())
        assert chosen==action.packet()
        assert trace['root_action_count']==160
        if expansions:assert trace['selected_status']=='model_error'


def test_feedback_branching_repeated_options_and_durations_preserved():
    transcript=[dict(OBS,decision=i,legal_pairs=p) for i,p in enumerate([
        [[3,3],[44,44]],[[3,3],[46,3]],[[34,34],[46,3]],[[34,34],[44,44]]])]
    free=[dict(a0=3,a1=3,ticks=1),dict(a0=3,a1=3,ticks=7),dict(a0=34,a1=34,ticks=40),dict(a0=34,a1=34,ticks=1)]
    for obs,expected in zip(transcript,free):
        action,_=Search(target_scorer(expected),Budget(expansions=1)).choose(obs,dict(observation=obs),Reject())
        assert action==expected


def test_multistep_can_choose_initial_motion_away_from_goal():
    obs=dict(OBS,legal_pairs=[[3,3],[44,44]])
    class Model:
        def __init__(self):self.plans=[]
        def evaluate(self,plan,deadline):
            self.plans.append(plan)
            distance=0 if len(plan)>1 and plan[0]['a0']==3 and plan[-1]['a0']==44 else 2
            return dict(state=dict(object=dict(x=distance,y=0)),observation=obs,ticks=sum(a['ticks'] for a in plan),termination='success' if distance==0 else None)
    def h(n,c):
        if n['prediction']:return n['prediction']['state']['object']['x']
        if any(a['ticks']!=1 for a in n['plan']):return 100
        if len(n['plan'])>1:return 0 if n['plan'][-1]['a0']==44 else 10
        return 1
    model=Model();action,trace=Search(Scorer(h),Budget(expansions=2)).choose(obs,{},model)
    assert action==dict(a0=3,a1=3,ticks=1)
    assert len(trace['selected_plan'])==2
    assert trace['events'][0]['score']>trace['selected_score']
    assert len(model.plans[1])==2


def test_no_merging_equal_poses_with_different_action_histories():
    obs=dict(OBS,legal_pairs=[[3,3]])
    class Model:
        def evaluate(self,plan,deadline):return dict(state=dict(object=dict(x=1,y=0)),observation=obs,ticks=sum(a['ticks'] for a in plan),termination=None)
    action,t=Search(Scorer(lambda n,c:0),Budget(expansions=3)).choose(obs,{},Model())
    assert len(t['events'])==3
    assert [len(e['plan']) for e in t['events']]==[1,2,3]
    assert t['nodes_materialized']==160


def test_fixed_budget_and_depth_do_not_remove_executable_roots():
    wanted=dict(a0=34,a1=34,ticks=40)
    action,t=Search(target_scorer(wanted),Budget(expansions=10,prediction_ticks=1,prediction_calls=3,max_depth=1)).choose(OBS,{},Reject())
    assert action==wanted
    assert t['prediction_ticks_reserved']<=1 and t['prediction_calls_reserved']<=3
    assert t['root_action_count']==160


@pytest.mark.parametrize('value',[float('nan'),float('inf'),'bad'])
def test_invalid_heuristic_is_reported_and_action_still_original_legal(value):
    action,t=Search(Scorer(lambda n,c:value),Budget(expansions=0)).choose(OBS,{},Reject())
    validate_action(action,OBS);assert t['scoring_errors']


def test_replay_model_retains_every_prefix_command_and_accounts_calls():
    class Client:
        def __init__(self):self.log=[]
        def create(self,h):self.log.append(('create',h));return dict(handle='own',termination=None)
        def step(self,h,a):self.log.append(('step',h,a));return dict(handle=h,termination=None)
        def close(self,h):self.log.append(('close',h))
    client=Client();model=ReplayModel(client,{'synthetic':'only'})
    plan=[dict(a0=3,a1=3,ticks=7),dict(a0=3,a1=3,ticks=13),dict(a0=34,a1=34,ticks=1)]
    model.evaluate(plan,time.monotonic()+2)
    assert [r[2] for r in client.log if r[0]=='step']==plan
    assert model.calls==5 and model.requested_ticks==21


@pytest.mark.parametrize('code,works',[
    ('def score(context,nodes,memory):\n print("worker output")\n for n in nodes: n["plan"][0]["a0"]=999\n return [-1.]*len(nodes)\n',True),
    ('def score(context,nodes,memory):\n while True: pass\n',False),
])
def test_worker_mutation_or_hang_cannot_change_parent_actions(tmp_path,monkeypatch,code,works):
    here=Path('experiments/planar_tasks/option_search').resolve()
    shutil.copyfile(here/'hook_worker.py',tmp_path/'hook_worker.py')
    (tmp_path/'heuristic.py').write_text(code)
    monkeypatch.chdir(tmp_path)
    hook=HookClient();before=deepcopy(OBS)
    try:
        start=time.monotonic()
        action,t=Search(hook,Budget(expansions=0,search_seconds=.3)).choose(OBS,dict(observation=OBS),Reject())
        assert OBS==before;validate_action(action,OBS)
        assert bool(t['scoring_errors'])!=works
        assert time.monotonic()-start<2
    finally:hook.close()


def test_package_verifier_rejects_rewritten_core_even_with_matching_manifest(tmp_path,monkeypatch):
    from experiments.planar_tasks.option_search import build
    monkeypatch.setattr(build,'verify',lambda: {})
    lock=tmp_path/'release.json';lock.write_text('{}')
    monkeypatch.setattr(build,'LOCK',lock)
    package=tmp_path/'package'
    build.build_package(package)
    build.verify_package(package)
    (package/'search.py').write_text('# tampered core\n')
    manifest=json.loads((package/'package_manifest.json').read_text())
    manifest['files_sha256']['search.py']=build.seal.digest(package/'search.py')
    (package/'package_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='fixed framework modified'):
        build.verify_package(package)
