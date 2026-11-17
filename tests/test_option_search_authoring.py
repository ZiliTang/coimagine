"""Restricted authoring, two scoring hooks, immutable packaging and accounting."""
from pathlib import Path
from copy import deepcopy
import json
import time
import pytest
from experiments.planar_tasks.option_search_v2 import build
from experiments.planar_tasks.option_search_v2.session import Session,verify_frozen
from experiments.planar_tasks.option_search_v2.search import Search,Budget,action_space
from experiments.planar_tasks.baseline import seal

VALID='def evaluate_nodes(context,nodes,memory): return [0.]*len(nodes)\ndef score_actions(context,nodes,memory): return [0.]*len(nodes)\n'

@pytest.fixture
def session(tmp_path,monkeypatch):
    # Development-time fixture lock; production verify is separately checked after freeze.
    lock=tmp_path/'release.json';lock.write_text('{}')
    monkeypatch.setattr(build,'LOCK',lock)
    monkeypatch.setattr(build,'verify',lambda: seal.verify_release())
    config=json.loads((seal.BASE/'development_config.json').read_text())
    config['model']=dict(provider='synthetic',name='no-network',settings={},source_access='hidden',accounting='reserved_output_tokens_and_input_bytes')
    config['prediction']=dict(version='nominal-v1',runtime=True)
    config['budget'].update(wall_seconds=90,tool_calls=100,control_ticks=4,policy_tests=3,resets=4,python_call_seconds=3,python_total_seconds=6,response_seconds=3,episode_wall_seconds=8)
    config.pop('finalization',None);config.pop('submission_settings',None)
    s=Session.create(tmp_path/'session',config,planner_budget=Budget(expansions=1,prediction_ticks=4,prediction_calls=3,search_seconds=1))
    yield s
    s.close()


def test_separate_action_priority_changes_expansion_not_direct_output():
    obs=dict(legal_pairs=[[3,3],[44,44]])
    class Scorer:
        def begin(self,*a):pass
        def register(self,*a):pass
        def score(self,nodes,deadline):
            return dict(values=[0 if n['plan'][0]['a0']==3 else 10 for n in nodes],
                        actions=[-100 if n['plan'][0]['a0']==44 else 0 for n in nodes])
    class Model:
        def evaluate(self,plan,deadline):return dict(ticks=plan[0]['ticks'],termination='terminal',observation=obs)
    a,t=Search(Scorer(),Budget(expansions=1)).choose(obs,{},Model())
    assert t['events'][0]['plan'][0]['a0']==44
    assert a==dict(a0=3,a1=3,ticks=1)
    assert t['root_action_count']==80


def test_every_legal_action_remains_selectable_with_two_hooks():
    obs=dict(legal_pairs=[[3,3],[34,34],[40,46],[44,44]])
    class Scorer:
        def begin(self,*a):pass
        def register(self,*a):pass
        def score(self,nodes,deadline):return dict(values=[-10000 if n['plan'][0]==wanted else 0 for n in nodes],actions=[0]*len(nodes))
    class Reject:
        def evaluate(self,*a):raise ValueError('cold model rejects')
    for action in action_space(obs):
        wanted=action.packet()
        chosen,trace=Search(Scorer(),Budget(expansions=1)).choose(obs,{},Reject())
        assert chosen==wanted and trace['selected_status']=='model_error'


def test_only_hooks_are_editable_and_deployable(session):
    for name in ['policy.py','search.py','planner_config.json','framework.json','../heuristic.py']:
        assert not session.tool('write_file',dict(path=name,text='bad'))['ok']
    assert session.tool('write_file',dict(path='heuristic.py',text=VALID))['ok']
    for files in [['policy.py'],['heuristic.py','helper.py'],['../heuristic.py']]:
        assert not session.tool('submit',dict(files=files))['ok']
    assert session.world is None and session.state['usage'].get('control_ticks',0)==0


def test_workspace_core_tampering_does_not_enter_package(session):
    (session.workspace/'search.py').write_text('raise RuntimeError("tampered")')
    (session.workspace/'framework.json').write_text('{}')
    result=session.tool('submit',dict(files=['heuristic.py'],contents={'heuristic.py':VALID}))
    assert result['ok'],result
    package=session.path/'frozen/package'
    assert (package/'search.py').read_bytes()==(build.HERE/'search.py').read_bytes()
    assert json.loads((package/'planner_config.json').read_text())['expansions']==1
    verify_frozen(session.path)
    assert not session.tool('write_file',dict(path='heuristic.py',text=VALID))['ok']


@pytest.mark.parametrize('code',[
 'def evaluate_nodes(c,n,m): return []\n',
 'def evaluate_nodes(c,n,m): return []\ndef score_actions(a): return []\n',
 'raise RuntimeError("import failure")\n',
 'while True: pass\n',
])
def test_invalid_hook_submit_can_be_corrected_without_physics(session,code):
    from dataclasses import replace
    if 'while True' in code:session.budget=replace(session.budget,response_seconds=.2)
    r=session.tool('submit',dict(files=['heuristic.py'],contents={'heuristic.py':code}))
    assert not r['ok'] and not (session.path/'frozen').exists()
    session.budget=replace(session.budget,response_seconds=3)
    r=session.tool('submit',dict(files=['heuristic.py'],contents={'heuristic.py':VALID}))
    assert r['ok'],r
    assert session.state['usage'].get('control_ticks',0)==0
    assert session.world is None


def test_real_test_uses_fixed_package_shared_prediction_ledger_and_legal_logs(session):
    task=session.config['development_tasks'][0]
    r=session.tool('test_policy',dict(files=['heuristic.py'],task=task,max_ticks=1))
    assert r['ok'],r
    assert r['result']['summary']['error'] is None,r
    assert r['result']['summary']['executed_control_ticks']==1
    assert session.state['usage']['control_ticks']==1
    assert session.state['prediction_usage']['ticks']>0
    assert r['result']['summary']['prediction_usage']['ticks']==session.state['prediction_usage']['ticks']
    tid=r['result']['test_id']
    assert session.tool('read_log',dict(test_id=tid,stream='sensors',count=1))['ok']
    assert not session.tool('read_log',dict(test_id=tid,stream='truth'))['ok']
    assert session.tool('submit',dict(files=['heuristic.py']))['ok']
    verify_frozen(session.path)


def test_failed_packaging_keeps_test_reservation(session):
    session.write_file('heuristic.py','def bad(:')
    r=session.tool('test_policy',dict(files=['heuristic.py'],task=session.config['development_tasks'][0],max_ticks=2))
    assert not r['ok'];assert session.state['usage']['control_ticks']==2
    assert session.state['usage']['policy_tests']==1
    assert session.state['tests'][0]['status']=='infrastructure_error'


def test_active_script_accesses_legal_environment_and_charges_real_steps(session):
    task=session.config['development_tasks'][0]
    script='import sys\nsys.path.insert(0,".")\nfrom client import DevelopmentEnv\nc=DevelopmentEnv()\nr=c.reset('+repr(task)+')\np=r["observation"]["legal_pairs"][0]\nprint(c.step(dict(a0=p[0],a1=p[1],ticks=1))["observation"]["decision"])\n'
    session.write_file('experiments/validate.py',script)
    r=session.tool('run_python',dict(path='experiments/validate.py',seconds=3))
    assert r['ok'],r
    assert r['result']['status']=='ok',r
    assert session.state['usage']['control_ticks']==1
    assert session.state['usage']['resets']==1


def test_submission_stage_only_allows_atomic_hooks(session):
    from test_baseline_budget import enable_reserve
    enable_reserve(session)
    session.finalize('test')
    assert not session.tool('test_policy',dict(files=['heuristic.py'],task=session.config['development_tasks'][0],max_ticks=1))['ok']
    assert not session.tool('write_file',dict(path='heuristic.py',text=VALID))['ok']
    assert session.tool('submit',dict(files=['heuristic.py'],contents={'heuristic.py':VALID}))['ok']
    verify_frozen(session.path)


def test_profile_and_submission_core_tampering_are_rejected(session):
    assert session.tool('submit',dict(files=['heuristic.py'],contents={'heuristic.py':VALID}))['ok']
    package=session.path/'frozen/package'
    (package/'search.py').write_text('# changed core')
    # Even rewriting both local manifests cannot make a changed fixed core valid.
    p=package/'package_manifest.json';m=json.loads(p.read_text());m['files_sha256']['search.py']=seal.digest(package/'search.py');p.write_text(json.dumps(m))
    p=session.path/'frozen/manifest.json';m=json.loads(p.read_text())
    for name in ('search.py','package_manifest.json'):m['files_sha256'][name]=seal.digest(package/name)
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='fixed framework modified'):verify_frozen(session.path)


def test_host_budget_change_is_rejected(session):
    p=session.path/'framework.json';m=json.loads(p.read_text());m['planner_budget']['expansions']=999;p.write_text(json.dumps(m))
    r=session.tool('submit',dict(files=['heuristic.py'],contents={'heuristic.py':VALID}))
    assert not r['ok'] and 'profile changed' in r['error']['message']


def test_mock_llm_driver_edits_tests_reads_and_submits_two_hooks(session,monkeypatch):
    from experiments.planar_tasks.option_search_v2 import driver
    calls=[('write_file',dict(path='heuristic.py',text=VALID)),
           ('test_policy',dict(files=['heuristic.py'],task=session.config['development_tasks'][0],max_ticks=1)),
           ('read_log',dict(test_id='test_0000',stream='decisions')),
           ('submit',dict(files=['heuristic.py']))]
    requests=[]
    def transport(unused,payload,timeout):
        requests.append(deepcopy(payload));name,args=calls.pop(0)
        return dict(ok=True,response=dict(model='no-network',usage=dict(total_tokens=1),choices=[dict(finish_reason='tool_calls',message=dict(content=None,tool_calls=[dict(id=str(len(requests)),type='function',function=dict(name=name,arguments=json.dumps(args)))]))]))
    monkeypatch.setattr(driver,'bounded_completion',transport)
    result=driver.drive(session,None)
    assert result['status']=='submitted',result
    assert result['usage']['model_calls']==4
    assert result['usage']['control_ticks']==1
    assert 'evaluate_nodes' in requests[0]['messages'][0]['content']
    assert all(json.loads(req['messages'][-1]['content'])['ok'] for req in requests[1:])
    verify_frozen(session.path)


@pytest.mark.parametrize('field',['values','actions'])
def test_invalid_hook_values_record_failure_without_removing_actions(field):
    class Scorer:
        def begin(self,*a):pass
        def score(self,nodes,deadline):
            r=dict(values=[0]*len(nodes),actions=[0]*len(nodes));r[field]=[float('nan')]*len(nodes);return r
    obs=dict(legal_pairs=[[44,44]])
    action,trace=Search(Scorer(),Budget(expansions=0)).choose(obs,{},None)
    assert action==dict(a0=44,a1=44,ticks=1)
    assert trace['root_action_count']==40 and trace['scoring_errors']
