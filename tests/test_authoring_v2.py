"""Budget, counterfactual replay, evidence boundary and real request projection checks."""
from pathlib import Path
from copy import deepcopy
import json,time
import pytest
from experiments.planar_tasks.authoring_v2 import create_session,open_session,verify_frozen,campaign_config
from experiments.planar_tasks.authoring_v2 import release,driver,context,reports
from experiments.planar_tasks.authoring_v2.profile import DEFAULT_PROFILE
from experiments.planar_tasks.baseline import seal
from experiments.planar_tasks.baseline.prediction import LIMITS
from experiments.planar_tasks.option_search_v3 import session as search_session
from experiments.planar_tasks.baseline import session as free_session

MODEL=dict(provider='synthetic',name='no-network',settings={},source_access='hidden',accounting='reserved_output_tokens_and_input_bytes')
ACT=dict(a0=44,a1=44,ticks=1)
VALID='def evaluate_state(node,memory): return 0.\ndef rank_actions(parent,candidates,memory): return dict(costs=[0.]*len(candidates))\n'


def observation(t=0):
    own=lambda x:dict(x=x,y=1.,angle=0.,vx=0.,vy=0.,omega=0.)
    return dict(protocol='test',decision=t,legal_pairs=[[44,44]],robots=[dict(robot_id=i,time=t*.05,elapsed_seconds=t*.05,tick=t,own=own(1+i+t*.01),teammate=own(2-i),execution=dict(stage='brake' if t==0 else 'move',done=False,exit_reason=None,pause_reason=None),contacting_object=t>0,contacting_static=False,contacting_teammate=False,patches=[],patch=None,goal_xy=[3.,2.],goal_radius=.15,arena=[4.,4.],scan={}) for i in range(2)])


class FakeWorld:
    def __init__(self,instance):
        self._env=self;self.total_periods=0;self.decisions=0;self.skills_used=0;self.termination=None;self.success=False
    def observe(self):return observation(self.total_periods)
    def joint_step(self,a,b,record=True):self.total_periods+=1


@pytest.fixture
def frozen(tmp_path,monkeypatch):
    monkeypatch.setattr(release,'LOCK',tmp_path/'release.json');release.freeze();return release.verify()


def config():
    c=campaign_config(MODEL,['planar_open_anchor'])
    c['budget'].update(wall_seconds=120,python_total_seconds=90,python_call_seconds=30,episode_wall_seconds=10,response_seconds=3,model_calls=4,model_call_output_tokens=512,model_output_tokens=2048)
    c.pop('finalization');return c


@pytest.fixture
def session(tmp_path,frozen):
    s=create_session(tmp_path/'session',config(),world_factory=FakeWorld)
    yield s
    s.close()


def test_large_defaults_and_deployment_limits_unchanged(session):
    c=campaign_config(MODEL)
    assert c['budget']['resets']==2000 and c['budget']['policy_tests']==100
    assert c['budget']['wall_seconds']==5400
    assert session.predictor.ledger.limits['ticks']==1000000
    for k in ('handles','decision_calls','decision_ticks','decision_cpu_seconds'):
        assert session.predictor.ledger.limits[k]==LIMITS[k]
    assert session.planner_budget.expansions==6 and session.planner_budget.prediction_ticks==480
    assert not session.state['usage'] and session.world is None


def test_hundred_branch_batch_replays_and_charges_all_prefixes(session):
    r=session.tool('run_branches',dict(task='planar_open_anchor',prefix=[ACT],candidates=[[dict(ACT,ticks=2)] for _ in range(100)]))
    assert r['ok'],r
    result=r['result'];assert result['attempted']==100 and result['completed']==100
    assert result['executed_ticks']==300 and len(result['preview'])==5
    assert len(json.dumps(r).encode())<12000
    assert session.state['usage']['resets']==100 and session.state['usage']['control_ticks']==300
    page=session.batch_rows(result['batch_id'],start=98,count=20)
    assert len(page['rows'])==2 and page['next_start'] is None
    traces=[session.read_trace(result['batch_id'],branch=i)['rows'] for i in (0,99)]
    assert traces[0]==traces[1]
    assert session.test_events(result['batch_id'],branch=0)['total']>0


def test_invalid_prefix_and_unstarted_branches_are_not_fabricated(session):
    r=session.run_branches('planar_open_anchor',[dict(ACT,a0=9)],[[ACT],[ACT]])
    assert r['errors']==2 and r['completed']==0
    rows=session.batch_rows(r['batch_id'])['rows']
    assert all(x['branch_start_row'] is None and x['estimated_object_delta'] is None for x in rows)
    expired=session.run_branches('planar_open_anchor',[],[[ACT]],deadline=time.monotonic()-1)
    assert expired['status']=='incomplete' and expired['attempted']==0
    assert session.state['usage']['resets']==2


def test_actual_test_accounting_does_not_consume_explicit_resets(session,monkeypatch):
    session.write_file('heuristic.py',VALID)
    def fake_run(task,command,directory,*args,**kwargs):
        directory.mkdir();summary=dict(status='complete',success=True,termination='success',executed_control_ticks=3,simulated_seconds=.15,error=None)
        (directory/'summary.json').write_text(json.dumps(summary));return summary
    monkeypatch.setattr(search_session,'run_episode',fake_run)
    result=session.test_policy(['heuristic.py'],'planar_open_anchor',1800)
    assert result['overview']['summary']['success']
    assert session.state['usage']['control_ticks']==3
    assert session.state['usage']['policy_tests']==1 and session.state['usage']['policy_resets']==1
    assert session.state['usage'].get('resets',0)==0
    def failure(*args,**kwargs):raise RuntimeError('no trustworthy summary')
    monkeypatch.setattr(search_session,'run_episode',failure)
    with pytest.raises(RuntimeError):session.test_policy(['heuristic.py'],'planar_open_anchor',20)
    assert session.state['usage']['control_ticks']==23  # failed unknown work retains reservation


def test_script_batch_rpc_and_actual_wall_settlement(session):
    session.write_file('experiments/batch.py',"import sys\nsys.path.insert(0, '/work')\nfrom batch_client import BatchEnv\nr=BatchEnv().branches('planar_open_anchor',[],[[dict(a0=44,a1=44,ticks=1)]])\nprint(r['completed'])\n")
    r=session.run_python('experiments/batch.py',10)
    assert r['status']=='ok',r
    assert '1' in r['stdout']
    assert session.state['usage']['control_ticks']==1
    assert session.state['usage']['python_total_seconds']==pytest.approx(r['wall_seconds'])
    assert session.state['usage']['python_total_seconds']<10


def test_evidence_projection_does_not_leak_future_or_hidden_fields(session):
    ident='test_0000';session.state['tests'].append(dict(id=ident,task='planar_open_anchor',status='finished'))
    p=session.path/'private'/ident/'episode';p.mkdir(parents=True)
    obs=[observation(),observation(2)]
    (p/'sensors.jsonl').write_text(''.join(json.dumps(dict(request=dict(observation=o)))+'\n' for o in obs))
    (p/'summary.json').write_text(json.dumps(dict(status='complete',success=False,final_goal_distance_diagnostic='SECRET',true_state='SECRET')))
    before=session.test_summary(ident)
    for o in obs:
        o['true_state']='SECRET'
        for r in o['robots']:
            r['own']['hidden_parameter']='SECRET';r['execution']['release']='SECRET';r['scan']['raw_labels']='SECRET'
    (p/'sensors.jsonl').write_text(''.join(json.dumps(dict(request=dict(observation=o)))+'\n' for o in obs))
    (p/'truth.json').write_text('SECRET DO NOT READ')
    assert session.test_summary(ident)==before
    data=session.read_trace(ident,fields=['own','execution','scan','contacts'])
    assert 'SECRET' not in json.dumps(data)
    with pytest.raises(ValueError):session.read_trace(ident,fields=['true_state'])
    with pytest.raises(ValueError):session.read_trace('../../truth')
    assert before['observations']['sampling_note']


def test_large_results_are_archived_with_error_history_and_retrievable(session):
    session.write_file('notes/large.txt','x'*20000)
    r=session.tool('read_file',dict(path='notes/large.txt',limit=64000))
    assert r['ok'] and r['archived'] and len(json.dumps(r).encode())<12000
    chunks=[];start=0
    while True:
        part=session.read_result(r['record_id'],start,count=8000);chunks.append(part['text'])
        if part['next_start'] is None:break
        start=part['next_start']
    assert json.loads(''.join(chunks))['response']['result']['text']=='x'*20000
    fail=session.tool('read_trace',dict(record_id='../truth'))
    assert not fail['ok']
    assert session.list_results()['rows'][-1]['ok'] is False
    with pytest.raises(ValueError):session.read_result('../truth')


def test_context_projection_keeps_recent_pairs_and_archive_refs_without_mutation():
    messages=[dict(role='system',content='immutable instructions'),dict(role='user',content='current budget')]
    for i in range(12):
        messages += [dict(role='assistant',content=None,tool_calls=[dict(id=str(i),type='function',function=dict(name='read_log',arguments='{}'))]),
            dict(role='tool',tool_call_id=str(i),content=json.dumps(dict(ok=i!=3,record_id=i,error='failure evidence' if i==3 else None,result=dict(text='sensor '*5000))))]
    saved=deepcopy(messages);projected,stats=context.project(messages,DEFAULT_PROFILE,'hypothesis references record 3')
    assert messages==saved and stats['compacted']
    assert stats['projected_bytes']<stats['original_bytes']/2
    assert 'failure evidence' in json.dumps(projected) and 'record_range' in json.dumps(projected)
    ids={c['id'] for m in projected if m['role']=='assistant' for c in m['tool_calls']}
    assert all(m['tool_call_id'] in ids for m in projected if m['role']=='tool')
    assert projected[0]==messages[0] and projected[1]==messages[1]


@pytest.mark.parametrize('method',['free','option_search_v3'])
def test_driver_metered_projection_submission_and_provenance(tmp_path,frozen,monkeypatch,method):
    s=create_session(tmp_path/'run',config(),method=method,world_factory=FakeWorld)
    try:
        s.notebook('Record failed trials as well as successes.')
        name='heuristic.py' if method!='free' else 'policy.py'
        source=VALID if method!='free' else 'class Policy:\n def __init__(self,contract): pass\n def act(self,obs):return dict(a0=44,a1=44,ticks=1)\n'
        payloads=[]
        def fake(complete,payload,timeout):
            payloads.append(deepcopy(payload));assert 'Experiment authoring v2' in payload['messages'][0]['content']
            return dict(ok=True,response=dict(model='no-network',usage={},choices=[dict(finish_reason='tool_calls',message=dict(content=None,tool_calls=[dict(id='s',type='function',function=dict(name='submit',arguments=json.dumps(dict(files=[name],contents={name:source}))))]))]))
        monkeypatch.setattr(driver,'bounded_completion',fake)
        assert driver.drive(s,None)['status']=='submitted'
        assert s.state['usage']['model_input_bytes']==sum(len(json.dumps(p,ensure_ascii=False).encode()) for p in payloads)
        assert verify_frozen(s.path)['experiment_release_sha256']==seal.digest(release.LOCK)
    finally:s.close()
    opened=open_session(tmp_path/'run');opened.close()


def test_unicode_archive_pages_and_prediction_event_labels(session):
    session.write_file('notes/unicode.txt','接触结果🙂'*100)
    r=session.tool('read_file',dict(path='notes/unicode.txt',limit=64000))
    start=0;parts=[]
    while True:
        p=session.read_result(r['record_id'],start,count=31);parts.append(p['text'])
        if p['next_start'] is None:break
        start=p['next_start']
    assert json.loads(''.join(parts))['response']['result']['text']=='接触结果🙂'*100
    traces=[dict(prediction_feedback=dict(duration_comparable=True,position_residual_m=.25,true_state='SECRET')),
            dict(prediction_feedback=dict(duration_comparable=False,position_residual_m=999))]
    events=reports.prediction_events(traces)
    assert events[0]['position_residual_m']==.25 and 'position_residual_m' not in events[1]
    assert 'SECRET' not in json.dumps(events)
    assert reports.search_summary(traces)['comparable_prediction_samples']==1


def test_protected_profile_and_materials_reject_tampering(session):
    with pytest.raises(ValueError):session.write_file('experiment_api.md','override')
    (session.workspace/'experiment_profile.json').write_text('{}')
    with pytest.raises(ValueError,match='material changed'):session._active()


def test_driver_compacts_actual_request_and_meters_projected_payload(session,monkeypatch):
    payloads=[]
    def fake(complete,payload,timeout):
        payloads.append(deepcopy(payload));n=len(payloads)
        call=dict(id=str(n),type='function',function=dict(name='list_files',arguments='{}')) if n==1 else dict(id=str(n),type='function',function=dict(name='submit',arguments=json.dumps(dict(files=['heuristic.py'],contents={'heuristic.py':VALID}))))
        return dict(ok=True,response=dict(model='no-network',usage={},choices=[dict(finish_reason='tool_calls',message=dict(content='x'*180000 if n==1 else None,tool_calls=[call]))]))
    monkeypatch.setattr(driver,'bounded_completion',fake)
    assert driver.drive(session,None)['status']=='submitted'
    assert len(payloads)==2
    assert 'HOST_CONTEXT_ARCHIVE_NOTICE' in json.dumps(payloads[1])
    assert len(json.dumps(payloads[1]['messages']).encode())<=DEFAULT_PROFILE['context_max_bytes']
    assert session.state['usage']['model_input_bytes']==sum(len(json.dumps(p,ensure_ascii=False).encode()) for p in payloads)
    assert session.list_results()['rows'][0]['tool']=='list_files'
    verify_frozen(session.path)
