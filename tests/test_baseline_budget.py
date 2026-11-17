"""Two budgeted stages and atomic explicit submission; no model API or physics."""
import json
from copy import deepcopy
import pytest
from test_baseline_session import campaign
from test_baseline_driver import reply
from experiments.planar_tasks.baseline import driver
from experiments.planar_tasks.baseline.config import validate_config
from experiments.planar_tasks.runner import write_json

POLICY='class Policy:\n def __init__(self, contract): pass\n def act(self, observation): return dict(a0=44,a1=44,ticks=1)\n'

def enable_reserve(campaign):
    config=deepcopy(campaign.config)
    config['budget'].update(model_calls=4,model_input_bytes=1000000,model_output_tokens=400,
                            model_call_output_tokens=100,tool_calls=30)
    config['finalization']=dict(model_calls=2,model_input_bytes=100000,
        model_output_tokens=200,tool_calls=8,wall_seconds=10)
    campaign.config=config;campaign.budget=validate_config(config)
    write_json(campaign.path/'session.json',config)
    from experiments.planar_tasks.baseline.seal import digest
    campaign.state['config_sha256']=digest(campaign.path/'session.json');campaign._save()


def test_two_stages_force_atomic_submission_and_allow_bounded_validation_correction(campaign,monkeypatch):
    enable_reserve(campaign)
    sequence=[reply('list_files',{}),reply('list_files',{}),
        reply('submit',{'files':['policy.py'],'contents':{'policy.py':'class Policy: pass'}}),
        reply('submit',{'files':['policy.py'],'contents':{'policy.py':POLICY}})]
    requests=[]
    def complete(fn,payload,timeout):
        requests.append(json.loads(json.dumps(payload)))
        return dict(ok=True,response=sequence.pop(0))
    monkeypatch.setattr(driver,'bounded_completion',complete)
    result=driver.drive(campaign,None)
    assert result['status']=='submitted'
    assert result['usage']['model_calls']==4 and result['usage']['model_output_tokens']==400
    assert requests[1]['tool_choice']=='auto'
    for req in requests[2:]:
        assert [t['function']['name'] for t in req['tools']]==['submit']
        assert req['tool_choice']=={'type':'function','function':{'name':'submit'}}
    assert '"finalizing": true' in requests[2]['messages'][1]['content']
    assert 'DEVELOPMENT IS CLOSED' in requests[2]['messages'][-1]['content']
    assert not json.loads(requests[3]['messages'][-1]['content'])['ok']
    assert (campaign.path/'frozen/package/policy.py').read_text()==POLICY
    assert campaign.state['submission_attempts']==2
    assert not (campaign.workspace/'policy.py').exists()  # no partial workspace writes
    assert campaign.world is None


def test_experiment_exhaustion_preserves_atomic_submission(campaign):
    enable_reserve(campaign)
    with pytest.raises(ValueError,match='control_ticks budget'):campaign._charge('control_ticks',6)
    assert campaign.state['finalizing'] and campaign.state['status']=='developing'
    for name,args in [('run_python',dict(path='x.py',seconds=1)),('write_file',dict(path='policy.py',text=POLICY))]:
        denied=campaign.tool(name,args)
        assert not denied['ok'] and 'finalization' in denied['error']['message']
    assert denied['budget']['phase_remaining']['model_calls']==2  # even if development ended early
    assert campaign.tool('submit',dict(files=['policy.py'],contents={'policy.py':POLICY}))['ok']
    assert campaign.state['usage'].get('control_ticks',0)==0


def test_rpc_cannot_spend_submission_tool_reserve(campaign):
    enable_reserve(campaign)
    campaign.state['usage']['tool_calls']=22
    with pytest.raises(ValueError,match='finalization'):campaign.rpc({'method':'reset','task':'alpha'},10**12)
    assert campaign.world is None and campaign.state['usage']['tool_calls']==22
    assert campaign.tool('submit',dict(files=['policy.py'],contents={'policy.py':POLICY}))['ok']


def test_reserve_is_within_total_and_integer(campaign):
    enable_reserve(campaign)
    for key,value in [('model_calls',4),('model_calls',1),('wall_seconds',True),('tool_calls',1.5),('model_output_tokens',100)]:
        config=deepcopy(campaign.config);config['finalization'][key]=value
        with pytest.raises(ValueError):validate_config(config)


def test_declared_development_configuration_is_consistent():
    from experiments.planar_tasks.baseline.session import BASE
    config=json.loads((BASE/'development_config.json').read_text())
    budget=validate_config(config)
    assert budget.model_calls*budget.model_call_output_tokens==budget.model_output_tokens
    assert len(config['development_tasks'])==9


def test_last_development_request_can_finish_experiment_before_phase_transition(campaign,monkeypatch):
    enable_reserve(campaign)
    script='print("last development call")'
    campaign.write_file('last.py',script)
    sequence=[reply('list_files',{}),reply('run_python',{'path':'last.py','seconds':1}),
        reply('submit',{'files':['policy.py'],'contents':{'policy.py':POLICY}})]
    results=[]
    def complete(fn,payload,timeout):
        results.append(json.loads(json.dumps(payload)))
        return dict(ok=True,response=sequence.pop(0))
    monkeypatch.setattr(driver,'bounded_completion',complete)
    assert driver.drive(campaign,None)['status']=='submitted'
    assert json.loads((campaign.path/'private/python_0000/result.json').read_text())['status']=='ok'
    assert results[1]['tool_choice']=='auto' and results[2]['tool_choice']['function']['name']=='submit'


def test_early_submission_stage_has_own_call_limit_not_all_unused_development_calls(campaign,monkeypatch):
    enable_reserve(campaign);campaign.finalize('early experimental limit')
    calls=[]
    def complete(fn,payload,timeout):
        calls.append(payload)
        return dict(ok=True,response=reply('submit',{'files':['missing.py']}))
    monkeypatch.setattr(driver,'bounded_completion',complete)
    result=driver.drive(campaign,None)
    assert len(calls)==2 and result['status']=='stopped_without_submission'
    assert not (campaign.path/'frozen').exists()


def test_submission_tool_and_wall_budgets_are_enforced(campaign):
    enable_reserve(campaign);campaign.finalize('test')
    campaign._charge('tool_calls',8)
    with pytest.raises(ValueError,match='submission tool_calls'):campaign._charge('tool_calls')
    campaign.state['status']='developing'
    campaign.state['finalization_started_monotonic']-=11
    with pytest.raises(TimeoutError):campaign._active()


def test_deepseek_thinking_mode_is_not_silently_changed(campaign):
    enable_reserve(campaign);config=deepcopy(campaign.config)
    config['model'].update(provider='deepseek',settings={'thinking':{'type':'enabled'}})
    with pytest.raises(ValueError,match='disabled thinking'):validate_config(config)


def test_tool_schema_exposes_real_limits():
    schema=next(t for t in driver.TOOLS if t['function']['name']=='test_policy')['function']['parameters']
    assert schema['properties']['max_ticks']['maximum']==1800


def test_submission_response_is_one_explicit_package_not_multiple_candidates(campaign,monkeypatch):
    enable_reserve(campaign);campaign.finalize('fixture')
    multi=reply('submit',{'files':['policy.py'],'contents':{'policy.py':POLICY}})
    second=deepcopy(multi['choices'][0]['message']['tool_calls'][0]);second['id']='2'
    multi['choices'][0]['message']['tool_calls'].append(second)
    responses=[multi,reply('submit',{'files':['policy.py'],'contents':{'policy.py':POLICY}})]
    def complete(fn,payload,timeout):return dict(ok=True,response=responses.pop(0))
    monkeypatch.setattr(driver,'bounded_completion',complete)
    assert driver.drive(campaign,None)['status']=='submitted'
    assert campaign.state['usage']['model_calls']==2
    assert campaign.state['submission_attempts']==1  # first multi-candidate response executed nothing
