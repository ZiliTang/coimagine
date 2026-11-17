"""Autonomous tool decisions and freeze gates; synthetic completion, no model API."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_baseline_session import campaign, SyntheticWorld
from experiments.planar_tasks.baseline import driver, session as session_module, seal


def reply(name,args,identifier='1'):
    return {'model':'unused','usage':{'prompt_tokens':7,'completion_tokens':11},'choices':[
        {'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'tool_calls':[
            {'id':identifier,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}}]}


def test_model_chooses_experiment_then_explicit_submission(campaign,monkeypatch):
    script='from client import DevelopmentEnv\ne=DevelopmentEnv()\nprint(e.reset("beta")["observation"]["decision"])\nprint(e.step({"a0":1,"a1":2,"ticks":2})["observation"]["decision"])\n'
    policy='class Policy:\n def __init__(self,contract): pass\n def act(self,obs): return {"a0":1,"a1":2,"ticks":1}\n'
    sequence=[reply('write_file',{'path':'probe.py','text':script}),
              reply('run_python',{'path':'probe.py','seconds':2}),
              reply('write_file',{'path':'policy.py','text':policy}),
              reply('submit',{'files':['policy.py']})]
    campaign.budget=type(campaign.budget)(**{**campaign.budget.as_dict(),'model_input_bytes':1000000,
        'model_output_tokens':1000,'model_call_output_tokens':100})
    requests=[]
    def complete(fn,payload,timeout):
        requests.append(json.loads(json.dumps(payload)))
        return {'ok':True,'response':sequence.pop(0)}
    monkeypatch.setattr(driver,'bounded_completion',complete)
    result=driver.drive(campaign,None)
    assert result['status']=='submitted'
    assert campaign.world.instance=='selected-beta'
    assert len(campaign.world.calls)==2
    assert result['usage']['model_calls']==4
    assert result['usage']['model_output_tokens']==400
    assert len(requests)==4
    assert requests[2]['messages'][-1]['role']=='tool'
    assert json.loads(requests[2]['messages'][-1]['content'])['result']['status']=='ok'
    manifest=session_module.verify_frozen(campaign.path,campaign.release_path)
    assert manifest['selection']=='agent_explicit_submission'
    (campaign.path/'frozen/package/policy.py').write_text('tampered')
    with pytest.raises(ValueError,match='hash changed'):
        session_module.verify_frozen(campaign.path,campaign.release_path)


def test_transport_failure_consumes_reservation_and_does_not_retry(campaign,monkeypatch):
    campaign.budget=type(campaign.budget)(**{**campaign.budget.as_dict(),'model_input_bytes':1000000})
    calls=[]
    def failed(*args):calls.append(1);return {'ok':False,'error':'timeout','billing':'unknown'}
    monkeypatch.setattr(driver,'bounded_completion',failed)
    result=driver.drive(campaign,None)
    assert len(calls)==1 and result['usage']['model_calls']==1
    assert result['usage']['model_output_tokens']==100
    assert result['status']=='stopped_without_submission'
    assert not (campaign.path/'frozen').exists()
    with pytest.raises(ValueError,match='already attempted'):driver.drive(campaign,None)


def test_actual_completion_worker_stops_on_deadline():
    def blocked(payload,timeout):time.sleep(5)
    started=time.monotonic();result=driver.bounded_completion(blocked,{},.1)
    assert time.monotonic()-started<2 and result['error']=='model_wall_timeout'


def test_release_lock_replacement_is_not_silent(campaign):
    campaign.release_path.write_text('{"different":"valid-but-different-lock"}')
    with pytest.raises(ValueError,match='release changed'):campaign._active()


def test_release_detects_source_drift(tmp_path,monkeypatch):
    f=tmp_path/'code.py';f.write_text('version=1')
    monkeypatch.setattr(seal,'ROOT',tmp_path)
    monkeypatch.setattr(seal,'sources',lambda:[f])
    monkeypatch.setattr(seal,'environment',lambda:{'test':'environment'})
    from experiments.planar_tasks.baseline import VERSION
    lock=tmp_path/'release.json'
    lock.write_text(json.dumps({'version':VERSION,'files':{'code.py':seal.digest(f)},'environment':{'test':'environment'}}))
    assert seal.verify_release(lock)['version']==VERSION
    f.write_text('version=2')
    with pytest.raises(ValueError,match='release hash changed'):seal.verify_release(lock)


def test_runner_tick_cap_cannot_be_bypassed_by_batch(tmp_path,monkeypatch):
    from experiments.planar_tasks import runner
    from experiments.planar_tasks.tasks import task_catalog
    class Env:
        termination=None;success=False;total_periods=0;reward_total=0;rows=[]
        config=SimpleNamespace(time_budget=90.,as_dict=lambda:{})
        def joint_step(self,*args,**kwargs):self.total_periods+=1
        def object_distance(self):return 1.
    env=Env()
    world=SimpleNamespace(_env=env,decisions=0,observe=lambda:{'legal_pairs':[[44,44]]})
    class Policy:
        last_response=None
        def __init__(self,*args):pass
        def request(self,*args):return {'a0':44,'a1':44,'ticks':40}
        def close(self):pass
    monkeypatch.setattr(runner,'TaskEnvironment',lambda instance:world)
    monkeypatch.setattr(runner,'JsonlPolicy',Policy)
    monkeypatch.setattr(runner,'source_manifest',lambda *args:{})
    monkeypatch.setattr(runner,'replay_record',lambda *args:{})
    out=runner.run_episode(task_catalog()['planar_open_anchor'],['mock'],tmp_path/'episode',max_control_ticks=3)
    assert env.total_periods==3
    assert out['termination']=='control_tick_budget_exhausted'
    assert not out['success'] and out['task_cost_seconds']==180


def test_recorded_rpc_is_queryable_without_truth(campaign):
    campaign.write_file('query.py','from client import DevelopmentEnv\ne=DevelopmentEnv()\nprint(e.reset("alpha")["terminated"])\n')
    result=campaign.tool('run_python',{'path':'query.py','seconds':2})
    assert result['ok'] and result['result']['status']=='ok'
    log=campaign.tool('read_log',{'test_id':result['result']['run_id'],'stream':'rpc','count':1})
    assert log['ok'] and log['result'][0]['request']['method']=='reset'
    assert 'PRIVATE_GEOMETRY' not in json.dumps(log)


def test_deepseek_adapter_only_calls_transport_not_legacy_synthesis(monkeypatch):
    from experiments.planar_tasks.baseline import transports
    sent=[]
    def request(payload,timeout):
        sent.append((payload,timeout))
        return {'ok':True,'response':reply('list_files',{})}
    monkeypatch.setattr(transports.deepseek_agent,'request',request)
    monkeypatch.setattr(transports.deepseek_agent,'synthesize',lambda *args:pytest.fail('legacy synthesis called'))
    payload={'tools':driver.TOOLS,'messages':[],'model':'unused'}
    result=transports.deepseek_complete(payload,3)
    assert result['choices'][0]['message']['tool_calls'][0]['function']['name']=='list_files'
    assert sent==[(payload,3)]


def test_smoke_profile_cannot_create_real_simulator(campaign):
    campaign.world_factory=session_module.TaskEnvironment
    with pytest.raises(ValueError,match='forbids real simulator'):
        campaign.rpc({'method':'reset','task':'alpha'},time.monotonic()+2)
    result=campaign.tool('test_policy',{'files':['policy.py'],'task':'alpha','max_ticks':1})
    assert not result['ok'] and 'forbids real physics' in result['error']['message']


def test_policy_adapter_is_memoryful_per_episode_but_fresh_between_processes(tmp_path):
    from experiments.planar_tasks.policy import JsonlPolicy
    package=tmp_path/'package';package.mkdir()
    (package/'__policy_host__.py').write_bytes((session_module.BASE/'policy_host.py').read_bytes())
    (package/'policy.py').write_text('''import os
class Policy:
 def __init__(self,contract):
  self.count=99 if os.path.exists('/tmp/episode_marker') else 0
  open('/tmp/episode_marker','w').write('current episode')
 def act(self,observation):
  self.count+=1
  return dict(a0=self.count,a1=2,ticks=1)
''')
    for index in range(2):
        client=session_module.JsonlPolicy if hasattr(session_module,'JsonlPolicy') else JsonlPolicy
        instance=client(session_module.package_command(package),tmp_path/f'stderr_{index}')
        try:
            first=instance.request({'reset':True,'contract':{},'observation':{}},3)
            second=instance.request({'observation':{}},3)
            assert first['a0']==1 and second['a0']==2
        finally:instance.close()


def test_completion_timeout_terminates_descendant(tmp_path):
    import os
    import subprocess
    pidfile=tmp_path/'child.pid'
    def spawning(payload,timeout):
        child=subprocess.Popen(['/usr/bin/python3','-c','import time; time.sleep(10)'])
        pidfile.write_text(str(child.pid))
        time.sleep(10)
    result=driver.bounded_completion(spawning,{},.3)
    assert result['error']=='model_wall_timeout'
    assert pidfile.exists()
    pid=int(pidfile.read_text())
    proc=Path(f'/proc/{pid}/stat')
    for _ in range(50):
        if not proc.exists() or proc.read_text().split()[2]=='Z':break
        time.sleep(.01)
    assert not proc.exists() or proc.read_text().split()[2]=='Z'
