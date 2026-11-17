import json
from experiments.planar_tasks.baseline import preflight as module
from experiments.planar_tasks.baseline.session import BASE


def test_native_preflight_is_one_call_no_retries(tmp_path,monkeypatch):
    monkeypatch.setattr(module.seal,'verify_release',lambda p:{})
    lock=tmp_path/'lock';lock.write_text('{}')
    calls=[]
    def complete(fn,body,timeout):
        calls.append(body)
        assert body['tool_choice']['function']['name']=='protocol_echo' and timeout==60
        return {'ok':True,'response':{'model':body['model'],'usage':{'total_tokens':9},'choices':[
            {'finish_reason':'tool_calls','message':{'tool_calls':[{'function':{
                'name':'protocol_echo','arguments':'{"value":"coimagine"}'}}]}}]}}
    monkeypatch.setattr(module,'bounded_completion',complete)
    config=json.loads((BASE/'development_config.json').read_text())
    result=module.preflight(config,tmp_path/'ok',release_path=lock)
    assert result['status']=='passed' and result['physics_episodes']==0
    assert result['reported_usage']['total_tokens']==9 and len(calls)==1
    monkeypatch.setattr(module,'bounded_completion',lambda *args:{'ok':False,'error':'timeout'})
    result=module.preflight(config,tmp_path/'failed',release_path=lock)
    assert result['status']=='failed' and result['model_calls']==1
