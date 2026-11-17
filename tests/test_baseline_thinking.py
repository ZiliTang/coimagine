"""Stage-specific settings and reasoning continuation, without paid calls."""
from copy import deepcopy
import json
import pytest
from test_baseline_session import campaign
from test_baseline_budget import enable_reserve, POLICY
from test_baseline_driver import reply
from experiments.planar_tasks.baseline import driver, preflight, seal
from experiments.planar_tasks.baseline.config import validate_config
from experiments.planar_tasks.baseline.session import BASE


def test_high_development_then_declared_non_thinking_submission(campaign, monkeypatch):
    enable_reserve(campaign)
    campaign.config['model'].update(provider='deepseek', settings={
        'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'})
    campaign.config['submission_settings'] = {'thinking': {'type': 'disabled'}, 'reasoning_effort': 'none'}
    campaign.budget = validate_config(campaign.config)
    path = campaign.path/'session.json'
    path.write_text(json.dumps(campaign.config))
    campaign.state['config_sha256'] = seal.digest(path)
    campaign._save()
    first = reply('list_files', {})
    first['choices'][0]['message']['reasoning_content'] = 'synthetic continuity marker'
    responses = [first, reply('list_files', {}),
                 reply('submit', {'files': ['policy.py'], 'contents': {'policy.py': POLICY}})]
    requests = []
    def complete(fn, payload, timeout):
        requests.append(deepcopy(payload))
        return {'ok': True, 'response': responses.pop(0)}
    monkeypatch.setattr(driver, 'bounded_completion', complete)
    assert driver.drive(campaign, None)['status'] == 'submitted'
    assert len(requests) == 3
    for request in requests[:2]:
        assert request['thinking']['type'] == 'enabled'
        assert request['reasoning_effort'] == 'high'
        assert request['tool_choice'] == 'auto'
    assert requests[2]['thinking']['type'] == 'disabled'
    assert requests[2]['reasoning_effort'] == 'none'
    assert requests[2]['tool_choice']['function']['name'] == 'submit'
    for request in requests[1:]:
        assert any(m.get('reasoning_content') == 'synthetic continuity marker' for m in request['messages'])


def test_high_requires_explicit_compatible_submission_settings():
    c = json.loads((BASE/'flash_high_config.json').read_text())
    validate_config(c)
    bad = deepcopy(c); bad.pop('submission_settings')
    with pytest.raises(ValueError, match='disabled thinking'): validate_config(bad)
    bad = deepcopy(c); bad['submission_settings']['model'] = 'different-model'
    with pytest.raises(ValueError, match='only override'): validate_config(bad)
    bad = deepcopy(c); bad.pop('finalization')
    with pytest.raises(ValueError, match='requires finalization'): validate_config(bad)


def test_high_preflight_uses_auto_native_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.seal, 'verify_release', lambda path: {})
    lock = tmp_path/'lock'; lock.write_text('{}')
    def complete(fn, payload, timeout):
        assert payload['thinking']['type'] == 'enabled'
        assert payload['reasoning_effort'] == 'high'
        assert payload['tool_choice'] == 'auto'
        r = reply('protocol_echo', {'value': 'coimagine'})
        r['model'] = payload['model']
        return {'ok': True, 'response': r}
    monkeypatch.setattr(preflight, 'bounded_completion', complete)
    c = json.loads((BASE/'flash_high_config.json').read_text())
    assert preflight.preflight(c, tmp_path/'result', release_path=lock)['status'] == 'passed'


def test_truncated_development_uses_reserved_delivery_without_executing_partial_tools(campaign, monkeypatch):
    enable_reserve(campaign)
    truncated = reply('write_file', {'path': 'partial.py', 'text': 'must not be written'})
    truncated['choices'][0]['finish_reason'] = 'length'
    truncated['usage'] = {'prompt_tokens': 50, 'completion_tokens': 100, 'total_tokens': 150}
    truncated['choices'][0]['message']['reasoning_content'] = 'unfinished synthetic reasoning'
    second = reply('submit', {'files': ['policy.py'], 'contents': {'policy.py': POLICY}})
    responses = [truncated, second]; requests = []
    def complete(fn, payload, timeout):
        requests.append(deepcopy(payload))
        return {'ok': True, 'response': responses.pop(0)}
    monkeypatch.setattr(driver, 'bounded_completion', complete)
    result = driver.drive(campaign, None)
    assert result['status'] == 'submitted' and result['usage']['model_calls'] == 2
    assert result['reported_model_usage'][0]['completion_tokens'] == 100
    assert campaign.state['finalization_reason'] == 'model_output_token_cap_reached'
    assert not (campaign.workspace/'partial.py').exists()
    assert requests[1]['tool_choice']['function']['name'] == 'submit'
    assert [t['function']['name'] for t in requests[1]['tools']] == ['submit']
    assert 'unfinished synthetic reasoning' not in json.dumps(requests[1])


def test_truncated_submission_never_extends_two_attempt_allowance(campaign, monkeypatch):
    enable_reserve(campaign); campaign.finalize('test')
    requests = []
    def complete(fn, payload, timeout):
        requests.append(deepcopy(payload))
        response = reply('submit', {'files': ['policy.py'], 'contents': {'policy.py': POLICY}})
        response['choices'][0]['finish_reason'] = 'length'
        return {'ok': True, 'response': response}
    monkeypatch.setattr(driver, 'bounded_completion', complete)
    result = driver.drive(campaign, None)
    assert result['status'] == 'stopped_without_submission'
    assert len(requests) == 2 and result['usage']['model_calls'] == 2
    assert not (campaign.path/'frozen').exists()
    assert len(result['reported_model_usage']) == 2
