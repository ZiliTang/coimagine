"""Wide reasoning requests must not retain hidden 120-second/2-MiB limits."""
import json
import pytest
from experiments.planar_tasks import deepseek_agent as transport
from experiments.planar_tasks.baseline.session import BASE
from experiments.planar_tasks.baseline.config import validate_config


def test_http_worker_honors_deadline_and_accepts_large_bounded_response(monkeypatch):
    payload = {'model': 'synthetic', 'text': 'x' * (3 * 1024 * 1024)}
    raw = json.dumps(payload).encode(); sent = []; calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit):
            assert limit == transport.MAX_RESPONSE_BYTES + 1
            return raw
    class Opener:
        def open(self, request, timeout):
            calls.append(timeout); return Response()
    class Connection:
        def send(self, value): sent.append(value)
        def close(self): pass
    monkeypatch.setattr(transport, '_credential', lambda: 'synthetic-no-network')
    monkeypatch.setattr(transport.urllib.request, 'build_opener', lambda *args: Opener())
    transport._request_worker({'model': 'synthetic'}, Connection(), 1200)
    assert calls == [1200] and sent == [{'ok': True, 'response': payload}]


def test_oversize_response_stays_bounded(monkeypatch):
    sent = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return b'x' * limit
    class Opener:
        def open(self, *args, **kwargs): return Response()
    class Connection:
        def send(self, value): sent.append(value)
        def close(self): pass
    monkeypatch.setattr(transport, '_credential', lambda: 'synthetic-no-network')
    monkeypatch.setattr(transport.urllib.request, 'build_opener', lambda *args: Opener())
    transport._request_worker({}, Connection(), 1200)
    assert sent == [{'ok': False, 'error': 'ValueError'}]


def test_wide_config_preserves_delivery_and_experiment_limits():
    c = json.loads((BASE/'flash_high_config.json').read_text()); b = validate_config(c)
    assert b.model_call_output_tokens == 131072 and b.model_call_seconds == 1200
    assert b.model_output_tokens == b.model_calls * b.model_call_output_tokens
    assert c['finalization']['model_output_tokens'] == 2 * b.model_call_output_tokens
    assert b.control_ticks == 36000 and b.resets == 40 and b.policy_tests == 20


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), True])
def test_invalid_http_deadline_fails_before_start(timeout):
    with pytest.raises(ValueError, match='timeout'): transport.request({}, timeout)
