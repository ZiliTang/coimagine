"""Bounded synthesis accounting and isolated deployment, without API or rollouts."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pytest

from experiments.planar_tasks import agentic, sandbox, runner
from experiments.planar_tasks.policy import JsonlPolicy


@pytest.fixture
def session(tmp_path, monkeypatch):
    class SensorsOnly:
        def __init__(self, instance):
            pass

        def observe(self):
            return {'robots': [], 'legal_pairs': [[44, 44]]}

    monkeypatch.setattr(agentic, 'TaskEnvironment', SensorsOnly)
    path = tmp_path/'session'
    agentic.initialize(path, 'test-model', 'test-author')
    return path


@pytest.fixture
def episodes(monkeypatch):
    calls = []

    def fake_episode(task, command, directory, budget, **kwargs):
        calls.append((task.instance.name, command, budget, kwargs))
        directory.mkdir(parents=True)
        (directory/'sensors.jsonl').write_text('')
        (directory/'decisions.json').write_text('[]')
        return dict(success=False, task_cost_seconds=180., final_goal_distance_diagnostic=3.)

    monkeypatch.setattr(agentic, 'run_episode', fake_episode)
    monkeypatch.setattr(agentic, 'policy_command', lambda path: ['isolated', str(path)])
    return calls


def test_attempt_budget_includes_invalid_program_and_development_is_separate(session, episodes):
    bad = agentic.trial(session, 'def broken(')
    assert bad['candidate'] == 0 and 'compile_error' in bad
    assert episodes == []
    for expected in (1, 2):
        result = agentic.trial(session, 'pass')
        assert result['candidate'] == expected
    with pytest.raises(ValueError, match='budget exhausted'):
        agentic.trial(session, 'pass')
    assert [call[0] for call in episodes] == list(agentic.DEV_TASKS)*2
    for _, _, budget, kwargs in episodes:
        assert budget.max_decisions == 1800
        assert kwargs['security_sandbox'] == sandbox.SANDBOX
        assert kwargs['provenance']['generated_agent_baseline'] is True
        assert kwargs['provenance']['model'] == 'test-model'
    config = json.loads((session/'session.json').read_text())
    assert set(config['transfer_tasks']).isdisjoint(config['development_tasks'])
    assert 'development' in config['split']


def test_freeze_selects_earliest_tie_and_transfer_keeps_exact_program(session, episodes):
    first = agentic.trial(session, 'x = 1\n')
    agentic.trial(session, 'x = 2\n')
    frozen = agentic.freeze(session)
    assert frozen['candidate'] == 0
    assert frozen['policy_sha256'] == first['policy_sha256']
    assert (session/'frozen_policy.py').read_text() == 'x = 1\n'
    assert agentic.freeze(session) == frozen
    with pytest.raises(ValueError, match='frozen'):
        agentic.trial(session, 'pass')
    episodes.clear()
    result = agentic.transfer(session)
    config = json.loads((session/'session.json').read_text())
    assert [c[0] for c in episodes] == config['transfer_tasks']
    assert result['attempts'] == len(config['transfer_tasks'])
    for _, command, _, kwargs in episodes:
        assert command[-1] == str(session/'frozen_policy.py')
        assert kwargs['provenance']['policy_sha256'] == frozen['policy_sha256']
    with pytest.raises(FileExistsError):
        agentic.transfer(session)


def test_freeze_and_transfer_reject_changed_programs(session, episodes):
    agentic.trial(session, 'pass')
    candidate = session/'candidate_00'/'policy.py'
    candidate.write_text('changed = True')
    with pytest.raises(ValueError, match='changed after trial'):
        agentic.freeze(session)
    candidate.write_text('pass')
    agentic.freeze(session)
    (session/'frozen_policy.py').write_text('changed = True')
    episodes.clear()
    with pytest.raises(ValueError, match='hash mismatch'):
        agentic.transfer(session)
    assert not episodes
    assert not (session/'transfer').exists()


def test_no_compilable_candidate_cannot_freeze(session, episodes):
    agentic.trial(session, 'def broken(')
    with pytest.raises(ValueError, match='no compilable candidate'):
        agentic.freeze(session)


def test_sandbox_requires_isolation_tools_without_fallback(tmp_path, monkeypatch):
    policy = tmp_path/'policy.py'
    policy.write_text('pass')
    monkeypatch.setattr(sandbox.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='no unsandboxed fallback'):
        sandbox.policy_command(policy)


def test_sandbox_command_has_minimal_mount_and_environment_boundary(tmp_path, monkeypatch):
    policy = tmp_path/'policy.py'
    policy.write_text('pass')
    monkeypatch.setattr(sandbox.shutil, 'which', lambda name: '/usr/bin/'+name)
    command = sandbox.policy_command(policy)
    assert '--unshare-all' in command and '--clearenv' in command
    assert '--share-net' not in command
    binds = [command[i+1:i+3] for i, arg in enumerate(command) if arg == '--ro-bind']
    assert all(source in ('/usr', '/lib', '/lib64', str(policy)) for source, _ in binds)
    assert [str(policy), '/work/policy.py'] in binds
    assert command[-4:] == ['/usr/bin/python3', '-I', '-u', '/work/policy.py']


def test_live_sandbox_denies_files_environment_network_and_times_out(tmp_path, monkeypatch):
    probe = tmp_path/'probe.py'
    probe.write_text('print("ready", flush=True)')
    try:
        command = sandbox.policy_command(probe)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    available = subprocess.run(command, capture_output=True, text=True, timeout=5)
    if available.returncode:
        pytest.skip('bubblewrap user namespace unavailable: '+available.stderr[:150])
    sentinel = tmp_path/'private.txt'
    sentinel.write_text('synthetic sentinel')
    monkeypatch.setenv('COIMAGINE_TEST_SECRET', 'synthetic-value')
    probe.write_text("import os,json,socket,sys,time\n"
        f"paths={ [str(sentinel), str(Path.cwd()/'pyproject.toml'), str(Path.home())]!r}\n"
        "denied=[not os.path.exists(p) for p in paths]\n"
        "s=socket.socket();s.settimeout(.1)\n"
        "try:\n s.connect(('1.1.1.1',443));network=True\n"
        "except OSError:\n network=False\n"
        "sys.stdin.readline()\n"
        "print(json.dumps(dict(denied=denied,secret=os.getenv('COIMAGINE_TEST_SECRET'),network=network)),flush=True)\n"
        "sys.stdin.readline();time.sleep(5)\n")
    policy = JsonlPolicy(sandbox.policy_command(probe), tmp_path/'stderr')
    try:
        result = policy.request({}, 2.)
        assert all(result['denied']) and result['secret'] is None
        assert result['network'] is False
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            policy.request({}, .1)
        assert time.monotonic()-started < 2.
    finally:
        policy.close()
    assert policy.process.poll() is not None


def test_source_manifest_archives_sandbox_policy_file(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    (root/'experiments').mkdir(parents=True)
    source = root/'experiments'/'test.py'
    source.write_text('source = 1\n')
    policy = tmp_path/'submitted.py'
    policy.write_text('policy = 2\n')
    out = tmp_path/'run'
    out.mkdir()
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'synthetic-git\n')
    manifest = runner.source_manifest(out, ['bwrap', '--ro-bind', str(policy), '/work/policy.py'])
    assert manifest['source_sha256']['experiments/test.py'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest['command_files_sha256'][str(policy.resolve())] == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert (out/'source_snapshot'/'experiments'/'test.py').read_bytes() == source.read_bytes()
    assert next((out/'policy_files').iterdir()).read_bytes() == policy.read_bytes()


@pytest.mark.parametrize('scores, winner', [
    ([(False, 1., 1.), (True, 100., 100.), (False, 0., 0.)], 1),
    ([(True, 10., 1.), (True, 5., 100.), (True, 20., 0.)], 1),
    ([(True, 5., 10.), (True, 5., 2.), (True, 5., 20.)], 1),
])
def test_selection_prioritizes_success_then_cost_then_distance(session, episodes, monkeypatch, scores, winner):
    calls = []

    def scored_episode(task, command, directory, budget, **kwargs):
        candidate = len(calls)//len(agentic.DEV_TASKS)
        calls.append(task)
        success, cost, distance = scores[candidate]
        return dict(success=success, task_cost_seconds=cost,
                    final_goal_distance_diagnostic=distance)

    monkeypatch.setattr(agentic, 'run_episode', scored_episode)
    monkeypatch.setattr(agentic, 'sensor_feedback', lambda directory: {})
    for i in range(3):
        agentic.trial(session, f'candidate = {i}\n')
    assert agentic.freeze(session)['candidate'] == winner


from experiments.planar_tasks import deepseek_agent


def api_reply(content='{"code":"pass","note":"test"}', finish='stop'):
    return dict(ok=True, response=dict(model='mock-model', usage={'total_tokens': 7},
        id='mock-id', choices=[dict(finish_reason=finish, message={'content': content})]))


def test_api_transport_error_stops_without_retry_and_records_incomplete(session, monkeypatch):
    calls = []

    def failed_request(body, timeout):
        calls.append((body, timeout))
        return dict(ok=False, error='http_error', http_status=429)

    monkeypatch.setattr(deepseek_agent, 'request', failed_request)
    result = deepseek_agent.synthesize(session)
    assert len(calls) == result['api_calls'] == 1
    assert result['status'] == 'incomplete' and result['frozen'] is None
    assert result['statuses'][0]['http_status'] == 429
    assert calls[0][1] == 180
    body = calls[0][0]
    assert body['max_tokens'] == 8192 and body['stream'] is False
    assert body['thinking'] == {'type': 'disabled'}
    with pytest.raises(ValueError, match='already attempted'):
        deepseek_agent.synthesize(session)
    assert len(calls) == 1


@pytest.mark.parametrize('content, finish', [
    ('not-json', 'stop'), ('{"code":"pass"}', 'stop'),
    ('{"code":1,"note":"test"}', 'stop'),
    ('{"code":"pass","note":"test","extra":true}', 'stop'),
    (json.dumps({'code': 'x'*65537, 'note': ''}), 'stop'),
    ('{"code":"pass","note":"test"}', 'length'),
])
def test_malformed_api_output_consumes_three_calls_without_trials(session, monkeypatch, content, finish):
    calls = []

    def response(body, timeout):
        calls.append(json.loads(json.dumps(body)))
        return api_reply(content, finish)

    monkeypatch.setattr(deepseek_agent, 'request', response)
    monkeypatch.setattr(deepseek_agent, 'trial', lambda *a: pytest.fail('invalid output reached trial'))
    result = deepseek_agent.synthesize(session)
    assert result['api_calls'] == len(calls) == 3
    assert result['status'] == 'incomplete'
    assert all('error' in status for status in result['statuses'])
    assert 'consumed one model call' in calls[1]['messages'][-1]['content']


def test_api_freezes_completed_candidate_when_later_transport_fails(session, episodes, monkeypatch):
    responses = iter([api_reply(), dict(ok=False, error='wall_timeout')])
    monkeypatch.setattr(deepseek_agent, 'request', lambda *a: next(responses))
    result = deepseek_agent.synthesize(session)
    assert result['api_calls'] == 2
    assert result['status'] == 'frozen'
    assert result['frozen']['candidate'] == 0
    assert result['statuses'][0]['candidate'] == 0
    assert result['statuses'][1]['error'] == 'wall_timeout'
    assert len(episodes) == len(agentic.DEV_TASKS)


def test_request_character_budget_stops_before_call(session, monkeypatch):
    packet_path = session/'packet.json'
    packet = json.loads(packet_path.read_text())
    packet['instructions'] = 'x'*200001
    packet_path.write_text(json.dumps(packet))
    monkeypatch.setattr(deepseek_agent, 'request', lambda *a: pytest.fail('oversized request sent'))
    result = deepseek_agent.synthesize(session)
    assert result['api_calls'] == 0 and result['status'] == 'incomplete'
    assert result['statuses'][0]['error'] == 'request_character_budget'


def test_request_worker_redacts_http_error_and_does_not_retry(monkeypatch):
    import urllib.error
    calls, sent = [], []

    class Connection:
        closed = False
        def send(self, data):
            sent.append(data)
        def close(self):
            self.closed = True

    class Opener:
        def open(self, request, timeout):
            calls.append((request.full_url, timeout))
            raise urllib.error.HTTPError(request.full_url, 401, 'synthetic-secret', {}, None)

    connection = Connection()
    monkeypatch.setattr(deepseek_agent, '_credential', lambda: 'synthetic-secret')
    monkeypatch.setattr(deepseek_agent.urllib.request, 'build_opener', lambda *a: Opener())
    deepseek_agent._request_worker({'model': 'mock'}, connection)
    assert calls == [(deepseek_agent.ENDPOINT, 120)]
    assert sent == [dict(ok=False, error='http_error', http_status=401)]
    assert connection.closed
    assert 'synthetic-secret' not in json.dumps(sent)
    with pytest.raises(ValueError, match='redirects disabled'):
        deepseek_agent.NoRedirect().redirect_request(None)


@pytest.mark.parametrize('poll, eof, error', [(False, False, 'wall_timeout'),
                                          (True, True, 'request_worker_exited')])
def test_request_timeout_or_worker_exit_cleans_process_without_retry(monkeypatch, poll, eof, error):
    class Pipe:
        closed = False
        def poll(self, timeout):
            assert timeout == .25
            return poll
        def recv(self):
            if eof:
                raise EOFError
            pytest.fail('unexpected recv')
        def close(self):
            self.closed = True

    class Process:
        starts = 0
        terminated = False
        def start(self):
            self.starts += 1
        def is_alive(self):
            return not self.terminated
        def terminate(self):
            self.terminated = True
        def join(self, timeout=None):
            pass

    parent, child, worker = Pipe(), Pipe(), Process()
    class Context:
        def Pipe(self, duplex):
            assert duplex is False
            return parent, child
        def Process(self, **kwargs):
            assert kwargs['args'][-1] == .25  # socket and parent share the deadline
            return worker

    monkeypatch.setattr(deepseek_agent.mp, 'get_context', lambda method: Context())
    result = deepseek_agent.request({}, timeout=.25)
    assert result['ok'] is False and result['error'] == error
    assert worker.starts == 1 and worker.terminated
    assert parent.closed and child.closed


def test_catastrophic_trial_retains_failed_episode_denominator(session, episodes, monkeypatch):
    def crash(*args, **kwargs):
        raise RuntimeError('synthetic infrastructure failure')
    monkeypatch.setattr(agentic, 'run_episode', crash)
    try:
        agentic.trial(session, 'pass')
    except RuntimeError:
        pass
    feedback = json.loads((session/'candidate_00'/'feedback.json').read_text())
    assert len(feedback['episodes']) == len(agentic.DEV_TASKS)
    assert all(not e['summary']['success'] for e in feedback['episodes'])
    assert all(e['summary']['task_cost_seconds'] == 180. for e in feedback['episodes'])


def test_api_trial_crash_still_writes_incomplete_aggregate(session, monkeypatch):
    monkeypatch.setattr(deepseek_agent, 'request', lambda *a: api_reply())
    def crash(*args):
        raise RuntimeError('synthetic infrastructure failure')
    monkeypatch.setattr(deepseek_agent, 'trial', crash)
    try:
        deepseek_agent.synthesize(session)
    except RuntimeError:
        pass
    result = json.loads((session/'synthesis.json').read_text())
    assert result['api_calls'] == 1 and result['status'] == 'incomplete'
    assert 'error' in result['statuses'][0]


def test_api_freeze_error_is_reported_in_aggregate(session, monkeypatch):
    monkeypatch.setattr(deepseek_agent, 'request', lambda *a: dict(ok=False, error='http_error'))
    def reject_freeze(*args):
        raise ValueError('candidate changed after trial')
    monkeypatch.setattr(deepseek_agent, 'freeze', reject_freeze)
    result = deepseek_agent.synthesize(session)
    assert result['status'] == 'incomplete'
    assert result.get('freeze_error') == 'candidate changed after trial'


@pytest.mark.parametrize('data', [None, [], 'invalid response'])
def test_malformed_response_envelope_records_error_and_consumes_call(session, monkeypatch, data):
    calls = []
    def malformed(*args):
        calls.append(1)
        return dict(ok=True, response=data)
    monkeypatch.setattr(deepseek_agent, 'request', malformed)
    result = deepseek_agent.synthesize(session)
    assert result['status'] == 'incomplete' and result['api_calls'] == 3
    assert len(calls) == 3
    assert all('error' in row for row in result['statuses'])
