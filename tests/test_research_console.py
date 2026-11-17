"""Read-only projection, information hygiene, and real shared-replay serving."""
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import pytest

from experiments.planar_tasks.console.catalog import Catalog
from experiments.planar_tasks.console.server import handler_for


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def recorded(tmp_path):
    run = tmp_path / 'campaign'
    write(run / 'session.json', {'model': {'provider': 'fake', 'name': 'fake-model',
        'settings': {'api_key': 'never-show-this'}, 'source_access': 'hidden'},
        'purpose': 'infrastructure_smoke', 'budget': {'model_output_tokens': 1000}})
    write(run / 'state.json', {'version': 'v1.2', 'status': 'submitted', 'usage': {'model_output_tokens': 200},
        'reported_model_usage': [{'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}]})
    package = run / 'private/test_0000/package'
    package.mkdir(parents=True)
    (package / 'policy.py').write_text('class Policy: pass\n')
    frozen = run / 'frozen/package'
    frozen.mkdir(parents=True)
    (frozen / 'policy.py').write_text('class Policy:\n    pass\n')
    write(run / 'frozen/manifest.json', {'selection': 'agent_explicit_submission'})
    write(run / 'private/test_0000/episode/summary.json', {'task': 'dev_alpha', 'success': False, 'status': 'complete'})
    write(run / 'private/test_0000/episode/replay.json', {'id': 'saved', 'frames': []})
    event = {'index': 0, 'kind': 'model', 'time_unix': 1, 'request': {'model': 'fake-model', 'messages': ['private-prompt']},
        'result': {'ok': True, 'response': {'choices': [{'message': {'content': 'Trying a script', 'reasoning_content': 'never-show-reasoning',
            'tool_calls': [{'function': {'name': 'run_python'}}]}}], 'usage': {'total_tokens': 10}}}}
    (run / 'private/events.jsonl').write_text(json.dumps(event) + '\n')
    evaluation = tmp_path / 'evaluation'
    write(evaluation / 'manifest.json', {'kind': 'frozen_evaluation', 'session_path': str(run), 'scheduled_episodes': 1})
    write(evaluation / 'results.jsonl', {'success': True, 'secret_in_progress_result': 'not-yet-public'})
    catalog = Catalog(tmp_path).scan()
    return catalog, next(iter(catalog.runs)), run, evaluation


def test_projects_usage_without_leaking_prompts_or_reasoning(recorded):
    catalog, run_id, _, _ = recorded
    detail = catalog.detail(run_id)
    assert detail['budget']['used']['model_output_tokens'] == 200
    assert detail['budget']['reported_model_usage']['completion_tokens'] == 3
    assert 'never-show-this' not in json.dumps(detail)
    events = catalog.events(run_id)
    assert events['items'][0]['calls'] == ['run_python']
    assert 'private-prompt' not in json.dumps(events)
    assert 'never-show-reasoning' not in json.dumps(events)
    assert catalog.list_evaluations(run_id)[0]['results'] == []


def test_completed_usage_correction_requires_matching_event_chain(recorded):
    catalog, run_id, run, _ = recorded
    state = json.loads((run/'state.json').read_text())
    state.update(events=5, last_event_hash='terminal-chain')
    write(run/'state.json', state)
    audit = dict(events=5, last_event_sha256='terminal-chain',
                 reported_usage=dict(prompt_tokens=107, completion_tokens=13, total_tokens=120),
                 basis='includes truncated final response')
    write(run/'provider_usage_audit.json', audit)
    assert catalog.detail(run_id)['budget']['reported_model_usage']['total_tokens'] == 120
    audit['last_event_sha256'] = 'stale'
    write(run/'provider_usage_audit.json', audit)
    assert catalog.detail(run_id)['budget']['reported_model_usage']['total_tokens'] == 10


def test_versions_diff_and_no_symlink_reads(recorded, tmp_path):
    catalog, run_id, run, _ = recorded
    versions = catalog.versions(run_id)
    assert len(versions) == 2 and versions[-1]['selected']
    assert '+    pass' in catalog.diff(run_id, versions[0]['id'], versions[1]['id'])['text']
    outside = tmp_path / 'outside.py'
    outside.write_text('SECRET')
    policy = run / versions[-1]['path']
    policy.unlink();policy.symlink_to(outside)
    assert len(catalog.versions(run_id)) == 1
    with pytest.raises(KeyError):
        catalog.policy(run_id, versions[-1]['id'])


def test_http_is_readonly_and_replay_uses_shared_renderer(recorded):
    catalog, run_id, run, _ = recorded
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(catalog))
    worker = threading.Thread(target=server.serve_forever, daemon=True);worker.start()
    root = 'http://127.0.0.1:' + str(server.server_port)
    try:
        detail = json.load(urlopen(root + '/api/experiments/' + run_id))
        replay_url = detail['episodes'][0]['replay']
        html = urlopen(root + replay_url).read().decode()
        assert 'sceneCanvas' in html and 'replay-readonly.js' in html
        replay_id = replay_url.split('=')[-1]
        assert json.load(urlopen(root + '/api/runs/' + replay_id))['id'] == 'saved'
        for method in ('POST', 'DELETE', 'PUT', 'PATCH'):
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(root + '/api/run', method=method, data=b'{}'))
            assert exc.value.code == 405
        for route in ('/../../session.json', '/api/experiments/unknown', '/api/experiments/' + run_id + '/policy/../../session.json'):
            with pytest.raises(HTTPError):
                urlopen(root + route)
    finally:
        server.shutdown();server.server_close();worker.join()


def test_completed_evaluation_has_separate_replay(recorded):
    catalog, run_id, run, evaluation = recorded
    write(evaluation / 'episodes/0000/replay.json', {'id': 'heldout'})
    write(evaluation / 'summary.json', {'status': 'complete', 'successful_episodes': 0, 'attempted_episodes': 1,
        'success_rate': 0, 'results': [{'index': 0, 'task': 'heldout_alpha', 'family': 'open', 'status': 'failed',
            'success': False, 'attempted': True, 'episode_path': 'episodes/0000'}]})
    result = catalog.list_evaluations(run_id)[0]
    assert result['results'][0]['replay'].startswith('/replay?run=')
    assert catalog.detail(run_id)['episodes'][0]['task'] == 'dev_alpha'


def test_finalization_stage_zero_counters_and_reserved_totals(recorded):
    catalog, run_id, run, _ = recorded
    state = json.loads((run / 'state.json').read_text())
    state['finalizing'] = True
    write(run / 'state.json', state)
    config = json.loads((run / 'session.json').read_text())
    config['budget'].update(model_calls=8, tool_calls=80, model_input_bytes=100000,
                            workspace_bytes=1000, model_call_seconds=5)
    config['finalization'] = {'model_calls': 2, 'model_output_tokens': 200,
                              'model_input_bytes': 10000, 'tool_calls': 8, 'wall_seconds': 15}
    write(run / 'session.json', config)
    detail = catalog.detail(run_id)
    assert detail['stage'] == 'submitted'
    assert detail['budget']['used']['model_calls'] == 0
    assert detail['budget']['used']['tool_calls'] == 0
    assert detail['budget']['used']['model_output_tokens'] == 200
    assert 'workspace_bytes' not in detail['budget']['used']
    assert 'model_call_seconds' not in detail['budget']['used']
    assert detail['budget']['finalization']['model_calls'] == 2
    assert detail['budget']['limits']['model_calls'] == 8
    state['status'] = 'developing';write(run / 'state.json', state)
    assert catalog.overview(run_id)['stage'] == 'finalizing'


def test_submission_phase_limit_is_distinct_from_unused_development_budget(recorded):
    catalog, run_id, run, _ = recorded
    state=json.loads((run/'state.json').read_text())
    state.update(finalizing=True,status='submitted',started_monotonic=50,finished_monotonic=107,
        finalization_started_monotonic=100,finalization_start_usage={'model_calls':5,'model_output_tokens':500},
        usage={'model_calls':6,'model_output_tokens':600})
    config=json.loads((run/'session.json').read_text())
    config['budget'].update(model_calls=40,model_output_tokens=4000,wall_seconds=1800)
    config['finalization']={'model_calls':2,'model_output_tokens':200,'wall_seconds':15}
    write(run/'session.json',config);write(run/'state.json',state)
    remaining=catalog.detail(run_id)['budget']['submission_remaining']
    assert remaining=={'model_calls':1,'model_output_tokens':100,'wall_seconds':8}
    attempt=run/'private/submission_000000/package/policy.py';attempt.parent.mkdir(parents=True);attempt.write_text('invalid syntax')
    versions=catalog.versions(run_id)
    assert any(v['kind']=='submission_attempt' and not v['selected'] for v in versions)


def test_search_versions_show_authored_heuristic_not_fixed_executor(recorded):
    catalog, run_id, run, _ = recorded
    write(run / 'framework.json', {'kind': 'option-search-v3'})
    (run / 'private/test_0000/package/heuristic.py').write_text('def evaluate_state(n,m): return 10.\n')
    (run / 'frozen/package/heuristic.py').write_text('def evaluate_state(n,m): return 20.\n')
    versions = catalog.versions(run_id)
    assert len(versions) == 2
    assert all(v['path'].endswith('/heuristic.py') for v in versions)
    assert 'return 20.' in catalog.policy(run_id, versions[-1]['id'])['text']
    assert '+def evaluate_state(n,m): return 20.' in catalog.diff(run_id, versions[0]['id'], versions[-1]['id'])['text']
