"""Independent-panel and frozen-batch tests; no model or physics calls."""
import json
from pathlib import Path
import time

import pytest

from experiments.planar_tasks.baseline import evaluation as module
from experiments.planar_tasks.baseline import panels
from experiments.planar_tasks.tasks import task_catalog


def test_panel_is_deterministic_new_roots_and_geometry_only(tmp_path):
    path = tmp_path / 'panel.json'
    original = panels.create_panel(path, seed=911, per_family=4)
    manifest, tasks = panels.load_panel(path)
    assert original == manifest
    assert manifest['tasks'] == panels.generate_panel(911, 4)['tasks']
    assert manifest['generated_unix'] <= time.time()
    assert len(tasks) == 12
    assert all(t.instance.split == 'independent_evaluation' for t in tasks)
    assert len({t.root_id for t in tasks}) == len(tasks)
    assert not ({t.root_id for t in tasks} & {t.root_id for t in task_catalog().values()})
    assert all(row['geometry']['legal'] for row in manifest['tasks'])
    assert all(row['task']['feasibility'] == 'unverified' for row in manifest['tasks'])
    assert manifest['physics_evaluations_at_creation'] == 0
    assert panels.generate_panel(912, 4)['content_sha256'] != manifest['content_sha256']
    with pytest.raises(FileExistsError):
        panels.create_panel(path, seed=911, per_family=4)


def test_panel_rejects_tampering_even_with_rewritten_outer_hash(tmp_path):
    path = tmp_path / 'panel.json'
    data = panels.create_panel(path, seed=911, per_family=1)
    data['tasks'][0]['task'] = next(iter(task_catalog().values())).record()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='hash changed'):
        panels.load_panel(path)
    data['content_sha256'] = panels.content_hash({k: v for k, v in data.items() if k != 'content_sha256'})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='frozen generator'):
        panels.load_panel(path)


@pytest.fixture
def frozen_batch(tmp_path, monkeypatch):
    session = tmp_path / 'synthesis'
    package = session / 'frozen/package'
    package.mkdir(parents=True)
    (package / 'policy.py').write_text('# immutable agent program\n')
    (package / '__policy_host__.py').write_text('# standard host\n')
    hashes = {p.name: module.seal.digest(p) for p in package.iterdir()}
    submission = {'selection': 'agent_explicit_submission', 'version': 'test-baseline', 'files_sha256': hashes}
    (session / 'frozen/manifest.json').write_text(json.dumps(submission))
    (session / 'session.json').write_text('{}')
    release = tmp_path / 'release.json'; release.write_text('{}')
    panel = tmp_path / 'panel.json'
    panels.create_panel(panel, seed=911, per_family=1)
    (session / 'state.json').write_text(json.dumps({'created_unix': time.time(), 'status': 'submitted'}))
    checks = []
    def verify(path, release_path):
        checks.append(path)
        module._verify_package(package, hashes)
        return submission
    monkeypatch.setattr(module, 'verify_frozen', verify)
    return session, panel, tmp_path / 'evaluation', release, checks


def test_batch_keeps_errors_in_denominator_and_uses_fresh_sandbox(frozen_batch, monkeypatch):
    session, panel, output, release, checks = frozen_batch
    calls = []
    def episode(task, command, destination, budget, **kwargs):
        calls.append((task, command, destination, budget, kwargs))
        assert kwargs['security_sandbox'] == {
            'kind': 'bubblewrap', 'read_only_package': True,
            'development_rpc': False, 'fresh_process_per_episode': True}
        assert kwargs['max_control_ticks'] == 17
        assert '--unshare-all' in command and '--ro-bind' in command
        assert '/tmp' in command and '--tmpfs' in command
        assert str(session / 'workspace') not in command
        if len(calls) == 2:
            raise RuntimeError('synthetic initialization failure')
        Path(destination).mkdir(parents=True)
        (Path(destination) / 'replay.json').write_text('{}')
        return {'status': 'complete', 'success': len(calls) == 1,
                'termination': 'success' if len(calls) == 1 else 'time_budget',
                'task_cost_seconds': 1. if len(calls) == 1 else 180., 'executed_control_ticks': 17}
    monkeypatch.setattr(module, 'run_episode', episode)
    result = module.evaluate(session, panel, output, module.EvaluationBudget(max_control_ticks=17), release_path=release)
    assert result['status'] == 'incomplete'
    assert result['scheduled_episodes'] == result['attempted_episodes'] == len(calls) == 3
    assert result['successful_episodes'] == 1 and result['success_rate'] == 1 / 3
    assert result['infrastructure_errors'] == 1 and result['total_task_cost_seconds'] == 361.
    assert len({str(call[2]) for call in calls}) == 3
    assert len(checks) >= 4  # gate plus revalidation before every instance
    assert len((output / 'results.jsonl').read_text().splitlines()) == 3
    assert (output / 'artifacts_sha256.json').exists()
    assert json.loads((session / 'state.json').read_text())['status'] == 'submitted'
    assert len(list((session / 'evaluation_claims').glob('*.json'))) == 1
    with pytest.raises(FileExistsError):
        module.evaluate(session, panel, output, release_path=release)
    with pytest.raises(FileExistsError):
        module.evaluate(session, panel, output.with_name('second_attempt'), release_path=release)
    assert not output.with_name('second_attempt').exists()


def test_batch_wall_budget_marks_unattempted_without_running_them(frozen_batch, monkeypatch):
    session, panel, output, release, _ = frozen_batch
    clock = [0.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    calls = []
    def episode(*args, **kwargs):
        calls.append(kwargs)
        clock[0] = 2.
        return {'status': 'incomplete', 'success': False, 'termination': 'wall_budget_exhausted',
                'task_cost_seconds': 180., 'executed_control_ticks': 1}
    monkeypatch.setattr(module, 'run_episode', episode)
    result = module.evaluate(session, panel, output, module.EvaluationBudget(wall_seconds=1.), release_path=release)
    assert len(calls) == result['attempted_episodes'] == 1
    assert result['not_run_episodes'] == 2
    assert result['scheduled_episodes'] == 3 and result['success_rate'] == 0.
    assert result['total_task_cost_seconds'] == 540.
    assert result['stop_reason'] == 'evaluation_wall_budget_exhausted'
    assert all(not row['attempted'] for row in result['results'][1:])


def test_package_drift_stops_remaining_panel_without_repair(frozen_batch, monkeypatch):
    session, panel, output, release, _ = frozen_batch
    calls = []
    def episode(*args, **kwargs):
        calls.append(kwargs)
        (output / 'package/policy.py').write_text('# changed after first episode')
        return {'status': 'complete', 'success': False, 'termination': 'time_budget',
                'task_cost_seconds': 180., 'executed_control_ticks': 1}
    monkeypatch.setattr(module, 'run_episode', episode)
    result = module.evaluate(session, panel, output, release_path=release)
    assert len(calls) == 1 and result['not_run_episodes'] == 2
    assert result['stop_reason'] == 'integrity_error'
    assert result['results'][1]['error']['type'] == 'ValueError'


def test_explicit_submission_gate_precedes_any_episode(frozen_batch, monkeypatch):
    session, panel, output, release, _ = frozen_batch
    monkeypatch.setattr(module, 'verify_frozen', lambda *a, **k: {'selection': 'researcher_auto_best'})
    monkeypatch.setattr(module, 'run_episode', lambda *a, **k: pytest.fail('must not execute'))
    with pytest.raises(ValueError, match='explicit agent submission'):
        module.evaluate(session, panel, output, release_path=release)
    assert not output.exists()


def test_evaluation_output_cannot_enter_synthesis_workspace(frozen_batch):
    session, panel, _, release, _ = frozen_batch
    with pytest.raises(ValueError, match='outside the synthesis'):
        module.evaluate(session, panel, session / 'workspace/eval', release_path=release)


def test_panel_must_predate_synthesis_creation(frozen_batch):
    session, panel, output, release, _ = frozen_batch
    (session / 'state.json').write_text(json.dumps({'created_unix': 1., 'status': 'submitted'}))
    with pytest.raises(ValueError, match='frozen before synthesis'):
        module.evaluate(session, panel, output, release_path=release)
    assert not output.exists()


def test_real_two_tick_pipeline_across_three_diagnostic_families(frozen_batch):
    """Physical integration smoke only; temporary seed 911, not the research panel."""
    session, panel, output, release, _ = frozen_batch
    submission=module.verify_frozen(session,release)
    package=session/'frozen/package'
    (package/'policy.py').write_text('class Policy:\n def __init__(self, contract): pass\n def act(self, observation): return dict(a0=44,a1=44,ticks=2)\n')
    (package/'__policy_host__.py').write_bytes((Path(module.__file__).parent/'policy_host.py').read_bytes())
    hashes=submission['files_sha256'];hashes.clear()
    hashes.update({p.name:module.seal.digest(p) for p in package.iterdir()})
    (session/'frozen/manifest.json').write_text(json.dumps(submission))
    result=module.evaluate(session,panel,output,module.EvaluationBudget(
        wall_seconds=30,episode_wall_seconds=10,response_seconds=3,max_control_ticks=2),release_path=release)
    assert result['attempted_episodes']==3 and result['executed_control_ticks']==6
    assert result['infrastructure_errors']==0
    for row in result['results']:
        assert row['termination']=='control_tick_budget_exhausted'
        replay=json.loads((output/row['episode_path']/'replay.json').read_text())
        assert len(replay['frames'])==2
