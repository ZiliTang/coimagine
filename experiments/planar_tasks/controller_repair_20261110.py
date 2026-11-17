"""Bounded reevaluation of two unchanged submissions under a recorded repair.

This host-only experiment does not alter or re-submit historical artifacts. The
ordinary evaluator retains its strict same-release gate. Original author release
and repaired execution release are separate provenance fields here.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from .baseline import seal
from .baseline.evaluation import EvaluationBudget, _verify_package, _write_once
from .baseline.panels import load_panel
from .baseline.session import package_command
from .runner import RunBudget, run_episode

ROOT = seal.ROOT / 'outputs/agentic/controller_radial_repair_20261110_01'


def read(path):
    return json.loads(Path(path).read_text())


def verify_originals(plan):
    for path, expected in plan['originals_sha256'].items():
        if seal.digest(path) != expected:
            raise ValueError('historical input changed: ' + path)
    if seal.digest(plan['panel_path']) != plan['panel_sha256']:
        raise ValueError('original panel changed')


def prepare():
    original = read(ROOT / 'original_inputs.json'); verify_originals(original)
    new = seal.verify_release()
    old_path = seal.BASE / 'release_v1_3_5.json'; old = read(old_path)
    assert seal.digest(old_path) == original['old_release_sha256']
    snapshot = Path(original['old_source_snapshot'])
    for name, digest in old['files'].items():
        assert seal.digest(snapshot / name) == digest, name
    assert new['contract'] == old['contract']
    assert new['development_tasks'] == old['development_tasks']
    assert new['environment'] == old['environment']
    changed = sorted(n for n in old['files'] if new['files'].get(n) != old['files'][n])
    assert changed == sorted(['experiments/option_rl_v13/push_controller.py',
        'experiments/planar_tasks/baseline/__init__.py',
        'experiments/planar_tasks/baseline/seal.py',
        'experiments/planar_tasks/baseline/README.md',
        'docs/research/Agentic_Options_Baseline_v1.md'])
    # Exact single-line controller patch, independently checked against old snapshot.
    name = 'experiments/option_rl_v13/push_controller.py'
    expected = (snapshot / name).read_text().replace('        raw = (0.0, 0.0)\n        if face is None:',
        '        raw = (0.0, 0.0)\n        radial = 0.0  # No normal command when no face or no driving stage is available.\n        if face is None:')
    assert (seal.ROOT / name).read_text() == expected
    plan = dict(original, prepared_unix=time.time(),
        original_inputs_sha256=seal.digest(ROOT / 'original_inputs.json'),
        execution_release_sha256=seal.digest(seal.LOCK),
        runner_sha256=seal.digest(__file__), changed_frozen_sources=changed,
        added_frozen_sources=sorted(set(new['files']) - set(old['files'])),
        same_contract_tasks_and_runtime=True)
    _write_once(ROOT / 'preflight.json', plan)
    print(json.dumps({'prepared': True, 'changed_frozen_sources': changed,
                      'scheduled_episodes': plan['scheduled_episodes']}))


def run(condition):
    plan = read(ROOT / 'preflight.json')
    assert seal.digest(__file__) == plan['runner_sha256']
    assert seal.digest(ROOT / 'original_inputs.json') == plan['original_inputs_sha256']
    verify_originals(plan); seal.verify_release()
    assert seal.digest(seal.LOCK) == plan['execution_release_sha256']
    entry = plan['policies'][condition]; session = Path(entry['session_path'])
    submission = read(session / 'frozen/manifest.json')
    assert submission == entry['submission']
    assert seal.digest(session / 'frozen/manifest.json') == entry['submission_sha256']
    assert submission['selection'] == 'agent_explicit_submission'
    assert submission['release_sha256'] == plan['old_release_sha256']
    assert submission['config_sha256'] == seal.digest(session / 'session.json')
    _verify_package(session / 'frozen/package', submission['files_sha256'])
    panel, tasks = load_panel(plan['panel_path'])
    assert len(tasks) == 30
    assert panel['generated_unix'] < read(session / 'state.json')['created_unix']
    budget = EvaluationBudget(**plan['evaluation_budget'])
    output = ROOT / condition / 'evaluation'
    _write_once(ROOT / f'{condition}_claim.json', {
        'original_submission_sha256': entry['submission_sha256'],
        'execution_release_sha256': plan['execution_release_sha256'],
        'panel_sha256': plan['panel_sha256'], 'output_path': str(output),
        'claimed_unix': time.time(), 'no_retry': True})
    output.mkdir(parents=True, exist_ok=False)
    package = output / 'package'; shutil.copytree(session / 'frozen/package', package)
    _write_once(output / 'submission.json', submission)
    _write_once(output / 'panel.json', panel)
    started = time.monotonic(); deadline = started + budget.wall_seconds
    manifest = {'kind': 'frozen_evaluation', 'session_path': str(session),
        'evaluation_variant': 'v1.3.6 controller radial repair; original v1.3.5 policy',
        'condition': condition, 'original_submission_release_sha256': plan['old_release_sha256'],
        'release_sha256': plan['execution_release_sha256'],
        'submission_manifest_sha256': entry['submission_sha256'],
        'panel_sha256': plan['panel_sha256'], 'budget': plan['evaluation_budget'],
        'scheduled_episodes': len(tasks), 'started_unix': time.time(),
        'source_policy_unchanged': True, 'new_synthesis': False,
        'panel_preregistered_before_synthesis': True,
        'panel_generated_unix': panel['generated_unix'],
        'synthesis_created_unix': read(session / 'state.json')['created_unix'],
        'cross_episode_state': False, 'development_feedback': False,
        'failure_cost_seconds': 180., 'denominator': 'all scheduled panel instances',
        'preflight_sha256': seal.digest(ROOT / 'preflight.json')}
    _write_once(output / 'manifest.json', manifest)
    results = []
    for index, task in enumerate(tasks):
        seal.verify_release()
        assert seal.digest(seal.LOCK) == plan['execution_release_sha256']
        assert seal.digest(plan['panel_path']) == plan['panel_sha256']
        _verify_package(package, submission['files_sha256'])
        relative = f'episodes/{index:04d}_{task.instance.name}'
        row = {'index': index, 'task': task.instance.name, 'family': task.family,
               'task_sha256': task.sha256, 'episode_path': relative, 'attempted': False}
        if time.monotonic() >= deadline:
            summary = {'status': 'not_run', 'success': False,
                'termination': 'evaluation_wall_budget_exhausted', 'task_cost_seconds': 180.,
                'executed_control_ticks': 0}
        else:
            row['attempted'] = True
            try:
                summary = run_episode(task, package_command(package), output / relative,
                    RunBudget(budget.max_decisions,
                        min(budget.episode_wall_seconds, max(.001, deadline-time.monotonic())),
                        budget.response_seconds),
                    provenance={'generated_agent_baseline': True,
                        'method': 'unchanged_policy_controller_repair_v1.3.6',
                        'original_submission_manifest_sha256': entry['submission_sha256'],
                        'original_submission_release_sha256': plan['old_release_sha256'],
                        'execution_release_sha256': plan['execution_release_sha256'],
                        'files_sha256': submission['files_sha256'],
                        'evaluation_panel_sha256': plan['panel_sha256']},
                    security_sandbox={'kind': 'bubblewrap', 'read_only_package': True,
                        'development_rpc': False, 'fresh_process_per_episode': True},
                    max_control_ticks=budget.max_control_ticks, wall_deadline=deadline)
            except Exception as exc:
                summary = {'status': 'infrastructure_error', 'success': False,
                    'termination': 'infrastructure_error', 'task_cost_seconds': 180.,
                    'executed_control_ticks': 0,
                    'error': {'type': type(exc).__name__, 'message': str(exc)}}
        row.update(status=summary['status'], success=bool(summary['success']),
                   termination=summary['termination'], summary=summary)
        results.append(row)
        with (output / 'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    verify_originals(plan)
    attempts = sum(r['attempted'] for r in results)
    errors = sum(r['status'] == 'infrastructure_error' for r in results)
    summary = {'kind': 'frozen_evaluation', 'session_path': str(session),
        'status': 'complete' if attempts == len(tasks) and not errors else 'incomplete',
        'scheduled_episodes': len(tasks), 'attempted_episodes': attempts,
        'successful_episodes': sum(r['success'] for r in results),
        'infrastructure_errors': errors,
        'execution_errors': sum(r['termination'] == 'execution_error' for r in results),
        'not_run_episodes': len(tasks)-attempts,
        'success_rate': sum(r['success'] for r in results)/len(tasks),
        'denominator': 'all scheduled panel instances',
        'total_task_cost_seconds': sum(r['summary']['task_cost_seconds'] for r in results),
        'executed_control_ticks': sum(r['summary'].get('executed_control_ticks', 0) for r in results),
        'wall_seconds': time.monotonic()-started, 'results': results}
    _write_once(output / 'summary.json', summary)
    _write_once(output / 'artifacts_sha256.json', {
        str(p.relative_to(output)): seal.digest(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'control', 'modular'))
    command = parser.parse_args().command
    prepare() if command == 'prepare' else run(command)
