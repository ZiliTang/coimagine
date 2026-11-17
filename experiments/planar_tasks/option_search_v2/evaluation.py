"""Bounded batch evaluation of an explicitly submitted immutable program.

This host-only entry point is never an agent development tool. Every episode uses
a fresh read-only sandboxed process; no API, simulator RPC, cross-episode program
state, policy selection, or automatic repair is offered.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import time

from experiments.planar_tasks.baseline import seal
from experiments.planar_tasks.baseline.panels import content_hash, load_panel
from experiments.planar_tasks.baseline.session import package_command, safe_path
from .session import verify_frozen
from ..runner import RunBudget, run_episode


@dataclass(frozen=True)
class EvaluationBudget:
    wall_seconds: float = 3600.
    episode_wall_seconds: float = 120.
    response_seconds: float = 20.
    max_control_ticks: int = 1800
    max_decisions: int = 1800

    def __post_init__(self):
        for name in ('wall_seconds', 'episode_wall_seconds', 'response_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive finite seconds')
        for name in ('max_control_ticks', 'max_decisions'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')


def _write_once(path, data):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def _verify_package(package, hashes):
    files = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    if files != set(hashes):
        raise ValueError('evaluation package file set changed')
    for name, expected in hashes.items():
        if seal.digest(safe_path(package, name)) != expected:
            raise ValueError('evaluation package hash changed: ' + name)


def evaluate(session_path, panel_path, output_path, budget=EvaluationBudget(), *, release_path=seal.LOCK):
    """Evaluate the full declared panel once; existing output directories refuse.

    All scheduled instances stay in the success denominator. Infrastructure errors
    and deadline-skipped instances are explicit failures, with full failure cost;
    skipped instances are not represented as attempted simulator episodes.
    """
    if not isinstance(budget, EvaluationBudget):
        raise TypeError('EvaluationBudget required')
    session_path = Path(session_path).resolve()
    panel_path = Path(panel_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.is_relative_to(session_path):
        raise ValueError('evaluation output must be outside the synthesis session')
    if panel_path.is_relative_to(session_path):
        raise ValueError('evaluation panel must be outside the synthesis session')
    submission = verify_frozen(session_path, release_path=release_path)
    if submission.get('selection') != 'agent_explicit_submission':
        raise ValueError('evaluation requires an explicit agent submission')
    panel, tasks = load_panel(panel_path)
    if not tasks:
        raise ValueError('empty evaluation panel')
    state = json.loads((session_path / 'state.json').read_text())
    prediction_profile = json.loads((session_path / 'session.json').read_text()).get('prediction')
    created = state.get('created_unix')
    if isinstance(created, bool) or not isinstance(created, (int, float)) or not math.isfinite(created):
        raise ValueError('synthesis creation timestamp required for panel preregistration')
    if panel['generated_unix'] > created:
        raise ValueError('evaluation panel must be frozen before synthesis session creation')
    if output_path.exists():
        raise FileExistsError('evaluation output already exists')
    submission_hash = seal.digest(session_path / 'frozen/manifest.json')
    panel_hash = seal.digest(panel_path)
    # Host-only ledger, never available through workspace tools. A failed or
    # interrupted attempt remains consumed; changing output directory cannot
    # silently repeat the same submitted program / panel pair.
    claim = safe_path(session_path, 'evaluation_claims/' + content_hash(
        {'submission_sha256': submission_hash, 'panel_instances_sha256': panel['instances_sha256']}) + '.json')
    claim.parent.mkdir(exist_ok=True)
    _write_once(claim, {'submission_manifest_sha256': submission_hash,
                       'panel_content_sha256': panel['content_sha256'],
                       'panel_instances_sha256': panel['instances_sha256'],
                       'output_path': str(output_path), 'claimed_unix': time.time()})
    output_path.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started + budget.wall_seconds
    package = output_path / 'package'
    shutil.copytree(session_path / 'frozen/package', package)
    _verify_package(package, submission['files_sha256'])
    _write_once(output_path / 'panel.json', panel)
    _write_once(output_path / 'submission.json', submission)
    manifest = {'kind': 'frozen_evaluation', 'session_path': str(session_path),
                'panel_sha256': panel_hash, 'panel_content_sha256': panel['content_sha256'],
                'submission_manifest_sha256': submission_hash,
                'release_sha256': seal.digest(release_path), 'budget': asdict(budget),
                'scheduled_episodes': len(tasks), 'started_unix': time.time(),
                'panel_generated_unix': panel['generated_unix'], 'synthesis_created_unix': created,
                'panel_preregistered_before_synthesis': True,
                'cross_episode_state': False, 'development_feedback': False,
                'selection': 'evaluate_explicit_submission_without_selection',
                'failure_cost_seconds': 180., 'denominator': 'all scheduled panel instances'}
    _write_once(output_path / 'manifest.json', manifest)
    results, stopped = [], None
    for index, task in enumerate(tasks):
        relative = f'episodes/{index:04d}_{task.instance.name}'
        row = {'index': index, 'task': task.instance.name, 'family': task.family,
               'task_sha256': task.sha256, 'episode_path': relative, 'attempted': False}
        if stopped is None and time.monotonic() >= deadline:
            stopped = 'evaluation_wall_budget_exhausted'
        if stopped is None:
            try:
                verify_frozen(session_path, release_path=release_path)
                if seal.digest(session_path / 'frozen/manifest.json') != submission_hash:
                    raise ValueError('submission manifest changed during evaluation')
                if seal.digest(panel_path) != panel_hash:
                    raise ValueError('panel changed during evaluation')
                _verify_package(package, submission['files_sha256'])
            except Exception as exc:
                stopped = 'integrity_error'
                row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        if stopped is not None:
            summary = {'status': 'not_run', 'success': False, 'termination': stopped,
                       'task_cost_seconds': 180., 'executed_control_ticks': 0}
        else:
            row['attempted'] = True
            try:
                summary = run_episode(task, package_command(package), output_path / relative,
                    RunBudget(budget.max_decisions, min(budget.episode_wall_seconds, max(.001, deadline - time.monotonic())),
                              budget.response_seconds),
                    provenance={'generated_agent_baseline': True, 'method': submission['version'],
                                'submission_manifest_sha256': submission_hash,
                                'files_sha256': submission['files_sha256'], 'evaluation_panel_sha256': panel_hash},
                    security_sandbox={'kind': 'bubblewrap', 'read_only_package': True,
                                      'development_rpc': False, 'fresh_process_per_episode': True},
                    max_control_ticks=budget.max_control_ticks, wall_deadline=deadline,
                    **({'prediction_profile': prediction_profile} if prediction_profile else {}))
            except Exception as exc:
                summary = {'status': 'infrastructure_error', 'success': False,
                           'termination': 'infrastructure_error', 'task_cost_seconds': 180.,
                           'executed_control_ticks': 0,
                           'error': {'type': type(exc).__name__, 'message': str(exc)}}
        row.update(status=summary['status'], success=bool(summary['success']),
                   termination=summary.get('termination'), summary=summary)
        results.append(row)
        with (output_path / 'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    attempts = sum(row['attempted'] for row in results)
    successes = sum(row['success'] for row in results)
    infrastructure_errors = sum(row['status'] == 'infrastructure_error' for row in results)
    summary = {'kind': 'frozen_evaluation', 'status': 'complete' if attempts == len(tasks) and not infrastructure_errors else 'incomplete',
               'session_path': str(session_path), 'scheduled_episodes': len(tasks),
               'attempted_episodes': attempts, 'successful_episodes': successes,
               'infrastructure_errors': infrastructure_errors, 'not_run_episodes': len(tasks) - attempts,
               'success_rate': successes / len(tasks), 'denominator': 'all scheduled panel instances',
               'total_task_cost_seconds': sum(row['summary']['task_cost_seconds'] for row in results),
               'executed_control_ticks': sum(row['summary'].get('executed_control_ticks', 0) for row in results),
               'wall_seconds': time.monotonic() - started, 'stop_reason': stopped,
               'results': results}
    _write_once(output_path / 'summary.json', summary)
    _write_once(output_path / 'artifacts_sha256.json', {
        str(p.relative_to(output_path)): seal.digest(p) for p in output_path.rglob('*') if p.is_file()})
    return summary
