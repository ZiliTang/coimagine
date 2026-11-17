"""One bounded policy episode with separate policy transcript and truth evidence."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

from coimagine.random_start import spec as R
from ..interface import TaskEnvironment, contract, json_safe, validate_action
from .runtime_policy import JsonlPolicy
from ..tasks import geometry_audit, scene_metadata

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RunBudget:
    max_decisions: int = 300
    wall_seconds: float = 120.
    response_seconds: float = 20.

    def __post_init__(self):
        if type(self.max_decisions) is not int or self.max_decisions < 1:
            raise ValueError('max_decisions must be a positive integer')
        if any(not math.isfinite(v) or v <= 0 for v in (self.wall_seconds, self.response_seconds)):
            raise ValueError('wall and response budgets must be positive finite seconds')


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False)+'\n')


def source_manifest(directory, command):
    """Archive project sources plus any explicit command file (not model weights)."""
    paths = sorted(p for base in ('src/coimagine', 'experiments')
                   for p in (ROOT/base).rglob('*.py') if '__pycache__' not in p.parts)
    snapshots = directory/'source_snapshot'
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        data = path.read_bytes()
        hashes[str(rel)] = hashlib.sha256(data).hexdigest()
        dest = snapshots/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    policy_files = {}
    for n, arg in enumerate(command[1:], 1):
        path = Path(arg)
        if path.is_file():
            data = path.read_bytes()
            dest = directory/'policy_files'/f'{n}_{path.name}'
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(data)
            policy_files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    return dict(head=git('rev-parse', 'HEAD'), dirty=git('status', '--short'),
                python=platform.python_version(), source_sha256=hashes,
                command_files_sha256=policy_files)


def replay_record(task, env, decisions, summary):
    frames, strip = [], []
    coordination = {row['tick']: row for row in env.coordinator.rows}
    for row in env.rows:
        states = [row[f'r{i}'] for i in (0, 1)]
        stages = [r.get('stage') for r in states]
        contacts = [r['contacting_object'] for r in states]
        coord = coordination.get(row['period']-1, {})
        frames.append(dict(t=row['time'], **row['true'], stage=stages, phase=coord.get('phase'),
            scan=[s['scan'] for s in states], target=[None, None], ref_centre=[None, None],
            mb=[dict(action=s['skill'], stage=s['stage'], contact=contacts[i],
                     force=row['forces'][i], reference=s['reference_velocity'],
                     desired=s['desired_velocity']) for i, s in enumerate(states)], progress=[{}, {}]))
        strip.append(dict(t=row['time'], tick=row['period'], stage=stages, phase=coord.get('phase'),
                          physical_contact=contacts, contact=contacts,
                          joint_active=coord.get('joint_active'), forces=row['forces']))
    # Commands use an exact time budget so UI replay preserves even 1-tick requests.
    sequence = [dict(a0=d['actions'][0], a1=d['actions'][1],
                     decisions=max(1, math.ceil((d['end_tick']-d['start_tick'])/4)),
                     time_budget_s=(d['end_tick']-d['start_tick'])*R.CONTROL_DT)
                for d in decisions if d['end_tick'] > d['start_tick']]
    meta = scene_metadata(task.instance)
    return dict(id=summary['run_id'], name=summary['run_id'], created=time.time(),
        case='planar_policy', scene=task.instance.name, scene_meta=meta,
        object=meta['initial']['object'], robots=meta['initial']['robots'],
        controller_version='option-v2', action_version=env.action_version,
        action_codebook_hash=env.action_codebook_hash, support_memory_enabled=False,
        sequence=sequence, overrides={}, frames=frames, strip=strip,
        success=summary['success'], termination=summary['termination'],
        periods=env.total_periods, reward=env.reward_total,
        diagnostic_note='truth visualization only; policy received sensors.jsonl, not this file')


def run_episode(task, command, directory, budget=RunBudget(), *, provenance=None,
                security_sandbox=None, max_control_ticks=None, wall_deadline=None,
                prediction_profile=None, prediction_ledger=None):
    if max_control_ticks is not None and (type(max_control_ticks) is not int or max_control_ticks < 1):
        raise ValueError('max_control_ticks must be a positive integer')
    if wall_deadline is not None and not math.isfinite(wall_deadline):
        raise ValueError('wall_deadline must be finite')
    audit = geometry_audit(task.instance)
    if not audit['legal']:
        raise ValueError(f'illegal scene: {audit["reasons"]}')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    predictor = None
    if prediction_profile is not None:
        from ..baseline.prediction import PredictionService, validate_profile, LIMITS
        validate_profile(prediction_profile)
        if prediction_profile['runtime']:
            predictor = PredictionService(ledger=prediction_ledger, log_path=directory/'prediction.jsonl', local_limits=LIMITS)
    manifest = source_manifest(directory, command)
    manifest.update(task=task.record(), task_sha256=task.sha256, geometry=audit,
                    command=command, budget=asdict(budget), interface=contract(),
                    started_unix=time.time(), protocol_is_security_sandbox=False,
                    policy_provenance=provenance, external_security_sandbox=security_sandbox,
                    max_control_ticks=max_control_ticks, wall_deadline_monotonic=wall_deadline,
                    prediction_profile=prediction_profile, planning_clock='execution physics paused')
    write_json(directory/'manifest.json', manifest)
    world = TaskEnvironment(task.instance)
    env = world._env  # harness-only access; never passed to policy process
    write_json(directory/'environment.json', env.config.as_dict())
    started = time.monotonic()
    deadline = min(started+budget.wall_seconds, wall_deadline) if wall_deadline is not None else started+budget.wall_seconds
    policy = None
    decisions = []
    reason, error, inference_seconds = None, None, 0.
    try:
        policy = JsonlPolicy(command, directory/'policy.stderr',
                             **({'rpc_handler': predictor.rpc} if predictor else {}))
        with (directory/'sensors.jsonl').open('x') as log:
            while env.termination is None:
                if max_control_ticks is not None and env.total_periods >= max_control_ticks:
                    reason = 'control_tick_budget_exhausted'
                    break
                if world.decisions >= budget.max_decisions:
                    reason = 'decision_budget_exhausted'
                    break
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    reason = 'wall_budget_exhausted'
                    break
                observation = world.observe()
                request = dict(observation=observation)
                if world.decisions == 0:
                    request.update(reset=True, contract=contract())
                log.write(json.dumps(dict(request=request), allow_nan=False)+'\n')
                log.flush()
                before = time.monotonic()
                if predictor:
                    predictor.begin_decision()
                try:
                    action = policy.request(request, min(remaining, budget.response_seconds))
                finally:
                    inference_seconds += time.monotonic()-before
                log.write(json.dumps(dict(response=action), allow_nan=False)+'\n')
                log.flush()
                validate_action(action, observation)
                # Enforce the wall deadline even during a multi-tick request.
                a0, a1, ticks = action['a0'], action['a1'], action['ticks']
                start_tick = env.total_periods
                for _ in range(ticks):
                    if max_control_ticks is not None and env.total_periods >= max_control_ticks:
                        reason = 'control_tick_budget_exhausted'
                        break
                    if time.monotonic() >= deadline:
                        reason = 'wall_budget_exhausted'
                        break
                    if env.termination is not None:
                        break
                    env.joint_step(a0, a1, record=True)
                world.decisions += 1
                env.skills_used = world.decisions
                decisions.append(dict(start_tick=start_tick, end_tick=env.total_periods,
                                      requested_ticks=ticks, actions=[a0, a1]))
                if reason:
                    break
            reason = reason or env.termination
    except Exception as exc:
        reason = 'policy_timeout' if isinstance(exc, TimeoutError) else 'execution_error'
        error = dict(type=type(exc).__name__, message=str(exc),
                     raw_response=policy.last_response if policy else None)
    finally:
        if policy:
            policy.close()
    elapsed = env.total_periods*R.CONTROL_DT
    complete = env.termination is not None
    success = bool(env.success and complete)
    # New reporting metric, never a reward/control modification. Every unsuccessful
    # attempt pays the full horizon plus a fixed horizon-sized failure charge,
    # including errors and early resource stops; failing early cannot be cheaper.
    summary = dict(run_id=directory.name, task=task.instance.name, task_sha256=task.sha256,
        status='complete' if complete else 'incomplete', success=success, termination=reason,
        attempted_episodes=1, successful_episodes=int(success), error=error,
        decisions=world.decisions, executed_control_ticks=env.total_periods,
        simulated_seconds=elapsed, wall_seconds=time.monotonic()-started,
        policy_seconds=inference_seconds, original_reward=env.reward_total,
        task_cost_seconds=elapsed if success else 2*env.config.time_budget,
        cost_definition='success: elapsed simulated seconds; all failures/incomplete: 2 * 90 s',
        final_goal_distance_diagnostic=env.object_distance(),
        generated_agent_baseline=bool(provenance and provenance.get('generated_agent_baseline')),
        policy_provenance=provenance)
    summary['process_exit']=getattr(policy,'exit_info',None) if policy else None
    if prediction_profile is not None:
        summary['prediction_usage'] = dict(predictor.local_usage) if predictor else {}
    write_json(directory/'summary.json', summary)
    write_json(directory/'decisions.json', decisions)
    write_json(directory/'truth.json', env.rows)
    write_json(directory/'replay.json', replay_record(task, env, decisions, summary))
    manifest.update(finished_unix=time.time(), status=summary['status'],
                    artifacts_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in directory.iterdir() if p.is_file() and p.name != 'manifest.json'})
    write_json(directory/'manifest.json', manifest)
    return summary
