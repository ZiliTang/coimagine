"""D002 v17: age-decayed switch-cost experiment (A stage first).

The A stage is read-only physics: each selected D001 boundary is reconstructed
from reset by replaying the recorded 5 Hz command prefix, then two 3.2 s windows
are compared (`recorded` vs `hold` the same same-push pair).  No policy, reward,
checkpoint or training code is changed here.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from coimagine.random_start import spec
from experiments.option_rl_v14.mode_env import LearningEnv
from experiments.option_rl_v14.interface import observe, ACTIONS
from experiments.option_rl_v14.scenes import geometry_audit
from experiments.option_rl_v16.live import atomic_json

TOLERANCE = 1e-5
WINDOW_SECONDS = 3.2
WINDOW_TICKS = 64
TICKS_PER_SAMPLE = 4
STAGE = 'A'


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class DiagEnv(LearningEnv):
    """No behaviour change; add the handshake/ledger-relevant fields to each 20 Hz row."""

    def _row(self, skill_index, period, action_0, action_1, infos, observations, forces,
             conditions, penetration):
        row = super()._row(skill_index, period, action_0, action_1, infos, observations,
                           forces, conditions, penetration)
        for i, action in enumerate((action_0, action_1)):
            slot = self.slots[i]
            skill = slot.skill
            info = row[f'r{i}']
            info.update(
                requested_label=action.label,
                requested_kind=action.kind,
                requested_type=int(getattr(action, 'type_index', -1)),
                requested_effect_kind=getattr(action, 'effect_kind', None),
                committed_label=slot.action.label,
                committed_kind=slot.action.kind,
                committed_type=int(getattr(slot.action, 'type_index', -1)),
                committed_effect_kind=getattr(slot.action, 'effect_kind', None),
                pending_label=slot.pending.label if slot.pending is not None else None,
                pause_reason=slot.pause_reason,
                contact_ready=bool(getattr(skill, 'contact_ready', False)),
                joint_active=bool(getattr(skill, 'joint_active', False)),
                ready_flag=bool(getattr(skill, 'ready', False)),
                failed=bool(skill.done),
            )
        return row


def make_diag_env(instance):
    if not geometry_audit(instance)['legal']:
        raise ValueError('illegal initial geometry')
    return DiagEnv(instance, context='extended')


def _requested_family(action):
    if action.kind == 'wait':
        return 'BRAKE'
    if action.kind == 'follow':
        return 'FOLLOW'
    if action.kind == 'reposition':
        return 'REPOSITION'
    if action.kind == 'push' and action.effect_kind == 'rotate':
        return 'ROTATE_CCW' if action.rotation_sign > 0 else 'ROTATE_CW'
    return 'TRANSLATE'


def _same_translate_push(row):
    a, b = row['r0'], row['r1']
    return (a['committed_kind'] == 'push' and b['committed_kind'] == 'push'
            and a['committed_effect_kind'] == 'translate'
            and b['committed_effect_kind'] == 'translate'
            and a['committed_type'] == b['committed_type'])


def _metrics(rows, start_distance):
    end_distance = float(rows[-1]['true_goal_distance']) if rows else start_distance
    min_distance = min([start_distance] + [float(r['true_goal_distance']) for r in rows])
    progress = float(start_distance - end_distance)
    engaged = 0.0
    both_push = 0.0
    one_push = 0.0
    switch = 0.0
    wait = 0.0
    failure = 0.0
    contact = 0.0
    for row in rows:
        dt = 0.05
        a, b = row['r0'], row['r1']
        if _same_translate_push(row) and a['contacting_object'] and b['contacting_object'] \
                and a['contact_ready'] and b['contact_ready'] and a['joint_active'] and b['joint_active']:
            engaged += dt
        if a['contacting_object'] and b['contacting_object']:
            contact += dt
        pa = a['stage'] == 'push'
        pb = b['stage'] == 'push'
        if pa and pb:
            both_push += dt
        elif pa or pb:
            one_push += dt
        if a['pending_label'] is not None or b['pending_label'] is not None \
                or a['stage'] == 'switching' or b['stage'] == 'switching':
            switch += dt
        if a['pause_reason'] in ('peer_wait', 'perception_wait', 'peer_push_switching', 'peer_push_failed') \
                or b['pause_reason'] in ('peer_wait', 'perception_wait', 'peer_push_switching', 'peer_push_failed') \
                or a['stage'] in ('peer_wait', 'perception_wait') or b['stage'] in ('peer_wait', 'perception_wait'):
            wait += dt
        if (a['exit_reason'] not in (None, 'continuing')) or (b['exit_reason'] not in (None, 'continuing')) \
                or a['failed'] or b['failed']:
            failure += dt
    return dict(start_distance=float(start_distance), ended_distance=end_distance, min_distance=float(min_distance), progress=progress,
                engaged_seconds=float(engaged), contact_seconds=float(contact),
                both_push_stage_seconds=float(both_push), one_push_stage_seconds=float(one_push),
                switching_seconds=float(switch), wait_seconds=float(wait),
                failure_seconds=float(failure), rows=len(rows), window_seconds=len(rows) * 0.05)


def _source_frame(npz, t_start):
    times = np.asarray(npz['trace_t'], dtype=np.float64)
    index = int(np.argmin(np.abs(times - t_start)))
    return index, dict(object=np.asarray(npz['trace_object'][index], dtype=np.float64),
                       robots=np.asarray(npz['trace_robots'][index], dtype=np.float64),
                       time=float(times[index]))


def _pose_error(env, frame):
    obj = env.object_state()
    robots = env.robot_states()
    values = [obj.x - frame['object'][0], obj.y - frame['object'][1], _wrap(obj.angle - frame['object'][2])]
    for i in (0, 1):
        values += [robots[i].x - frame['robots'][i][0],
                   robots[i].y - frame['robots'][i][1],
                   _wrap(robots[i].angle - frame['robots'][i][2])]
    return float(max(abs(v) for v in values))


def run_branch(job):
    source = job['source']
    branch_kind = job['branch']
    d001 = Path(job['d001'])
    out = Path(job['out'])
    config = job['config']
    episode_id = f"joint_types__{source['root']}__random__{source['seed']}"
    episode_json = json.loads((d001 / 'episodes' / f'{episode_id}.json').read_text())
    npz = np.load(d001 / 'episodes' / f'{episode_id}.npz', allow_pickle=False)
    from experiments.option_rl_v16.learnability import load_panel
    train, _ = load_panel(job['panel'])
    instance = {i.name: i for i in train}[source['root']]
    env = make_diag_env(instance)
    prefix = [s for s in episode_json['execution_samples'] if float(s['time']) <= float(source['t_start']) + 1e-9]
    after = [s for s in episode_json['execution_samples'] if float(s['time']) > float(source['t_start']) + 1e-9]
    source_action = (int(source['a0']), int(source['a1']))
    if not after:
        raise RuntimeError('no execution sample after source')
    first = after[0]
    if (int(first['a0']), int(first['a1'])) != source_action:
        raise RuntimeError(f'source action mismatch: {first} vs {source_action}')

    for sample in prefix:
        env.joint_step(int(sample['a0']), int(sample['a1']), record=False)
        if env.termination is not None:
            raise RuntimeError('termination during prefix replay')

    obs, m0, m1 = observe(env)
    di = int(source['decision_index'])
    raw_error = float(np.max(np.abs(obs.astype(np.float64) - np.asarray(npz['raw_obs'][di], dtype=np.float64))))
    mask_error = int(np.sum(m0 != npz['mask0'][di]) + np.sum(m1 != npz['mask1'][di]))
    frame_index, frame = _source_frame(npz, float(source['t_start']))
    pose_error = _pose_error(env, frame)
    total_periods = int(env.total_periods)
    expected_periods = int(round(float(source['t_start']) / spec.CONTROL_DT))
    source_ok = (raw_error <= TOLERANCE and mask_error == 0 and pose_error <= TOLERANCE
                 and total_periods == expected_periods)
    source_report = dict(raw_error=raw_error, mask_error=mask_error,
                         pose_error=pose_error, total_periods=total_periods,
                         expected_periods=expected_periods, frame_time=frame['time'],
                         source_frame_index=frame_index, ok=bool(source_ok))
    if not source_ok:
        atomic_json(out / 'branches' / f"{episode_id}__{branch_kind}.json",
                    dict(episode_id=episode_id, branch=branch_kind, source=source,
                         source_report=source_report, status='incomplete'))
        return dict(episode_id=episode_id, branch=branch_kind, pair=f"{episode_id}__d{di}",
                    source=source, source_report=source_report, status='incomplete',
                    branch_metrics=None)

    start_distance = float(env.object_distance())
    env.rows = []
    actions = []
    if branch_kind == 'recorded':
        window = after[:16]
        for sample in window:
            env.joint_step(int(sample['a0']), int(sample['a1']), record=True)
            actions.append(dict(t=float(sample['time']), a0=int(sample['a0']), a1=int(sample['a1'])))
            if env.termination is not None:
                break
    else:
        for k in range(16):
            env.joint_step(source_action[0], source_action[1], record=True)
            actions.append(dict(t=float(env.total_periods) * spec.CONTROL_DT,
                                a0=source_action[0], a1=source_action[1]))
            if env.termination is not None:
                break
    rows = [r for r in env.rows]
    metrics = _metrics(rows, start_distance)
    termination = env.termination
    success = bool(env.success)
    anomaly = termination in ('numeric_failure', 'penetration_limit', 'out_of_bounds')
    budget = termination == 'task_budget_exhausted'
    truncated = bool(metrics['rows'] < WINDOW_TICKS)
    # The recorded branch must reproduce the old D001 5 Hz trace over its overlap.
    overlap_error = None
    overlap_count = 0
    if branch_kind == 'recorded':
        errors = []
        for k, row in enumerate(rows):
            if (k + 1) % TICKS_PER_SAMPLE != 0:
                continue
            times = np.asarray(npz['trace_t'], dtype=np.float64)
            idx = int(np.argmin(np.abs(times - float(row['time']))))
            truth = np.asarray(npz['trace_object'][idx], dtype=np.float64)
            robots = np.asarray(npz['trace_robots'][idx], dtype=np.float64)
            values = [row['true']['object'][0] - truth[0], row['true']['object'][1] - truth[1],
                      _wrap(row['true']['object'][2] - truth[2])]
            for i in (0, 1):
                values += [row['true']['robots'][i][0] - robots[i][0],
                           row['true']['robots'][i][1] - robots[i][1],
                           _wrap(row['true']['robots'][i][2] - robots[i][2])]
            errors.append(max(abs(v) for v in values))
            overlap_count += 1
        overlap_error = float(max(errors)) if errors else None
    overlap_ok = bool(overlap_error is not None and overlap_error <= TOLERANCE) if branch_kind == 'recorded' else True
    branch_id = f"{episode_id}__{branch_kind}"
    branch = dict(episode_id=episode_id, branch=branch_kind,
                  pair=f"{episode_id}__d{di}", root=source['root'], seed=int(source['seed']),
                  decision_index=di, t_start=float(source['t_start']),
                  source_action=list(source_action), actions=actions, rows=rows,
                  metrics=metrics, termination=termination, success=success,
                  anomaly=bool(anomaly), budget=bool(budget), truncated=truncated,
                  source_report=source_report, recorded_overlap_error=overlap_error,
                  recorded_overlap_count=overlap_count, recorded_overlap_ok=overlap_ok,
                  status='ok' if source_ok and overlap_ok else 'incomplete')
    atomic_json(out / 'branches' / f'{branch_id}.json', branch)
    return dict(episode_id=episode_id, branch=branch_kind, pair=branch['pair'],
                root=source['root'], seed=int(source['seed']), decision_index=di,
                t_start=float(source['t_start']), source=source, branch_metrics=metrics,
                termination=termination, success=success, anomaly=bool(anomaly),
                budget=bool(budget), truncated=truncated, source_report=source_report,
                recorded_overlap_error=overlap_error, recorded_overlap_ok=overlap_ok,
                actions=actions, status=branch['status'])


def _validate_sources(config, d001):
    sources = config['stage_a']['selected_sources']
    errors = []
    for source in sources:
        episode_id = f"joint_types__{source['root']}__random__{source['seed']}"
        json_path = d001 / 'episodes' / f'{episode_id}.json'
        npz_path = d001 / 'episodes' / f'{episode_id}.npz'
        if not json_path.exists() or not npz_path.exists():
            errors.append(f'missing {episode_id}')
            continue
        data = json.loads(json_path.read_text())
        actions = data['actions']
        di = int(source['decision_index'])
        t = float(source['t_start'])
        if di >= len(actions):
            errors.append(f'{episode_id}: decision {di} out of range')
            continue
        prev = float(actions[di - 1]['time']) if di > 0 else 0.0
        if abs(prev - t) > 1e-9:
            errors.append(f'{episode_id}: t_start {t} != previous end {prev}')
        if (int(actions[di]['a0']), int(actions[di]['a1'])) != (int(source['a0']), int(source['a1'])):
            errors.append(f'{episode_id}: action mismatch')
        first = next((s for s in data['execution_samples'] if float(s['time']) > t + 1e-9), None)
        if first is None or (int(first['a0']), int(first['a1'])) != (int(source['a0']), int(source['a1'])):
            errors.append(f'{episode_id}: first execution sample mismatch')
    return sources, errors


def _replay(out, branches, config):
    from experiments.option_rl_v45.render_episodes import TEMPLATE
    from experiments.option_rl_v16.learnability import load_panel
    train, _ = load_panel(config['panel'])
    by_name = {i.name: i for i in train}
    episodes = []
    arena = None
    for branch in branches:
        if branch['branch_metrics'] is None:
            continue
        branch_file = out / 'branches' / f"{branch['episode_id']}__{branch['branch']}.json"
        data = json.loads(branch_file.read_text())
        rows = data['rows']
        instance = by_name[branch['root']]
        arena = list(instance.arena)
        frames = [dict(t=float(r['time']),
                       o=[float(v) for v in r['true']['object'][:3]],
                       r=[[float(v) for v in rr[:3]] for rr in r['true']['robots']],
                       ct=[bool(r['r0']['contacting_object']), bool(r['r1']['contacting_object'])],
                       st=[str(r['r0']['stage']), str(r['r1']['stage'])],
                       fo=[float(r['r0']['force_norm']), float(r['r1']['force_norm'])])
                  for r in rows]
        decisions = [dict(t=float(a['t']), a0=ACTIONS[int(a['a0'])].label,
                          a1=ACTIONS[int(a['a1'])].label) for a in data['actions']]
        episodes.append(dict(
            name=f"{branch['episode_id']}__{branch['branch']}",
            goal=list(instance.goal_centre), radius=float(instance.goal_radius),
            object_size=[spec.OBJECT_WIDTH, spec.OBJECT_HEIGHT],
            start_goal_distance=round(float(data['metrics'].get('start_distance', 0.0)), 3),
            frames=frames, decisions=decisions, success=bool(branch['success']),
            termination=data.get('termination'),
            summary=dict(branch=branch['branch'], root=branch['root'], seed=branch['seed'],
                         decision=branch['decision_index'],
                         progress=data['metrics']['progress'], engaged=data['metrics']['engaged_seconds']),
        ))
    payload = dict(checkpoint='D002 A | recorded vs hold 3.2s | 20 Hz true-state replay',
                   arena=arena or [5.0, 4.0], episodes=episodes)
    html = TEMPLATE.replace('__DATA__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    html = html.replace('ARENA[1]', 'DATA.arena[1]').replace('ARENA[0]', 'DATA.arena[0]')
    (out / 'replay.html').write_text(html)
    (out / 'replay_payload.json').write_text(json.dumps(payload, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--d001', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--deadline-seconds', type=float, default=120.)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'output directory must be new/empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'branches').mkdir(exist_ok=True)
    config = json.loads(args.config.read_text())
    d001 = args.d001.resolve()
    if not (d001 / 'manifest.json').exists():
        raise SystemExit('D001 output missing manifest')
    d001_manifest = json.loads((d001 / 'manifest.json').read_text())
    if d001_manifest.get('status') != 'complete':
        raise SystemExit('D001 is not complete')
    sources, source_errors = _validate_sources(config, d001)
    if source_errors:
        raise SystemExit('source validation failed: ' + '; '.join(source_errors))
    if len(sources) != 12:
        raise SystemExit(f'expected 12 sources, got {len(sources)}')
    parent = Path(config['parent_checkpoint'])
    panel = Path(config['panel'])
    input_hashes = dict(parent=_sha256(parent), panel=_sha256(panel),
                        d001_manifest=_sha256(d001 / 'manifest.json'))
    if input_hashes['parent'] != config['parent_sha256']:
        raise SystemExit('parent hash mismatch')
    if input_hashes['panel'] != config['panel_sha256']:
        raise SystemExit('panel hash mismatch')
    preflight = dict(stage=STAGE, sources=len(sources), source_errors=[],
                     input_hashes=input_hashes, d001=str(d001),
                     config=str(args.config.resolve()), workers=args.workers,
                     deadline_seconds=args.deadline_seconds)
    atomic_json(out / 'preflight.json', preflight)
    if args.preflight:
        print(json.dumps(preflight, indent=1))
        return

    started = time.monotonic()
    jobs = []
    for source in sources:
        for branch in ('recorded', 'hold'):
            jobs.append(dict(source=source, branch=branch, out=str(out), d001=str(d001),
                             panel=str(args.panel), config=config))
    if len(jobs) != 24:
        raise SystemExit(f'expected 24 branches, got {len(jobs)}')
    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=__import__('multiprocessing').get_context('spawn')) as executor:
        futures = {executor.submit(run_branch, job): job for job in jobs}
        pending = set(futures)
        while pending:
            remaining = args.deadline_seconds - (time.monotonic() - started)
            if remaining <= 0:
                for future in pending:
                    future.cancel()
                errors.append(dict(error='stage_a_deadline'))
                break
            done, pending = wait(pending, timeout=min(20., remaining), return_when=FIRST_COMPLETED)
            for future in done:
                job = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(dict(job=job, error=repr(exc)))
    stage_seconds = time.monotonic() - started
    # Pair by source.
    pairs = {}
    for result in results:
        pairs.setdefault(result['pair'], {})[result['branch']] = result
    pair_rows = []
    for pair_id in sorted(pairs):
        if set(pairs[pair_id]) != {'recorded', 'hold'}:
            continue
        rec, hold = pairs[pair_id]['recorded'], pairs[pair_id]['hold']
        if rec.get('branch_metrics') is None or hold.get('branch_metrics') is None:
            continue
        delta = float(hold['branch_metrics']['progress'] - rec['branch_metrics']['progress'])
        delta_eng = float(hold['branch_metrics']['engaged_seconds'] - rec['branch_metrics']['engaged_seconds'])
        pair_rows.append(dict(
            pair=pair_id, root=rec['root'], seed=rec['seed'], decision_index=rec['decision_index'],
            t_start=rec['t_start'], source_action=[int(rec['source']['a0']), int(rec['source']['a1'])],
            recorded_progress=rec['branch_metrics']['progress'],
            hold_progress=hold['branch_metrics']['progress'], delta_progress=delta,
            recorded_engaged=rec['branch_metrics']['engaged_seconds'],
            hold_engaged=hold['branch_metrics']['engaged_seconds'], delta_engaged=delta_eng,
            recorded_both_push=rec['branch_metrics']['both_push_stage_seconds'],
            hold_both_push=hold['branch_metrics']['both_push_stage_seconds'],
            recorded_wait=rec['branch_metrics']['wait_seconds'],
            hold_wait=hold['branch_metrics']['wait_seconds'],
            recorded_failure=rec['branch_metrics']['failure_seconds'],
            hold_failure=hold['branch_metrics']['failure_seconds'],
            recorded_success=bool(rec['success']), hold_success=bool(hold['success']),
            recorded_anomaly=bool(rec['anomaly']), hold_anomaly=bool(hold['anomaly']),
            recorded_truncated=bool(rec['truncated']), hold_truncated=bool(hold['truncated']),
            recorded_source_ok=bool(rec['source_report']['ok']),
            hold_source_ok=bool(hold['source_report']['ok']),
            recorded_overlap_ok=bool(rec['recorded_overlap_ok']),
            hold_overlap_ok=True,
        ))
    # Gate.
    def q(values, p):
        return float(np.quantile(values, p)) if values else None
    improved = [p for p in pair_rows if p['delta_progress'] >= 0.01]
    severe = [p for p in pair_rows if p['delta_progress'] <= -0.02]
    engaged_improved = [p for p in pair_rows if p['delta_engaged'] >= 0.4]
    per_root = {root: sum(1 for p in pair_rows if p['root'] == root and p['delta_progress'] >= 0.01)
                for root in ('dual_2', 'dual_0')}
    deltas = [p['delta_progress'] for p in pair_rows]
    gate = dict(
        complete_pairs=len(pair_rows) == 12,
        all_sources_reproducible=bool(pair_rows and all(p['recorded_source_ok'] and p['hold_source_ok'] for p in pair_rows)),
        recorded_overlap_ok=bool(pair_rows and all(p['recorded_overlap_ok'] for p in pair_rows)),
        min_improved_pairs=len(improved) >= 6,
        min_improved_per_root=all(v >= 2 for v in per_root.values()),
        min_median_delta=(q(deltas, .5) is not None and q(deltas, .5) >= 0.01),
        max_severe_regressions=len(severe) <= 2,
        min_engagement_pairs=len(engaged_improved) >= 4,
        hold_no_new_anomalies=all(not p['hold_anomaly'] for p in pair_rows),
    )
    gate['pass'] = all(bool(v) for v in gate.values())
    atomic_json(out / 'gate.json', gate)
    atomic_json(out / 'pairs.json', pair_rows)
    _write_csv(out / 'pairs.csv', pair_rows)
    _write_csv(out / 'branches.csv', [r for r in results if r.get('branch_metrics') is not None])
    branch_files = sorted((out / 'branches').glob('*.json'))
    summary = dict(stage=STAGE, task='D002', gate=gate, pairs=len(pair_rows),
                   deltas=dict(median=q(deltas, .5), q25=q(deltas, .25), q75=q(deltas, .75),
                               min=min(deltas) if deltas else None, max=max(deltas) if deltas else None),
                   improved=len(improved), severe=len(severe), engaged_improved=len(engaged_improved),
                   per_root=per_root, stage_a_seconds=stage_seconds, errors=errors,
                   branch_files=[str(p.relative_to(out)) for p in branch_files],
                   input_hashes=input_hashes, d001_manifest=d001_manifest.get('status'))
    atomic_json(out / 'summary.json', summary)
    if branch_files:
        _replay(out, results, config)
    source_files = [Path(__file__),
                    ROOT / 'experiments/option_rl_v14/interface.py',
                    ROOT / 'experiments/option_rl_v14/mode_env.py',
                    ROOT / 'experiments/option_rl_v14/scenes.py',
                    ROOT / 'experiments/option_rl_v14/controller.py',
                    ROOT / 'experiments/option_rl_v13/controller.py',
                    ROOT / 'experiments/option_rl_v13/mode_env.py',
                    ROOT / 'experiments/option_rl_v13/push_controller.py',
                    ROOT / 'experiments/option_rl_v16/learnability.py',
                    ROOT / 'experiments/option_rl_v16/live.py',
                    ROOT / 'experiments/option_rl_v45/render_episodes.py',
                    ROOT / 'src/coimagine/random_start/env.py',
                    ROOT / 'src/coimagine/random_start/spec.py',
                    ROOT / 'src/coimagine/random_start/skills.py',
                    ROOT / 'src/coimagine/random_start/effect_push.py',
                    ROOT / 'docs/agent_exchange/tasks/D002_Switch_Cost_and_Persistence.md',
                    ROOT / 'docs/agent_exchange/tasks/D002_config.json']
    source_hashes = {str(p.relative_to(ROOT)): _sha256(p) for p in source_files if p.exists()}
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for p in source_files:
        if p.exists():
            dest = snapshot / str(p.relative_to(ROOT))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    (out / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=1))
    manifest = dict(task='D002', version=1, stage=STAGE,
                    implementation_version='v17-age-decay-switch-cost-v1',
                    status='complete' if gate['pass'] and not errors else ('incomplete' if errors or not gate['complete_pairs'] else 'failed_gate'),
                    stage_a_seconds=stage_seconds, deadline_seconds=args.deadline_seconds,
                    workers=args.workers, pairs=len(pair_rows), branches=len(results),
                    errors=errors, gate=gate, input_hashes=input_hashes, source_hashes=source_hashes,
                    source_errors=source_errors, config=str(args.config.resolve()),
                    d001=str(d001), head=_git('HEAD'), dirty_files=len(_git_status().splitlines()),
                    command=sys.argv, started_at=time.time() - stage_seconds, ended_at=time.time())
    atomic_json(out / 'manifest.json', manifest)
    (out / 'task.md').write_text(ROOT.joinpath('docs/agent_exchange/tasks/D002_Switch_Cost_and_Persistence.md').read_text())
    (out / 'config.json').write_text(args.config.read_text())
    print(json.dumps(dict(stage=STAGE, status=manifest['status'], pairs=len(pair_rows),
                          stage_seconds=round(stage_seconds, 1), gate=gate), indent=1))
    if not gate['pass']:
        raise SystemExit(0 if not errors else 1)


def _git(*args):
    import subprocess
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _git_status():
    return _git('status', '--porcelain')


def _write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    fields = list(rows[0].keys())
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            out = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    out[key] = json.dumps(value, ensure_ascii=False)
                else:
                    out[key] = value
            writer.writerow(out)


if __name__ == '__main__':
    main()
