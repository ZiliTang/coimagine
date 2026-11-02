"""Analyse the already-completed D001 episode budget without re-running rollouts.

The first D001 process completed all 92 episodes and wrote episode/decision CSVs,
then stopped in the gradient-recombination scalar conversion.  This script reads
the frozen episode artifacts and runs only the B analysis, same-push gradient and
replay/manifest generation.  No policy, normalizer, checkpoint or episode is changed.
"""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.option_rl_v16.diagnose_dual_credit import (
    BATCH_SEEDS, DUAL_ROOTS, EXPECTED, TASK_VERSION, _make_replay, _sha256, _write_csv,
    batch_analysis, same_push_gradient,
)
from experiments.option_rl_v16.train import load_checkpoint
from experiments.option_rl_v16.live import atomic_json
from coimagine.random_start import spec


def _a_summary(episode_rows):
    result = {}
    for model in ('independent', 'joint_types'):
        for root in DUAL_ROOTS:
            greedy = [e for e in episode_rows if e['model'] == model and e['root'] == root and e['mode'] == 'greedy']
            random = [e for e in episode_rows if e['model'] == model and e['root'] == root and e['mode'] == 'random']
            result[f'{model}/{root}'] = dict(
                greedy=dict(n=len(greedy), success=int(sum(e['success'] for e in greedy)),
                            return_=float(np.mean([e['return_'] for e in greedy])) if greedy else None,
                            seconds=float(np.mean([e['seconds'] for e in greedy])) if greedy else None,
                            min_distance=float(np.mean([e['min_distance'] for e in greedy])) if greedy else None),
                random=dict(n=len(random), success=int(sum(e['success'] for e in random)),
                            return_=float(np.mean([e['return_'] for e in random])) if random else None,
                            seconds=float(np.mean([e['seconds'] for e in random])) if random else None,
                            min_distance=float(np.mean([e['min_distance'] for e in random])) if random else None,
                            entered_goal_not_success=int(sum(e['entered_goal_not_success'] for e in random)),
                            anomalies=int(sum(e['anomaly'] for e in random)),
                            same_push_prob_mean=float(np.mean([e['same_push_prob_mean'] for e in random])) if random else None,
                            same_push_selected=int(sum(e['same_push_selected'] for e in random)),
                            same_push_selected_seconds=float(sum(e['same_push_selected_seconds'] for e in random)),
                            same_push_both_push_seconds=float(sum(e['same_push_both_push_seconds'] for e in random)),
                            same_push_wait_seconds=float(sum(e['same_push_wait_seconds'] for e in random))),
            )
    return result


def _eval_inputs(episode_meta):
    inputs = []
    for root in DUAL_ROOTS:
        episode_id = f'joint_types__{root}__greedy__greedy'
        meta = episode_meta.get(episode_id)
        if not meta:
            continue
        npz = np.load(meta['npz'], allow_pickle=False)
        for i in range(min(10, len(npz['a0']))):
            inputs.append(dict(source=episode_id, decision=i,
                               obs=npz['obs'][i].copy(),
                               mask0=npz['mask0'][i].copy(),
                               mask1=npz['mask1'][i].copy()))
    unique = []
    seen = set()
    for item in inputs:
        key = hashlib.sha256(item['obs'].tobytes()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--independent', type=Path, required=True)
    parser.add_argument('--joint', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    analysis_started = time.monotonic()
    preflight = json.loads((out / 'preflight.json').read_text())
    input_hashes = preflight['input_hashes']
    for key, expected in EXPECTED.items():
        actual = input_hashes['panel'] if key == 'panel' else input_hashes[key]
        if actual != expected:
            raise SystemExit(f'{key} SHA256 mismatch: {actual} != {expected}')

    episode_files = sorted((out / 'episodes').glob('*.json'))
    episode_rows = []
    for path in episode_files:
        data = json.loads(path.read_text())
        episode_rows.append(data['episode'])
    episode_rows.sort(key=lambda e: e['episode_id'])
    if len(episode_rows) != 92:
        raise SystemExit(f'expected 92 completed episodes, found {len(episode_rows)}')
    episode_meta = {e['episode_id']: e for e in episode_rows}
    episodes = {e['episode_id']: e['npz'] for e in episode_rows}
    for episode_id, path in episodes.items():
        if not Path(path).exists():
            raise SystemExit(f'missing episode NPZ: {episode_id} {path}')

    A = _a_summary(episode_rows)

    joint_policy, joint_normalizer, joint_data = load_checkpoint(args.joint)
    joint_policy.eval()
    batch_summaries = []
    advantage_rows = []
    batch_grads = {}
    for batch_seed in BATCH_SEEDS:
        result = batch_analysis(joint_policy, joint_data, episodes, episode_meta, batch_seed, out)
        if 'summary' in result:
            batch_summaries.append(result['summary'])
            advantage_rows.extend(result['advantage_rows'])
            batch_grads[f'batch_{batch_seed}'] = result['g']
        else:
            batch_summaries.append(result)
    _write_csv(out / 'advantage_summary.csv', advantage_rows)

    eval_inputs = _eval_inputs(episode_meta)
    same_grad = same_push_gradient(joint_policy, eval_inputs, batch_grads) if eval_inputs else dict(status='no_inputs')
    atomic_json(out / 'same_push_gradient.json', same_grad)

    selected = []
    for episode_id in [e['episode_id'] for e in episode_rows if e['mode'] == 'greedy' and not e['success']]:
        if episode_id not in selected:
            selected.append(episode_id)
    for model in ('independent', 'joint_types'):
        for root in DUAL_ROOTS:
            random = [e for e in episode_rows if e['model'] == model and e['root'] == root and e['mode'] == 'random']
            for entry in (next((e for e in random if e['success']), None),
                          next((e for e in random if not e['success']), None)):
                if entry and entry['episode_id'] not in selected:
                    selected.append(entry['episode_id'])
    if selected:
        _make_replay(out, selected, episodes, episode_meta)

    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout
    (out / 'git_status.txt').write_text(dirty)
    source_files = [Path(__file__), ROOT / 'experiments/option_rl_v16/diagnose_dual_credit.py',
                    ROOT / 'experiments/option_rl_v16/train.py', ROOT / 'experiments/option_rl_v16/interface.py',
                    ROOT / 'experiments/option_rl_v15/interface.py', ROOT / 'experiments/option_rl_v14/interface.py',
                    ROOT / 'experiments/option_rl_v14/mode_env.py', ROOT / 'experiments/option_rl_v13/controller.py',
                    ROOT / 'experiments/option_rl_v13/mode_env.py', ROOT / 'src/coimagine/random_start/env.py',
                    ROOT / 'src/coimagine/random_start/spec.py']
    source_hashes = {str(p.relative_to(ROOT)): _sha256(p) for p in source_files if p.exists()}
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for p in source_files:
        if p.exists():
            dest = snapshot / str(p.relative_to(ROOT))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    (out / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=1))

    # Simulation phase: preflight completion -> episode CSV write. Analysis phase: this run.
    episode_start = (out / 'preflight.json').stat().st_mtime
    episode_end = (out / 'episode_metrics.csv').stat().st_mtime
    episode_seconds = max(0.0, episode_end - episode_start)
    analysis_seconds = time.monotonic() - analysis_started
    run_seconds = episode_seconds + analysis_seconds
    batch_complete = all(b.get('status') == 'complete' for b in batch_summaries)
    status = 'complete' if (len(episode_rows) == 92 and batch_complete and run_seconds <= 900.) else 'incomplete'
    reason = None
    if status != 'complete':
        reason = 'run_seconds_exceeded_900' if run_seconds > 900. else 'incomplete_batches'
    manifest = dict(
        task_version=TASK_VERSION, status=status, incomplete_reason=reason,
        out=str(out), started_at=episode_start, ended_at=time.time(),
        episode_seconds=episode_seconds, analysis_seconds=analysis_seconds,
        run_seconds=run_seconds, wall_seconds_total=time.time() - episode_start,
        workers=4, deadline_seconds=900., jobs=92, completed_jobs=len(episode_rows),
        errors=[], analysis_retry=dict(
            reason='first analysis crashed on tensor scalar conversion at gradient recombination; '
                   'episodes were not rerun; only the analysis was rerun',
            episode_reruns=0),
        episode_budget=92, input_hashes=input_hashes, source_hashes=source_hashes,
        head=head, dirty_files=len(dirty.splitlines()), command=sys.argv,
        preflight=preflight,
    )
    atomic_json(out / 'manifest.json', manifest)
    summary = dict(task_version=TASK_VERSION, manifest=manifest, A=A,
                   B=dict(batches=batch_summaries, advantage_rows=len(advantage_rows)),
                   same_push_gradient=same_grad, replay=selected, eval_inputs=len(eval_inputs))
    atomic_json(out / 'summary.json', summary)
    print(json.dumps(dict(status=status, episodes=len(episode_rows), run_seconds=round(run_seconds, 1),
                          episode_seconds=round(episode_seconds, 1), analysis_seconds=round(analysis_seconds, 1),
                          batch_complete=batch_complete), indent=1))
    if status != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
