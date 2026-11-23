"""Supervised host for the first trace-value construction/training/test round.

No LLM/provider calls. New implementation is isolated from frozen historical
learners. User's 2026-11-18 instruction authorizes direct implementation and runs;
the prior cross-machine old-feature discrepancy is retained, not a new-run gate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import dataset as D
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import evaluation as E
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.instrumentation import instrument
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.model import train_all
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.runtime import TraceLibrary, entries
from experiments.planar_tasks.semantic_rl.memory_flow import MemoryHistory

ROOT = Path(__file__).resolve().parents[3]


def source_hashes():
    base = ROOT/'experiments/planar_tasks/semantic_rl/program_trace_value_v1'
    paths = [base/name for name in ('__init__.py', 'instrumentation.py', 'runtime.py', 'dataset.py', 'model.py', 'evaluation.py')]
    paths.append(Path(__file__).resolve())
    paths.extend((base/'pose_kernel').glob('*.py'))
    paths.extend((base/'pose_kernel').glob('*.c'))
    paths.extend((base/'pose_kernel').glob('*.h'))
    paths.append(base.parent/'trusted_native.py')
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def verify_frozen(out):
    frozen = D.read(Path(out)/'inputs.json')
    if source_hashes() != frozen['new_sources']:
        raise ValueError('implementation changed after this run was frozen')
    bundle = Path(frozen['bundle'])
    if hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest() != frozen['bundle_manifest_sha256']:
        raise ValueError('input bundle manifest changed')


def prepare(bundle, out):
    library = entries(bundle)
    roots = D.split(bundle)
    selected = roots['pilot_train'] + roots['pilot_validation']
    rows = [r for r in D.read(bundle/'index.json') if r['case_id'] in selected]
    rows.sort(key=lambda r: (selected.index(r['case_id']), r['cap_id']))
    assert len(rows) == 336
    D.save(out/'split.json', roots)
    D.save(out/'selected_runs.json', rows)
    schemas = {}
    for entry in library:
        source, schema = instrument(entry['source'])
        schemas[entry['name']] = schema
        path = out/'instrumented'/ (entry['name']+'.py')
        path.parent.mkdir(exist_ok=True)
        path.write_text(source)
    D.save(out/'schema.json', schemas)
    frozen = dict(bundle=str(bundle), bundle_manifest_sha256=hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),
                  new_sources=source_hashes(), protocol=dict(arms=['GA', 'LOCAL', 'GLOBAL'],
                    seed=17, updates=2000, batch_size=64, optimizer='Adam', learning_rate=.0003,
                    train_roots=32, validation_roots=16, trajectories=336, evaluation_episodes=48,
                    reward='terminal success=1, other=0', gamma=1, max_ticks=1800, max_decisions=300,
                    target='CAP continuation V; online policy switching; no Q or RL optimizer in pilot',
                    sampling='uniform decision boundaries; all successes and failures',
                    arm_mask='trace values AND missingness masks zeroed when inaccessible',
                    train_wall_per_arm=1200, evaluation_wall=3600, fit_to_finish_wall=10800,
                    extraction_workers=4, evaluation_workers=8, lidar='numpy',
                    user_override='Direct construction/training/testing; no repeat full data audit; old 4.27e-17 feature difference remains historical'))
    D.save(out/'inputs.json', frozen)
    snapshot = out/'sources'
    for rel in frozen['new_sources']:
        p = snapshot/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT/rel).read_bytes())
    # One common legal prefix, eight boundaries, all seven original vs instrumented programs.
    # This is a code smoke test, not a separate research acceptance campaign.
    runtime = TraceLibrary(library, parity=True)
    with gzip.open(bundle/rows[0]['path'], 'rt') as f:
        trajectory = json.load(f)
    history = MemoryHistory()
    for row in trajectory['public_rows'][:8]:
        history._append_projected(row['observation'], row['action'])
        runtime.advance(history)
    D.save(out/'smoke.json', dict(original_instrumented_context_memory_exact=True,
                                 boundaries=history.index+1, caps=7,
                                 trace_widths=[s['width'] for s in runtime.schemas]))
    print('SMOKE', D.read(out/'smoke.json'), flush=True)
    return roots, rows


def run(bundle, out):
    if out.exists():
        raise FileExistsError('fresh output required; never overwrite or silently resume')
    out.mkdir(parents=True)
    started = time.time()
    D.save(out/'progress.json', dict(stage='prepare', started=started))
    roots, rows = prepare(bundle, out)
    jobs = [(str(bundle), str(out), row, (roots['pilot_train']+roots['pilot_validation']).index(row['case_id'])) for row in rows]
    context = mp.get_context('spawn')
    shards = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=4, mp_context=context, initializer=D.initialize, initargs=(str(bundle),)) as pool:
        futures = {pool.submit(D.extract, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            shards.append(result)
            progress = dict(stage='extract', completed=len(shards), total=len(rows),
                            boundaries=sum(r['boundaries'] for r in shards), seconds=time.monotonic()-start,
                            latest=result['case_id']+'/'+result['cap_id'])
            D.save(out/'extract_progress.json', progress)
            print('EXTRACT', progress, flush=True)
    shards.sort(key=lambda r: ((roots['pilot_train']+roots['pilot_validation']).index(r['case_id']), r['cap_id']))
    D.save(out/'shards.json', shards)
    data = D.assemble(out, shards, roots)
    print('DATASET', data, flush=True)
    verify_frozen(out)
    fit_started = time.time()
    D.save(out/'progress.json', dict(stage='fit', started=started, fit_started=fit_started))
    train_all(out)
    verify_frozen(out)
    D.save(out/'progress.json', dict(stage='evaluation', started=started, fit_started=fit_started))
    scenes = {r['case_id']: r for r in D.read(bundle/'scenes.json')}
    jobs = [(str(out), arm, scenes[c]) for arm in ('GA', 'LOCAL', 'GLOBAL') for c in roots['pilot_validation']]
    results = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=8, mp_context=context, initializer=E.initialize, initargs=(str(bundle), str(out))) as pool:
        futures = {pool.submit(E.evaluate, job): job for job in jobs}
        for future in as_completed(futures, timeout=3600):
            result = future.result()
            results.append(result)
            D.save(out/'evaluation/progress.json', dict(completed=len(results), total=48, seconds=time.monotonic()-start))
            print('EVALUATE', result, flush=True)
    complete = len(results) == 48 and all(r['complete'] and not r['error'] for r in results)
    summary = dict(complete=complete, episodes=len(results), results=results,
                   per_arm={arm: dict(success=sum(r['success'] for r in results if r['arm']==arm),
                                      total=sum(r['arm']==arm for r in results)) for arm in ('GA', 'LOCAL', 'GLOBAL')},
                   seconds=time.time()-started, fit_to_finish_seconds=time.time()-fit_started)
    D.save(out/'evaluation/result.json', summary)
    D.save(out/'FINISHED.json', summary)
    D.save(out/'progress.json', dict(stage='complete' if complete else 'incomplete', seconds=time.time()-started))
    print('FINISHED', summary['per_arm'], flush=True)
    if not complete:
        raise RuntimeError('incomplete physical evaluation; see preserved errors')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.environ.setdefault('COIMAGINE_LIDAR_KERNEL', 'numpy')
    out = Path(args.out).resolve()
    started = time.time()
    try:
        run(Path(args.bundle).resolve(), out)
    except BaseException as exc:
        D.save(out/'FAILED.json', dict(type=type(exc).__name__, message=str(exc), seconds=time.time()-started))
        raise


if __name__ == '__main__':
    main()
