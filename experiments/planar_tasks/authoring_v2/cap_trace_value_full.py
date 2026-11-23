"""Autonomous full trace/value/RL comparison authorized on 2026-11-18."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import dataset as D
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_data as F
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_worker as W
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_learner import train_offline, train_online_round
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.runtime import CAPS, entries
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.instrumentation import instrument

ROOT = Path(__file__).resolve().parents[3]
ARMS = ('GA', 'LOCAL', 'GLOBAL')
SEEDS = (17, 29, 43)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    base = ROOT/'experiments/planar_tasks/semantic_rl/program_trace_value_v1'
    paths = list(base.glob('*.py')) + list((base/'pose_kernel').glob('*.py'))
    paths.extend((base/'pose_kernel').glob('*.c'))
    paths.extend((base/'pose_kernel').glob('*.h'))
    paths.append(base.parent/'trusted_native.py')
    paths += [Path(__file__).resolve(), Path(__file__).with_name('cap_trace_value_panels.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def verify_frozen(out):
    inputs = D.read(out/'inputs.json')
    assert inputs['new_sources'] == sources(), 'new implementation changed during run'
    panels = D.read(out/'panels/manifest.json')
    for name, value in panels['files'].items():
        assert sha(out/'panels'/name) == value, 'scene panel changed'
    assert sha(Path(inputs['bundle'])/'manifest.json') == inputs['bundle_manifest_sha256']


def phase(out, stage, **values):
    value = dict(stage=stage, at=time.time(), **values)
    D.save(out/'progress.json', value)
    print('PHASE', value, flush=True)


def prepare(bundle, priority, out):
    if (out/'inputs.json').exists():
        verify_frozen(out)
        return
    roots = D.split(bundle)
    scenes = D.read(bundle/'scenes.json')
    by_id = {r['case_id']:r for r in scenes}
    monitor = []
    for count in range(2, 6):
        monitor.extend([c for c in roots['validation'] if len(by_id[c]['fixture']['movable_objects'])+1 == count][:8])
    assert len(monitor) == 32
    roots['monitor'] = monitor
    D.save(out/'split.json', roots)
    schemas = {}
    for entry in entries(bundle):
        generated, schema = instrument(entry['source'])
        schemas[entry['name']] = schema
        destination = out/'instrumented'/(entry['name']+'.py')
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(generated)
    D.save(out/'schema.json', schemas)
    inputs = dict(bundle=str(bundle), priority_bundle=str(priority), bundle_manifest_sha256=sha(bundle/'manifest.json'),
                  new_sources=sources(), protocol=dict(arms=ARMS, seeds=SEEDS, caps=CAPS,
                    v_updates=16000, q_init_updates=8000, online_updates=6000,
                    online_rounds=4, episodes_per_round=50, updates_per_round=1500,
                    epsilon=[.2, .15, .1, .05], offline_online_replay_fraction=[.5, .5],
                    online_new_roots=200, online_root_distribution='same generator, new content-disjoint seeds',
                    offline_train_roots=len(roots['train']), offline_validation_roots=len(roots['validation']),
                    normalization='offline training boundaries only; frozen online',
                    reward='success terminal 1, otherwise 0; finite 1800 ticks / 300 decisions', gamma=1,
                    budget_truncation='finite-task terminal mask; no bootstrap at either fixed task budget',
                    value_target='expert continuation Monte Carlo; behavior CAP labels only',
                    q_target='one real action followed by common live candidate selector',
                    optimizer='Adam', learning_rate=.0003, batch=64, clip_grad=1., target_sync=250,
                    feature_workers=8, physical_workers=12, fit_workers=1, lidar='numpy',
                    monitor_roots=32, monitor_checkpoints=['offline', 'online100', 'online200'],
                    test_roots=200, test_model_executions=3600, test_cap_executions=2000,
                    single_cap_baselines='all original 10; primary development-selected CAP009 plus descriptive test-best',
                    full_physical_budget=8264, pilot_physical_budget=48,
                    fixed_final_checkpoint=True, test_used_for_selection=False,
                    offline_fit_wall_seconds=3600, online_model_wall_seconds=3600,
                    whole_run_wall_seconds=129600,
                    authorization='User explicitly requested autonomous complete experiment/report and additional disjoint online scenes.'))
    D.save(out/'inputs.json', inputs)
    for rel in inputs['new_sources']:
        destination = out/'sources'/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, destination)
    verify_frozen(out)


def extract_all(bundle, out):
    if (out/'dataset.json').exists():
        return D.read(out/'dataset.json')
    phase(out, 'full_extract', completed=0, total=5320)
    roots = D.read(out/'split.json')
    all_ids = roots['train'] + roots['validation']
    rows = D.read(bundle/'index.json')
    rows.sort(key=lambda r:(all_ids.index(r['case_id']), r['cap_id']))
    jobs = [(str(bundle), str(out), row, all_ids.index(row['case_id'])) for row in rows]
    collected = []
    began = time.monotonic()
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'), initializer=F.initialize, initargs=(str(bundle),)) as pool:
        futures = {pool.submit(F.extract, job):job for job in jobs}
        for future in as_completed(futures):
            collected.append(future.result())
            progress = dict(completed=len(collected), total=5320, seconds=time.monotonic()-began,
                            boundaries=sum(r['boundaries'] for r in collected))
            D.save(out/'extract_progress.json', progress)
            if len(collected)%50 == 0:
                print('FULL_EXTRACT', progress, flush=True)
    collected.sort(key=lambda r:(all_ids.index(r['case_id']), r['cap_id']))
    assert len(collected) == 5320 and sum(r['boundaries'] for r in collected) == 369726
    D.save(out/'shards.json', collected)
    phase(out, 'assemble')
    result = dict(layout=collected[0]['layout'], extraction_seconds=time.monotonic()-began)
    for name in ('train', 'validation'):
        selected = set(roots[name])
        shards = [r for r in collected if r['case_id'] in selected]
        result[name] = F.assemble(out/'arrays'/name, shards)
    D.save(out/'dataset.json', result)
    return result


def physical_batch(pool, out, jobs, name):
    result_path = out/(name+'.json')
    if result_path.exists():
        result = D.read(result_path)
        assert result['complete']
        return result['results']
    results = []
    started = time.monotonic()
    futures = {pool.submit(W.run, job):job for job in jobs}
    for future in as_completed(futures):
        row = future.result()
        results.append(row)
        D.save(out/(name+'_progress.json'), dict(completed=len(results), total=len(jobs), seconds=time.monotonic()-started))
        if len(results)%12 == 0 or len(results) == len(jobs):
            print('PHYSICAL', name, len(results), '/', len(jobs), 'seconds', round(time.monotonic()-started, 1), flush=True)
    complete = len(results) == len(jobs) and all(r['complete'] and not r.get('error') for r in results)
    D.save(result_path, dict(complete=complete, results=results, seconds=time.monotonic()-started))
    if not complete:
        raise RuntimeError('incomplete physical batch: '+name)
    return results


def monitor(pool, out, model_name, checkpoint, label, cases):
    jobs = [dict(directory=str(out/'monitor'/model_name/label/c['case_id']), case=c,
                 condition=model_name+'_'+label, checkpoint=str(checkpoint)) for c in cases]
    return physical_batch(pool, out, jobs, f'monitor/{model_name}/{label}')


def online(pool, out, arm, seed, cases, monitor_cases):
    name = f'{arm}_s{seed}'
    directory = out/'models'/name
    if (directory/'online_complete.json').exists():
        return
    phase(out, 'online', model=name, model_started=time.time())
    started = time.monotonic()
    # Paired strata-balanced order, fixed before any outcomes.
    rng = np.random.default_rng(seed)
    strata = [[c for c in cases if c['movable_count']==n] for n in range(2, 6)]
    for group in strata:
        rng.shuffle(group)
    order = [strata[j][i] for i in range(50) for j in range(4)]
    D.save(directory/'online_order.json', [c['case_id'] for c in order])
    all_shards = []
    for round_number, epsilon in enumerate((.2, .15, .1, .05), 1):
        checkpoint = directory/('offline.pt' if round_number == 1 else f'online_r{round_number-1}.pt')
        selected = order[(round_number-1)*50:round_number*50]
        jobs = [dict(directory=str(out/'online'/name/f'r{round_number}'/c['case_id']), case=c,
                     condition=name+f'_online_r{round_number}', checkpoint=str(checkpoint), online=True,
                     epsilon=epsilon, action_seed=seed*10000+(round_number-1)*50+j,
                     root_number=760+cases.index(c)) for j, c in enumerate(selected)]
        results = physical_batch(pool, out, jobs, f'online/{name}/r{round_number}')
        order_index = {case['case_id']:i for i,case in enumerate(selected)}
        results.sort(key=lambda result:order_index[result['case_id']])
        all_shards.extend(r['transition_shard'] for r in results)
        replay = out/'replay'/name/f'r{round_number}'
        if not (replay/'manifest.json').exists():
            F.assemble(replay, all_shards)
        if not (directory/f'online_r{round_number}.pt').exists():
            train_online_round(out, arm, seed, round_number, replay)
        if round_number in (2, 4):
            monitor(pool, out, name, directory/f'online_r{round_number}.pt',
                    'online100' if round_number == 2 else 'online200', monitor_cases)
        if time.monotonic()-started > 3600:
            raise RuntimeError('one-hour online model budget reached')
    D.save(directory/'online_complete.json', dict(complete=True, episodes=200, updates=6000, seconds=time.monotonic()-started))


def run(bundle, priority, out):
    started = time.time()
    torch.set_num_threads(1)
    prepare(bundle, priority, out)
    extract_all(bundle, out)
    verify_frozen(out)
    for arm in ARMS:
        for seed in SEEDS:
            phase(out, 'offline_fit', model=f'{arm}_s{seed}', model_started=time.time())
            train_offline(out, arm, seed)
    scenes = {r['case_id']:r for r in D.read(bundle/'scenes.json')}
    monitor_cases = [scenes[c] for c in D.read(out/'split.json')['monitor']]
    new_online = D.read(out/'panels/online_training.json')['cases']
    with ProcessPoolExecutor(max_workers=12, mp_context=mp.get_context('spawn'), initializer=W.initialize,
                             initargs=(str(bundle), str(priority))) as pool:
        for arm in ARMS:
            for seed in SEEDS:
                name = f'{arm}_s{seed}'
                monitor(pool, out, name, out/'models'/name/'offline.pt', 'offline', monitor_cases)
                online(pool, out, arm, seed, new_online, monitor_cases)
        verify_frozen(out)
        models = {str(p.relative_to(out)):sha(p) for p in (out/'models').rglob('*.pt')}
        D.save(out/'ALL_MODELS_FROZEN.json', dict(at=time.time(), files=models, test_outcomes_seen=False))
        phase(out, 'final_test', episodes=5600)
        cases = D.read(out/'panels/final_test.json')['cases']
        jobs = []
        for arm in ARMS:
            for seed in SEEDS:
                for label, filename in [('offline', 'offline.pt'), ('online', 'online_r4.pt')]:
                    name = f'{arm}_s{seed}_{label}'
                    for case in cases:
                        jobs.append(dict(directory=str(out/'evaluation'/name/case['case_id']), case=case, condition=name,
                                         checkpoint=str(out/'models'/f'{arm}_s{seed}'/filename)))
        for i in range(1, 11):
            name = f'CAP-RR-{i:03d}'
            for case in cases:
                jobs.append(dict(directory=str(out/'evaluation'/name/case['case_id']), case=case, condition=name, cap=name))
        results = physical_batch(pool, out, jobs, 'final_test')
    summary = dict(complete=True, seconds=time.time()-started, final_executions=len(results),
                   conditions={name:dict(success=sum(r['success'] for r in results if r['condition']==name),
                                         total=sum(r['condition']==name for r in results)) for name in sorted({r['condition'] for r in results})})
    D.save(out/'FINISHED.json', summary)
    phase(out, 'complete', seconds=time.time()-started)
    print('FINISHED', summary, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--priority', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.environ.setdefault('COIMAGINE_LIDAR_KERNEL', 'numpy')
    out = Path(args.out).resolve()
    try:
        run(Path(args.bundle).resolve(), Path(args.priority).resolve(), out)
    except BaseException as exc:
        D.save(out/'FAILED.json', dict(at=time.time(), type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
