"""Fixed 10x online-budget continuation of the completed GLOBAL experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_data as F, full_worker as W
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import long_online as L
from .random_room_tasks import generate

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (17, 29, 43)
DIAGNOSTIC_ROUNDS = (8, 16, 24, 40)


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def phase(out, stage, **extra):
    row = dict(stage=stage, at=time.time(), **extra)
    save(out/'progress.json', row)
    print('LONG_PHASE', row, flush=True)


def freeze_confirmation(parent, bundle, out):
    known = read(bundle/'used_scene_identities.json')
    for name in ('final_test', 'online_training'):
        known += read(parent/'panels'/(name+'.json'))['cases']
    seen = {r['content_sha256'] for r in known}
    seen_ids = {r['case_id'] for r in known}
    rows, rejected = [], []
    for attempt in range(100):
        generation_seed = 2026111812+1000*attempt
        for row in generate(generation_seed)['cases']:
            count = row['movable_count']
            number = sum(c['movable_count'] == count for c in rows)
            if number >= 50:
                continue
            fixture = row['fixture']
            content = digest({k:v for k,v in fixture.items() if k != 'scene_id'})
            case_id = f'long_confirm_{count}_{number:03d}'
            if content in seen or case_id in seen_ids:
                rejected.append(content)
                continue
            seen.add(content)
            seen_ids.add(case_id)
            rows.append(dict(case_id=case_id, fixture=fixture, fixture_sha256=digest(fixture),
                             content_sha256=content, movable_count=count, generation_seed=generation_seed))
        if len(rows) == 200:
            break
    assert len(rows) == 200
    save(out/'panels/confirmation.json', dict(cases=rows, frozen_at=time.time(),
         known_unique_contents=len({r['content_sha256'] for r in known}), rejected=rejected,
         outcome_filtering=False, new_physical_executions=0))


def prepare(parent, out):
    if out.exists():
        raise FileExistsError('New run requires a new output directory')
    inputs = read(parent/'inputs.json')
    assert read(parent/'FINISHED.json')['complete']
    for rel, expected in inputs['new_sources'].items():
        assert sha(ROOT/rel) == expected, 'Parent implementation changed: '+rel
    bundle, priority = Path(inputs['bundle']), Path(inputs['priority_bundle'])
    out.mkdir(parents=True)
    freeze_confirmation(parent, bundle, out)
    frozen = read(parent/'ALL_MODELS_FROZEN.json')['files']
    parent_paths = [parent/'models'/f'GLOBAL_s{s}'/'online_r4.pt' for s in SEEDS]
    for path in parent_paths:
        assert sha(path) == frozen[str(path.relative_to(parent))]
    source_paths = list(inputs['new_sources']) + [str(Path(__file__).relative_to(ROOT)),
        str(Path(L.__file__).relative_to(ROOT)), 'tools/cap_trace_value_long_report.py']
    source_paths += ['tests/test_cap_trace_value_long.py']
    record = dict(parent=str(parent), bundle=str(bundle), priority_bundle=str(priority),
        started=time.time(), head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        sources={rel:sha(ROOT/rel) for rel in source_paths},
        parent_checkpoints={str(p):sha(p) for p in parent_paths},
        parent_inputs_sha256=sha(parent/'inputs.json'),
        parent_online_panel_sha256=sha(parent/'panels/online_training.json'),
        diagnostic_panel_sha256=sha(parent/'panels/final_test.json'),
        confirmation_panel_sha256=sha(out/'panels/confirmation.json'),
        protocol=dict(arm='GLOBAL', seeds=SEEDS, initial_episodes=200, total_episodes=2000,
            initial_online_updates=6000, total_online_updates=60000, new_episodes=5400,
            new_unique_online_roots=0, repeated_roots=200, episodes_per_round=50,
            updates_per_round=1500, additional_epsilon=.05, normalizer='parent frozen',
            reward='success terminal 1 else 0', gamma=1, max_ticks=1800, max_decisions=300,
            optimizer='Adam3e-4; restore optimizer/target/RNG', batch=64, offline_online=[32,32],
            target_sync=250, clip_grad=1, replay='all previous own transitions; uniform boundaries',
            entropy='exact epsilon-greedy action distribution nats; no actor or entropy bonus',
            monitor_every_episodes=200, monitor_roots=32, diagnostic_episodes=[200,400,800,1200,2000],
            confirmation='new frozen 200; paired original200 vs final2000, all10CAP',
            final_selection='fixed2000; no checkpoint selection', new_physical_budget=11864,
            workers=12, torch_threads=1, whole_run_wall_seconds=43200,
            learning_round_wall_seconds=1800, report_interpretation='budget bundle of data and updates; fixed training roots',
            test_status='old200 now diagnostic; fresh confirmation only after all models freeze'))
    save(out/'inputs.json', record)
    for rel in record['sources']:
        dest = out/'sources'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, dest)
    for seed in SEEDS:
        save(out/'models'/f'GLOBAL_s{seed}'/'parent.json',
             dict(checkpoint=str(parent/'models'/f'GLOBAL_s{seed}'/'online_r4.pt'),
                  replay=str(parent/'replay'/f'GLOBAL_s{seed}'/'r4'),
                  order=read(parent/'models'/f'GLOBAL_s{seed}'/'online_order.json')))
    return record


def verify_frozen(out):
    record = read(out/'inputs.json')
    assert all(sha(ROOT/rel) == expected for rel,expected in record['sources'].items())
    assert all(sha(path) == expected for path,expected in record['parent_checkpoints'].items())
    assert sha(out/'panels/confirmation.json') == record['confirmation_panel_sha256']
    parent = Path(record['parent'])
    assert sha(parent/'panels/online_training.json') == record['parent_online_panel_sha256']
    assert sha(parent/'panels/final_test.json') == record['diagnostic_panel_sha256']


def batch(pool, out, jobs, name):
    path = out/(name+'.json')
    if path.exists():
        raise FileExistsError('No implicit rerun of an existing physical batch')
    results = []
    started = time.monotonic()
    future_jobs = {pool.submit(L.run_job, job):job for job in jobs}
    for future in as_completed(future_jobs):
        result = future.result()
        results.append(result)
        save(out/(name+'_progress.json'), dict(completed=len(results), total=len(jobs),
             seconds=time.monotonic()-started))
        if len(results) % 12 == 0 or len(results) == len(jobs):
            print('LONG_PHYSICAL', name, len(results), '/', len(jobs),
                  'seconds', round(time.monotonic()-started, 1), flush=True)
    assert len({r['case_id']+'|'+r['condition'] for r in results}) == len(jobs)
    complete = len(results) == len(jobs) and all(r['complete'] and not r.get('error') for r in results)
    save(path, dict(complete=complete, results=results, seconds=time.monotonic()-started))
    if not complete:
        raise RuntimeError('Incomplete physical batch: '+name)
    return results


def evaluate(pool, out, checkpoint, seed, episodes, cases, panel):
    condition = f'GLOBAL_s{seed}_e{episodes}'
    name = f'{panel}/{condition}'
    jobs = [dict(directory=str(out/name/c['case_id']), case=c, checkpoint=str(checkpoint),
                 condition=condition) for c in cases]
    return batch(pool, out, jobs, name)


def render(out):
    from tools.cap_trace_value_long_report import render_live
    render_live(out)


def run(parent, out):
    record = prepare(parent, out)
    torch.set_num_threads(1)
    bundle, priority = Path(record['bundle']), Path(record['priority_bundle'])
    cases = read(parent/'panels/online_training.json')['cases']
    by_id = {c['case_id']:c for c in cases}
    development = {c['case_id']:c for c in read(bundle/'scenes.json')}
    monitor = [development[c] for c in read(parent/'split.json')['monitor']]
    diagnostic = read(parent/'panels/final_test.json')['cases']
    replay_paths = {s:[parent/'replay'/f'GLOBAL_s{s}'/'r4'] for s in SEEDS}
    orders = {s:read(parent/'models'/f'GLOBAL_s{s}'/'online_order.json') for s in SEEDS}
    phase(out, 'prepared', seeds=SEEDS, total_episodes_per_seed=2000)
    render(out)
    with ProcessPoolExecutor(max_workers=12, mp_context=mp.get_context('spawn'), initializer=W.initialize,
                             initargs=(str(bundle), str(priority))) as pool:
        # Round-major order gives all three paired training seeds timely progress.
        for round_number in range(5, 41):
            for seed in SEEDS:
                if (out/'STOP').exists():
                    raise RuntimeError('STOP requested')
                name = f'GLOBAL_s{seed}'
                directory = out/'models'/name
                checkpoint = (parent/'models'/name/'online_r4.pt' if round_number == 5
                              else directory/f'online_r{round_number-1}.pt')
                phase(out, 'collect', seed=seed, round=round_number, episodes_before=(round_number-1)*50)
                start = ((round_number-1)*50) % 200
                selected = [by_id[c] for c in orders[seed][start:start+50]]
                jobs = [dict(directory=str(out/'online'/name/f'r{round_number}'/c['case_id']),
                    case=c, condition=f'{name}_online_r{round_number}', checkpoint=str(checkpoint),
                    online=True, epsilon=.05, action_seed=seed*10000+(round_number-1)*50+j,
                    root_number=760+cases.index(c)) for j,c in enumerate(selected)]
                results = batch(pool, out, jobs, f'online/{name}/r{round_number}')
                indices = {c['case_id']:i for i,c in enumerate(selected)}
                results.sort(key=lambda r:indices[r['case_id']])
                replay = out/'replay'/name/f'block{round_number}'
                F.assemble(replay, [r['transition_shard'] for r in results])
                replay_paths[seed].append(replay)
                phase(out, 'fit', seed=seed, round=round_number, episodes=round_number*50)
                L.train_round(parent, out, seed, round_number, replay_paths[seed])
                checkpoint = directory/f'online_r{round_number}.pt'
                if round_number % 4 == 0:
                    evaluate(pool, out, checkpoint, seed, round_number*50, monitor, 'monitor')
                if round_number in DIAGNOSTIC_ROUNDS:
                    verify_frozen(out)
                    evaluate(pool, out, checkpoint, seed, round_number*50, diagnostic, 'diagnostic')
                render(out)
        verify_frozen(out)
        frozen = {str(p.relative_to(out)):sha(p) for p in (out/'models').rglob('*.pt')}
        save(out/'ALL_MODELS_FROZEN.json', dict(at=time.time(), files=frozen,
             confirmation_outcomes_seen=False, final_round=40))
        phase(out, 'confirmation', total=3200)
        confirmation = read(out/'panels/confirmation.json')['cases']
        for seed in SEEDS:
            for episodes, checkpoint in [(200, parent/'models'/f'GLOBAL_s{seed}'/'online_r4.pt'),
                                          (2000, out/'models'/f'GLOBAL_s{seed}'/'online_r40.pt')]:
                evaluate(pool, out, checkpoint, seed, episodes, confirmation, 'confirmation')
                render(out)
        jobs = []
        for number in range(1, 11):
            name = f'CAP-RR-{number:03d}'
            for case in confirmation:
                jobs.append(dict(directory=str(out/'confirmation'/name/case['case_id']),
                                 case=case, condition=name, cap=name))
        batch(pool, out, jobs, 'confirmation/caps')
    verify_frozen(out)
    save(out/'FINISHED.json', dict(complete=True, seconds=time.time()-record['started'],
         new_training_episodes=5400, new_monitor_episodes=864, diagnostic_episodes=2400,
         confirmation_episodes=3200, new_physical_executions=11864,
         episodes_per_seed=2000, updates_per_seed=60000, seeds=SEEDS))
    phase(out, 'complete')
    from tools.cap_trace_value_long_report import final_report
    final_report(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.environ.setdefault('COIMAGINE_LIDAR_KERNEL', 'numpy')
    out = Path(args.out).absolute()
    try:
        run(Path(args.parent).resolve(), out)
    except BaseException as exc:
        if out.exists():
            save(out/'FAILED.json', dict(at=time.time(), type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    main()
