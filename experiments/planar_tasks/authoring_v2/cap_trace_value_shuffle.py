"""Fixed seed17 diagnostic: stratified cyclic shuffle and 500 updates per batch."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_data as F
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime as W
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.fast_learner import train_round
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.resident_replay import ReplayCache
from .cap_trace_value_long import ROOT, sha, phase
from .cap_trace_value_single import batch, evaluate


def schedule(cases, original_order, seed=2026111817, cycles=9):
    """No outcome conditioning. Separate schedule RNG; pair action seeds by root/visit."""
    assert len(cases) == len(original_order) == 200
    rank = {case_id: i for i, case_id in enumerate(original_order)}
    assert len(rank) == 200 and set(rank) == {c['case_id'] for c in cases}
    strata = {n: [c['case_id'] for c in cases if c['movable_count'] == n] for n in range(2, 6)}
    assert all(len(v) == 50 for v in strata.values())
    rng = np.random.default_rng(seed)
    result = []
    for cycle in range(cycles):
        shuffled = {n: rng.permutation(ids).tolist() for n, ids in strata.items()}
        groups = rng.permutation(list(strata)).tolist()
        interleaved = [shuffled[n][i] for i in range(50) for n in groups]
        for block in range(4):
            ids = rng.permutation(interleaved[block*50:(block+1)*50]).tolist()
            result.append(dict(round=5+cycle*4+block, cycle=cycle+1, cases=[dict(
                case_id=c, action_seed=170000+(cycle+1)*200+rank[c]) for c in ids]))
    return result


def prepare(parent, baseline, performance, out):
    if out.exists():
        raise FileExistsError(out)
    old = read(baseline/'inputs.json')
    assert read(baseline/'REPORT_FINISHED.json')['complete']
    for rel, expected in old['sources'].items():
        assert sha(ROOT/rel) == expected, 'Frozen source changed: '+rel
    checkpoint = parent/'models/GLOBAL_s17/online_r4.pt'
    assert sha(checkpoint) == read(parent/'ALL_MODELS_FROZEN.json')['files'][str(checkpoint.relative_to(parent))]
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['online_updates'] == 6000 and state['seed'] == 17 and state['arm'] == 'GLOBAL'
    assert 'target_model' in state
    cases = read(parent/'panels/online_training.json')['cases']
    order = read(parent/'models/GLOBAL_s17/online_order.json')
    frozen_schedule = schedule(cases, order)
    out.mkdir(parents=True)
    save(out/'schedule.json', frozen_schedule)
    files = list(old['sources']) + [str(Path(__file__).relative_to(ROOT)),
        'tools/cap_trace_value_shuffle_report.py', 'tests/test_cap_trace_value_shuffle.py']
    record = dict(parent=str(parent), baseline=str(baseline), performance=str(performance),
        paused=old['paused'], started=time.time(), bundle=old['bundle'], priority_bundle=old['priority_bundle'],
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), schedule_sha256=sha(out/'schedule.json'),
        sources={rel: sha(ROOT/rel) for rel in files},
        original_online_panel_sha256=sha(parent/'panels/online_training.json'),
        diagnostic_panel_sha256=sha(parent/'panels/final_test.json'),
        sealed_confirmation_sha256=sha(Path(old['paused'])/'panels/confirmation.json'),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        protocol=dict(seed=17, initial_episodes=200, initial_online_updates=6000,
            total_episodes=2000, total_online_updates=24000, new_training_episodes=1800,
            updates_per_round=500, episodes_per_round=50, new_updates=18000,
            new_monitor_executions=288, new_diagnostic_executions=800, new_physical_executions=2888,
            confirmation_executions=0, fixed_online_roots=200, epsilon=.05,
            sampling='reshuffle each 200-root cycle within 2/3/4/5-object strata; four balanced batches without replacement',
            schedule_seed=2026111817, action_rng='paired with fixed-order baseline by root and visit',
            unchanged='network/reward/gamma/normalizer/optimizer/target update interval/clip/replay ratio/boundary sampling',
            workers=12, learner_threads=4, worker_threads=1, wall_limit_seconds=21600,
            comparison='two factors changed together; no separate causal attribution',
            selection='fixed final2000; viewed diagnostic only; confirmation sealed'))
    save(out/'inputs.json', record)
    for rel in record['sources']:
        dest = out/'sources'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, dest)
    # Only original offline data and the original 200-episode replay are inherited.
    for source in (parent/'arrays/train', parent/'replay/GLOBAL_s17/r4'):
        key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:20]
        cached = performance/'packed'/key
        if cached.exists():
            shutil.copytree(cached, out/'packed'/key, copy_function=os.link)
    return record


def verify(out):
    d = read(out/'inputs.json')
    assert all(sha(ROOT/p) == v for p, v in d['sources'].items())
    assert sha(d['checkpoint']) == d['checkpoint_sha256']
    assert sha(out/'schedule.json') == d['schedule_sha256']
    assert sha(Path(d['parent'])/'panels/online_training.json') == d['original_online_panel_sha256']
    assert sha(Path(d['parent'])/'panels/final_test.json') == d['diagnostic_panel_sha256']
    assert sha(Path(d['paused'])/'panels/confirmation.json') == d['sealed_confirmation_sha256']


def run(parent, baseline, performance, out):
    record = prepare(parent, baseline, performance, out)
    torch.set_num_threads(4)
    from tools.cap_trace_value_shuffle_report import render_live, final_report
    phase(out, 'load_resident_replay', seed=17)
    render_live(out)
    cache = ReplayCache(out/'packed')
    offline = cache.get(parent/'arrays/train')
    paths = [parent/'replay/GLOBAL_s17/r4']
    online = cache.online(paths)
    cases = read(parent/'panels/online_training.json')['cases']
    by_id = {c['case_id']: c for c in cases}
    roots = {c['case_id']: 760+i for i, c in enumerate(cases)}
    development = {c['case_id']: c for c in read(Path(record['bundle'])/'scenes.json')}
    monitor = [development[c] for c in read(parent/'split.json')['monitor']]
    diagnostic = read(parent/'panels/final_test.json')['cases']
    checkpoint = Path(record['checkpoint'])
    with ProcessPoolExecutor(max_workers=12, mp_context=mp.get_context('spawn'), initializer=W.initialize,
                             initargs=(record['bundle'], record['priority_bundle'])) as pool:
        for block in read(out/'schedule.json'):
            r = block['round']
            if (out/'STOP').exists():
                raise RuntimeError('STOP requested')
            phase(out, 'collect', seed=17, round=r, episodes_before=(r-1)*50)
            jobs = [dict(directory=str(out/'online/GLOBAL_s17'/f'r{r}'/item['case_id']),
                case=by_id[item['case_id']], condition=f'GLOBAL_s17_online_r{r}', checkpoint=str(checkpoint),
                online=True, epsilon=.05, action_seed=item['action_seed'], root_number=roots[item['case_id']])
                for item in block['cases']]
            rows = batch(pool, out, jobs, f'online/GLOBAL_s17/r{r}')
            indices = {item['case_id']: i for i, item in enumerate(block['cases'])}
            rows.sort(key=lambda row: indices[row['case_id']])
            replay = out/'replay/GLOBAL_s17'/f'block{r}'
            F.assemble(replay, [row['transition_shard'] for row in rows])
            paths.append(replay)
            online = cache.online(paths)
            save(out/'memory.json', dict(resident_replay_bytes=cache.resident_bytes,
                offline_boundaries=len(offline), online_boundaries=len(online)))
            phase(out, 'fit', seed=17, round=r, episodes=r*50)
            train_round(checkpoint, out, 17, r, offline, online, updates=500)
            checkpoint = out/'models/GLOBAL_s17'/f'online_r{r}.pt'
            if r % 4 == 0:
                evaluate(pool, out, checkpoint, r*50, monitor, 'monitor')
            if r in (8, 16, 24, 40):
                verify(out)
                evaluate(pool, out, checkpoint, r*50, diagnostic, 'diagnostic')
            render_live(out)
    verify(out)
    save(out/'ALL_MODELS_FROZEN.json', dict(at=time.time(), files={str(p.relative_to(out)): sha(p)
        for p in (out/'models').rglob('*.pt')}, new_confirmation_executions=0))
    save(out/'FINISHED.json', dict(complete=True, seed=17, episodes=2000, online_updates=24000,
        new_training_executions=1800, new_monitor_executions=288, new_diagnostic_executions=800,
        new_physical_executions=2888, new_confirmation_executions=0, seconds=time.time()-record['started']))
    phase(out, 'complete', seed=17)
    final_report(out)


def main():
    parser = argparse.ArgumentParser()
    for name in ('parent', 'baseline', 'performance', 'out'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    try:
        run(Path(args.parent).resolve(), Path(args.baseline).resolve(), Path(args.performance).resolve(), out)
    except BaseException as error:
        if out.exists():
            save(out/'FAILED.json', dict(at=time.time(), type=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
