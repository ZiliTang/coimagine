"""Benchmark the opt-in frozen CAP009 pose kernel on archived CAP and mixed-policy histories."""
import argparse
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter
from unittest.mock import patch

from experiments.planar_tasks.authoring_v2 import cap_retention_supervised as study
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import pose_kernel as pk
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.instrumentation import instrument, load_native
from experiments.planar_tasks.semantic_rl.trusted_native import owned_copy

ARCHIVE = study.BASE / 'cap_retention_supervised_20261119_01'
CAP = 'CAP-RR-009'
ROUTER = 'boundary_s17_free'
LONG = [f'retention_dev_{count}_00' for count in range(2, 6)]
SUCCESSES = ['retention_dev_2_01','retention_dev_3_06','retention_dev_4_02','retention_dev_5_01']


def result(directory):
    with gzip.open(directory / 'result.json.gz', 'rt') as f:
        return json.load(f)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def replay_check(source, kernel, out):
    _, schema = instrument(source)
    mod, optimized = pk.make_route(source, kernel, traced=True)
    reference = mod.pose_route
    original_mod, untraced = pk.make_route(source, kernel)
    plain = original_mod.pose_route
    reference_paths, kernel_paths = [], []
    audit_ref = pk._route_function(source, original_mod, audit=reference_paths.append)
    _, audit_new = pk.make_route(source, kernel, audit=kernel_paths.append)
    width = schema['width']
    rows = []
    for policy, ids in [(CAP, LONG+SUCCESSES), (ROUTER, LONG)]:
        for idx, case in enumerate(ids):
            record = result(ARCHIVE/'evaluation'/policy/case)
            memories = [mod.initial_memory(), mod.initial_memory()]
            times = Counter()
            modes = Counter()
            calls = 0
            prev = None
            for i, decision in enumerate(record['decisions']):
                pub = record['public_rows'][i]
                obs = pub['observation']
                event = dict(observation=obs, action=pub['action'], executed_ticks=None if prev is None else obs['tick']-prev)
                prev = obs['tick']
                contexts, traces, routes = [], [], []
                inputs = []
                for arm, fn in enumerate((reference, optimized)):
                    trace = [math.nan]*width
                    def route(*args, **kwargs):
                        if arm == 0:
                            inputs.append(args)
                        answer = fn(*args, **kwargs)
                        routes.append(answer)
                        return answer
                    with patch.object(mod, 'pose_route', route):
                        memories[arm] = mod.update_memory(memories[arm], owned_copy(event), _tb=trace)
                        ctx = mod.prepare_context(owned_copy(memories[arm]), owned_copy(obs), _tb=trace)
                    contexts.append(encoded(ctx)); traces.append(array('d', trace).tobytes())
                assert contexts[0] == contexts[1] and traces[0] == traces[1], (case, i, 'context/trace')
                assert encoded(memories[0]) == encoded(memories[1]), (case, i, 'memory')
                if policy == CAP:
                    assert ctx['proposals'] == [decision['action']]
                else:
                    assert ctx['proposals'] == [decision['proposals'][study.CAPS.index(CAP)]]
                if not inputs:
                    assert not routes
                    continue
                assert len(inputs)==1 and len(routes)==2 and routes[0]==routes[1]
                args = inputs[0]
                # Native (untraced) calls timed in alternating order. I/O, audits
                # and profile hooks are outside the interval; packing is inside.
                arms = [('reference',plain),('kernel',untraced)]
                for name, fn in arms[::1 if (idx+i)%2 else -1]:
                    begin=perf_counter()
                    answer=fn(*args)
                    times[name]+=perf_counter()-begin
                    assert answer == routes[0]
                assert audit_ref(*args) == audit_new(*args) == routes[0]
                assert reference_paths.pop() == kernel_paths.pop(), (case,i,'full path')
                modes[routes[0][0]]+=1
                calls+=1
            rows.append(dict(policy=policy,case=case,boundaries=len(record['decisions']),
                             calls=calls,modes=dict(modes),seconds=dict(times)))
            save(out/'replay.json',rows)
    return rows


def benchmark(out):
    out.mkdir(parents=True,exist_ok=False)
    source_path = Path(pk.__file__).with_name('pose_route.c')
    cmd = ['gcc','-std=c99','-shared','-fPIC','-O2','-fno-fast-math','-ffp-contract=off',
           '-Wall','-Wextra','-Werror',str(source_path),'-lm','-o',str(out/'pose.so')]
    subprocess.run(cmd,check=True)
    kernel=pk.PoseKernel(out/'pose.so')
    source=Path('outputs/transfers/cap_trace_value_dev_v1/policies/CAP-RR-009.py').read_text()
    save(out/'build.json',dict(command=cmd,source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                               binary_sha256=hashlib.sha256((out/'pose.so').read_bytes()).hexdigest(),cap_source_sha256=pk.SOURCE_SHA))
    ref=study.read(study.prior.SHUFFLE/'inputs.json')
    study.prior.init_worker(ref['bundle'],ref['priority_bundle'])
    rows=replay_check(source,kernel,out)
    print('replay checks complete',len(rows),flush=True)
    cases={c['case_id']:c for c in study.read(ARCHIVE/'panel.json')['cases']}
    episode_rows=[]
    for name, mode in [(CAP,'cap'),(ROUTER,'free')]:
        def run(case,dest):
            return study.execute((cases[case],mode,ARCHIVE/'boundary_s17.pt',dest,name))
        run(SUCCESSES[0],out/'warm'/name)
        if name==CAP:
            module=study.worker.CAP_FEATURES[CAP].worker.native_module
        else:
            module=study.worker.LIBRARY.modules[study.CAPS.index(CAP)]
        _,route=pk.make_route(source,kernel,traced=name==ROUTER)
        original=module.pose_route
        for rep in range(2):
            for idx, case in enumerate(LONG):
                order=[('reference',original),('kernel',route)]
                for arm, fn in order[::1 if (idx+rep)%2 else -1]:
                    dest=out/arm/str(rep)/name/case
                    with patch.object(module,'pose_route',fn):
                        start=perf_counter()
                        episode=run(case,dest)
                        elapsed=perf_counter()-start
                    assert episode['complete']
                    got=result(dest); frozen=result(ARCHIVE/'evaluation'/name/case)
                    for row in [got,frozen]:row['summary'].pop('wall_seconds',None)
                    assert encoded(got)==encoded(frozen),(arm,name,case)
                    episode_rows.append(dict(arm=arm,case=case,rep=rep,policy=name,seconds=elapsed,
                                             ticks=episode['ticks'],decisions=episode['decisions']))
                    save(out/'episodes.json',episode_rows)
    save(out/'complete.json',dict(episodes=len(episode_rows),warmups=2,
                                  checked_boundaries=sum(r['boundaries'] for r in rows),
                                  checked_paths=sum(r['calls'] for r in rows)))

def init_throughput(out, barrier):
    global pool_barrier, pool_cases, pool_module, pool_reference, pool_fast
    ref = study.read(study.prior.SHUFFLE/'inputs.json')
    study.prior.init_worker(ref['bundle'], ref['priority_bundle'])
    source = study.worker.CAP_LIBRARY[CAP]
    pool_module = study.worker.LIBRARY.modules[study.CAPS.index(CAP)]
    pool_reference = pool_module.pose_route
    _, pool_fast = pk.make_route(source, pk.PoseKernel(out/'pose.so'), traced=True)
    pool_cases = study.read(ARCHIVE/'panel.json')['cases']
    pool_barrier = barrier


def throughput_job(job):
    out, index = job
    def execute(case, dest):
        row = study.execute((case, 'free', ARCHIVE/'boundary_s17.pt', dest, ROUTER))
        assert row['complete']
    execute(pool_cases[1], out/'pool'/f'warm_{index}')
    cases = [pool_cases[50*(index%4) + index//4 + offset] for offset in (2, 5)]
    results = []
    for block, arm in enumerate(('reference', 'kernel', 'kernel', 'reference')):
        with patch.object(pool_module, 'pose_route', pool_fast if arm == 'kernel' else pool_reference):
            pool_barrier.wait(timeout=240)
            start = perf_counter()
            for case in cases:
                execute(case, out/'pool'/str(block)/case['case_id'])
            end = perf_counter()
        results.append(dict(block=block, arm=arm, start=start, end=end))
    return results


def throughput(out):
    if not (out/'complete.json').exists():
        raise ValueError('complete the serial parity benchmark first')
    (out/'pool').mkdir(exist_ok=False)
    ctx = get_context('spawn')
    barrier = ctx.Barrier(12)
    with ProcessPoolExecutor(12, mp_context=ctx, initializer=init_throughput,
                             initargs=(out,barrier)) as pool:
        workers = list(pool.map(throughput_job, [(out, i) for i in range(12)]))
    blocks = []
    for block in range(4):
        workers_block = [worker[block] for worker in workers]
        seconds = max(r['end'] for r in workers_block)-min(r['start'] for r in workers_block)
        episodes = list((out/'pool'/str(block)).iterdir())
        assert len(episodes)==24
        for case in episodes:
            assert encoded(result(case))==encoded(result(ARCHIVE/'evaluation'/ROUTER/case.name))
        blocks.append(dict(block=block, arm=workers_block[0]['arm'], seconds=seconds,
                           episodes=len(episodes), episodes_per_minute=60*len(episodes)/seconds))
    save(out/'pool.json',blocks)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--pool-only',action='store_true')
    args=parser.parse_args()
    # Both arms patch the reference module explicitly in this benchmark.
    os.environ['COIMAGINE_CAP009_POSE'] = 'reference'
    os.environ['COIMAGINE_CAP_SHARED'] = 'reference'
    (throughput if args.pool_only else benchmark)(args.out.resolve())
