"""Same-input helper and trace comparison, followed by paired seven-CAP rollouts."""
import argparse
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import gzip
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from experiments.planar_tasks.authoring_v2 import cap_retention_supervised as study
from experiments.planar_tasks.semantic_rl.memory_flow import MemoryHistory

ARCHIVE = study.BASE/'cap_retention_supervised_20261119_01'
POLICY = 'boundary_s17_free'
MODES = ('reference','clearances','native','navigate')
FUNCTIONS = ('clearances','navigate')


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def read_result(directory):
    with gzip.open(directory/'result.json.gz','rt') as stream:
        return json.load(stream)


def initialize():
    ref = study.read(study.prior.SHUFFLE/'inputs.json')
    study.prior.init_worker(ref['bundle'], ref['priority_bundle'])
    return study.read(ARCHIVE/'panel.json')['cases']


def check_result(case, directory):
    assert read_result(directory) == read_result(ARCHIVE/'evaluation'/POLICY/case), directory


def check_calls(out):
    cases = initialize()
    library = study.worker.LIBRARY
    rows = []
    counts, times = Counter(), Counter()
    calls = Counter()

    def measured(name, cap, original, native):
        def checked(*args, **kwargs):
            original_kwargs = dict(kwargs)
            original_kwargs['_tb'] = list(kwargs['_tb'])
            key = cap+'/'+name
            order = ('reference','native') if counts[key]%2 else ('native','reference')
            answers = {}
            for arm in order:
                fn, kw = (native,kwargs) if arm=='native' else (original,original_kwargs)
                start = perf_counter()
                answers[arm] = fn(*args,**kw)
                times[key+'/'+arm] += perf_counter()-start
            assert json.dumps(answers['reference']) == json.dumps(answers['native']), (cap,name,'value')
            assert array('d',kwargs['_tb']).tobytes()==array('d',original_kwargs['_tb']).tobytes(),(cap,name,'trace')
            counts[key]+=1
            calls[key]+=1
            return answers['native']
        return checked

    with ExitStack() as stack:
        for cap,module in zip(study.CAPS,library.modules):
            for name in FUNCTIONS:
                if hasattr(module,name):
                    original=getattr(module,'_reference_'+name)
                    fast=getattr(module,'_native_'+name)
                    assert getattr(module,name) is fast
                    stack.enter_context(patch.object(module,name,measured(name,cap,original,fast)))
        for i in (0,2,4,50,52,54,100,102,104,150,152,154):
            case = cases[i]['case_id']
            counts.clear(); times.clear()
            record = read_result(ARCHIVE/'evaluation'/POLICY/case)
            h = MemoryHistory()
            digests=[]
            for pub, decision in zip(record['public_rows'],record['decisions']):
                h._append_projected(pub['observation'],pub['action'])
                packet=library.advance(h)
                assert packet['proposals']==decision['proposals']
                digest=hashlib.sha256(packet['trace'].tobytes()+packet['advice'].tobytes())
                digest.update(json.dumps(library.memory,sort_keys=True).encode())
                digests.append(digest.hexdigest())
            rows.append(dict(case=case,boundaries=len(digests),counts=dict(counts),times=dict(times),digests=digests))
            save(out/'calls.json',rows)
    assert len(calls)==12 and all(calls.values())

    # The separate plain interpreter path has no trace slots. Exercise all
    # seven native single-CAP policies through their actual rollout workers.
    for i,cap in enumerate(study.CAPS):
        case=cases[(i%4)*50+2]
        dest=out/'fixed'/cap/case['case_id']
        summary=study.execute((case,'cap',None,dest,cap))
        assert summary['complete']
        actual=read_result(dest); frozen=read_result(ARCHIVE/'evaluation'/cap/case['case_id'])
        for item in (actual,frozen):
            item['summary'].pop('wall_seconds',None)
        assert actual==frozen,(cap,case['case_id'])


def selection(mode):
    stack = ExitStack()
    for module in study.worker.LIBRARY.modules:
        for name in FUNCTIONS:
            if hasattr(module,name):
                tag='_native_' if mode in ('native',name) else '_reference_'
                stack.enter_context(patch.object(module,name,getattr(module,tag+name)))
    return stack


def execute(case, dest):
    summary=study.execute((case,'free',ARCHIVE/'boundary_s17.pt',dest,POLICY))
    assert summary['complete']
    return summary


def check_independent(out):
    cases=initialize()
    ref=study.read(out/'calls.json')
    by_id={case['case_id']:case for case in cases}
    with selection('reference'):
        for saved in ref:
            record=read_result(ARCHIVE/'evaluation'/POLICY/by_id[saved['case']]['case_id'])
            h=MemoryHistory()
            for i,pub in enumerate(record['public_rows'][:-1]):
                h._append_projected(pub['observation'],pub['action'])
                packet=study.worker.LIBRARY.advance(h)
                digest=hashlib.sha256(packet['trace'].tobytes()+packet['advice'].tobytes())
                digest.update(json.dumps(study.worker.LIBRARY.memory,sort_keys=True).encode())
                assert digest.hexdigest()==saved['digests'][i],(saved['case'],i)
    save(out/'parity.json',dict(fixed_caps=7,boundaries=sum(x['boundaries'] for x in ref),
         matched_calls=sum(sum(x['counts'].values()) for x in ref)))


def init_pool(barrier):
    global panel, sync
    panel=initialize()
    sync=barrier


def worker(job):
    out,index=job
    execute(panel[1],out/'warm'/str(index))
    rows=[]
    for block,mode in enumerate(MODES+MODES[::-1]):
        with selection(mode):
            sync.wait(timeout=240)
            start=perf_counter()
            for offset in (2,5):
                case=panel[50*(index%4)+index//4+offset]
                begin=perf_counter()
                execute(case,out/'pool'/str(block)/case['case_id'])
                end=perf_counter()
                rows.append(dict(case=case['case_id'],worker=index,block=block,mode=mode,
                                 start=start,end=end,episode_seconds=end-begin))
    return rows


def throughput(out):
    ctx=get_context('spawn')
    with ProcessPoolExecutor(12,mp_context=ctx,initializer=init_pool,initargs=(ctx.Barrier(12),)) as pool:
        rows=[r for group in pool.map(worker,[(out,i) for i in range(12)]) for r in group]
    save(out/'pool_times.json',rows)
    blocks=[]
    for b in range(8):
        batch=[r for r in rows if r['block']==b]
        assert len(batch)==24
        blocks.append(dict(mode=batch[0]['mode'],seconds=max(r['end'] for r in batch)-min(r['start'] for r in batch)))
        for r in batch:
            check_result(r['case'],out/'pool'/str(b)/r['case'])
    save(out/'batches.json',blocks)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    os.environ['COIMAGINE_CAP_SHARED']='native'
    os.environ['COIMAGINE_CAP009_POSE']='native'
    check_calls(out)
    check_independent(out)
    print('paired public histories and fixed rollouts verified',flush=True)
    throughput(out)
