"""Attribute archived unmasked actor cycles to selected IDs and actual legal proposals."""
import collections
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from tools import cap_regional_cycle_mask as mask

ref = mask.ref
study = mask.study
OUT = mask.OUT/'source_audit'
DOC = mask.DOC
NAMES = ref.cpp.SID_ORDER


@torch.inference_mode()
def replay(fixture, *, entry=None, lock=None, stop=None):
    env = ref.ppo.environment(fixture)
    lib = ref.cpp.Proposals()
    rows=[]
    past=collections.deque(maxlen=mask.REPEATS*max(mask.PERIODS))
    first_cycle=None
    try:
        for step in range(300):
            x,_,sources,_,pool,raw=lib.panel(env.history)
            if entry is None or step<entry:
                scores=ref.ID(torch.from_numpy(x[None]))[0]
                cap=int(scores.argmax()) if scores.isfinite().any() else None
            else:
                cap=lock
            if cap is None:
                assert len(pool)==1 and not sources.any()
                act=pool[0]
            else:
                act=raw[cap]
                if act not in pool or not sources[pool.index(act),cap]:
                    return rows,dict(complete=False,invalid=step)
            support=sources[pool.index(act)].nonzero()[0].tolist()
            action=mask.identity(act)
            obs=study.dq.current_event(env.history)['observation']
            xy=tuple((r['own']['x'],r['own']['y']) for r in obs['robots'])
            if first_cycle is None:
                hit=mask.cycle(past,xy,action)
                if hit:
                    first_cycle=dict(step=step,period=len(hit[0]),pattern=hit[0])
            past.append((action,xy))
            rows.append(dict(prefix=env.history.sha256,cap=cap,support=support,action=action))
            if stop==step:
                break
            env.step(act,remaining_ticks=1800-env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        return rows,dict(complete=True,success=int(env.env.success),history=env.history.sha256,first_cycle=first_cycle)
    finally:
        env.close_reads()
        study.dq.fast_runtime.ACTIVE=None


def attribute(trace,w):
    p=w['period']
    start=w['step']-mask.REPEATS*p
    segment=trace[start:w['step']+1]
    assert [list(r['action']) for r in segment]==w['pattern']*mask.REPEATS+w['pattern'][:1]
    assert segment[-1]['prefix']==w['prefix']
    ids=[r['cap'] for r in segment]
    assert all(i is not None for i in ids)
    common=set.intersection(*(set(r['support']) for r in segment))
    category='same' if len(set(ids))==1 else 'shared' if common else 'switch_required'
    return dict(step=w['step'],start=start,period=p,ids=ids,common=sorted(common),
                entry_cap=ids[0],category=category,entry_supports=ids[0] in common,
                support=[r['support'] for r in segment],pattern=w['pattern'])


def job(args):
    group,cid,n,f=args
    file=OUT/group/(cid+'.json')
    if file.exists():return study.read(file)
    original=study.read(mask.OUT/'rooms'/group/(cid+'.json'))['original']
    trace,end=replay(f)
    assert end['complete'] and end['history']==original['history']
    windows=[attribute(trace,w) for w in original['windows']]
    first=windows[0]
    start,trigger,cap=first['start'],first['step'],first['entry_cap']
    locked,branch=replay(f,entry=start,lock=cap)
    assert locked[start]['prefix']==trace[start]['prefix']
    reproduced=(len(locked)>trigger and all(locked[t]['prefix']==trace[t]['prefix'] and locked[t]['action']==trace[t]['action']
                                           for t in range(start,trigger+1)))
    assert reproduced==first['entry_supports']
    if start==0:
        pure=branch
    else:
        _,pure=replay(f,entry=0,lock=cap)
    row=dict(panel=group,cid=cid,objects=n,original=end['success'],windows=windows,lock=branch,
             reproduced=reproduced,pure=pure)
    study.save(file,row)
    return row


def main():
    ref.init()
    protocol=study.read(mask.OUT/'protocol.json')
    assert protocol['source']==study.sha(mask.__file__) and protocol['actor']==study.sha(ref.base.OUT/'fixed.pt')
    assert protocol['binary']==ref.cpp.load()[1]['binary']
    jobs=study.read(mask.OUT/'panels.json')
    assert study.ext.old.digest(jobs)==protocol['panels']
    jobs=[j for j in jobs if study.read(mask.OUT/'rooms'/j[0]/(j[1]+'.json'))['original']['windows']]
    with ProcessPoolExecutor(16,mp_context=mp.get_context('fork')) as pool:
        rows=list(pool.map(job,jobs))
    with (DOC/'cycle_sources.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['panel','cid','original_success','first_trigger','start','trigger','period','category',
                         'selected_ids','common_support','entry_reproduced','entry_lock_success','from_reset_cycle','from_reset_success'])
        for r in rows:
            for j,w in enumerate(r['windows']):
                writer.writerow([r['panel'],r['cid'],r['original'],int(j==0),w['start'],w['step'],w['period'],w['category'],
                                 ' '.join(NAMES[i] for i in w['ids']),' '.join(NAMES[i] for i in w['common']),
                                 int(r['reproduced']) if not j else '',r['lock'].get('success','') if not j else '',
                                 int(r['pure']['first_cycle'] is not None) if not j and r['pure']['complete'] else '',
                                 r['pure'].get('success','') if not j else ''])
    print('Rooms',len(rows),'windows',sum(len(r['windows']) for r in rows),flush=True)


if __name__=='__main__':
    main()
