"""Room-disjoint single-opportunity entry calibration of frozen trial06 over trial09."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

from experiments.planar_tasks.relational_search.pipeline import save, read
from tools import cap_fault_curriculum_rooms as rooms

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/semantic_rl/cap_entry_residual_20261124_01'
PREV=OUT.parent/'cap_open_recovery_20261124_01'
SIZES={'train':2400,'validation':800,'held':1000}
SEEDS={'train':741103901,'validation':741103903,'held':741103907}
CAP=3


def immutable(path,value):
    if path.exists():
        if read(path)!=value:
            raise ValueError(f'frozen input changed: {path}')
    else:
        save(path,value)


def panels():
    seen=set()
    for path in [PREV/'fixtures.json',OUT.parent/'cap_cluster_module_20261124_01/fixtures.json']:
        seen.update(rooms.content_hash(f) for f in read(path).values())
    prior=read(OUT.parent/'cap_fault_curriculum_20261124_01/rooms.json')
    seen.update(r['content_sha256'] for r in prior['records'].values())
    hashes={}
    for split,count in SIZES.items():
        path=OUT/'fixtures'/(split+'.json')
        name='entry_'+split
        if not path.exists():
            rooms.SEEDS[name]=SEEDS[split]
            fixtures,_=rooms.panel(name,count)
            save(path,fixtures)
        fixtures=read(path)
        ids={rooms.content_hash(f) for f in fixtures.values()}
        if len(fixtures)!=count or len(ids)!=count or seen & ids:
            raise ValueError('fixture overlap or count mismatch')
        hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
        seen.update(ids)
    immutable(OUT/'protocol.json',dict(sizes=SIZES,seeds=SEEDS,fixture_hashes=hashes,version=1,
         authoring='no author calls; immutable previous two sealed CAPs',parent='trial09',combined='trial06',
         entries='eligible actual new recovery proposals on frozen natural route; retain up to first/middle/last per training room',
         target='terminal success advantage of deferring ONE opportunity versus taking, then identical frozen rules',
         evaluation='held fresh 4-arm natural full start, room-level paired outcomes; prior panels used for engineering parity only',
         entry_cap=CAP,deploy='only decide where original program proposes distinct eligible recovery, leave exits unchanged'))


def output(split,arm,room):
    return OUT/'runs'/split/arm/(room+'.json')


def selected(entries):
    if len(entries)<=CAP:return entries
    return [entries[0],entries[len(entries)//2],entries[-1]]


def scan(job):
    from tools import cap_entry_residual_engine as engine
    split,arm,room,fixture,gate=job
    path=output(split,arm,room)
    if path.exists():return
    row=engine.rollout(room,fixture,gate=gate,parent=arm=='parent')
    save(path,row)


def label(job):
    from tools import cap_entry_residual_engine as engine
    split,room,fixture=job
    root=OUT/'labels'/split/(room+'.json')
    if root.exists():return
    zero=read(output(split,'zero',room))
    rows=[]
    for entry in selected(zero['entries']):
        deferred=engine.rollout(room,fixture,defer_at=entry)
        if not deferred['matched']:raise ValueError('target entry not reproduced')
        rows.append(dict(room=room,features=entry['features'],take=int(zero['result']['success']),
                         defer=int(deferred['result']['success']),anchor=entry,
                         take_ticks=zero['result']['ticks'],defer_ticks=deferred['result']['ticks'],
                         take_history=zero['result']['history'],defer_history=deferred['result']['history']))
    save(root,dict(room=room,rows=rows))


def batch(pool,worker,jobs):
    futures=[pool.submit(worker,j) for j in jobs]
    for f in as_completed(futures):f.result()


def parity():
    from tools import cap_entry_residual_engine as engine
    rooms_dev=read(OUT.parent/'cap_fault_curriculum_20261124_01/rooms.json')['rooms']['dev']
    checks=[]
    for key in ['027','126','084','030']:
        room='diag_dev_'+key
        fixture=rooms_dev[room]
        zero=engine.rollout(room,fixture)
        baseline=engine.rollout(room,fixture,parent=True)
        for arm,r in [('P',baseline),('trial06',zero)]:
            expected=read(PREV/'runs/dev'/arm/'full'/(room+'.json'))['result']
            assert all(r['result'][k]==expected[k] for k in ('history','ticks','decisions','success')),(room,arm)
        if key in ('027','126','084') and not zero['entries']:raise ValueError('missed known intervention '+room)
        if zero['entries']:
            paired=engine.rollout(room,fixture,defer_at=zero['entries'][0])
            assert paired['matched'],room
        checks.append(dict(room=room,zero_history=zero['result']['history'],parent_history=baseline['result']['history'],opportunities=len(zero['entries'])))
    save(OUT/'parity.json',checks)


def labels(split):
    return [row for f in sorted((OUT/'labels'/split).glob('*.json')) for row in read(f)['rows']]


def held_diagnostic(out=OUT):
    """Audit one-step intervention labels out of sample, after freezing the gate."""
    from tools import cap_entry_residual_engine as engine, cap_entry_residual_fit as fit
    if Path(out).resolve()!=OUT.resolve():
        raise ValueError('held intervention scan requires the frozen run directory')
    path=OUT/'held_diagnostic.json'
    if path.exists():return read(path)
    fixtures=read(OUT/'fixtures/held.json')
    for arm in ('parent','zero','learned','always_defer'):
        if any(not output('held',arm,k).exists() for k in fixtures):
            raise ValueError('held closed-loop scan is incomplete')
    gate=read(OUT/'fit.json')['gate']
    missing=[('held',k,v) for k,v in fixtures.items() if read(output('held','zero',k))['entries']
             and not (OUT/'labels/held'/(k+'.json')).exists()]
    with ProcessPoolExecutor(20,mp_context=mp.get_context('fork'),initializer=engine.init_worker) as pool:
        batch(pool,label,missing)
    rows=labels('held')
    result={name:fit._summary(rows,g) for name,g in [('zero',fit.ZERO),('always_defer',fit.ALWAYS_DEFER),('learned',gate)]}
    result['discordant_room_weighted_auc']=fit._ranking(rows,gate)
    # Resample whole rooms, retaining the same within-room weight as the reported AUC.
    from collections import Counter
    import numpy as np
    counts=Counter(r['room'] for r in rows)
    ordered=sorted(counts)
    group={k:i for i,k in enumerate(ordered)}
    paired=[r for r in rows if r['take']!=r['defer']]
    pos=[r for r in paired if r['defer']]
    neg=[r for r in paired if r['take']]
    if pos and neg:
        ps=np.array([fit._advantage(gate,r['features']) for r in pos])
        ns=np.array([fit._advantage(gate,r['features']) for r in neg])
        concordance=(ps[:,None]>ns).astype(float)+.5*(ps[:,None]==ns)
        rng=np.random.default_rng(17)
        draws=rng.multinomial(len(ordered),[1/len(ordered)]*len(ordered),size=10000)
        def weights(records):
            return draws[:,[group[r['room']] for r in records]]/np.array([counts[r['room']] for r in records])
        wp,wn=weights(pos),weights(neg)
        denom=wp.sum(axis=1)*wn.sum(axis=1)
        keep=denom>0
        values=np.sum((wp @ concordance)*wn,axis=1)[keep]/denom[keep]
        result['auc_room_bootstrap_95']=np.quantile(values,[.025,.975]).tolist()
        result['auc_valid_bootstraps']=int(keep.sum())
    immutable(path,result)
    return result


def study(workers):
    from tools import cap_entry_residual_engine as engine
    panels()
    engine.prepare()
    if not (OUT/'parity.json').exists():
        engine.init_worker()
        parity()
        engine.close()
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('fork'),initializer=engine.init_worker) as pool:
        for split in ('train','validation'):
            fixtures=read(OUT/'fixtures'/(split+'.json'))
            batch(pool,scan,[(split,'zero',k,v,None) for k,v in fixtures.items() if not output(split,'zero',k).exists()])
            missing=[(split,k,v) for k,v in fixtures.items() if not (OUT/'labels'/split/(k+'.json')).exists()
                     and read(output(split,'zero',k))['entries']]
            batch(pool,label,missing)
        from tools import cap_entry_residual_fit as fit
        model=fit.train(labels('train'),labels('validation'))
        immutable(OUT/'fit.json',model)
        fixtures=read(OUT/'fixtures/held.json')
        for arm,gate in [('parent',None),('zero',None),('learned',model['gate']),('always_defer',{'kind':'always_defer'})]:
            batch(pool,scan,[('held',arm,k,v,gate) for k,v in fixtures.items() if not output('held',arm,k).exists()])
    from tools import cap_entry_residual_report as report
    report.build(OUT)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['panels','run']);p.add_argument('--workers',type=int,default=20)
    a=p.parse_args()
    if a.stage=='panels':panels();return
    try:study(a.workers)
    except BaseException:
        save(OUT/'ERROR.json',dict(time=time.time(),error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
