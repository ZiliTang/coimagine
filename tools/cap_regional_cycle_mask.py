"""Fixed, public-history anti-cycle mask for the frozen ten-ID regional actor."""
import collections
import csv
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from tools import cap_regional_advantage as ref

study = ref.study
OUT = ref.OUT.parent / 'cap_regional_cycle_mask_20261120_01'
DOC = study.DOCS.parent / '2026-11-20_regional_cycle_mask'
SEED = 6820491
PERIODS = (1, 2, 3)
REPEATS = 3
METRES = .05
WINDOW = 3


def identity(a):
    return (a['target_id'], a['a0'], a['a1'], a['ticks'])


def cycle(history, current_xy, preferred):
    for p in PERIODS:
        n = REPEATS*p
        if len(history)<n:
            continue
        tail = list(history)[-n:]
        actions = [action for action,_ in tail]
        pattern = actions[:p]
        if preferred!=pattern[0] or actions!=pattern*REPEATS:
            continue
        positions = [xy for _,xy in tail]+[current_xy]
        # Compare every visit to the same phase, including the current cycle entrance.
        maximum = 0.
        for phase in range(p):
            same = positions[phase::p]
            for j, early in enumerate(same):
                for late in same[j+1:]:
                    maximum = max(maximum, *(math.dist(a,b) for a,b in zip(early,late,strict=True)))
        if maximum<=METRES:
            return pattern,maximum
    return None


@torch.inference_mode()
def episode(fixture, apply):
    env = ref.ppo.environment(fixture)
    lib = ref.cpp.Proposals()
    past = collections.deque(maxlen=REPEATS*max(PERIODS))
    windows = []
    remaining = 0
    first_change = None
    try:
        for step in range(300):
            x,_,_,_,pool,raw = lib.panel(env.history)
            logits = ref.ID(torch.from_numpy(x[None]))[0].numpy()
            ranked = [raw[i] for i in np.argsort(-logits,kind='stable') if np.isfinite(logits[i])]
            actions = list(dict.fromkeys(identity(a) for a in ranked or pool[:1]))
            obs = study.dq.current_event(env.history)['observation']
            xy = tuple((r['own']['x'],r['own']['y']) for r in obs['robots'])
            preferred = actions[0]
            if remaining==0:
                hit = cycle(past,xy,preferred)
                if hit:
                    pattern,dist = hit
                    forbidden = set(pattern)
                    windows.append(dict(step=step,tick=obs['tick'],prefix=env.history.sha256,
                                        period=len(pattern),pattern=pattern,max_cm=100*dist,
                                        seen=0,no_alternative=0,overrides=0,available=0))
                    remaining = WINDOW
            chosen = preferred
            if remaining:
                w = windows[-1]
                alternative = next((a for a in actions if a not in forbidden),None)
                w['seen']+=1
                if alternative is None:
                    w['no_alternative']+=1
                else:
                    w['available']+=1
                    if apply and preferred in forbidden:
                        chosen=alternative
                        w['overrides']+=1
                        if first_change is None:
                            first_change=dict(step=step,tick=obs['tick'],prefix=env.history.sha256,
                                              preferred=preferred,chosen=chosen)
                remaining-=1
            act=next(a for a in pool if identity(a)==chosen)
            past.append((chosen,xy))
            if windows and remaining==0 and step==windows[-1]['step']+WINDOW-1:
                past.clear()
            env.step(act,remaining_ticks=1800-env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        return dict(success=int(env.env.success),ticks=env.env.total_periods,steps=step+1,
                    history=env.history.sha256,termination=env.env.termination or 'decision_budget',
                    windows=windows,first_change=first_change)
    finally:
        env.close_reads()
        study.dq.fast_runtime.ACTIVE=None


def freeze():
    assert study.sha(ref.base.OUT/'fixed.pt')==study.read(ref.OUT/'protocol.json')['id_sha']
    assert ref.cpp.load()[1]['binary']==study.read(ref.OUT/'protocol.json')['binary']
    original = study.read(study.OUT/'held.json')
    assert study.ext.old.digest(original)==study.read(study.OUT/'frozen.json')['held']
    jobs = [('reviewed',cid,int(cid.split('_')[-2]),f) for cid,f in sorted(original.items())
            if cid.startswith('fresh_')]
    used = {study.ext.input_hash(f) for f in original.values()}
    used.update(study.ext.input_hash(f) for f in study.read(study.OUT/study.ext.DEV).values())
    used.update(study.read(OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
    for p in study.read(ref.OUT/'panel.json'):
        used.update(p['hashes'])
    for p in (OUT.parent/'cap_regional_actor_iteration_20261120_01/panels').glob('*.json'):
        used.update(study.ext.input_hash(r['fixture']) for r in study.read(p))
    used.update(r['hash'] for r in study.read(OUT.parent/'cap_regional_once_20261120_01/cases.json').values())
    for r in study.ext.old.rooms.generate(SEED)['cases']:
        h=study.ext.input_hash(r['fixture'])
        assert h not in used
        used.add(h)
        jobs.append(('new',f"cm{SEED}_{r['id']}",r['movable_count'],r['fixture']))
    assert collections.Counter(j[0] for j in jobs)=={'new':200,'reviewed':200}
    protocol = dict(seed=SEED,actor=study.sha(ref.base.OUT/'fixed.pt'),binary=ref.cpp.load()[1]['binary'],
                    source=study.sha(__file__),panels=study.ext.old.digest(jobs),
                    periods=PERIODS,repeats=REPEATS,metres=METRES,mask_boundaries=WINDOW,
                    identity='target_id,a0,a1,requested ticks; all duplicate CAP proposals masked together',
                    position='both public robot odometries, pairwise Euclidean distance of all visits to each phase, including next entrance',
                    fallback='original highest legal CAP when all masked, count boundary against window; clear history at expiry',
                    primary='new panel, paired terminal success change; reviewed panel separately; one frozen rule')
    for name,data in [('protocol',protocol),('panels',jobs)]:
        path=OUT/(name+'.json')
        if path.exists(): assert study.ext.old.digest(study.read(path))==study.ext.old.digest(data)
        else: study.save(path,data)
    return jobs


def room(job):
    group,cid,n,f=job
    file=OUT/'rooms'/group/(cid+'.json')
    if file.exists(): return study.read(file)
    original=episode(f,False)
    if group=='reviewed':
        old=study.read(ref.base.OUT/'fixed/evaluation/fresh'/(cid+'.json'))
        assert (old['history'],old['success'],old['ticks'])==(original['history'],original['success'],original['ticks'])
    masked=episode(f,True)
    if masked['first_change'] is None:
        assert original==masked
    else:
        assert original['windows'][0]['prefix']==masked['windows'][0]['prefix']
    row=dict(panel=group,cid=cid,objects=n,original=original,masked=masked)
    study.save(file,row)
    return row


def report(rows):
    DOC.mkdir(parents=True,exist_ok=True)
    simple=[]
    for r in sorted(rows,key=lambda s:s['cid']):
        old,new=r['original'],r['masked']
        w=new['windows']
        simple.append(dict(panel=r['panel'],cid=r['cid'],objects=r['objects'],original=old['success'],
                           masked=new['success'],delta=new['success']-old['success'],
                           original_ticks=old['ticks'],masked_ticks=new['ticks'],windows=len(w),
                           original_windows=len(old['windows']),changed=sum(v['overrides'] for v in w),
                           window_steps=sum(v['seen'] for v in w),no_alternative=sum(v['no_alternative'] for v in w),
                           first_change_tick=(new['first_change'] or {}).get('tick'),
                           p1=sum(v['period']==1 for v in w),p2=sum(v['period']==2 for v in w),
                           p3=sum(v['period']==3 for v in w)))
    with (DOC/'rooms.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(simple[0]))
        writer.writeheader()
        writer.writerows(simple)
    metrics={}
    for panel in ['reviewed','new']:
        a=[r for r in simple if r['panel']==panel]
        rng=np.random.default_rng(17)
        boot=np.zeros(20000)
        for n in range(2,6):
            ds=np.array([r['delta'] for r in a if r['objects']==n])
            boot+=rng.choice(ds,(len(boot),len(ds))).sum(1)
        metrics[panel]=dict(n=len(a),original=sum(r['original'] for r in a),masked=sum(r['masked'] for r in a),
                            wins=sum(r['delta']==1 for r in a),losses=sum(r['delta']==-1 for r in a),
                            triggered=sum(r['windows']>0 for r in a),redirected=sum(r['changed']>0 for r in a),
                            original_triggered=sum(r['original_windows']>0 for r in a),
                            triggered_by_baseline_result=[[sum(r['original']==y and bool(r['windows'])==t for r in a)
                                                           for t in [False,True]] for y in [0,1]],
                            **{k:sum(r[k] for r in a) for k in ['windows','changed','window_steps','no_alternative','p1','p2','p3']},
                            ci_pp=(100*np.quantile(boot,[.025,.975])/len(a)).tolist())
    study.save(DOC/'metrics.json',metrics)
    study.save(DOC/'protocol.json',study.read(OUT/'protocol.json'))
    study.save(OUT/'FINISHED.json',metrics)


def main():
    ref.init()
    jobs=freeze()
    with ProcessPoolExecutor(16,mp_context=mp.get_context('fork')) as pool:
        rows=[]
        for f in as_completed([pool.submit(room,j) for j in jobs]):
            rows.append(f.result())
            study.save(OUT/'progress.json',dict(pairs=len(rows),total=len(jobs)))
    report(rows)


if __name__=='__main__':
    main()
