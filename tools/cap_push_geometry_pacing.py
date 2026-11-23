"""Pace post-contact transport by the remaining distance along the proposed ray."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
import math

import numpy as np

from experiments.planar_tasks.relational_search.motion import ACTIONS
from experiments.planar_tasks.semantic_rl.candidate_semantics import fitted
from experiments.planar_tasks.semantic_rl.phase_aware_cap import signature
from tools import cap_push_pacing as pacing
from tools.cap_push_phase_comparison_report import paired_mean

EXP=pacing.exp
OUT=EXP.OUT/'geometry_pacing'
DOC=EXP.DOC/'geometry_pacing'
CAPS=pacing.CAPS
ARMS=('fixed','distance_gate','ray_distance')


class PaceProfile:
    def __init__(self,name):
        self.name=name

    def choose_push(self, obs, command, remaining):
        if self.name=='fixed':
            return BASE.choose_push(obs,command,remaining)
        cap=min(command['ticks'],remaining)
        nominal=dict(command,ticks=cap)
        target,robot,axis=signature(command)
        if target!=obs['goal']['target_id']:
            return nominal
        box=fitted(obs,target)
        if box is None:
            return nominal
        delta=np.subtract(obs['goal']['xy'],[box['x'],box['y']])
        length=np.linalg.norm(delta)
        proj=float(delta@axis)
        if self.name=='distance_gate':
            return BASE.choose_push(obs,command,remaining) if length<=.33 else nominal
        side2=max(0.,float(delta@delta)-proj**2)
        radius=obs['goal']['radius']
        # The ray needs to enter an inner goal disk to leave room for settling.
        inset=max(.05,radius-.04)
        if length>.48 or proj<.82*length or side2>inset**2:
            return nominal
        drive=max(0.,proj-math.sqrt(inset**2-side2))
        if drive>.15:
            return nominal
        code=command[f'a{robot}']
        slow=code-code%4+(0 if drive<=.045 else 1)
        pair=[nominal['a0'],nominal['a1']]
        pair[robot]=slow
        if ACTIONS[slow].speed<ACTIONS[code].speed and pair in obs['entity_legal_pairs'].get(target,[]):
            nominal[f'a{robot}']=slow
        velocity=min(.30,max(.12,ACTIONS[nominal[f'a{robot}']].speed))
        # Continue if the estimated along-ray distance requires the full pulse.
        nominal['ticks']=min(cap,max(5,math.ceil((drive+.025)/(.05*velocity))))
        return nominal


BASE=EXP.PRECISION


def job(values):
    sid,arm,room,f,panel=values
    path=OUT/'runs'/panel/sid/arm/(room+'.json')
    if path.exists():
        return
    original=EXP.PRECISION
    EXP.PRECISION=PaceProfile(arm)
    try:
        pacing.run_one((sid,'pace_after',room,f,None,path))
    finally:
        EXP.PRECISION=original


def run(panel,data,arms,workers):
    jobs=[(sid,a,k,f,panel) for k,f in data.items() for sid in CAPS for a in arms
          if not (OUT/'runs'/panel/sid/a/(k+'.json')).exists()]
    if jobs:
        with ProcessPoolExecutor(workers,initializer=EXP.phase.init) as pool:
            future=[pool.submit(job,r) for r in jobs]
            for n,f in enumerate(as_completed(future),1):
                f.result()
                if n%50==0 or n==len(jobs):
                    print(panel,n,len(jobs),flush=True)


def read_group(panel,cases,arms):
    table={}
    for a in arms:
        by_cap=[]
        for sid in CAPS:
            if panel=='held' and a=='fixed':
                root=pacing.OUT/'runs'/sid/'pace_after'
            else:
                root=OUT/'runs'/panel/sid/a
            rows=[EXP.read(root/(k+'.json')) for k in cases]
            assert all(0<r['ticks']<=1800 and 0<r['decisions']<=300 for r in rows)
            by_cap.append([r['success'] for r in rows])
        table[a]=np.array(by_cap)
    return table


def main():
    cases=EXP.read(EXP.phase.OUT/'development.json')
    reference=EXP.read(EXP.phase.OUT/'protocol.json')
    assert EXP.phase.cap.digest(cases)==reference['split']['development']
    run('development',cases,ARMS,8)
    dev=read_group('development',cases,ARMS)
    candidate=max(ARMS[1:],key=lambda a:(int(dev[a].sum()),int((dev[a]>dev['fixed']).sum()),-ARMS.index(a)))
    EXP.save(OUT/'selected.json',dict(candidate=candidate,dev={a:r.sum(1).tolist() for a,r in dev.items()},
                                    selection='highest development wins among two timed-geometry variants'))
    test=EXP.read(pacing.OUT/'rooms.json')
    assert EXP.phase.cap.digest(test)==EXP.read(pacing.OUT/'protocol.json')['rooms']
    assert (pacing.OUT/'FINISHED.json').is_file()
    run('held',test,[candidate],8)
    held=read_group('held',test,('fixed',candidate))
    objects=np.array([1+len(f['movable_objects']) for f in test.values()])
    compared={}
    for i,sid in enumerate(CAPS):
        compared[sid]=EXP.phase.paired(held['fixed'][i],held[candidate][i],objects)
    effect=paired_mean(held['fixed'],held[candidate],objects)
    result=dict(selected=candidate,development={a:r.sum(1).tolist() for a,r in dev.items()},held=compared,effect=effect,
                rooms={k:{a:held[a][:,i].tolist() for a in held} for i,k in enumerate(test)})
    DOC.mkdir(exist_ok=True,parents=True)
    EXP.save(DOC/'results.json',result)
    EXP.save(OUT/'FINISHED.json',result)
    print(compared,effect,flush=True)


if __name__=='__main__':
    main()
