"""Fallback to the dev-best CAP on measured failure to advance over real PUSHes."""
import argparse
from collections import Counter,deque
import hashlib

import numpy as np

from experiments.planar_tasks.semantic_rl.candidate_semantics import fitted
from experiments.planar_tasks.semantic_rl.phase_aware_cap import translating
from tools import cap_selective_semantic_route as stored
from tools import cap_option_semantic_factorial_report as report

EXP=stored.EXP
OUT=EXP.OUT/'progress_handoff'
DOC=EXP.DOC/'progress_handoff'
EX='split_precision'
BANK='specialized'


def patch_position(obs):
    box=fitted(obs,obs['goal']['target_id'])
    if box is None:return None
    return np.array([box['x'],box['y']])


def trial(job):
    room,fixture,origin,fallback,unused,path=job
    if origin==fallback:
        orig=EXP.read(stored.OUT/'runs'/EX/origin/(room+'.json'))
        EXP.save(path,dict(orig,start=origin,result=origin,handoffs=[]))
        return
    act=origin
    events=deque()
    handoffs=[]
    counts=Counter()
    with EXP.phase.world(fixture) as (w,_):
        modules={id_:EXP.phase.FrozenCAP(EXP.MODULES[id_]) for id_ in (origin,fallback)}
        def update(h):
            for x in modules.values():x.update(h)
        update(w.history)
        while not EXP.ended(w):
            obs=EXP.observe(w)
            a=modules[act].propose(obs)
            target=obs['goal']['target_id']
            thrust=a['target_id']==target and bool(translating(a))
            if thrust and act!=fallback and len(events)>=2:
                oldest=events[0]
                newest=events[-1]
                if newest['tick']-oldest['before_tick']>=55 and obs['tick']-oldest['before_tick']<=140:
                    progress=np.linalg.norm(np.subtract(newest['end'],oldest['start']))
                    if progress<=.07 and sum(x['bad'] for x in events)>=2:
                        try:backup=modules[fallback].propose(obs)
                        except ValueError:backup=None
                        if backup is not None and backup!=a:
                            handoffs.append(dict(tick=obs['tick'],estimated_progress_m=float(progress),
                                                 timeouts_or_low_progress=sum(x['bad'] for x in events),
                                                 proposal=a,alternative=backup))
                            act=fallback;a=backup
            start=patch_position(obs) if thrust and act!=fallback else None
            timeouts=counts['contact_contact_timeout']
            EXP.tick_action(w,a,EX,update,counts)
            after=EXP.observe(w)
            if start is not None:
                end=patch_position(after)
                if end is not None:
                    bad=(counts['contact_contact_timeout']>timeouts or np.linalg.norm(end-start)<.025)
                    events.append(dict(before_tick=obs['tick'],tick=after['tick'],start=start,end=end,bad=bool(bad)))
            while events and after['tick']-events[0]['before_tick']>140:
                events.popleft()
        result=EXP.output(w,counts)
    EXP.save(path,dict(result,start=origin,result=act,handoffs=handoffs))


def freeze():
    cases=EXP.read(stored.OUT/'held.json')
    plan=EXP.read(stored.OUT/'selections.json')[EX]
    protocol=dict(held=EXP.phase.cap.digest(cases),assignments=EXP.phase.cap.digest(plan),
        agent_code=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        trigger='at least two unsuccessful PUSH outcomes across 55-140 ticks with observed target displacement <=7cm; once to dev-best CAP',
        comparison='same opening full evidence assignment, compare no handoff and fixed best')
    p=OUT/'protocol.json'
    if p.exists():assert EXP.read(p)==protocol
    else:EXP.save(p,protocol)
    return cases,plan


def run(cases,plan,workers):
    EXP.initialize()
    jobs=[(k,f,plan['always'][k],plan['single'][k],None,OUT/'runs'/(k+'.json')) for k,f in cases.items()]
    save=EXP.save
    def progress(p,data):
        if p.name=='progress.json':p=OUT/'progress.json'
        save(p,data)
    EXP.save=progress
    EXP.execute(trial,jobs,'persistent low-progress handoff',workers)


def result(cases,plan):
    rooms=sorted(cases)
    panels=np.array([r[:7] for r in rooms])
    strata=np.array([f'{p}/{1+len(cases[k]["movable_objects"])}' for p,k in zip(panels,rooms)])
    rows=[]
    for k in rooms:
        no=EXP.read(stored.OUT/'runs'/EX/plan['always'][k]/(k+'.json'))
        one=EXP.read(OUT/'runs'/(k+'.json'))
        fixed=EXP.read(stored.OUT/'runs'/EX/plan['single'][k]/(k+'.json'))
        if not one['handoffs']:
            assert all(no[m]==one[m] for m in ('history','ticks','decisions','success'))
        rows.append(dict(room=k,opening=no['success'],fixed=fixed['success'],handoff=one['success'],
                         trigger=one['handoffs']))
    cells={a:np.array([r[a] for r in rows]) for a in ('opening','fixed','handoff')}
    d=cells['handoff']-cells['opening']
    data=dict(wins={k:int(v.sum()) for k,v in cells.items()},exposed=sum(bool(r['trigger']) for r in rows),
         rescued=int(sum(d>0)),lost=int(sum(d<0)),over_opening=report.contrast(d,strata,panels),
         over_fixed=report.contrast(cells['handoff']-cells['fixed'],strata,panels))
    DOC.mkdir(exist_ok=True,parents=True)
    EXP.save(DOC/'results.json',data)
    EXP.save(DOC/'rooms.json',rows)
    EXP.save(OUT/'FINISHED.json',data)
    print(data,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    cases,plan=freeze()
    if not a.freeze:
        assert (stored.OUT/'FINISHED.json').exists()
        run(cases,plan,a.workers)
        result(cases,plan)
