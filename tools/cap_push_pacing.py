"""Separate the effect of near-goal pacing from the decision time of contact."""
import argparse
from collections import Counter

import numpy as np

from experiments.planar_tasks.semantic_rl import push_phases
from experiments.planar_tasks.semantic_rl.phase_aware_cap import translating
from tools import cap_option_semantic_factorial as exp
from tools import cap_push_phase_comparison_report as report

OUT=exp.OUT/'pacing'
DOC=exp.DOC/'pacing'
ARMS=('coupled','pace_before','pace_if_ready','split','pace_after')
CAPS=exp.phase.CAPS


def run_one(job):
    sid,arm,k,fixture,unused,path=job
    policy=exp.phase.FrozenCAP(exp.phase.MODULES[sid])
    counts=Counter()
    with exp.phase.world(fixture) as (w,_):
        policy.update(w.history)
        while not exp.ended(w):
            obs=exp.observe(w)
            action=policy.propose(obs)
            movers=translating(action)
            if len(movers)!=1:
                exp.tick_action(w,action,'coupled',policy.update,counts)
                continue
            if arm=='coupled':
                exp.tick_action(w,action,'coupled',policy.update,counts)
                continue
            if arm in ('pace_before','pace_if_ready'):
                i,axis=push_phases.pusher(action)
                if arm=='pace_before' or push_phases.contact_state(w,action,i,axis)=='ready':
                    changed=exp.PRECISION.choose_push(obs,action,action['ticks'])
                    counts['retimed']+=changed!=action
                    action=changed
                exp.tick_action(w,action,'coupled',policy.update,counts)
                continue
            if arm=='pace_after':
                exp.tick_action(w,action,'split_precision',policy.update,counts)
                continue
            touch=push_phases.contact(w,action,min(exp.remaining(w),action['ticks']))
            counts['contact_'+touch.status]+=1
            if touch.ticks:policy.update(w.history)
            left=action['ticks']-touch.ticks
            if touch.status!='ready' or left<=0 or exp.ended(w):
                assert touch.ticks
                continue
            drive=push_phases.push_from_contact(w,dict(action,ticks=left),min(left,exp.remaining(w)))
            if drive.ticks:policy.update(w.history)
            elif not touch.ticks:
                w.step(dict(target_id=None,a0=44,a1=44,ticks=1),remaining_ticks=exp.remaining(w))
                policy.update(w.history)
            counts['drive_'+drive.status]+=1
        result=exp.output(w,counts)
    exp.save(path,result)


def summarize(panels):
    counts=np.array([1+len(f['movable_objects']) for f in panels.values()])
    y={}
    overview={}
    table=[]
    for sid in CAPS:
        y[sid]={}
        overview[sid]={}
        for arm in ARMS:
            rr=[exp.read(OUT/'runs'/sid/arm/(k+'.json')) for k in panels]
            assert len(rr)==200 and all(0<r['ticks']<=1800 and 0<r['decisions']<=300 for r in rr)
            seq=report.outcomes(rr)
            y[sid][arm]=seq
            overview[sid][arm]=dict(wins=int(seq.sum()),retimed=sum(r['counts'].get('retimed',0)>0 for r in rr),
                                    mean_ticks=float(np.mean([r['ticks'] for r in rr])))
            for k,r in zip(panels,rr):
                table.append(dict(room=k,cap=sid,arm=arm,success=r['success'],ticks=r['ticks'],
                                  boundaries=r['decisions'],pacing=r['counts'].get('retimed',0)))
    arrays={a:np.stack([y[sid][a] for sid in CAPS]) for a in ARMS}
    pairs={}
    for a,b in [('coupled','split'),('coupled','pace_before'),('coupled','pace_if_ready'),('coupled','pace_after'),
                ('split','pace_after'),('pace_before','pace_after'),('pace_if_ready','pace_after')]:
        pairs[b+'_vs_'+a]=report.paired_mean(arrays[a],arrays[b],counts)
        for sid in CAPS:
            overview[sid][b][f'vs_{a}']=exp.phase.paired(y[sid][a],y[sid][b],counts)
    interaction=report.paired_mean(arrays['pace_if_ready']-arrays['coupled'],arrays['pace_after']-arrays['split'],counts)
    data=dict(paired=pairs,interaction=interaction,cap=overview)
    DOC.mkdir(exist_ok=True,parents=True)
    exp.save(DOC/'results.json',data)
    import csv
    with (DOC/'rooms.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    exp.save(OUT/'FINISHED.json',data)
    print(overview,pairs,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    exp.phase.init()
    known=exp.collect_identities()
    for name in ('development','held'):
        known.update(exp.phase.input_hash(f) for f in exp.read(exp.OUT/(name+'.json')).values())
    seed=662314125
    cases={f'pace_{r["id"]}':r['fixture'] for r in exp.phase.cap.rooms.generate(seed)['cases']}
    assert len(cases)==len({exp.phase.input_hash(f) for f in cases.values()})==200
    path=OUT/'rooms.json'
    if path.exists():assert exp.read(path)==cases
    else:
        assert all(exp.phase.input_hash(f) not in known for f in cases.values())
        exp.save(path,cases)
    protocol=dict(seed=seed,rooms=exp.phase.cap.digest(cases),arms=ARMS,caps=CAPS,ticks=1800,boundaries=300)
    path=OUT/'protocol.json'
    if path.exists():assert exp.phase.cap.digest(exp.read(path))==exp.phase.cap.digest(protocol)
    else:exp.save(path,protocol)
    jobs=[(sid,arm,k,f,None,OUT/'runs'/sid/arm/(k+'.json')) for k,f in cases.items() for sid in CAPS for arm in ARMS]
    exp.execute(run_one,jobs,'timing of push pacing',a.workers)
    summarize(cases)


if __name__=='__main__':
    main()
