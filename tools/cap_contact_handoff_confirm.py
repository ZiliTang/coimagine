"""Commit to the opening expert, permitting one handoff after a verified contact stall."""
import argparse
from collections import Counter
import csv
import hashlib

import numpy as np

from experiments.planar_tasks.semantic_rl.phase_aware_cap import position,recovery_options,signature,translating
from tools import cap_initial_specialization_confirm as prior
from tools import cap_option_semantic_factorial_report as report

EXP=prior.EXP
OUT=EXP.OUT/'contact_handoff'
DOC=EXP.DOC/'contact_handoff'
CONTROLLER='split_precision'


def exercise(job):
    bank,member,key,fixture,unused,path=job
    pickup=[]
    counts=Counter()
    previous={}
    with EXP.phase.world(fixture) as (world,_):
        policies,update=EXP.banks(world,bank)
        active=EXP.LIBRARIES[bank].index(member)
        while not EXP.ended(world):
            obs=EXP.observe(world)
            command=policies[active].propose(obs)
            if len(translating(command))==1:
                stall_key=signature(command)
                _,i,_=stall_key
                enough=previous.get(stall_key,[])
                if not pickup and len(enough)>=2 and obs['tick']-enough[-2]<=120:
                    choices={signature(a):quality for quality,a in recovery_options(obs,command,{stall_key})}
                    menu=[]
                    for other,policy in enumerate(policies):
                        if other==active:continue
                        try:d=policy.propose(obs)
                        except ValueError:continue
                        if len(translating(d))!=1:continue
                        key_=signature(d)
                        if key_==stall_key or d['target_id']!=command['target_id']:continue
                        robot,axis=EXP.push_phases.pusher(d)
                        touching=EXP.push_phases.contact_state(world,d,robot,axis)=='ready'
                        if touching or key_ in choices:
                            menu.append((int(touching),choices.get(key_,0.),-other,d))
                    if menu:
                        _,_,minus,alternative=max(menu,key=lambda r:r[:3])
                        pickup.append(dict(tick=obs['tick'],old=active,new=-minus,
                                           previous=command,action=alternative))
                        active=-minus
                        command=alternative
                        stall_key=signature(command)
                        _,i,_=stall_key
                start=position(obs,i)
                failed=counts['contact_contact_timeout']
                EXP.tick_action(world,command,CONTROLLER,update,counts)
                aft=EXP.observe(world)
                if counts['contact_contact_timeout']>failed and aft['tick']-obs['tick']>=12 and np.linalg.norm(position(aft,i)-start)<.05:
                    entries=previous.get(stall_key,[])
                    previous[stall_key]=[*[v for v in entries if aft['tick']-v<=120],aft['tick']]
                else:
                    previous.pop(stall_key,None)
            else:
                EXP.tick_action(world,command,CONTROLLER,update,counts)
        out=EXP.output(world,counts)
    EXP.save(path,dict(out,handoff=pickup))


def summarize(held,selection):
    ids=sorted(held)
    cl=np.array([1+len(held[k]['movable_objects']) for k in ids])
    panel=np.array([k[:8] for k in ids])
    strata=np.array([f'{a}/{b}' for a,b in zip(panel,cl)])
    output={}
    rows=[]
    for bank,initial in selection.items():
        a=[];b=[];count=0
        for k in ids:
            sid=initial[k]
            original=EXP.read(prior.path(CONTROLLER,sid,k))
            changed=EXP.read(OUT/'runs'/bank/(k+'.json'))
            if not changed['handoff']:
                assert all(original[key]==changed[key] for key in ('history','success','ticks','decisions'))
            count+=bool(changed['handoff'])
            a.append(original['success']);b.append(changed['success'])
            rows.append(dict(bank=bank,room=k,before=a[-1],after=b[-1],handoff=changed['handoff']))
        d=np.subtract(b,a)
        output[bank]=dict(original=sum(a),handoff=sum(b),changed_selected=count,rescued=int(sum(d>0)),lost=int(sum(d<0)),
                          effect=report.contrast(d,strata,panel))
    DOC.mkdir(exist_ok=True,parents=True)
    EXP.save(DOC/'results.json',output)
    EXP.save(DOC/'cases.json',rows)
    EXP.save(OUT/'FINISHED.json',output)
    print(output,flush=True)


def freeze():
    h=EXP.read(prior.OUT/'held.json')
    selections=EXP.read(prior.OUT/'selections.json')[CONTROLLER]
    sel={bank:arms['evidence'] for bank,arms in selections.items()}
    manifest=dict(held=EXP.phase.cap.digest(h),selection=EXP.phase.cap.digest(sel),
                  controller=CONTROLLER,code=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
                  policy='one handoff only after two contact timeouts with <5 cm robot travel in 120 ticks to a distinct feasible same-target proposal',
                  comparison='arbitration only, frozen opening assignment and split+pace control identical')
    p=OUT/'protocol.json'
    if p.exists():assert EXP.read(p)==manifest
    else:EXP.save(p,manifest)
    return h,sel


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    h,sel=freeze()
    if a.freeze:return
    assert (prior.OUT/'FINISHED.json').exists()
    EXP.initialize()
    objs=[(bank,sid,k,h[k],None,OUT/'runs'/bank/(k+'.json')) for bank,m in sel.items() for k,sid in m.items()]
    save=EXP.save
    def progress(p,data):
        if p.name=='progress.json':p=OUT/'progress.json'
        save(p,data)
    EXP.save=progress
    EXP.execute(exercise,objs,'one contact-aware expert handoff',a.workers)
    summarize(h,sel)


if __name__=='__main__':main()
