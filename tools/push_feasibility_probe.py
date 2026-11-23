"""Can public local geometry predict one-option fulfillment on unseen prefixes?"""
from concurrent.futures import ProcessPoolExecutor
import csv
import math

import numpy as np
import torch

from coimagine.random_start.spec import ROBOT_RADIUS
from experiments.option_rl_v19.unknown_shape import direction_matches, visible_faces
from experiments.planar_tasks.semantic_rl.candidate_semantics import endpoints
from experiments.planar_tasks.semantic_rl.phase_aware_cap import position
from experiments.planar_tasks.semantic_rl.push_contact_recovery import clearance
from tools import push_contract_bench as bench
from tools import cap_option_semantic_factorial as study

OUT=study.OUT/'feasibility'
DOC=study.DOC/'feasibility'


def measure(case,off,baseline):
    fixture=(bench.HELD if case['split']=='stress' else bench.DEV)[case['room']]
    cmd=case['command']
    i,axis=study.push_phases.pusher(cmd)
    direction=np.array(axis)
    with bench.world(fixture) as (w,_):
        prev=None
        for step in bench.archive(case['sid'],case['room'])[:case['index']]:
            prev=study.observe(w)
            w.step(step['a'],remaining_ticks=1800-w.env.total_periods)
            assert w.env.total_periods==step['tick']
        assert w.history.sha256==baseline['prefix']
        if off!='zero':
            n,t=bench.OFFSETS[off]
            assert bench.perturb(w,i,n*direction+t*np.array([-direction[1],direction[0]])) is None
        obs=study.observe(w)
        own=position(obs,i);peer=position(obs,1-i)
        view=obs['robots'][i]
        env=w.env
        sensor=env._sensor(i,obs['tick'],obs['tick']*.05,cmd['target_id'])
        faces=[f for f in visible_faces(sensor) if direction_matches(f[1],axis)]
        returns,_,labels=endpoints(view)
        returns=returns[labels!=cmd['target_id']]
        arena=np.asarray(view['arena'])
        rays=sum(k==cmd['target_id'] for k in view['scan']['entity_ids'])
        base=baseline['geometry']
        raw=[cmd['ticks']*.05,bench.ACTIONS[cmd[f'a{i}']].speed]
        compact=raw+[float(base[k]) for k in ('visible','points','contact','goal_gap','tangent','station_clear','peer')]
        result={}
        for q_rule in ('near','center'):
            options=[]
            for face in faces:
                n,t,rho,lo,hi=face[1:6]
                margin=min(.04,(hi-lo)/2)
                qc=.5*(lo+hi) if q_rule=='center' else np.clip(own@t,lo+margin,hi-margin)
                q=n*rho+t*qc
                station=q-n*(ROBOT_RADIUS+.15)
                end=q-n*ROBOT_RADIUS
                track=clearance(own,station,returns)
                final=clearance(station,end,returns)
                wall=min(*station,*(arena-station),*end,*(arena-end))-ROBOT_RADIUS
                peer_gap=clearance(own,station,peer[None])-2*ROBOT_RADIUS
                angle=float(view['own']['angle'])
                align=float(np.array([math.cos(angle),math.sin(angle)])@n)
                travel=float(np.linalg.norm(station-own))
                options.append(dict(score=travel+2*max(0.,ROBOT_RADIUS-min(track,final))+max(0.,-wall),
                                    travel=travel,gap=track,final=final,wall=max(-.5,wall),
                                    peer=max(-.5,peer_gap),heading=align,span=float(hi-lo),face=float(n@direction)))
            best=min(options,key=lambda q:q['score']) if options else dict.fromkeys(
                ('travel','gap','final','wall','peer','heading','span','face'),0.)
            result[q_rule]=list(best[k] for k in ('travel','gap','final','wall','peer','heading','span','face'))
        contact_ready=study.push_phases.contact_state(w,cmd,i,axis)=='ready'
        recent=0. if prev is None else float(np.linalg.norm(own-position(prev,i)))
        full=compact+[len(faces),int(contact_ready),rays,recent,view['own']['vx'],view['own']['vy'],
                      *result['near'],*result['center']]
    candidates={}
    for name in ('base','near5'):
        row=bench.read(bench.OUT/'runs'/name/off/(case['id']+'.json'))
        assert row['status']=='executed'
        candidates[name]=int(row['effect'])
    return dict(room=case['room'],split=case['split'],kind=case['kind'],off=off,
                command=raw,compact=compact,full=full,success=candidates,ready=int(contact_ready))


def job(args):
    c,o,b=args
    dest=OUT/'inputs'/o/(c['id']+'.json')
    if not dest.exists():
        bench.save(dest,measure(c,o,b))
    return bench.read(dest)


def fit(x,y,rooms):
    count={k:sum(rooms==k) for k in set(rooms)}
    weight=np.array([1/count[k] for k in rooms])
    weight/=weight.sum()
    mean=weight@x
    sd=np.maximum(.05,np.sqrt(weight@np.square(x-mean)))
    values=torch.tensor((x-mean)/sd,dtype=torch.float64)
    labels=torch.tensor(y,dtype=torch.float64)
    weights=torch.from_numpy(weight)
    w=torch.zeros(x.shape[1]+1,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([w],max_iter=80,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        losses=torch.nn.functional.binary_cross_entropy_with_logits(values@w[:-1]+w[-1],labels,reduction='none')
        z=weights@losses+.05*w[:-1].square().sum()
        z.backward()
        return z
    opt.step(closure)
    val=w.detach().numpy()
    return lambda x:1/(1+np.exp(-np.clip(((x-mean)/sd)@val[:-1]+val[-1],-30,30)))


def metrics(y,p,rooms):
    match=p[y==1,None]-p[None,y==0]
    comparative=[]
    for r in sorted(set(rooms)):
        ids=rooms==r
        a=p[ids & (y==1)]
        b=p[ids & (y==0)]
        if len(a) and len(b):
            d=a[:,None]-b[None,:]
            comparative.append(float(np.mean((d>0)+.5*(d==0))))
    return dict(n=len(y),positive=int(sum(y)),auc=float(np.mean((match>0)+.5*(match==0))),
                brier=float(np.mean((y-p)**2)),within_room_pairs=dict(rooms=len(comparative),
                     correctness=float(np.mean(comparative)) if comparative else None))


def report(rows):
    result={}
    for arm in ('base','near5'):
        train=[r for r in rows if r['split']=='development']
        y=np.array([r['success'][arm] for r in train])
        names=np.array([r['room'] for r in train])
        table={}
        for feature in ('command','compact','full'):
            predict=fit(np.array([r[feature] for r in train]),y,names)
            table[feature]={}
            for split in ('validation','stress'):
                part=[r for r in rows if r['split']==split]
                p=predict(np.array([r[feature] for r in part]))
                z=np.array([r['success'][arm] for r in part])
                groups=np.array([r['room'] for r in part])
                table[feature][split]=metrics(z,p,groups)
        result[arm]=table
    DOC.mkdir(exist_ok=True,parents=True)
    bench.save(DOC/'results.json',result)
    bench.save(DOC/'rows.json',rows)
    bench.save(OUT/'FINISHED.json',result)
    print(result,flush=True)


def main():
    torch.set_num_threads(1)
    cases=[c for k in ('protocol.json','protocol_stress.json') for c in bench.read(bench.OUT/k)['cases'] if c['kind']!='dual']
    jobs=[]
    for c in cases:
        for off in bench.OFFSETS:
            base=bench.read(bench.OUT/'runs'/'base'/off/(c['id']+'.json'))
            if base['status']=='executed':
                jobs.append((c,off,base))
    with ProcessPoolExecutor(4) as pool:
        rows=[]
        for i,x in enumerate(pool.map(job,jobs,chunksize=1),1):
            rows.append(x)
            if i%100==0 or i==len(jobs):print(i,len(jobs),flush=True)
    report(rows)


if __name__=='__main__':
    main()
