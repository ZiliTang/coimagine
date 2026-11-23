"""Independent expert terminal value and single-command sensitivity on optimized PUSH."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import math
import os
import numpy as np

from tools import cap_route_information as info

study=info.study
OUT=study.OUT.parent/'cap_terminal_value_20261123_01'
DOC=study.DOC.parent/'2026-11-23_cap_terminal_value'
EX='split_precision'
FRESH_SEED=662315879
FRACTIONS=(0.,.25,.5,.75,.9,1.)


def panels():
    libraries=study.read(study.OUT/'protocol.json')['libraries']
    experts={}
    for bank,ms in libraries.items():
        for m in ms:
            experts['basis' if m['id']=='basis' else bank+'_'+m['id']]=m['sha']
    dev=study.read(study.OUT/'development.json')
    fresh={'v_'+r['id']:r['fixture'] for r in study.phase.cap.rooms.generate(FRESH_SEED)['cases']}
    denied=set()
    for f in (study.OUT/'development.json',study.OUT/'held.json',info.parent.OUT/'confirmation/panel.json',
              study.OUT/'initial_confirmation/held.json',study.OUT/'selective_semantics/held.json'):
        denied.update(map(study.phase.input_hash,study.read(f).values()))
    denied.update(study.phase.input_hash(r['fixture']) for r in study.phase.cap.rooms.generate(info.NEW_SEED)['cases'])
    denied.update(v['content_sha256'] for v in study.read(study.phase.cap.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json'))
    hashes=set(map(study.phase.input_hash,fresh.values()))
    assert len(hashes)==len(fresh)==200 and not hashes&denied
    # Select by room identifier and movable count before any evaluation outcomes.
    subset=[]
    for n in range(1,5):
        group=[k for k,v in fresh.items() if len(v['movable_objects'])==n]
        subset+=sorted(group,key=lambda k:hashlib.sha256(('local/'+k).encode()).digest())[:14]
    protocol=dict(experts=experts,dev=study.phase.cap.digest(dev),test=study.phase.cap.digest(fresh),seed=FRESH_SEED,
                  rooms=[len(dev),len(fresh)],branch_rooms=sorted(subset),executor=EX,
                  target='negative true terminal object-center to goal-center distance in meters; binary task success independently supervised',
                  coordinates='last available pre-command boundary on retrospective fractions of actual ticks, plus fixed real tick landmarks among still active episodes',
                  branch='first single translating target PUSH with nominal duration >=12 at or after tick 300, one tick shorter, all later proposals by same CAP',
                  regression='per-CAP independent 64x64 MLP; separate distance and success models; 3 seeds; 2000 steps uniform episodes then uniform command boundaries; no bootstrap')
    path=OUT/'protocol.json'
    if path.exists():assert study.read(path)==protocol
    else:study.save(path,protocol)
    return experts,dev,fresh,subset


def snapshot(w,goal_id):
    be=w.env.env.backend
    obj=be.bodies[goal_id]
    return [*obj.position,obj.angle,*be.robot_bodies[0].position,*be.robot_bodies[1].position]


def view(obs,cmd,history,counts):
    g=study.old.base.geom(obs)
    tick=obs['tick']
    if g[3]:
        distance=g[0]
        history['last']=distance
        history['last_tick']=tick
        history['best']=min(history.get('best',distance),distance)
    gap=history.get('last',0.)
    if 'start' not in history:history['start']=gap
    past=history.setdefault('past',[])
    lag=[gap-past[max(0,len(past)-k)] if past else 0. for k in (1,3,10)]
    past.append(gap)
    robots=[]
    for r in obs['robots']:
        own=r['own']
        robots.extend([own['x']/6,own['y']/4.8,np.sin(own['angle']),np.cos(own['angle']),own['vx'],own['vy'],
                       int(r['contacting_object']),int(r['contacting_static']),int(r['contacting_teammate'])])
    return np.asarray([gap,g[3],(tick-history.get('last_tick',0))/1800,history.get('best',0.),history['start'],
                       *lag,tick/1800,obs['decision']/300,*g[1:],
                       *[counts[k]/len(past) for k in ('contact_ready','contact_contact_timeout','retimed')],
                       *robots,*info.action_vector(cmd,obs)],np.float32)


def path_for(panel,expert,room):
    return OUT/'runs'/panel/expert/(room+'.npz')


def trial(job):
    panel,expert,sid,room,fixture,branch=job
    dest=path_for(panel,expert,room)
    if dest.exists():return
    policy=study.phase.FrozenCAP(study.MODULES[sid])
    fs=[]; ticks=[]; geo_history={}; counts=Counter()
    cut=-1; origin=''; before=[]; after=[]; post_tick=-1
    with study.phase.world(fixture) as (w,_):
        policy.update(w.history)
        while not study.ended(w):
            obs=study.observe(w)
            cmd=policy.propose(obs)
            is_cut=cut<0 and obs['tick']>=300 and cmd['ticks']>=12 and cmd['target_id']==obs['goal']['target_id'] and len(study.translating(cmd))==1
            if is_cut:
                cut=len(fs);origin=w.history.sha256
                before=snapshot(w,obs['goal']['target_id'])
                if branch:cmd=dict(cmd,ticks=cmd['ticks']-1)
            fs.append(view(obs,cmd,geo_history,counts));ticks.append(obs['tick'])
            study.tick_action(w,cmd,EX,policy.update,counts)
            if is_cut:
                after=snapshot(w,obs['goal']['target_id'])
                post_tick=w.env.total_periods
        res=study.output(w,counts)
        obs=study.observe(w)
        obj=w.env.env.backend.bodies[obs['goal']['target_id']]
        distance=math.dist(obj.position,obs['goal']['xy'])
    assert res['termination'] in ('success','task_budget_exhausted','decision_budget') or res['decisions']>=300
    if panel=='development':
        orig=info.parent.label(panel,sid,room)
        assert all(res[k]==orig[k] for k in ('history','success','ticks','decisions'))
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp,x=np.stack(fs),t=ticks,d=distance,y=res['success'],end=res['ticks'],reason=str(res['termination']),
                        final_hash=res['history'],cut=cut,origin=origin,before=before,after=after,post_tick=post_tick)
    tmp.replace(dest)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('panels',nargs='+',choices=('development','fresh','branch'))
    p.add_argument('--workers',type=int,default=18)
    a=p.parse_args()
    experts,dev,fresh,subset=panels()
    study.initialize()
    with ProcessPoolExecutor(a.workers) as pool:
        for panel in a.panels:
            db=dev if panel=='development' else fresh
            jobs=[(panel,e,s,k,db[k],panel=='branch') for e,s in experts.items() for k in (subset if panel=='branch' else db)
                  if not path_for(panel,e,k).exists()]
            for i,_ in enumerate(pool.map(trial,jobs,chunksize=1),1):
                if i%200==0 or i==len(jobs):print(panel,i,len(jobs),flush=True)


if __name__=='__main__':main()
