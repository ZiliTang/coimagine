"""Test the actual proposed PUSH direction at a subset of paired graph vertices."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import hashlib
import multiprocessing as mp

import numpy as np

from coimagine.random_start.geometry import point_segment_distance
from experiments.option_rl_v19.unknown_shape import direction_matches
from tools import cap_topology_graph as g


def physical(w,obs,cmd,robot):
    backend=w.env.env.backend
    target=cmd['target_id']
    xy=np.asarray(backend.robot_bodies[robot].position)
    axis=np.asarray(g.study.ACTIONS[cmd[f'a{robot}']].effect_direction)
    verts=np.unique(np.concatenate([g.polygon(s) for s in backend.shapes[target]]),axis=0)
    centre=verts.mean(0)
    v=verts-centre
    poly=verts[np.argsort(np.arctan2(v[:,1],v[:,0]))]
    compatible=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        n=(b-a)[[1,0]]*[-1,1]
        if n@(centre-(a+b)/2)<0:n=-n
        n/=np.linalg.norm(n)
        if direction_matches(n,axis):compatible.append((a,b,n))
    contacts=[c for c in backend.contacts() if c['pair']=='robot_object' and c.get('entity_id',target)==target and c['robot_id']==robot]
    rear_contact=any(point_segment_distance(c['point'],a,b)<=.035 and (xy-a)@n<0 for c in contacts for a,b,n in compatible)
    arena=np.asarray(obs['robots'][0]['arena'])
    delta=.4*axis
    swept=np.vstack([poly,poly+delta])
    occupied=bool((np.minimum(swept,arena-swept)<.05).any())
    obstacles=[s for s in backend.static_shapes if hasattr(s,'get_vertices')]
    obstacles.extend(s for k,ss in backend.shapes.items() if k!=target for s in ss)
    blocked=occupied or any(g.sweep_overlaps(poly,delta,g.polygon(s)) for s in obstacles)
    return dict(compatible=bool(compatible),touch=bool(contacts),support=rear_contact,clear=not blocked),axis,centre


def branch(fixture,origin,depth,preparer,expert,expected,ids):
    ps=[g.study.phase.FrozenCAP(g.study.MODULES[s]) for s in ids]
    counts=Counter()
    with g.study.phase.world(fixture) as (w,_):
        def update(h):
            for p in ps:p.update(h)
        def act(i,obs):
            return g.prep.prior.try_action(ps[i],obs) or g.prep.prior.try_action(ps[1-i],obs) or g.prep.BRAKE
        update(w.history)
        for _ in range(depth):
            g.study.tick_action(w,act(origin,g.study.observe(w)),g.prep.prior.EXECUTOR,update,counts)
        if preparer is not None:
            end=w.env.total_periods+g.WINDOW
            while w.env.total_periods<end:
                cmd=act(preparer,g.study.observe(w))
                cmd=dict(cmd,ticks=min(end-w.env.total_periods,cmd['ticks']))
                g.study.tick_action(w,cmd,g.prep.prior.EXECUTOR,update,counts)
        assert w.history.sha256==expected
        obs=g.study.observe(w)
        cmd=g.prep.prior.try_action(ps[expert],obs)
        if not cmd or cmd['target_id']!=obs['goal']['target_id']:return None
        movers=g.study.translating(cmd)
        if len(movers)!=1:return None
        flags,axis,centre=physical(w,obs,cmd,movers[0])
        t=w.env.total_periods
        counts.clear()
        g.study.tick_action(w,cmd,g.prep.prior.EXECUTOR,update,counts)
        delta=np.asarray(w.env.env.backend.bodies[cmd['target_id']].position)-centre
        along=float(delta@axis)
        lateral=abs(float(delta@np.array([-axis[1],axis[0]])))
        return dict(**flags,action=cmd,tick=t,executed=w.env.total_periods-t,along=along,lateral=lateral,
                    effect=bool(along>=.02 and lateral<=max(.035,.75*along)),ready=bool(counts['contact_ready']))


def run(job):
    fold,key,fixture,ids=job
    room=g.study.read(g.OUT/'runs'/fold/(key+'.json'))
    items=[]
    seen=set()
    for rec in room['records']:
        for n,prep,branch_id in ((0,None,'AA'),(1,0,'AA'),(1,1,'BA')):
            states=rec['branches'][branch_id]['states']
            if len(states)<=n:continue
            h=states[n]
            for i in range(2):
                if (h,i) in seen:continue
                seen.add((h,i))
                row=branch(fixture,room['origin'],rec['depth'],prep,i,h,ids)
                if row:items.append(dict(history=h,expert='AB'[i],**row))
    return dict(room=key,attempts=len(seen),pushes=items)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--fold',choices=['dev','fresh'],required=True)
    cli.add_argument('--workers',type=int,default=16)
    opt=cli.parse_args()
    plan=g.study.read(g.OUT/'protocol.json')
    fixtures={'dev':g.study.read(g.study.OUT/'development.json'),
              'fresh':{r['id']:r['fixture'] for r in g.study.phase.cap.rooms.generate(g.SEED)['cases']}}
    g.study.initialize()
    jobs=[]
    for fold,keys in [(opt.fold,plan['rooms'][opt.fold])]:
        selected=sorted(keys,key=lambda k:hashlib.sha256(('option-precondition/'+k).encode()).digest())[:60]
        jobs.extend((fold,k,fixtures[fold][k],plan['pair']) for k in selected)
    with ProcessPoolExecutor(opt.workers,mp_context=mp.get_context('fork')) as pool:
        rows=list(pool.map(run,jobs))
    g.study.save(g.OUT/('pushes_'+opt.fold+'.json'),rows)


if __name__=='__main__':
    main()
