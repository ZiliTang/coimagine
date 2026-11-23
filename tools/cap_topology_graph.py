"""Paired A/B transition and continuation records at authentic CAP boundaries."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp

import numpy as np

from coimagine.random_start.geometry import point_segment_distance
from experiments.planar_tasks.semantic_rl.corridor_semantics import sweep_bounds
from tools import cap_preparation_pairs as prep

study=prep.study
OUT=study.OUT.parent/'cap_topology_graph_20261123_01'
DOC=study.phase.cap.ROOT/'docs/research/key_findings/2026-11-23_cap_topology_graph'
SEED=662315932
WINDOW=120
PREFIX=(4,24)
LABELS=('station_free','approach_clear','staged','target_contact','rear_contact','peer_blocks','sweep_static','sweep_movable')
ROUTES=('AA','AB','BA','BB')
R=study.ROBOT_RADIUS


def polygon(shape):
    return np.asarray([shape.body.local_to_world(v) for v in shape.get_vertices()])


def sweep_overlaps(poly, delta, obstacle):
    hull,axes=sweep_bounds(poly,delta)
    ed=np.roll(obstacle,-1,axis=0)-obstacle
    normals=np.vstack([axes,ed[:,[1,0]]*[-1.,1.]])
    a,b=hull@normals.T,obstacle@normals.T
    return bool(((a.max(0)>=b.min(0)) & (b.max(0)>=a.min(0))).all())


def straight_blocked(xy, station, obstacles):
    return any(s.segment_query(tuple(xy),tuple(station),R-.01).shape is not None for s in obstacles)


def topology(w, obs):
    """Offline labels from physical geometry; no expert receives this vector."""
    backend=w.env.env.backend
    target=obs['goal']['target_id']
    vertices=np.unique(np.concatenate([polygon(s) for s in backend.shapes[target]]),axis=0)
    assert len(vertices)==4
    centre=vertices.mean(0)
    offsets=vertices-centre
    box=vertices[np.argsort(np.arctan2(offsets[:,1],offsets[:,0]))]
    goal=np.asarray(obs['goal']['xy'])
    axis=goal-centre
    gap=float(np.linalg.norm(axis))
    axis/=max(gap,1e-8)
    robots=[np.asarray(r.position) for r in backend.robot_bodies]
    edges=np.roll(box,-1,axis=0)-box
    normals=edges[:,[1,0]]*[-1.,1.]
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    mid=box+edges*.5
    normals*=np.sign(((centre-mid)*normals).sum(axis=1))[:,None]
    side=int(np.argmax(normals@axis))
    face=box[side]
    edge=edges[side]
    normal=normals[side]
    static=list(backend.static_shapes)
    movable=[shape for label,shapes in backend.shapes.items() if label!=target for shape in shapes]
    obstacles=static+movable
    stationary=approach=staged=peer_veto=False
    for fraction in (.2,.5,.8):
        station=face+fraction*edge-(R+.035)*normal
        if any(s.point_query(tuple(station)).distance<R for s in obstacles):
            continue
        stationary=True
        for i,xy in enumerate(robots):
            # Same-side path avoids crossing the target; the disk sweep also tests walls and loose bodies.
            if (xy-face)@normal > -(R-.025):
                continue
            if straight_blocked(xy,station,obstacles):
                continue
            if point_segment_distance(robots[1-i],xy,station)<2*R:
                peer_veto=True
                continue
            approach=True
            staged |= bool(np.linalg.norm(xy-station)<=.35)
    contacts=[c for c in backend.contacts() if c['pair']=='robot_object' and c.get('entity_id',target)==target]
    rear=any(point_segment_distance(c['point'],face,face+edge)<=.035 and
             (robots[c['robot_id']]-face)@normal < 0 for c in contacts)
    delta=min(.4,gap)*axis
    arena=np.asarray(obs['robots'][0]['arena'])
    swept=np.vstack([box,box+delta])
    blocked_wall=bool((np.minimum(swept,arena-swept)<.05).any())
    static_polys=[polygon(s) for s in static if hasattr(s,'get_vertices')]
    movable_polys=[polygon(s) for s in movable]
    bits=[stationary,approach,staged,bool(contacts),rear,peer_veto and not approach,
          blocked_wall or any(sweep_overlaps(box,delta,p) for p in static_polys),
          any(sweep_overlaps(box,delta,p) for p in movable_polys)]
    drawing=dict(box=box.round(4).tolist(),robots=np.round(robots,4).tolist(),goal=goal.tolist(),
                 static=[p.round(4).tolist() for p in static_polys],movable=[p.round(4).tolist() for p in movable_polys],
                 face=[face.tolist(),(face+edge).tolist()])
    return [int(v) for v in bits],gap,drawing


def live_frame(obs):
    target=obs['goal']['target_id']
    xy=np.asarray([study.position(obs,i) for i in (0,1)])
    entity=next((e for e in obs['entities'] if e['id']==target and e['points'] and e['last_seen_tick']==obs['tick']),None)
    seen=entity is not None
    centre=np.mean(entity['points'],axis=0) if seen else xy.mean(0)
    axis=np.asarray(obs['goal']['xy'])-centre
    gap=np.linalg.norm(axis)
    axis/=max(gap,1e-8)
    rel=xy-centre
    relative=(rel@np.array([axis,axis[[1,0]]*[-1,1]]).T) if seen else np.zeros((2,2))
    touch=[float(obs['active_target']==target and r['contacting_object']) for r in obs['robots']]
    return dict(tick=obs['tick'],xy=xy,seen=seen,gap=gap if seen else 0.,rel=relative.ravel(),touch=touch)


def public_vector(obs, hist):
    now=hist[-1]
    station=prep.public_state(obs,None)
    corr=study.corridor_semantics(obs)
    out=[float(now['seen']),now['gap'],*now['rel'],*now['touch'],np.linalg.norm(now['xy'][0]-now['xy'][1]),
         float(station['distance'] is not None),station['distance'] or 0.,station['clearance'] or 0.]
    out.extend(float(corr[k]) for k in ('target_estimated','static_hit','movable_hit','swept_visibility','unknown'))
    for lag in (1,3):
        then=hist[max(0,len(hist)-1-lag)]
        visible=now['seen'] and then['seen']
        out.extend([int(visible),then['gap']-now['gap'] if visible else 0.,
                    *np.linalg.norm(now['xy']-then['xy'],axis=1),*then['touch']])
    return np.round(out,6).tolist()


def sample(w,obs,hist):
    labels,gap,draw=topology(w,obs)
    return dict(history=w.history.sha256,tick=obs['tick'],x=public_vector(obs,hist),
                y=labels,gap=gap,draw=draw)


def rollout(fixture, pair, origin, depth, route):
    policies=[study.phase.FrozenCAP(study.MODULES[k]) for k in pair]
    counters=Counter()
    hist=deque(maxlen=4)
    states=[]
    n=0
    first=None
    clipped=0
    invalid=0
    trace=[]
    with study.phase.world(fixture) as (w,_):
        def update(history):
            for p in policies:p.update(history)
        update(w.history)
        while not study.ended(w):
            obs=study.observe(w)
            tick=obs['tick']
            if len(states)<3:
                hist.append(live_frame(obs))
            if n==depth:
                first=tick
            if first is not None and len(states)<3 and tick==first+len(states)*WINDOW:
                states.append(sample(w,obs,hist))
            stage=len(states)
            idx=origin if not stage else int(route[0 if stage==1 else 1]=='B')
            command=prep.prior.try_action(policies[idx],obs)
            if command is None:
                invalid+=1
                command=prep.prior.try_action(policies[1-idx],obs) or prep.BRAKE
            if 0<stage<3:
                left=first+stage*WINDOW-tick
                assert left>0
                clipped+=int(command['ticks']>left)
                command=dict(command,ticks=min(command['ticks'],left))
                robots=hist[-1]['xy'].round(3).tolist()
                box=w.env.env.backend.bodies[obs['goal']['target_id']]
                trace.append([tick,stage,*np.round(box.position,3).tolist(),*robots[0],*robots[1]])
            study.tick_action(w,command,prep.prior.EXECUTOR,update,counters)
            if 0<stage<3:
                assert w.env.total_periods<=first+stage*WINDOW
            n+=1
        result=study.output(w,counters)
        goal=fixture['goal']
        finalgap=float(np.linalg.norm(np.asarray(w.env.env.backend.bodies[w.env.primary_id].position)-[goal['x'],goal['y']]))
    return dict(states=states,success=result['success'],end=result['ticks'],endhash=result['history'],
                reason=result['termination'],gap=finalgap,trim=clipped,invalid=invalid,trace=trace)


def run_room(job):
    fold,room,fixture,origin,pair=job
    path=OUT/'runs'/fold/(room+'.json')
    if path.exists():return
    records=[]
    for depth in PREFIX:
        branches={}
        points={}
        for route in ROUTES:
            source=branches.get(route[0]+'A')
            if source and len(source['states'])<2:
                branches[route]=source
                continue
            data=rollout(fixture,pair,origin,depth,route)
            for state in data['states']:
                h=state['history']
                assert h not in points or points[h]==state
                points[h]=state
            data['states']=[v['history'] for v in data['states']]
            branches[route]=data
            if not data['states']:
                branches=dict.fromkeys(ROUTES,data)
                break
        assert len({tuple(v['states'][:1]) for v in branches.values()})==1
        for r in ('A','B'):
            assert branches[r+'A']['states'][:2]==branches[r+'B']['states'][:2]
        records.append(dict(depth=depth,points=points,branches=branches))
    study.save(path,dict(room=room,origin=origin,records=records))


def main():
    old=study.read(prep.prior.OUT/'protocol.json')
    dev=study.read(study.OUT/'development.json')
    dev={k:dev[k] for k in old['rooms']['development']}
    fresh={r['id']:r['fixture'] for r in study.phase.cap.rooms.generate(SEED)['cases']}
    hashes={study.phase.input_hash(f) for f in fresh.values()}
    known=study.collect_identities()
    known.update(study.phase.input_hash(f) for name in ('held','development') for f in study.read(study.OUT/(name+'.json')).values())
    for seed in (662315862,662315879,prep.SEED):
        known.update(study.phase.input_hash(r['fixture']) for r in study.phase.cap.rooms.generate(seed)['cases'])
    assert len(hashes)==len(dev)==200 and not hashes&known
    jobs=[]
    for fold,rooms in [('dev',dev),('fresh',fresh)]:
        keys=sorted(rooms,key=lambda k:hashlib.sha256(('topology/'+k).encode()).digest())
        jobs.extend((fold,k,rooms[k],i%2,old['pair']) for i,k in enumerate(keys))
    protocol=dict(seed=SEED,rooms={fold:{j[1]:study.phase.input_hash(j[2]) for j in jobs if j[0]==fold}
                                   for fold in ('dev','fresh')},pair=old['pair'],depth=PREFIX,window=WINDOW,
                  labels=LABELS,k=6,executor=prep.prior.EXECUTOR,selection='largest dev paired edge per source; >=20 rooms, abs difference >=10pp',
                  switch='A prefix 120 ticks then B if dev node AB minus AA terminal advantage positive, minimum 20 distinct rooms. Else A.')
    path=OUT/'protocol.json'
    if path.exists():
        assert study.phase.cap.digest(protocol)==study.phase.cap.digest(study.read(path))
    else:study.save(path,protocol)
    study.initialize()
    with ProcessPoolExecutor(18,mp_context=mp.get_context('fork')) as pool:
        tasks=[pool.submit(run_room,j) for j in jobs]
        for i,f in enumerate(as_completed(tasks),1):
            f.result()
            if i%10==0 or i==1:
                study.save(OUT/'progress.json',dict(done=i,total=len(jobs)))
                print(i,flush=True)


if __name__=='__main__':
    main()
