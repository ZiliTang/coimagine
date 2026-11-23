"""Localize public proposal divergence and drift under geometrically matched CAP commands."""
from collections import Counter
import math
import numpy as np
from experiments.planar_tasks.relational_search.contract import validate_action
from tools import cap_room_symmetry_check as s

e=s.e
OUT=s.OUT/'mechanism'


def vector(w):
    backend=w.env.env.backend
    bodies=[*backend.robot_bodies,*(backend.bodies[k] for k in sorted(backend.bodies))]
    return np.array([[b.position.x,b.position.y,w.env.heading[i] if i<2 else b.angle,
                      b.velocity.x,b.velocity.y,w.env.heading_rate[i] if i<2 else b.angular_velocity]
                      for i,b in enumerate(bodies)])


def deviation(before,w,name):
    sx,sy=s.TRANSFORMS[name]
    now=vector(w)
    target=before.copy()
    if sx<0:target[:,0]=6-target[:,0]
    if sy<0:target[:,1]=4.8-target[:,1]
    target[:,2]=np.arctan2(sy*np.sin(before[:,2]),sx*np.cos(before[:,2]))
    target[:,3:]*=[sx,sy,sx*sy]
    delta=now-target
    angle=np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2]))
    return dict(m=float(np.linalg.norm(delta[:,:2],axis=1).max()),deg=float(np.degrees(np.abs(angle)).max()),
                speed=float(np.linalg.norm(delta[:,3:5],axis=1).max()),rate=float(np.abs(delta[:,5]).max()))


def perception(obs):
    rows=[]
    for r in obs['robots']:
        scan=r['scan']
        rows.append((np.array(scan['ray_angles']),np.array([0. if x is None else x for x in scan['distances']]),
                     np.array(list(zip(scan['category'],scan['entity_ids'],scan['valid'])),dtype=object)))
    return rows


def senses(old,new,name):
    sx,sy=s.TRANSFORMS[name]
    rays=0
    max_gap=0.
    max_angle=0.
    for (a,d,c),(b,f,g) in zip(old,perception(new)):
        target=np.arctan2(sy*np.sin(a),sx*np.cos(a))
        diff=target[:,None]-b[None,:]
        angle=np.abs(np.arctan2(np.sin(diff),np.cos(diff)))
        lookup=angle.argmin(1)
        rays+=int(np.any(c!=g[lookup],axis=1).sum())
        max_gap=max(max_gap,float(np.abs(d-f[lookup]).max()))
        max_angle=max(max_angle,float(np.degrees(angle.min(1).max())))
    return dict(rays=rays,range_error=max_gap,angle_error=max_angle)


def original(f,mod):
    cap=e.s.phase.FrozenCAP(mod)
    trace=[]
    with e.s.phase.world(f) as (w,_):
        cap.update(w.history)
        counts=Counter()
        while not e.s.ended(w):
            o=e.s.observe(w);a=cap.propose(o)
            trace.append(dict(tick=o['tick'],state=vector(w),sensors=perception(o),action=a))
            e.s.tick_action(w,a,e.g.prep.prior.EXECUTOR,cap.update,counts)
        return trace,int(w.env.success)


def replay(f,mod,trace,name):
    cap=e.s.phase.FrozenCAP(mod)
    first=None
    drift=None
    peak=dict(m=0.,deg=0.,speed=0.,rate=0.)
    strict=None
    status='original_commands_exhausted'
    first_sensor=None
    with e.s.phase.world(s.transform(f,name)) as (w,_):
        cap.update(w.history)
        counts=Counter()
        for i,step in enumerate(trace):
            if e.s.ended(w):
                status='terminated';break
            obs=e.s.observe(w)
            actual=cap.propose(obs)
            mapped=s.command(step['action'],name)
            diff=deviation(step['state'],w,name)
            peak={k:max(diff[k],v) for k,v in peak.items()}
            timing=obs['tick']!=step['tick']
            if strict is None and (timing or diff['m']>1e-6 or diff['deg']>1e-4):
                strict=i
            if drift is None and (timing or diff['m']>.001 or diff['deg']>.1):
                drift=i
            if first is None:
                sens=senses(step['sensors'],obs,name)
                if first_sensor is None and (sens['rays'] or sens['range_error']>.001 or sens['angle_error']>.1):
                    first_sensor=i
                if s.command(actual,'original')!=mapped:
                    first=dict(i=i,tick=obs['tick'],ref_tick=step['tick'],delta=diff,peak=peak.copy(),drift_before_or_now=drift is not None,
                               strict_before_or_now=strict is not None,sensor_before_or_now=first_sensor is not None,
                               sensors=sens,original=step['action'],mapped=mapped,actual=actual)
            try:
                validate_action(mapped,obs)
            except ValueError:
                status='unavailable';break
            e.s.tick_action(w,mapped,e.g.prep.prior.EXECUTOR,cap.update,counts)
        if e.s.ended(w):
            status='terminated'
        return dict(first=first,first_drift=drift,first_strict=strict,first_sensor=first_sensor,steps=len(trace),status=status,
                    replay_success=int(w.env.success) if status=='terminated' else None)


def worker(job):
    k,f=job
    path=OUT/'rooms'/(k+'.json')
    if path.exists():return
    natural=e.read(s.OUT/'rooms'/(k+'.json'))
    result=[]
    for arm,mod in [('A',e.AM),('B',e.BM)]:
        ref=natural['original']['outcomes'][arm]
        flipped=[name for name in s.TRANSFORMS if natural[name]['outcomes'][arm]!=ref]
        if not flipped:continue
        trace,original_success=original(f,mod)
        assert original_success==ref
        for name in flipped:
            v=replay(f,mod,trace,name)
            result.append(dict(room=k,arm=arm,name=name,original_success=ref,**v))
    e.save(path,result)


if __name__=='__main__':
    s.t.d.init(profile=e.perception.LEGACY)
    keys=e.read(s.OUT/'summary.json')['parents']
    cases=s.t.p.dev()
    e.pool(worker,[(k,cases[k]) for k in keys],5)


def first_example():
    """Feed the unmodified CAP centre estimator an exact reflection of its real input."""
    s.t.d.init(profile=e.perception.LEGACY)
    k='1_dev_3_13'
    f=s.t.p.dev()[k]
    mod=e.AM
    with e.s.phase.world(f) as (w,_):
        o=e.s.observe(w)
        cap=e.s.phase.FrozenCAP(mod)
        cap.update(w.history)
        c,faces,pts=mod.geometry(o,cap.memory['c'])
    matrix=np.diag([1.,-1.]);offset=np.array([0.,4.8])
    mv=lambda p:(matrix@np.asarray(p)).tolist()
    point=lambda p:(matrix@np.asarray(p)+offset).tolist()
    mirrored=[]
    for normal,tangent,dist,lo,hi in sum(faces,[]):
        n=mv(normal)
        tan=[-n[1],n[0]]
        mirrored.append((tuple(n),tuple(tan),dist+np.dot(offset,n),
                         -hi+np.dot(offset,tan),-lo+np.dot(offset,tan)))
    mirrored_points=[point(q) for q in pts]
    inner=mod.compact_center(mirrored_points,mirrored,None)
    synthetic=mod.compact_center(mirrored_points,mirrored,inner)
    expected=[*point(c[:2]),*mv(c[2:4]),*c[4:]]
    # Pure orthogonal reflection about coordinate zero isolates the translation term.
    zero_faces=[(n,t,d-np.dot(offset,n),lo-np.dot(offset,t),hi-np.dot(offset,t))
                for n,t,d,lo,hi in mirrored]
    zero_points=[mv(q) for q in pts]
    zero_prior=mod.compact_center(zero_points,zero_faces,None)
    zero=mod.compact_center(zero_points,zero_faces,zero_prior)
    zero_error=math.dist(zero[:2],mv(c[:2]))
    with e.s.phase.world(s.transform(f,'mirror_y')) as (w,_):
        o=e.s.observe(w)
        cap=e.s.phase.FrozenCAP(mod)
        cap.update(w.history)
        actual,fs,actual_pts=mod.geometry(o,cap.memory['c'])
        first=cap.propose(o)
        def approach(center):
            h=mod.route_goal(center,o['goal']['xy'],o,actual_pts)
            goals=mod.waypoints(o,cap.memory,center,h)
            return [mod.move(r,g,44,[],False) for r,g in zip(o['robots'],goals)]
        actual_pair=approach(actual)
        corrected=approach(expected)
        assert actual_pair==[first['a0'],first['a1']]
    return dict(room=k,actual=actual[:2],expected=expected[:2],synthetic=synthetic[:2],
                actual_error_m=math.dist(actual[:2],expected[:2]),exact_input_error_m=math.dist(synthetic[:2],expected[:2]),
                zero_origin_error_m=zero_error,actual_pair=actual_pair,only_center_corrected=corrected)
