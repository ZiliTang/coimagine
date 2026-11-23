"""Development-only lidar sampling and CAP centre diagnostics. No deployment changes."""
from collections import Counter
from dataclasses import asdict,replace
from math import atan2,cos,sin,pi,hypot
from time import process_time
import sys
import numpy as np

from coimagine.random_start import spec
from coimagine.random_start.lidar_service import RayFrame
from coimagine.random_start.lidar import scan_patches
from tools import cap_room_symmetry_check as s

e=s.e
OUT=s.t.OUT.parent/'cap_lidar_density_20261123_01'
DOC=s.t.DOC.parent/'2026-11-23_lidar_density'
MODES=('baseline','dense','stagger','rolling')


def subsample(frame,parity):
    scan=frame.scan
    sl=slice(parity,None,2)
    return RayFrame(replace(scan,ray_angles=scan.ray_angles[sl],valid=scan.valid[sl],distances=scan.distances[sl],
                            category=scan.category[sl]),frame.entity_ids[sl])


def acquire(env,i,t,time,rays,phase=0):
    backend=env.env.backend
    x,y,h=env._legal_robot_states()[i][:3]
    frame=env.lidar.physical(backend.space,backend.robot_bodies,backend.labels,i,t,time,(x,y),h+phase,
                             max_range=spec.LIDAR_RANGE,rays=rays,native=True)
    if phase:
        return RayFrame(replace(frame.scan,heading=h),frame.entity_ids)
    return frame


def merge(now,old,parity):
    """Retain current measurements, fill only complementary bins with old measured hits."""
    scan=now.scan
    step=2*pi/144
    angles=[scan.heading+j*step for j in range(144)]
    valid=[False]*144
    distances=[0.]*144
    category=[0]*144
    ids=[None]*144
    for j,angle in enumerate(scan.ray_angles):
        k=2*j+parity
        angles[k],valid[k],distances[k],category[k],ids[k]=angle,scan.valid[j],scan.distances[j],scan.category[j],now.entity_ids[j]
    errors=[float('inf')]*144
    if old is not None:
        raw=old.scan
        for j,(a,dist,hit) in enumerate(zip(raw.ray_angles,raw.distances,raw.valid)):
            if not hit:continue
            dx=raw.origin[0]+dist*cos(a)-scan.origin[0]
            dy=raw.origin[1]+dist*sin(a)-scan.origin[1]
            r=hypot(dx,dy)
            if r>spec.LIDAR_RANGE:continue
            ang=atan2(dy,dx)
            frac=(ang-scan.heading)/step
            k=round(frac)
            err=abs(frac-k)
            k%=144
            if k%2==parity or err>=errors[k]:continue
            angles[k]=ang
            valid[k],distances[k],category[k],ids[k]=True,r,raw.category[j],old.entity_ids[j]
            errors[k]=err
    return RayFrame(replace(scan,ray_angles=tuple(angles),distances=tuple(distances),
                            valid=tuple(valid),category=tuple(category)),tuple(ids))


def patches(env,frame,target):
    view=env.lidar.target_view(frame,target)
    return tuple(scan_patches(view,view.origin))


def local_center(mod,pts,fs,prior):
    """Evaluate the identical frozen estimator about the centroid of observed hits."""
    if not pts:return prior
    o=np.mean(pts,axis=0)
    xy=[(x-o[0],y-o[1]) for x,y in pts]
    local=[]
    for n,t,d,l,h in fs:
        local.append((n,t,d-np.dot(n,o),l-np.dot(t,o),h-np.dot(t,o)))
    prev=None if prior is None else [prior[0]-o[0],prior[1]-o[1],*prior[2:]]
    v=mod.compact_center(xy,local,prev)
    return [v[0]+o[0],v[1]+o[1],*v[2:]]


def fixed(f):
    res={}
    with e.s.phase.world(f) as (w,_):
        obs=e.s.observe(w);env=w.env
        target=obs['goal']['target_id']
        pts=e.AM.get_points(obs,target)
        orig=[env.ray_frame(i,0,0.) for i in range(2)]
        hi=[acquire(env,i,0,0.,144) for i in range(2)]
        for i in range(2):
            assert subsample(hi[i],0)==orig[i]
        for mode in ('72','144'):
            frames=orig if mode=='72' else hi
            fs=[];npts=0
            for frame in frames:
                hits=patches(env,frame,target)
                rec={'patches':[asdict(p) for p in hits]}
                fs.extend(e.AM.visible_faces(rec))
                npts+=sum(label==target for label in frame.entity_ids)
            rec=dict(rays=npts,faces=len(fs))
            for coord in ('world','local'):
                fn=e.AM.compact_center if coord=='world' else lambda points,faces,last: local_center(e.AM,points,faces,last)
                if pts:
                    c=fn(pts,fs,None)
                    c=fn(pts,fs,c)
                    rec[coord]=list(map(float,c[:2]))
                else:rec[coord]=None
            res[mode]=rec
    return res


class ControlProbe:
    def __init__(self,w,mode,diagnose=False):
        self.env=env=w.env
        self.mode=mode
        self.history={}
        self.capture=[]
        self.diagnose=diagnose
        self.in_control=False
        self.calls=0
        self.diag_frames={}
        self.original_observe=env.observe
        native_step=env.joint_step
        def step(a,b,record=False):
            self.in_control=True
            try:return native_step(a,b,record)
            finally:self.in_control=False
        env.joint_step=step
        env.observe=self.observe
        if mode=='dense':
            env.ray_frame=lambda i,t,time,**kw: env.lidar.memo(('audit_public72',i,t,time),lambda:subsample(acquire(env,i,t,time,144),0))

    def staggered(self,i,t,time):
        if i not in self.history or self.history[i][0]!=t:
            previous=self.history.get(i)
            frame=acquire(self.env,i,t,time,72,(t%2)*pi/72)
            self.history[i]=(t,frame,previous[1] if previous and previous[0]+1==t else None)
        return self.history[i][1:]

    def observe(self,i,t,time,contacts=None):
        base=self.original_observe(i,t,time,contacts)
        if not self.in_control:return base
        self.calls+=1
        env=self.env;target=env.active_target
        if self.diagnose:
            self.sample(i,t,time,base)
        if self.mode=='baseline':return base
        if self.mode=='dense':
            frame=acquire(env,i,t,time,144)
        else:
            now,old=self.staggered(i,t,time)
            frame=merge(now,old,t%2) if self.mode=='rolling' and old is not None else now
        ps=patches(env,frame,target)
        patch=min((p for p in ps if p.valid),key=lambda p:p.distance,default=None)
        return replace(base,patches=ps,patch=patch)

    def sample(self,i,t,time,base):
        # Sample two actual consecutive ticks. Mirror and all sensor alternatives see the same physics within a run.
        if t%35 not in (0,1):return
        dense=acquire(self.env,i,t,time,144)
        now=subsample(dense,t%2)
        if t%35==0:
            body=self.env.env.backend.bodies[self.env.active_target]
            self.diag_frames[i]=(t,now,self.env.active_target,tuple(body.position),float(body.angle))
            return
        prev=self.diag_frames.get(i)
        if not prev or prev[0]!=t-1 or prev[2]!=self.env.active_target:return
        target=self.env.active_target
        if target!=self.env.primary_id:return
        body=self.env.env.backend.bodies[target]
        angle=float(body.angle)
        moving=hypot(body.position.x-prev[3][0],body.position.y-prev[3][1])>.001 or abs(angle-prev[4])>pi/360
        obs=dict(t=t,i=i,moving=moving)
        for mode,frame in [('baseline',subsample(dense,0)),('dense',dense),('stagger',now),('rolling',merge(now,prev[1],t%2))]:
            ps=[p for p in patches(self.env,frame,target) if p.valid]
            # Ground-truth rectangle normals and supporting lines are audit labels only.
            def metrics(p):
                a=atan2(p.normal_world[1],p.normal_world[0])
                ix=round((a-angle)/(pi/2))
                ax=angle+ix*pi/2
                center=np.asarray(body.position)
                true_n=np.array([cos(ax),sin(ax)])
                width=.125 if ix%2==0 else .5
                residuals=np.abs((np.asarray(p.points_world)-center)@true_n+width)
                return abs((a-ax)*180/pi),float(np.mean(residuals))
            best=min(ps,key=lambda p:p.distance) if ps else None
            obs[mode]=metrics(best) if best else None
        self.capture.append(obs)


def run(f,mode,diagnose=False):
    cap=e.s.phase.FrozenCAP(e.AM)
    clock=process_time()
    with e.s.phase.world(f) as (w,_):
        probe=ControlProbe(w,mode,diagnose)
        cap.update(w.history);counts=Counter()
        while not e.s.ended(w):
            obs=e.s.observe(w)
            cmd=cap.propose(obs)
            e.s.tick_action(w,cmd,e.g.prep.prior.EXECUTOR,cap.update,counts)
        assert probe.calls>0
        return dict(success=int(w.env.success),ticks=w.env.total_periods,seconds=process_time()-clock,
                    samples=probe.capture)


def worker(job):
    k,original=job
    path=OUT/'rooms'/(k+'.json')
    if path.exists():return
    v={}
    for kind in ('original','mirror_y','jitter'):
        f=s.transform(original,'mirror_y' if kind=='mirror_y' else 'original')
        if kind=='jitter':
            for robot in f['robots']:robot['angle']+=pi/720
        v[kind]=dict(fixed=fixed(f),truth=[f['object']['x'],f['object']['y']])
        if kind=='jitter':continue
        for mode in MODES:
            row=run(f,mode,diagnose=(kind=='original' and mode=='baseline'))
            if mode=='baseline':
                expected=e.read(s.OUT/'rooms'/(k+'.json'))[kind]['outcomes']['A']
                assert row['success']==expected
            v[kind][mode]=row
    e.save(path,v)


def verify_snapshots():
    rooms = s.t.p.dev()
    ids = e.read(s.OUT/'summary.json')['parents']
    checks = Counter()
    cpu = Counter()
    worst = 0.
    for key in ids:
        fixture = rooms[key]
        for name in ('original', 'mirror_y'):
            with e.s.phase.world(s.transform(fixture, name)) as (w, _):
                env = w.env
                expected = env.public()
                result = e.read(OUT/'rooms'/(key+'.json'))[name]
                native, _, _ = e.AM.geometry(expected, None)
                native, _, _ = e.AM.geometry(expected, native)
                assert np.allclose(native[:2], result['fixed']['72']['world'], rtol=0, atol=1e-12)
                for body_shape in env.env.backend.bodies[env.primary_id].shapes:
                    assert all(abs(abs(p.x)-.125)<1e-12 and abs(abs(p.y)-.5)<1e-12 for p in body_shape.get_vertices())
                for i in range(2):
                    high = acquire(env, i, 0, 0., 144)
                    low = env.ray_frame(i,0,0.)
                    assert low == subsample(high, 0)
                    odd = acquire(env,i,1,.05,72,pi/72)
                    merged = merge(odd,low,1)
                    assert merged.entity_ids == high.entity_ids and merged.scan.valid == high.scan.valid
                    worst = max(worst, max(abs(a-b) for a,b in zip(merged.scan.distances,high.scan.distances)))
                    checks['static_sensors'] += 1
                    for n in (72,144):
                        start = process_time()
                        for _ in range(12):
                            frame = acquire(env,i,0,0.,n)
                            patches(env,frame,env.primary_id)
                        cpu[n] += process_time()-start
                for mode in MODES:
                    # Fresh installation per arm to avoid nesting probe wrappers.
                    with e.s.phase.world(s.transform(fixture,name)) as (v, _):
                        probe = ControlProbe(v,mode)
                        probe.in_control = True
                        probe.observe(0,0,0.)
                        probe.in_control = False
                        assert v.env.public() == expected
                        checks['public_packets'] += 1
    audit=dict(checks=dict(checks), static_max_range_error_m=worst,
               native_cast_and_fit_us={n:t*1e6/(12*checks['static_sensors']) for n,t in cpu.items()})
    e.save(OUT/'audit.json',audit)
    print(audit,flush=True)


if __name__=='__main__':
    s.t.d.init(profile=e.perception.LEGACY)
    rooms=s.t.p.dev()
    ids=e.read(s.OUT/'summary.json')['parents']
    if 'verify' in sys.argv:
        verify_snapshots()
    elif 'smoke' in sys.argv:
        f=rooms['1_dev_3_13'];print(fixed(f),flush=True)
        for mode in MODES:
            v=run(f,mode,mode=='baseline'); print(mode,v['success'],v['ticks'],len(v['samples']),v['seconds'],flush=True)
    else:
        e.pool(worker,[(k,rooms[k]) for k in ids],16)
