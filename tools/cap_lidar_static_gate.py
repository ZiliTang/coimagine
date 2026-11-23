"""Paired CAP A closed loops with an observation-only static gate and local centre."""
from collections import Counter, deque
from dataclasses import replace
from inspect import getsource
from math import atan2, cos, degrees, dist, pi, sin
from pathlib import Path
import csv
import hashlib
import sys

import numpy as np
from experiments.planar_tasks.relational_search.motion import ACTIONS
from tools import cap_lidar_density_probe as prior

e, s = prior.e, prior.s
OUT = prior.OUT.parent / 'cap_lidar_gate_20261123_01'
DOC = prior.DOC.parent / '2026-11-23_lidar_gate'
CENTERS = ('world', 'local')
SCANS = ('fixed', 'stagger', 'gated')


def compact_center(points, faces, previous):
    if not points:
        return previous
    origin = (sum(p[0] for p in points)/len(points), sum(p[1] for p in points)/len(points))
    shifted = [(p[0]-origin[0], p[1]-origin[1]) for p in points]
    planes = [(n,t,d-dot(n,origin),lo-dot(t,origin),hi-dot(t,origin)) for n,t,d,lo,hi in faces]
    last = None if previous is None else [previous[0]-origin[0],previous[1]-origin[1],*previous[2:]]
    center = fit_relative(shifted,planes,last)
    return [center[0]+origin[0],center[1]+origin[1],*center[2:]]


def is_quiet(frames, target):
    # Same-phase comparison over 100 ms; both intervening 50 ms steps must be quiet.
    if len(frames) != 3 or not target:
        return False
    first, middle, last = frames
    for a,b in ((first,middle),(middle,last)):
        if b[0] != a[0]+1 or a[1] != target or b[1] != target or a[2] or b[2]:
            return False
        scan,other = a[3].scan,b[3].scan
        if dist(scan.origin,other.origin) > .0025 or abs(np.arctan2(sin(scan.heading-other.heading),cos(scan.heading-other.heading))) > pi/900:
            return False
    a,b = first[3],last[3]
    hits = 0
    for i,(left,right) in enumerate(zip(a.entity_ids,b.entity_ids)):
        if (left==target) != (right==target):
            return False
        if left == target:
            if abs(a.scan.distances[i]-b.scan.distances[i]) > .003:
                return False
            hits += 1
    return hits >= 3


def face_good(patches, pose):
    face = min((p for p in patches if p.valid),key=lambda p:p.distance,default=None)
    if face is None:
        return False
    x,y,angle = pose
    normal = atan2(face.normal_world[1],face.normal_world[0])
    axis = round((normal-angle)/(pi/2))
    direction = angle+axis*pi/2
    width = .125 if axis%2==0 else .5
    offset = sum(abs((px-x)*cos(direction)+(py-y)*sin(direction)+width)
                 for px,py in face.points_world)/len(face.points_world)
    return abs(degrees(normal-direction)) <= 5 and offset <= .01


class GatedProbe(prior.ControlProbe):
    def __init__(self, world):
        super().__init__(world,'stagger')
        self.recent = [deque(maxlen=3),deque(maxlen=3)]
        self.counts = Counter()
        self.audit_pose = [None,None]
        inner = self.env.joint_step

        def step(a,b,record=False):
            self.push = any(ACTIONS[k].kind=='push' for k in (a,b))
            return inner(a,b,record)
        self.env.joint_step = step

    def observe(self,i,t,time,contacts=None):
        base = self.original_observe(i,t,time,contacts)
        if not self.in_control:
            return base
        self.calls += 1
        target = self.env.active_target
        frame,old = self.staggered(i,t,time)
        history = self.recent[i]
        assert not history or history[-1][0]<t
        history.append((t,target,self.push or base.contacting_object,frame))
        fuse = is_quiet(history,target)
        assert not (fuse and self.push)
        current = prior.patches(self.env,frame,target)
        chosen = prior.patches(self.env,prior.merge(frame,old,t%2),target) if fuse else current
        self.counts['push' if self.push else 'other'] += 1
        if fuse:
            self.counts['fused'] += 1
        # Audit truth after choosing the scan. No pose or audit counter enters the gate.
        if target == self.env.primary_id:
            body = self.env.env.backend.bodies[target]
            pose = (*body.position,body.angle)
            previous = self.audit_pose[i]
            if fuse and previous is not None and previous[0] == t-1:
                still = dist(previous[1][:2],pose[:2])<=.001 and abs(pose[2]-previous[1][2])<=pi/360
                prefix = 'quiet' if still else 'moving'
                self.counts[prefix] += 1
                before,after = face_good(current,pose),face_good(chosen,pose)
                self.counts[prefix+'_current'] += before
                self.counts[prefix+'_fusion'] += after
            self.audit_pose[i] = (t,pose)
        else:
            self.audit_pose[i] = None
        patch = min((p for p in chosen if p.valid),key=lambda p:p.distance,default=None)
        return replace(base,patches=chosen,patch=patch)


def episode(fixture,module,scan):
    cap = e.s.phase.FrozenCAP(module)
    counts = Counter()
    with e.s.phase.world(fixture) as (w,_):
        probe = GatedProbe(w) if scan=='gated' else prior.ControlProbe(w,'baseline' if scan=='fixed' else 'stagger')
        cap.update(w.history)
        while not e.s.ended(w):
            cmd = cap.propose(e.s.observe(w))
            e.s.tick_action(w,cmd,e.g.prep.prior.EXECUTOR,cap.update,counts)
        assert probe.calls
        return dict(success=int(w.env.success), ticks=w.env.total_periods,history=w.history.sha256,
                    gate=dict(probe.counts) if scan=='gated' else {})


def worker(job):
    panel,key,fixture = job
    path = OUT/panel/(key+'.json')
    result = {}
    for center in CENTERS:
        for scan in SCANS:
            row = episode(fixture,MODULES[center],scan)
            if center=='world' and scan!='gated':
                if panel!='natural':
                    ref = e.read(prior.OUT/'rooms'/(key+'.json'))[panel]['baseline' if scan=='fixed' else 'stagger']
                    assert (row['success'],row['ticks'])==(ref['success'],ref['ticks'])
                elif scan=='fixed':
                    assert row['success']==e.read(s.t.OUT/'confirmation'/(key+'.json'))['A']
            result[center+'_'+scan] = row
    e.save(path,result)


def initialize():
    global MODULES
    s.t.d.init(profile=e.perception.LEGACY)
    original=(e.WORK/'basis.py').read_text()
    assert e.base.digest(original)==e.A
    signature='def compact_center(points,faces,previous):'
    assert original.count(signature)==1
    local = original.replace(signature,getsource(compact_center)+'\n'+signature.replace('compact_center','fit_relative'))
    MODULES={'world':e.AM,'local':e.s.old.dq.load_native(local)[0]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'local_cap.py').write_text(local)
    cases=s.t.p.dev()
    parents=e.read(s.OUT/'summary.json')['parents']
    jobs=[(variant,k,s.transform(cases[k],variant)) for k in parents for variant in ('original','mirror_y')]
    natural=s.t.d.rooms(s.t.FRESH)
    keys=np.random.default_rng(4117).choice(sorted(natural),64,replace=False).tolist()
    jobs.extend(('natural',k,natural[k]) for k in keys)
    ids={panel+':'+key:e.s.phase.input_hash(f) for panel,key,f in jobs}
    assert len(set(ids.values()))==128
    for k in parents:
        with e.s.phase.world(cases[k]) as (w,_):
            obs=e.s.observe(w)
            first,_,_=MODULES['local'].geometry(obs,None)
            second,_,_=MODULES['local'].geometry(obs,first)
            ref=e.read(prior.OUT/'rooms'/(k+'.json'))['original']['fixed']['72']['local']
            assert np.allclose(second[:2],ref,rtol=0,atol=1e-9)
    protocol=dict(ids=ids,old=e.A,local=e.base.digest(local),scans=SCANS,centers=CENTERS,
                  implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    e.save(OUT/'protocol.json',protocol)
    return jobs


if __name__=='__main__':
    jobs=initialize()
    e.pool(worker,jobs,16)
