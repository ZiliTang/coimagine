"""Frozen development-to-validation test of a public scan gate at contact PUSH."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random

from experiments.planar_tasks.authoring_v2.option_effect_model import EffectModel, PUBLIC_SCALES, PUBLIC_FLAGS
from tools import cap_guarded_wall_pair as bench
from tools.cap_guarded_wall_pair import forks, loop, push_phases

ROOT = forks.OUT.parent / 'cap_guarded_wall_validation_v1'
DOSES = (.10, .40)
ARMS = ('dose_only', 'nine_inputs', 'wall_scan')
SIZES = (24, 72)


def seed_plan():
    plan = dict(doses=DOSES, thresholds=[.2,.8], max_neighbours=5, min_neighbours=3,
                safe='forward >= .02m and no object/static contact on any PUSH or subsequent tick and new-face ready within 120 ticks',
                fit_sizes=SIZES, source=loop.sha(Path(__file__)), parent=loop.sha(Path(bench.__file__)), collector=forks.frozen_hash(),
                partitions={})
    for split, seed, size in [('development', 416703, 72), ('validation', 982161, 96)]:
        rng=random.Random(seed)
        plan['partitions'][split]=[dict(id=f'{split}-{i:03}',gap=rng.uniform(.065,.575),
            normal_offset=rng.uniform(20,70), lateral=rng.uniform(-.15,.15), width=rng.uniform(.65,.86),
            dx=rng.uniform(-.02,.02) if i%2 else 0.,dy=rng.uniform(-.02,.02) if i%2 else 0.)
            for i in range(size)]
    # Convert tuples to the same JSON representation used on a resumed run.
    return json.loads(json.dumps(plan))


@lru_cache(maxsize=1)
def scene_geometry():
    from experiments.planar_tasks.relational_search.motion import LEGACY_VERSION
    anchor=bench.reference()
    record=forks.record_for('dev',bench.ROOM)
    fixture=forks.panels()['dev'][bench.ROOM]
    legacy = dict(fixture, overrides={**fixture['overrides'], 'push_face_contract': LEGACY_VERSION})
    with forks.archive.S.phase.world(legacy) as (world,_):
        for row in record['steps'][:anchor['step_index']]:
            forks.old.restore_step(world,row)
        backend=world.env.env.backend
        vertices=[tuple(s.body.local_to_world(v)) for s in backend.shapes[world.env.primary_id] for v in s.get_vertices()]
    return record, anchor, fixture, vertices


@lru_cache(maxsize=1)
def setup():
    record,anchor,fixture,vertices=scene_geometry()
    old_model=EffectModel.from_dict(loop.read(forks.OUT/'models/v2.json')['model'])
    return record,anchor,fixture,vertices,old_model,loop.donor_index(loop.packets('dev'))


def room(spec):
    record,anchor,fixture,vertices=scene_geometry()
    fixture=deepcopy(fixture)
    fixture['robots'][0]['x']+=spec['dx']
    fixture['robots'][0]['y']+=spec['dy']
    _,axis=push_phases.pusher(anchor['original_action'])
    phi=math.atan2(axis[1],axis[0])+math.radians(spec['normal_offset'])
    nx,ny=math.cos(phi),math.sin(phi)
    plane=max(x*nx+y*ny for x,y in vertices)+spec['gap']
    x,y,_=record['steps'][anchor['step_index']]['truth']['object']
    offset=plane+.04-x*nx-y*ny
    fixture['room']['obstacles'].append(dict(angle=phi,cx=x+nx*offset-ny*spec['lateral'],
        cy=y+ny*offset+nx*spec['lateral'],half_w=.04,half_h=spec['width']))
    return fixture,plane,phi


def scan_descriptor(obs,action):
    """Fit short visible static segments. Distances are to the visible robot-facing box footprint."""
    target=action['target_id']
    points=next((e['points'] for e in obs['entities'] if e['id']==target),[])
    _,axis=push_phases.pusher(action)
    candidates=[]
    if not points:
        return None
    centre=[sum(p[k] for p in points)/len(points) for k in range(2)]
    for r in obs['robots']:
        scan=r['scan']
        hits=[]
        for ok,category,angle,d in zip(scan['valid'],scan['category'],scan['ray_angles'],scan['distances']):
            ox,oy=scan['origin']
            hits.append((ox+d*math.cos(angle),oy+d*math.sin(angle)) if ok and category==2 else None)
        # Three neighbouring returns must support the same approximately straight surface.
        for i,a in enumerate(hits):
            b,c=hits[(i+1)%len(hits)],hits[(i+2)%len(hits)]
            if not (a and b and c):
                continue
            length=math.dist(a,c)
            if not .04<=length<=.85:
                continue
            tx,ty=(c[0]-a[0])/length,(c[1]-a[1])/length
            nx,ny=-ty,tx
            if abs((b[0]-a[0])*nx+(b[1]-a[1])*ny)>.008:
                continue
            if nx*(a[0]-centre[0])+ny*(a[1]-centre[1])<0:
                nx,ny=-nx,-ny
            alignment=nx*axis[0]+ny*axis[1]
            along=(centre[0]-a[0])*tx+(centre[1]-a[1])*ty
            gap=min((a[0]-x)*nx+(a[1]-y)*ny for x,y in points)
            if alignment>.25 and -.80<=along<=length+.80 and -.04<=gap<=1.20:
                candidates.append((gap,math.atan2(axis[0]*ny-axis[1]*nx,alignment)))
    return list(min(candidates)) if candidates else None


def risk_decision(train, features, wall, dose, arm):
    peers=[]
    for row in train:
        if row['dose']!=dose:
            continue
        other=row['features']
        if arm=='dose_only':
            distance=0.
        else:
            if any(features[k]!=other[k] for k in PUBLIC_FLAGS):
                continue
            distance=max(abs(features[k]-other[k])/scale for k,scale in PUBLIC_SCALES.items())
            if arm=='wall_scan':
                if wall is None or row['wall'] is None:
                    continue
                distance=max(distance,abs(wall[0]-row['wall'][0])/.13,
                             abs(wall[1]-row['wall'][1])/math.radians(25))
        if distance<=1:
            peers.append((distance,row['safe']))
    if len(peers)<3:
        return 'defer'
    radius=sorted(d for d,_ in peers)[min(len(peers),5)-1]
    labels=[y for d,y in peers if d<=radius+1e-10]
    good=sum(labels)/len(labels)
    return 'admit' if good>=.8 else ('reject' if good<=.2 else 'defer')


def collect(job):
    spec,split,models=job
    record,anchor,_,_,old,particles=setup()
    fixture,plane,phi=room(spec)
    identifier=spec['id']
    path=ROOT/split/(identifier+'.json')
    if path.exists():
        return loop.read(path)
    packet=dict(id=identifier,split=split,fixture_hash=forks.archive.content_hash(fixture),branches=[])
    with forks.archive.S.phase.world(fixture) as (world,_):
        for row in record['steps'][:anchor['step_index']]:
            for seg in row['segments']:
                obs=forks.archive.S.observe(world)
                if world.env.termination or not forks.legal(seg['requested'],obs):
                    packet['status']='preparation_unavailable'
                    loop.seal(path,packet)
                    return packet
                world.step(seg['requested'],remaining_ticks=seg['end_tick']-seg['start_tick'])
        obs=forks.archive.S.observe(world)
        action=anchor['original_action']
        robot,axis=push_phases.pusher(action)
        if not forks.legal(action,obs) or push_phases.contact_state(world,action,robot,axis)!='ready':
            packet['status']='no_ready_contact'
            loop.seal(path,packet)
            return packet
        features=forks.features(obs,action)
        wall=scan_descriptor(obs,action)
        packet.update(public_history=world.history.sha256,features=features,wall=wall)
        predictions={dose:{f'{arm}-{n}':risk_decision(model,features,wall,dose,arm)
            for n,model in models.items() for arm in ARMS} for dose in DOSES}
    for dose in DOSES:
        outcome=bench.branch(fixture,plane,phi,record,anchor,dose,old,particles)
        if outcome['public_history']!=packet['public_history']:
            raise ValueError('prefix changed')
        if outcome['status']!='measured':
            packet['status']=outcome['status']
            break
        safe=outcome['useful'] and outcome['collision_free_through_contact_check']
        decisions=predictions[dose]
        prior=outcome.get('decision',{})
        decisions['old_frozen']=('admit' if prior.get('admitted') else
            'reject' if prior.get('model_decision')=='reject' or prior.get('source')=='fallback' else 'defer')
        packet['branches'].append(dict(dose=dose,safe=safe,outcome=outcome,decisions=decisions))
    else:
        packet['status']='measured'
    loop.seal(path,packet)
    return packet


def main(workers):
    plan=seed_plan()
    loop.seal(ROOT/'plan.json',plan)
    models={}
    hashes=set()
    for split,specs in plan['partitions'].items():
        with ProcessPoolExecutor(workers) as pool:
            packets=list(pool.map(collect,[(s,split,models) for s in specs]))
        ids={p['fixture_hash'] for p in packets}
        if len(ids)!=len(specs) or ids & hashes:
            raise ValueError('duplicate fixture')
        hashes.update(ids)
        print(split,Counter(p['status'] for p in packets),flush=True)
        if split=='development':
            for n in SIZES:
                models[n]=[dict(id=p['id'],features=p['features'],wall=p['wall'],dose=b['dose'],safe=b['safe'])
                    for p in packets[:n] if p['status']=='measured' for b in p['branches']]
                loop.seal(ROOT/f'model-{n}.json',models[n])
    print('complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    main(parser.parse_args().workers)
