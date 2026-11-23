"""Collect causal boundary observations, privileged fork state and observed route futures."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import math
from pathlib import Path

import numpy as np

from tools import cap_route_horizon as parent

study = parent.study
OUT = study.OUT.parent/'cap_route_information_20261123_01'
DOC = study.DOC.parent/'2026-11-23_cap_route_information'
HORIZONS = (10,30)
SEEDS = (17,29,43)
NEW_SEED = 662315862


def plan():
    old = study.read(parent.OUT/'protocol.json')
    db = {k:study.read(study.OUT/(k+'.json')) for k in parent.FOLDS}
    db = {k:{r:db[k][r] for r in old['rooms'][k]} for k in db}
    new = {'info_'+r['id']:r['fixture'] for r in study.phase.cap.rooms.generate(NEW_SEED)['cases']}
    denied = set()
    for p in (study.OUT/'held.json', study.OUT/'development.json', parent.OUT/'confirmation/panel.json',
              study.OUT/'selective_semantics/held.json', study.OUT/'initial_confirmation/held.json'):
        denied.update(map(study.phase.input_hash,study.read(p).values()))
    denied.update(x['content_sha256'] for x in study.read(study.phase.cap.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json'))
    identities = set(map(study.phase.input_hash,new.values()))
    assert len(identities)==200 and not denied&identities
    db['fresh']=new
    spec=dict(pair=old['pair'],executor=parent.EXECUTOR,labels='full B success minus full A success from the first legal divergence',
              panels={k:study.phase.cap.digest(v) for k,v in db.items()},room_counts={k:len(v) for k,v in db.items()},
              windows=list(HORIZONS),primary=10,seeds=list(SEEDS),model='shared frame MLP 32 and GRU 32; 30 epochs, AdamW .001, batch 24; paired MSE; average 3 seeds, switch when score > 0',
              matched='both routes alive after H commands; same training and test rooms for all information arms; separate public deployment fits all divergent development rooms')
    path=OUT/'protocol.json'
    if path.exists():assert study.read(path)==spec
    else:study.save(path,spec)
    return db,old['pair']


def catcode(value,n):
    x=np.zeros(n)
    if value is not None:
        x[int.from_bytes(hashlib.blake2s(str(value).encode(),digest_size=4).digest(),'little')%n]=1.
    return x


def target_code(label):
    return [float(label==k) for k in (None,'entity-a','entity-b','entity-c','entity-d','entity-e')]


def action_vector(action,obs):
    if not action:return np.zeros(35)
    compact=study.old.dq.action_encoding(action,obs['goal']['target_id'],study.reference_angle(obs))[:-40]
    return np.r_[compact,target_code(action.get('target_id'))]


def public_frame(event):
    obs=event['observation']
    parts=[study.old.base.geom(obs),np.array(obs['goal']['xy'])/6,
           [obs['tick']/1800,obs['decision']/300],target_code(obs['active_target']),action_vector(event['action'],obs)]
    for robot in obs['robots']:
        own=robot['own']
        parts.extend(([own['x']/6,own['y']/4.8,math.sin(own['angle']),math.cos(own['angle']),own['vx'],own['vy'],own['omega']],
            [float(robot[k]) for k in ('contacting_object','contacting_static','contacting_teammate')],
            [float(robot['contact_point_robot'] is not None),*(robot['contact_point_robot'] or (0,0))]))
        scan=robot['scan']
        categories=np.asarray(scan['category'])
        labels=np.asarray(scan['entity_ids'],dtype=object)
        distances=np.asarray(scan['distances'],dtype=float)
        valid=np.asarray(scan['valid'],dtype=bool)
        rays=[np.where(valid,distances,0)/6,valid,*[categories==i for i in range(4)],
              *[labels==k for k in ('entity-a','entity-b','entity-c','entity-d','entity-e')]]
        parts.append(np.stack(rays,axis=1).ravel())
        ex=robot['execution']
        for name in ('stage','exit_reason','pause_reason'):
            parts.append(catcode(ex[name],16))
        for name in ('desired_velocity','reference_velocity','force'):
            parts.append(np.asarray(ex[name] or [0.,0.])/(15 if name=='force' else 1))
        parts.append([float(ex[k] or False) for k in ('done','direction_valid','direction_unavailable')])
    x=np.concatenate(parts).astype(np.float32)
    assert np.isfinite(x).all()
    return x


def physical(w,fixture):
    backend=w.env.env.backend
    goal=np.asarray(fixture['goal']['x']),np.asarray(fixture['goal']['y'])
    parts=[]
    sizes={o['label']:(o['width'],o['height']) for o in fixture['movable_objects']}
    for label in ('entity-a','entity-b','entity-c','entity-d','entity-e'):
        b=backend.bodies.get(label)
        if b is None:
            parts.extend([0.]*10)
        else:
            width,height=sizes.get(label,(.25,1.))
            parts.extend([1.,(b.position.x-goal[0])/6,(b.position.y-goal[1])/4.8,math.cos(b.angle),math.sin(b.angle),
                          b.velocity.x,b.velocity.y,b.angular_velocity,width,height])
    for i in range(3):
        boxes=fixture['room']['obstacles']
        if i>=len(boxes):
            parts.extend([0.]*7)
        else:
            o=boxes[i]
            parts.extend([1.,(o['cx']-goal[0])/6,(o['cy']-goal[1])/4.8,math.cos(o['angle']),math.sin(o['angle']),o['half_w'],o['half_h']])
    return np.asarray(parts,dtype=np.float32)


def route(fixture,pair,who,fresh):
    policies=[study.phase.FrozenCAP(study.MODULES[s]) for s in pair]
    prefix, future, ends = [], [], {}
    origin=None
    command_count=0
    keep=True
    with study.phase.world(fixture) as (w,_):
        def update(history):
            for p in policies:p.update(history)
            if keep:
                seq=prefix if origin is None else future
                seq.append(public_frame(study.phase.cap.current_event(history)))
        update(w.history)
        while not study.ended(w):
            obs=study.observe(w)
            cmd=policies[who].propose(obs)
            if origin is None:
                other=parent.try_action(policies[1-who],obs)
                if other is not None and cmd!=other:
                    origin=w.history.sha256
                    cmds=(cmd,other) if who==0 else (other,cmd)
                    actions=np.concatenate([action_vector(a,obs) for a in cmds]).astype(np.float32)
                    raw_state=physical(w,fixture)
            if origin is not None and command_count in HORIZONS:
                ends[command_count]=len(future)
                if command_count==max(HORIZONS):
                    keep=False
                    if not fresh:break
            study.tick_action(w,cmd,parent.EXECUTOR,update,Counter())
            if origin is not None:command_count+=1
        result=study.output(w,Counter())
    return dict(prefix=prefix,future=future,ends=ends,origin=origin,actions=actions if origin else None,
                raw=raw_state if origin else None,success=result['success'],history=result['history'])


def record(item,pair,fold):
    room,fixture=item
    path=OUT/'samples'/fold/(room+'.npz')
    if path.exists():return
    fresh=fold=='fresh'
    a,b=[route(fixture,pair,i,fresh) for i in (0,1)]
    assert a['origin']==b['origin'] and np.array_equal(a['prefix'],b['prefix'])
    if fresh:
        labels=[a['success'],b['success']]
    else:
        orig=study.read(parent.OUT/'runs'/fold/(room+'.json'))
        fork=orig['baseline']['first']
        assert a['origin']==(fork['history'] if fork else None)
        labels=[orig['baseline']['success'],orig['horizons']['0']['success']]
        for r,l in zip((a,b),labels):
            if max(HORIZONS) not in r['ends']:assert r['success']==l
    values=dict(labels=labels,valid=int(bool(a['origin'])),prefix=np.stack(a['prefix']),origin=a['origin'] or '',
                histories=[a['history'],b['history']])
    if a['origin']:
        width=values['prefix'].shape[1]
        values.update(actions=a['actions'],truth=a['raw'],A=np.asarray(a['future']).reshape(-1,width),
            B=np.asarray(b['future']).reshape(-1,width),ends=np.asarray([[r['ends'].get(h,-1) for h in HORIZONS] for r in (a,b)]))
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**values)
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folds',nargs='*',default=list(parent.FOLDS))
    parser.add_argument('--workers',type=int,default=14)
    args=parser.parse_args()
    db,pair=plan()
    study.initialize()
    with ProcessPoolExecutor(args.workers) as pool:
        for fold in args.folds:
            for i,_ in enumerate(pool.map(partial(record,pair=pair,fold=fold),db[fold].items()),1):
                if i%50==0:print(fold,i,flush=True)


if __name__=='__main__':
    main()
