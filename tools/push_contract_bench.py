"""Frozen-prefix PUSH robustness bench for random-room CAP development logs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import math
from pathlib import Path
import random

import numpy as np

from experiments.option_rl_v19 import unknown_shape as push
from experiments.planar_tasks.relational_search.contract import canonical, digest, validate_action
from experiments.planar_tasks.relational_search.motion import ACTIONS
from experiments.planar_tasks.semantic_rl.candidate_semantics import endpoints, fitted
from experiments.planar_tasks.semantic_rl import push_phases
from experiments.planar_tasks.semantic_rl.push_contact_recovery import ContactRecovery, clearance
from tools.cap_general_actor_availability import world
from tools.cap_regional_five import current_event, read, save
from tools.cap_standoff_diagnostic import SOURCE_DIR

OUT=SOURCE_DIR.parent/'push_contract_20261123_01'
DEV=read(SOURCE_DIR/'development.json')
HELD=read(SOURCE_DIR/'held.json')
ROUND_ONE={'base':None,'near8':(8,'near',.15),'clear8':(8,'clear',.20),
         'clear12':(12,'clear',.20),'clear8_wide':(8,'clear',.30)}
ROUND_TWO={'near5':(5,'near',.15),'near12':(12,'near',.15),'near8_wide':(8,'near',.28),
           'margin8':(8,'margin',.08),'margin12':(12,'margin',.08)}
ROUND_THREE={'short8':(8,'standoff',.08),'short12':(12,'standoff',.10)}
CONFIGS=ROUND_ONE|ROUND_TWO|ROUND_THREE
DRIVE={'drive04':(0,'heading',.04),'drive08':(0,'heading',.08)}
SPLIT={'split':None,'split_near8':(8,'near',.15),'split_near5':(5,'near',.15),'split_feedback':(5,'near',.15)}
OFFSETS={'zero':(0.,0.),'t+':(0.,.015),'t-':(0.,-.015),'n+':(.010,0.),'n-':(-.010,0.)}
CONTACT_M=.02
SIDE_M=.035


def translation(command):
    return [i for i in (0,1) if getattr(ACTIONS[command[f'a{i}']], 'effect_kind', None)=='translate']


def archive(sid,room):
    with gzip.open(SOURCE_DIR/'runs'/sid/(room+'.json.gz'),'rt') as stream:
        return json.load(stream)


def freeze(stress=False):
    cases=[]
    rng=random.Random(20261124 if stress else 20261123)
    rooms=sorted(HELD if stress else DEV)
    rng.shuffle(rooms)
    for room in rooms:
        choices={}
        for sid in (('s025','s027','s045') if not stress else ('s025',)):
            rows=archive(sid,room)
            for index,(last,row) in enumerate(zip(rows,rows[1:]),1):
                cmd=row['a']
                if cmd['target_id']!='entity-a' or cmd['ticks']<24 or row['tick']-last['tick']<10:
                    continue
                movers=translation(cmd)
                if not movers:
                    continue
                i=movers[0]
                past,cur=last['robots'][i],row['robots'][i]
                robot_travel=math.dist(past[:2],cur[:2])
                observed_motion=math.dist(last['xy'],row['xy']) if last['xy'] and row['xy'] else None
                if len(movers)==2:
                    label='dual'
                elif cur[2]=='standoff' and robot_travel<.045:
                    label='standoff'
                elif cur[2]=='push' and observed_motion is not None:
                    label='push_progress' if observed_motion>.06 else 'push_low' if observed_motion<.025 else None
                else:
                    label=None
                if label and label not in choices:
                    choices[label]=dict(room=room,sid=sid,index=index,starts=last['tick'],tick=row['tick'],kind=label,command=cmd)
        for case in choices.values():
            cases.append(case)
    selected=[]
    counts={}
    for c in cases:
        k=c['kind']
        n=counts.get(k,0)
        if n>=(18 if stress else 32):continue
        if stress and k=='dual':continue
        counts[k]=n+1
        c['split']='stress' if stress else 'validation' if rooms.index(c['room'])%3==0 else 'development'
        c['id']=f'{c["kind"]}_{c["room"]}'
        selected.append(c)
    assert len({c['id'] for c in selected})==len(selected)
    OUT.mkdir(exist_ok=True)
    protocol=dict(cases=selected,thresholds=dict(progress_m=CONTACT_M,side_m=SIDE_M,stable_contact_ticks=3),
                  offsets=OFFSETS,configs=ROUND_ONE,study='continue frozen histories into one fixed command, matched perturbed starts')
    path=OUT/('protocol_stress.json' if stress else 'protocol.json')
    if path.exists():assert digest(read(path))==digest(protocol)
    else:save(path,protocol)
    print(counts,flush=True)


def perturb(w, i, delta):
    env=w.env
    backend=env.env.backend
    body=backend.robot_bodies[i]
    body.position=tuple(np.asarray(body.position)+delta)
    backend.space.reindex_shapes_for_body(body)
    arena=env.instance.arena
    r=backend.robot_shapes[i]
    if min(body.position.x,body.position.y,arena[0]-body.position.x,arena[1]-body.position.y)<r.radius:
        return 'perturbation_outside'
    q=backend.space.shape_query(r)
    if any(info.shape is not r and any(p.distance<-.001 for p in info.contact_point_set.points) for info in q):
        return 'perturbation_overlap'
    backend._contact_result=None
    env.lidar.invalidate()
    env._patch_cache.clear()
    history=w.history
    event=dict(current_event(history),observation=env.public())
    state=history.snapshot()
    state['event']=canonical(event)
    state['chain']=digest(dict(version=state['version'],previous=state['previous_chain'],index=state['index'],
                               previous_tick=state['previous_tick'],event=event))
    w.history=history.restore(state)
    return None


def position(env, target):
    b=env.env.backend.bodies[target]
    return [float(b.position.x),float(b.position.y),float(b.angle)]


def geometry(obs, i, target, axis):
    robot=obs['robots'][i]
    own=np.asarray([robot['own']['x'],robot['own']['y']])
    points,cat,labels=endpoints(robot)
    returns=points[labels!=target]
    other=obs['robots'][1-i]['own']
    peer=np.array([other['x'],other['y']])
    box=fitted(obs,target)
    hit=points[labels==target]
    if box is None or not len(hit):
        return dict(visible=False,points=len(hit),contact=bool(robot['contacting_object']),goal_gap=0.,tangent=0.,station_clear=0.,peer=math.dist(own,peer))
    n=np.asarray(axis)
    tangent=np.array([-n[1],n[0]])
    selected=hit[np.argsort(hit@n)[:max(1,len(hit)//3)]]
    q=np.mean(selected,axis=0)
    station=q-.33*n
    return dict(visible=True,points=len(hit),contact=bool(robot['contacting_object']),
                goal_gap=float((q-own)@n),tangent=float(abs((q-own)@tangent)),
                station_clear=clearance(own,station,returns),
                peer=math.dist(own,peer))


def branch(job):
    case,arm,off=job
    key=case['id']
    dest=OUT/'runs'/arm/off/(key+'.json')
    if dest.exists():return
    record=archive(case['sid'],case['room'])
    cmd=case['command']
    movers=translation(cmd)
    robot=movers[0]
    axis=np.array(ACTIONS[cmd[f'a{robot}']].effect_direction)
    tangent=np.array([-axis[1],axis[0]])
    a,b=OFFSETS[off]
    config=(CONFIGS|SPLIT|DRIVE)[arm]
    recovery=ContactRecovery(*config) if config is not None else None
    fixture=(HELD if case['split']=='stress' else DEV)[case['room']]
    with world(fixture) as (w,_):
        for step in record[:case['index']]:
            ev=w.step(step['a'],remaining_ticks=1800-w.env.total_periods)
            assert ev['end_tick']==step['tick'],(case,ev)
        assert w.env.total_periods==case['starts']
        result=dict(id=key,arm=arm,offset=off,prefix=w.history.sha256,robot=robot,dir=axis.tolist())
        if off!='zero':
            reason=perturb(w,robot,a*axis+b*tangent)
            if reason:
                save(dest,dict(result,status=reason))
                return
        observation=current_event(w.history)['observation']
        try:validate_action(cmd,observation)
        except ValueError as e:
            save(dest,dict(result,status='unavailable',reason=str(e)))
            return
        env=w.env
        geo=geometry(observation,robot,cmd['target_id'],axis)
        start=position(env,cmd['target_id'])
        frames=[]
        prev=w.detector.update
        def observe(packet):
            detected=prev(packet)
            backend=env.env.backend
            contacts=backend.contacts()
            bound=env.active_target==cmd['target_id']
            packet_robots=packet['robots']
            hit=[bound and p['contacting_object'] for p in packet_robots]
            info=[p['execution'] for p in packet_robots]
            bots=[[float(v) for v in rb.position] for rb in backend.robot_bodies]
            frames.append(dict(t=env.total_periods,xy=position(env,cmd['target_id']),bots=bots,
                               contact=hit,stage=[p['stage'] for p in info],q=[env._commit_q.get(i) for i in (0,1)],
                               peer=any(c['pair']=='robot_robot' for c in contacts),
                               other=any(c['pair']=='robot_wall' or c['pair']=='robot_object' and c['entity_id']!=cmd['target_id'] for c in contacts)))
            if recovery is not None and detected not in ('contact_lost','blocked','controller_end') and env.total_periods<case['starts']+cmd['ticks'] and env.termination is None:
                recovery.tick(env,packet,cmd)
            return detected
        w.detector.update=observe
        phases=None
        if arm in SPLIT:
            cap=min(cmd['ticks'],1800-env.total_periods)
            touch=push_phases.contact(w,cmd,cap)
            left=cap-touch.ticks
            drive=None
            if touch.status=='ready' and left>0 and not env.termination:
                drive=push_phases.push_from_contact(w,cmd,left,measure_effect=arm=='split_feedback')
            phases=dict(contact=touch.status,contact_ticks=touch.ticks,drive=drive.status if drive else None,
                        drive_ticks=drive.ticks if drive else 0,observed_effect=drive.observed_effect if drive else None)
            total=phases['contact_ticks']+phases['drive_ticks']
            reason=phases['drive'] or phases['contact']
            assert total<=cap and len(frames)==total
        else:
            event=w.step(cmd,remaining_ticks=1800-env.total_periods)
            total=event['periods'];reason=event['reason']
            assert frames and len(frames)==total
            if arm=='base' and off=='zero':assert event['end_tick']==case['tick']
        exact=position(env,cmd['target_id'])
        delta=np.subtract(exact[:2],start[:2])
        contacts=[all(f['contact'][i] for i in movers) for f in frames]
        stable=next((j+1 for j in range(2,len(contacts)) if all(contacts[j-2:j+1])),None)
        along=float(delta@axis)
        lateral=float(delta@tangent)
        effect=stable is not None and along>=CONTACT_M and abs(lateral)<=max(SIDE_M,along*.75)
        save(dest,dict(result,status='executed',reason=reason,ticks=total,geometry=geo,start=start,phases=phases,
                       effect=effect,stable=stable,along=along,lateral=lateral,contacts=sum(contacts),
                       peer=sum(f['peer'] for f in frames),other=sum(f['other'] for f in frames),
                       recovery=[] if recovery is None else recovery.applied,
                       recovery_status=None if recovery is None else recovery.status,frames=frames))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=14)
    p.add_argument('--arms',nargs='+',choices=CONFIGS|SPLIT|DRIVE)
    p.add_argument('--split',choices=('development','validation','stress'))
    p.add_argument('--freeze',action='store_true')
    a=p.parse_args()
    if any(arm in ROUND_TWO for arm in a.arms or ()):
        save(OUT/'round_two.json',ROUND_TWO)
    if any(arm in ROUND_THREE for arm in a.arms or ()):
        save(OUT/'round_three.json',ROUND_THREE)
    if a.freeze:freeze(stress=a.split=='stress')
    protocol=read(OUT/('protocol_stress.json' if a.split=='stress' else 'protocol.json'))
    cases=[c for c in protocol['cases'] if a.split is None or c['split']==a.split]
    jobs=[(c,arm,o) for c in cases for arm in a.arms or ('base',) for o in OFFSETS
          if arm not in (SPLIT|DRIVE) or len(translation(c['command']))==1]
    tasks=[(c,arm,o) for c,arm,o in jobs if not (OUT/'runs'/arm/o/(c['id']+'.json')).exists()]
    with ProcessPoolExecutor(a.workers) as pool:
        for n,_ in enumerate(pool.map(branch,tasks,chunksize=1),1):
            if n%50==0 or n==len(tasks):
                print(n,len(tasks),flush=True)
                save(OUT/'progress.json',dict(arms=a.arms,split=a.split,done=n,total=len(tasks)))


if __name__=='__main__':
    main()
