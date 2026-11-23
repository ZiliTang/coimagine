"""Paired geometry and visible-face angles at the first policy departure and first rejection."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch

from tools import cap_general_actor_availability as audit

ref=audit.ref
OUT=audit.OUT/'geometry'


def inspect(env,action):
    obs=ref.dq.current_event(env.history)['observation']
    angles=[]
    for robot in range(2):
        spec=ref.ACTIONS[action[f'a{robot}']]
        angle=None
        if action['target_id'] is not None and getattr(spec,'effect_kind',None)=='translate' and spec.kind=='push':
            sensor=env.env._sensor(robot,obs['tick'],obs['tick']*audit.CONTROL_DT,action['target_id'])
            faces=audit.visible_faces(sensor)
            if faces:
                cosine=max(float(np.dot(f[1],spec.effect_direction)) for f in faces)
                angle=float(np.degrees(np.arccos(np.clip(cosine,-1,1))))
        angles.append(angle)
    xy,robots=ref.position(env)
    return dict(h=env.history.sha256,tick=obs['tick'],object_xy=xy,robot_xy=robots,angles=angles)


@torch.inference_mode()
def measure(job):
    fixture,r,old=job
    arm=r['actor']
    steps={'first':r['first']['step'],'illegal':r['rejection']['step']}
    reference=[]
    result={}
    for mode in ('original','perturbed'):
        with audit.world(fixture) as (env,lib):
            record={}
            for i in range(steps['illegal']+1):
                p=lib.panel(env.history)
                if mode=='original':
                    a=ref.choose(ref.MODELS[arm],p)
                    reference.append(a)
                else:
                    a=reference[i]
                for key,j in steps.items():
                    if i==j:
                        marker=r['first' if key=='first' else 'rejection']
                        assert a==marker['action']
                        z=inspect(env,a)
                        z['pool']=p[4]
                        z['raw']=p[5]
                        if mode=='perturbed':
                            assert z['h']==marker['history']
                        record[key]=z
                if i==steps['illegal']:
                    break
                if i==r['cut'] and mode=='perturbed':
                    assert env.history.sha256==old['prefix']
                    a=dict(a,ticks=a['ticks']+old['micro'][arm]['requested_delta'])
                audit.advance(env,a)
            result[mode]=record
    return dict(room=old['room'],arm=arm,**result)


def main():
    torch.set_num_threads(1)
    _,rooms,ref.SIDS=ref.scale.inputs()
    ref.MODELS=[ref.sets.checkpoint(ref.scale.OUT),ref.fit.load_best()]
    ref.dq.MODULES=[ref.dq.load_native((ref.campaign.CAM/'semantics'/(s+'.py')).read_text())[0] for s in ref.SIDS]
    jobs=[]
    for p in sorted((audit.OUT/'rooms').glob('*.json')):
        for row in ref.read(p)['branches']:
            if row['primary']:
                jobs.append((rooms[p.stem],row,ref.read(ref.OUT/'rooms'/p.name)))
    assert len(jobs)==71
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
        for f in as_completed([pool.submit(measure,j) for j in jobs]):
            v=f.result()
            ref.save(OUT/(v['room']+'_'+str(v['arm'])+'.json'),v)


if __name__=='__main__':
    main()
