"""Five-round library versus single-program CAP experiment, authoring_v2 host."""
import argparse
import ast
import contextlib
import fcntl
import gzip
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from experiments.planar_tasks.authoring_v2 import random_room_author as host
from experiments.planar_tasks.authoring_v2 import random_room_tasks as rooms
from experiments.planar_tasks.authoring_v2 import release
from experiments.planar_tasks.authoring_v2.cap_family_augmentation import variant
from experiments.planar_tasks.relational_search.contract import public_observation, digest
from experiments.planar_tasks.semantic_rl.memory_flow import current_event
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime as fast
from experiments.planar_tasks.semantic_rl.typed_compiler import stream_config
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.environment import EventConfig
from experiments.planar_tasks.semantic_rl.runner import rollout

ROOT=host.ROOT
OUT=ROOT/'outputs/semantic_rl/cap_regional_five_20261119_01'
WS=ROOT.parent/'coimagine_author_workspaces/regional5_v1'
ARMS=('regular','regional','single')
read,save=host.read,host.save
CFG=EventConfig(mode='duration')


def geom(o):
    goal=np.array(o['goal']['xy'])
    obs={e['id']:e['points'] for e in o['entities'] if e['points']}
    pts=np.array(obs.get(o['goal']['target_id'],[]))
    bots=np.array([[r['own']['x'],r['own']['y']] for r in o['robots']])
    center=pts.mean(0) if len(pts) else bots.mean(0)
    delta=goal-center
    dist=np.linalg.norm(delta)
    u=delta/max(1e-8,dist)
    side=np.array([-u[1],u[0]])
    other=np.array([p for eid,cloud in obs.items() if eid!=o['goal']['target_id'] for p in cloud])
    clearance=2.
    if len(other):
        rel=other-center
        clearance=min(clearance,np.linalg.norm(rel-np.clip(rel@u,0,dist)[:,None]*u,axis=1).min())
    return np.array([dist,clearance,len(obs)-int(len(pts)>0),float(len(pts)>0),*[(b-center)@k for b in bots for k in (u,side)],
                     min(center[0],6-center[0],center[1],4.8-center[1])])


def public(f):
    return public_observation(host.X.RecordedEnv(*host.X.fixture(f)).public())


def setup():
    # RoomCampaign is the random-room v2 profile, with its own current runtime/source seal.
    import hashlib
    materials=read(release.LOCK)
    for name,expected in materials['files'].items():
        assert hashlib.sha256((release.HERE/name).read_bytes()).hexdigest()==expected
    guide=release.knowledge.GUIDE
    assert hashlib.sha256(guide.read_bytes()).hexdigest()==read(release.knowledge.LOCK)['files'][guide.name]
    if (OUT/'protocol.json').exists():raise FileExistsError('fresh study directory required')
    OUT.mkdir(parents=True,exist_ok=True)
    dev=read(WS/'regular/development_fixtures.json')
    bundle=ROOT/'outputs/transfers/cap_trace_value_dev_v1'
    used={r['content_sha256'] for r in read(bundle/'used_scene_identities.json')}
    used.update(read(OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
    for pat in ('*/panel.json','*/evaluation_panel.json','*/batches/*_plan.json','cap_semantic_disagreement*/*_panel.json'):
        for p in OUT.parent.glob(pat):
            rows=read(p)
            for r in (rows if isinstance(rows,list) else rows.get('cases',[])):
                if isinstance(r,dict) and 'content_sha256' in r:used.add(r['content_sha256'])
    panels={}
    for r in rooms.generate(2026111997)['cases']:
        panels['fresh_'+r['id']]=r['fixture']
    for j,(key,f) in enumerate(sorted(dev.items())):
        panels['near_'+key]=variant(dict(case_id=key,fixture=f),2197000+j)
    for f in panels.values():
        key=digest({k:v for k,v in f.items() if k!='scene_id'})
        assert key not in used
        used.add(key)
    ids=sorted(dev)
    x=np.stack([geom(public(dev[i])) for i in ids])
    save(OUT/'regions.json',dict(ids=ids,x=x.tolist()))
    save(OUT/'held.json',panels)
    save(OUT/'protocol.json',dict(model='gpt-6.1-sol',reasoning='high',arms=list(ARMS),dev=160,fresh=200,near=160,seed=17,
         rounds=5,probe_per_round=32,attempts=960,ticks=1728000,author_wall_seconds=4*3600,dev_sha=digest(dev),held_sha=digest(panels),
         region='nearest 24 by standardized public geometry; choose dense neighborhood with most union failures; prefer distance to previous centers then compactness',
         single='one continuous author; 160 flexibly allocated probes, at most 5 full feedback sweeps; deploy highest dev count, earliest tie',
         eval='fixed dev-best member, commit by public geometry k=15, per-library union, frozen single; near-parent and natural panels separately, no online training',
         semantics='typed-stream-v1, isolated native code; 1800 ticks/300 decisions; prior source from same arm freely available; token metering unavailable'))
    for arm in ARMS:
        p=OUT/arm
        if not p.exists():
            c=host.RoomCampaign.create(p,dev)
            with c.locked():
                c.state['deadline']=time.time()+4*3600
                c.state['limits'].update(attempts=960,resets=960,execution_ticks=1728000,feature_batches=300000)
                c.state['author_protocol'].update(max_library_policies=1 if arm=='single' else 5,wall_seconds=4*3600)
                c._save()
        save(p/'study.json',dict(arm=arm,probes=0,probes_this_round=0,sweeps=[],active=None,final=None))
    for arm in ARMS:
        p=WS/arm/'lab'
        p.write_text(f'#!/bin/bash\ncd {ROOT} || exit 1\nexport PYTHONPATH={ROOT} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nexec {ROOT}/outputs/transfers/venv_py311/bin/python -m tools.cap_regional_five {arm} "$@"\n')
        p.chmod(0o755)


def region(s):
    d=read(OUT/'regions.json'); ids=d['ids']; x=np.array(d['x']); z=(x-x.mean(0))/np.maximum(.1,x.std(0))
    dist=np.linalg.norm(z[:,None]-z[None,:],axis=2)
    nn=np.argsort(dist,axis=1,kind='stable')[:,:24]
    radius=dist[np.arange(len(ids)),nn[:,-1]]
    covered=set().union(*(r['wins'] for r in s['sweeps']))
    prior=[ids.index(r['center']) for r in s['sweeps']]
    candidates=np.flatnonzero(radius<=np.median(radius))
    def score(i):
        return (sum(ids[j] not in covered for j in nn[i]),min(dist[i,prior]) if prior else 0.,-radius[i],-i)
    center=max(candidates,key=score)
    return ids[center],[ids[j] for j in nn[center]]


def snapshot(arm,s):
    r={k:v for k,v in s.items() if k!='sweeps'}
    r['sweeps']=[dict(sid=v['sid'],wins=len(v['wins']),added=v['added'],center=v['center']) for v in s['sweeps']]
    ids=read(OUT/'regions.json')['ids']
    r['unresolved']=sorted(set(ids)-set().union(*(v['wins'] for v in s['sweeps'])))
    if arm=='regional' and len(s['sweeps'])<5:
        r['center'],r['focus']=region(s)
    return r


class RecordingEnv(fast.ScopedEnvironment):
    def step(self,requested,remaining_ticks=None):
        event=super().step(requested,remaining_ticks)
        o=current_event(self.history)['observation']
        target=next((e['points'] for e in o['entities'] if e['id']==o['goal']['target_id']),[])
        trace=tuple([round(sum(p[i] for p in target)/len(target),3) for i in range(2)]) if target else None
        self.rows.append(dict(a=requested,tick=event['end_tick'],reason=event['reason'],xy=trace,
              author_pre_state=dict(zip(FEATURES.worker.protocol['state_names'],FEATURES.worker.context['state'])),goal=o['goal']['xy'],
              robots=[[round(r['own'][k],3) for k in ('x','y')]+[r['execution']['stage'],r['execution']['exit_reason']] for r in o['robots']]))
        return event


FEATURES=None
SOURCE=None


def start_worker(source):
    global FEATURES,SOURCE
    torch.set_num_threads(1)
    SOURCE=source
    FEATURES=Features(source,runtime='typed_stream',fast_embedding=True,action_protocol=protocol(CFG))


def execute(job):
    cid,fixture,directory=job
    path=Path(directory)/(cid+'.json')
    if path.exists():return read(path)
    rows=[]
    def make(*args,**kwargs):
        kwargs.update(history_factory=fast.AuditedCachedHistory)
        env=RecordingEnv(*args,**kwargs);env.rows=rows
        return env
    try:
        result,_=rollout(fixture,SOURCE,features=FEATURES,config=CFG,fast_feedback=True,environment_factory=make,
                         deterministic=True,record_details=False,max_ticks=1800,max_decisions=300,wall_seconds=600)
        summary=result['summary']
        rec={k:summary[k] for k in ('success','ticks','decisions','error','stop','termination')}
        rec.update(id=cid,complete=summary['error'] is None and summary['stop'] in ('terminal','execution_budget','decision_budget'),
                   history=fast.ACTIVE.history.sha256)
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path.with_suffix('.json.gz'),'wt',compresslevel=1) as f:json.dump(rows,f,separators=(',',':'))
        save(path,rec)
        return rec
    finally:
        if fast.ACTIVE:fast.ACTIVE.close_reads();fast.ACTIVE=None


def run_batch(source,fixtures,directory,workers):
    with ProcessPoolExecutor(min(len(fixtures),workers),mp_context=mp.get_context('fork'),initializer=start_worker,initargs=(source,)) as pool:
        pending=[pool.submit(execute,(k,f,str(directory))) for k,f in fixtures.items()]
        return sorted((f.result() for f in as_completed(pending)),key=lambda r:r['id'])


def cli(arm,command,args):
    root=OUT/arm; study=root/'study.json'; c=host.RoomCampaign(root)
    s=read(study)
    if command=='status':return snapshot(arm,s)
    if command=='inspect':
        sid,task=args[:2]
        p=sorted((root/'logs').glob('*/'+task+'.json'))
        p=[f for f in p if f.parent.name.endswith('_'+sid)]
        if not p:raise ValueError('no matching run')
        with gzip.open(p[-1].with_suffix('.json.gz'),'rt') as f:return dict(summary=read(p[-1]),boundaries=json.load(f))
    if s['final'] is not None:raise ValueError('already frozen for evaluation')
    if command=='submit':
        p=(WS/arm/args[0]).resolve()
        if not p.is_relative_to(WS/arm):raise ValueError('submit a file in your author directory')
        source=p.read_text()
        cfg=stream_config(source)
        names=next(ast.literal_eval(t.value) for t in ast.parse(source).body if isinstance(t,ast.Assign) and getattr(t.targets[0],'id','')=='NAMES')
        groups=dict(candidate={n:'A' for n in names},state={n:'A' for n in cfg['state_names']})
        state=read(root/'campaign.json')
        for sub in state['submissions']:
            if sub['sha256']==digest(source):return dict(sid=sub['id'],identical=True)
        item=host.RoomSession(root).tool('submit',dict(source=source,summary=' '.join(args[1:]) or p.name,groups=groups))
        return dict(sid=item['id'],sha=item['sha256'])
    if command=='final':
        valid=[v['sid'] for v in s['sweeps']]
        if not valid or (arm!='single' and len(valid)!=5):raise ValueError('complete required sweeps first')
        picked=max(s['sweeps'],key=lambda v:len(v['wins']))['sid'] if arm=='single' else valid[-1]
        if args and args[0]!=picked:raise ValueError('expected '+picked)
        s['final']=[picked] if arm=='single' else valid
        save(study,s)
        return dict(frozen_sids=s['final'])
    if command not in ('test','sweep'):raise ValueError('status / submit / test / inspect / sweep / final')
    sid=args[0]
    sub=next(row for row in read(root/'campaign.json')['submissions'] if row['id']==sid)
    source=(root/'semantics'/(sid+'.py')).read_text();assert digest(source)==sub['sha256']
    dev=read(WS/arm/'development_fixtures.json'); assert digest(dev)==read(OUT/'protocol.json')['dev_sha']
    idx=len(s['sweeps'])
    if idx>=5:raise ValueError('all sweep slots used')
    if command=='test':
        ids=args[1:]
        if not ids or len(set(ids))!=len(ids) or not set(ids)<=dev.keys():raise ValueError('test SID distinct development IDs')
        if s['probes']+len(ids)>160 or (arm!='single' and s['probes_this_round']+len(ids)>32):raise ValueError('probe budget')
    else:
        ids=sorted(dev)
        if any(row['sid']==sid for row in s['sweeps']):raise ValueError('sweep slot requires a new revision')
    jobs={i:dev[i] for i in ids}
    key=f'{time.time_ns()}_{command}_{sid}'
    s['active']=dict(key=key,sid=sid,kind=command,ids=ids,round=idx+1)
    with c.locked():
        c._check()
        c._charge(attempts=len(ids),resets=len(ids),execution_ticks=1800*len(ids),feature_batches=301*len(ids))
    if command=='test':
        s['probes']+=len(ids);s['probes_this_round']+=len(ids)
    save(study,s)
    rows=run_batch(source,jobs,root/'logs'/key,5)
    with c.locked():
        c.state['used']['execution_ticks']-=1800*len(ids)-sum(r['ticks'] for r in rows)
        c.state['used']['feature_batches']-=301*len(ids)-sum(r['decisions']+int(bool(r['error'])) for r in rows)
        c._save()
    s['active']=None
    if command=='sweep' and all(r['complete'] for r in rows):
        wins=sorted(r['id'] for r in rows if r['success'])
        covered=set().union(*(v['wins'] for v in s['sweeps']))
        center,focus=region(s) if arm=='regional' else (None,[])
        s['sweeps'].append(dict(sid=sid,key=key,wins=wins,added=len(set(wins)-covered),center=center,focus=focus))
        s['probes_this_round']=0
    save(study,s)
    if command=='test':return rows
    errors=[r for r in rows if not r['complete']]
    return dict(tested=len(rows),success=sum(r['success'] for r in rows),errors=errors,
                wins=[r['id'] for r in rows if r['success']],state=snapshot(arm,s))


def main():
    if sys.argv[1]=='setup':setup();return
    arm,command,*args=sys.argv[1:]
    if arm not in ARMS:raise ValueError(arm)
    if command in ('status','inspect'):
        result=cli(arm,command,args)
    else:
        with (OUT/(arm+'.author.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            result=cli(arm,command,args)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
