"""Growing development room pool, sparse counterfactual sampling, frozen natural evaluation."""
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tools import cap_union_iteration as p
from tools import cap_union_trend as trend

d,e=p.d,p.e
OUT=p.OUT.parent/'cap_expanding_pool_20261123_01'
DOC=p.DOC.parent/'2026-11-23_expanding_pool'
TRAIN_SEEDS=(662319301,662319303,662319307,662319311)
HOLD_SEED=662319379
WORKERS=len(os.sched_getaffinity(0))
ARMS=('A','B')
PROFILE=None
ROOMS={}
WAVES={}
MODULES={}


def save_npz(path, **fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending.npz')
    np.savez_compressed(temp,**fields)
    temp.replace(path)


def setup():
    global PROFILE
    p.torch.set_num_threads(1)
    PROFILE=d.init()
    MODULES.update(A=e.AM,B=e.BM)
    for stage,seed in enumerate((*TRAIN_SEEDS,HOLD_SEED),1):
        fixtures=d.rooms(seed)
        keys=[]
        for label,fixture in fixtures.items():
            key=e.s.phase.input_hash(fixture)
            assert key not in ROOMS
            ROOMS[key]=fixture
            keys.append(key)
        WAVES[stage]=keys
    assert len(ROOMS)==1000
    manifest=dict(profile=PROFILE,parent=d.source.graph.file_sha(d.OUT/'fit/semantic.npz'),
        executor=e.g.prep.prior.EXECUTOR,train_seeds=TRAIN_SEEDS,held_seed=HOLD_SEED,ids=WAVES,
        stages=4,seed=17,selection='each wave all A-only / B-only, 20 both, 8 neither; up to 32 revisits from previous visited rooms',
        revisits='rank past observed success loss, exclusive last failure with remaining unseen boundaries, other remaining candidates, then prior informative; stable hashed ties',
        forks='up to 3 temporally stratified distinct prefix-action requests; prefer unsampled physical prefixes under this policy family',
        fit='same round preferences and current actor anchors; 300 Adam updates as union iteration',
        evaluation='primary final vs initial on all 200 held rooms; diagnostic 64 previously viewed natural rooms per round')
    path=OUT/'protocol.json'
    if path.exists():
        assert e.read(path)==json.loads(json.dumps(manifest))
    else:
        known=e.s.collect_identities()
        for values in e.read(d.OUT/'protocol.json')['rooms']:
            known.update(values.values())
        for f in (trend.OUT/'confirmation_ids.json',p.OUT/'fresh_ids.json',d.source.OUT/'confirmation_ids.json',e.OUT/'confirmation_ids.json',e.old.OUT/'fresh_panel.json'):
            known.update(e.read(f).values())
        assert not set(ROOMS).intersection(known)
        e.save(path,manifest)


def screen_path(expert,room):
    return OUT/'screen'/PROFILE[expert]/(room+'.json')


def single(job):
    expert,key=job
    started=time.process_time()
    policy=e.s.phase.FrozenCAP(MODULES[expert])
    with e.s.phase.world(ROOMS[key]) as (w,_):
        counts=Counter()
        policy.update(w.history)
        while not e.s.ended(w):
            cmd=policy.propose(e.s.observe(w))
            e.s.tick_action(w,cmd,e.g.prep.prior.EXECUTOR,policy.update,counts)
        e.save(screen_path(expert,key),dict(success=int(w.env.success),ticks=w.env.total_periods,
                                         cpu=time.process_time()-started))


def timed_pool(name,fn,jobs):
    start=time.monotonic()
    progress=OUT/'progress'/(name+'.json')
    if jobs:
        with ProcessPoolExecutor(max_workers=min(WORKERS,len(jobs)),mp_context=mp.get_context('fork')) as pool:
            tasks=[pool.submit(fn,j) for j in jobs]
            for done,task in enumerate(as_completed(tasks),1):
                task.result()
                if done%4==0 or done==len(tasks):
                    e.save(progress,dict(done=done,total=len(tasks),percent=100*done//len(tasks)))
    else:
        e.save(progress,dict(done=0,total=0,percent=100))
    path=OUT/'timing'/(name+'.json')
    prev=e.read(path) if path.exists() else dict(wall=0,tasks=0)
    e.save(path,dict(wall=prev['wall']+time.monotonic()-start,tasks=prev['tasks']+len(jobs),workers=WORKERS))


def screen():
    keys=[k for stage in range(1,5) for k in WAVES[stage]]
    timed_pool('screen',single,[(arm,k) for k in keys for arm in ARMS if not screen_path(arm,k).exists()])


def stratum(k):
    a,b=(e.read(screen_path(arm,k))['success'] for arm in ARMS)
    return ('neither','B_only','A_only','both')[2*a+b]


if __name__=='__main__' and sys.argv[1]=='screen':
    setup()
    screen()

NEIGHBORS={'A':('B',),'B':('A',)}


def stage_dir(stage):
    return OUT/f'round{stage}'


def baseline(job):
    stage,k=job
    start=time.process_time()
    result,points=p.run_fork(ROOMS[k],comparison=NEIGHBORS)
    directory=stage_dir(stage)/'parents'
    features=np.asarray([q[1] for q in points],dtype=np.float32)
    save_npz(directory/(k+'.npz'),x=features,b=np.array([q[2] for q in points],dtype=int))
    e.save(directory/(k+'.json'),dict(success=result['success'],ticks=result['ticks'],cpu=time.process_time()-start,
            points=[[q[0],q[3],q[4],q[5]] for q in points]))


def branch(job):
    stage,k,index=job
    start=time.process_time()
    directory=stage_dir(stage)
    row=e.read(directory/'parents'/(k+'.json'))
    boundary,prefix,cmd,who=row['points'][index]
    with np.load(directory/'parents'/(k+'.npz')) as data:
        ref=(boundary,data['x'][index],int(data['b'][index]),prefix,cmd,who)
    result,_=p.run_fork(ROOMS[k],ref,boundary,comparison=NEIGHBORS)
    e.save(directory/'branches'/f'{k}_{index}.json',dict(success=result['success'],ticks=result['ticks'],cpu=time.process_time()-start,
        prefix=prefix,alternate_cmd=cmd,expert=who))


def shuffled(keys,salt):
    return sorted(keys,key=lambda k:e.key(f'{salt}/{k}'))


def request_key(point):
    return (point[1],tuple(point[2]))


def select_points(points,used,stage,k):
    picks=[]
    for j,block in enumerate(np.array_split(np.arange(len(points)),3)):
        ids=list(map(int,block))
        if ids:
            novel=[i for i in ids if request_key(points[i]) not in used]
            picks.append(shuffled(novel or ids,f'fork/{stage}/{k}/{j}')[0])
    return picks


def old_visits(stage):
    history={}
    for step in range(1,stage):
        for visit in e.read(stage_dir(step)/'visits.json'):
            history.setdefault(visit['room'],[]).append(visit)
    return history


def select_rooms(stage,history):
    directory=stage_dir(stage)
    keys=WAVES[stage]
    strata={k:stratum(k) for k in [*keys,*history]}
    fresh=[k for k in keys if strata[k].endswith('_only')]
    for grp,limit in [('both',20),('neither',8)]:
        fresh+=shuffled([k for k in keys if strata[k]==grp],f'{stage}/{grp}')[:limit]
    sentinels=shuffled([k for k in history if history[k][-1]['success']],f'{stage}/retention')[:32]
    timed_pool(f'r{stage}_retention',baseline,[(stage,k) for k in sentinels if not (directory/'parents'/(k+'.json')).exists()])
    lost={k for k in sentinels if not e.read(directory/'parents'/(k+'.json'))['success']}
    def priority(k):
        visits=history[k]
        last=visits[-1]
        remaining=last['remaining']
        return (k not in lost,not(strata[k].endswith('_only') and not last['success'] and remaining),
                not remaining,not last['informative'],len(visits),e.key(f'{stage}/repeat/{k}'))
    repeat=sorted(history,key=priority)[:32]
    return dict(new=fresh,repeat=repeat,sentinels=sentinels,lost=sorted(lost))


def train_round(stage):
    directory=stage_dir(stage)
    history=old_visits(stage)
    selection=select_rooms(stage,history)
    e.save(directory/'selection.json',selection)
    keys=selection['new']+selection['repeat']
    timed_pool(f'r{stage}_parent',baseline,[(stage,k) for k in keys if not (directory/'parents'/(k+'.json')).exists()])
    requests={}
    previously={}
    for k in keys:
        used={(prefix,tuple(cmd)) for v in history.get(k,[]) for prefix,cmd in v['sampled']}
        points=e.read(directory/'parents'/(k+'.json'))['points']
        requests[k]=select_points(points,used,stage,k)
        previously[k]=used
    jobs=[(stage,k,i) for k in keys for i in requests[k] if not (directory/'branches'/f'{k}_{i}.json').exists()]
    e.save(directory/'schedule.json',dict(rooms=len(keys),forks=sum(map(len,requests.values()))))
    timed_pool(f'r{stage}_fork',branch,jobs)
    visits=[]
    for k in keys:
        row=e.read(directory/'parents'/(k+'.json'))
        selected=requests[k]
        outcomes=[e.read(directory/'branches'/f'{k}_{i}.json')['success'] for i in selected]
        with np.load(directory/'parents'/(k+'.npz')) as data:
            save_npz(directory/'fit_data'/(k+'.npz'),x=data['x'],b=data['b'],index=np.asarray(selected,int),
                     baseline=row['success'],alternate=np.asarray(outcomes,int))
        sampled=[request_key(row['points'][i]) for i in selected]
        available=set(map(request_key,row['points']))
        used=previously[k]
        visits.append(dict(room=k,stratum=stratum(k),fresh=k not in history,success=row['success'],candidates=len(available),
             remaining=len(available-used-set(sampled)),repeat_prefix=sum(q in used for q in sampled),sampled=sampled,
             informative=sum(v!=row['success'] for v in outcomes),rescue=sum(v>row['success'] for v in outcomes)))
    p.fit(stage,[directory/'fit_data'/(k+'.npz') for k in sorted(keys)])
    e.save(directory/'visits.json',visits)


def evaluate_one(job):
    stage,k,folder=job
    start=time.process_time()
    result=d.episode(ROOMS[k],'semantic',label=False)[0]
    e.save(OUT/folder/f's{stage}'/(k+'.json'),dict(success=result['success'],ticks=result['ticks'],cpu=time.process_time()-start))


def eval_stage(stage,keys,folder):
    timed_pool(f'{folder}_{stage}',evaluate_one,[(stage,k,folder) for k in keys if not (OUT/folder/f's{stage}'/(k+'.json')).exists()])


def run():
    setup()
    assert all(screen_path(a,k).exists() for w in range(1,5) for k in WAVES[w] for a in ARMS)
    p.OUT=OUT
    diagnostic=[]
    for f in d.rooms(trend.FRESH).values():
        k=e.s.phase.input_hash(f)
        assert k not in ROOMS
        ROOMS[k]=f
        diagnostic.append(k)
    diagnostic=shuffled(diagnostic,'diagnostic64')[:64]
    p.load_stage(0)
    eval_stage(0,diagnostic,'diagnostic')
    for stage in range(1,5):
        p.load_stage(stage-1)
        if not (stage_dir(stage)/'visits.json').exists():
            train_round(stage)
        p.load_stage(stage)
        eval_stage(stage,diagnostic,'diagnostic')
    held=WAVES[5]
    timed_pool('held_singles',single,[(a,k) for k in held for a in ARMS if not screen_path(a,k).exists()])
    for stage in (0,4):
        p.load_stage(stage)
        eval_stage(stage,held,'held')
    e.save(OUT/'lock.json',dict(final=d.source.graph.file_sha(OUT/'actor4.npz'),held=held))


if __name__=='__main__' and sys.argv[1]=='run':
    run()


def training_outcomes():
    """Measure retention across the same visited training rooms at both endpoints."""
    setup()
    assert e.read(OUT/'lock.json')['final']==d.source.graph.file_sha(OUT/'actor4.npz')
    p.OUT=OUT
    keys=sorted(old_visits(5))
    # The initial policy's first-wave parent rollouts are already exact baseline outcomes.
    for k in e.read(stage_dir(1)/'selection.json')['new']:
        initial=e.read(stage_dir(1)/'parents'/(k+'.json'))
        e.save(OUT/'train_check'/'s0'/(k+'.json'),dict(success=initial['success'],ticks=initial['ticks'],source='round1 parent'))
    for stage in (0,4):
        p.load_stage(stage)
        eval_stage(stage,keys,'train_check')
    rows=[dict(room=k,stratum=stratum(k),initial=e.read(OUT/'train_check/s0'/(k+'.json'))['success'],
               final=e.read(OUT/'train_check/s4'/(k+'.json'))['success']) for k in keys]
    import csv
    DOC.mkdir(parents=True,exist_ok=True)
    with (DOC/'train_check.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    e.save(OUT/'train_check.json',dict(rooms=len(rows),initial=sum(r['initial'] for r in rows),
        final=sum(r['final'] for r in rows),gain=sum(r['final']>r['initial'] for r in rows),
        loss=sum(r['final']<r['initial'] for r in rows)))


if __name__=='__main__' and sys.argv[1]=='train-check':
    training_outcomes()
