"""Archived expert-prefix option forks; GA inputs and public one-option targets."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from . import cap_successor_value as old
from .cap_inverse_fit import rows_for
from experiments.planar_tasks.semantic_rl.memory_flow import current_event
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime as fast
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save, flatten
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import candidate_arrays
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import all_v
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.runtime import CAPS

OUT=old.OUT.parent/'cap_ga_effect_20261119_01'
TASKS=['开发场景采集','留出分叉特征','A / B / C 训练更新','新面板闭环（含七个 CAP）']
SIZES=[608,196,3600,2200]
W=fast.W


def publish(stage,detail='',task=None,count=None,**extra):
    path=OUT/'live/state.json'
    s=read(path) if path.exists() else {}
    numbers={k:s.get('tasks',{}).get(k,[0,n]) for k,n in zip(TASKS,SIZES)}
    if task is not None:
        numbers[TASKS[task]]=[count,SIZES[task]]
    s.update(stage=stage,detail=detail,time=time.time(),tasks=numbers,**extra)
    save(path,s)


def order(key):
    return hashlib.sha256(('ga-effect-01/'+key).encode()).hexdigest()


def plan():
    family=old.scale.family; bundle=family.BUNDLE
    grouped=defaultdict(list)
    pool=set(read(family.PARENT/'split.json')['train'])
    scenes={r['case_id']:r['fixture'] for r in read(bundle/'scenes.json')}
    for r in read(bundle/'index.json'):
        if r['case_id'] in pool:
            grouped[r['case_id']].append(dict(path=str(bundle/r['path']),cap=r['cap_id'],sha=r['sha256']))
    original=sorted(grouped,key=order)[:304]
    for p in old.scale.OUT.glob('batches/*_plan.json'):
        scenes.update({r['case_id']:r['fixture'] for r in read(p)})
    augmented=defaultdict(list)
    for r in rows_for(1001):
        augmented[r['case_id']].append(dict(path=r['replay'],cap=r['cap'],sha=r['sha256']))
    extra=sorted(augmented,key=order)[:304]
    grouped.update(augmented)
    train=[]
    for j,cid in enumerate(original+extra):
        donors=sorted(grouped[cid],key=lambda r:r['cap'])
        train.append(dict(case_id=cid,fixture=scenes[cid],**donors[j%7]))
    used=set(read(old.scale.OUT/'forbidden.json'))
    for p in [*old.scale.OUT.glob('batches/*_plan.json'),old.OUT/'panel.json',
              *(OUT.parent/'cap_semantic_disagreement_20261119_01').glob('*_panel.json')]:
        used.update(r['content_sha256'] for r in read(p))
    test=[]
    for r in family.rooms.generate(2026111971)['cases']:
        h=family.content(r['fixture'])
        assert h not in used
        used.add(h)
        test.append(dict(case_id='ef_'+r['id'],fixture=r['fixture'],movable_count=r['movable_count'],content_sha256=h))
    assert len(test)==200 and len(train)==608
    for name,data in [('training',train),('panel',test)]:
        p=OUT/(name+'.json')
        if p.exists():assert read(p)==data
        else:save(p,data)
    spec=dict(q=old.sha(old.Q),v=old.sha(old.V),train=old.sha(OUT/'training.json'),panel=old.sha(OUT/'panel.json'),
              warm_start='GA_s17 highest dose, frozen first affine incl normalizers; train shared second 64-dimensional layer and scoring heads',
              training='all distinct legal proposals at up to four archive boundaries per donor; require exact full-prefix hash; same 1200 minibatches of 64, Adam 3e-4; fixed last step; no trace inputs',
              effects=['visible target centroid goalward displacement m','visible target PCA axis signed change rad modulo pi','target contact robot0','target contact robot1','actual periods'],
              aux='0.05 times equal-channel standardized masked MSE; train-set mean and scale; C reads predicted effects with state-dependent coefficients',
              ranking='existing 196 diagnostic pairs with archived common GA continuation; untouched by training, previously viewed outcomes',
              closed_loop='fresh 200, arms original/A/B/C plus seven frozen CAPs; primary C minus B paired, single seed17')
    if (OUT/'protocol.json').exists():assert spec==read(OUT/'protocol.json')
    else:save(OUT/'protocol.json',spec)
    return train


def tensors(packet):
    x=torch.from_numpy(flatten(packet))[None]
    a,s,_=candidate_arrays(packet,W.LIBRARY)
    return x,torch.from_numpy(a)[None],torch.from_numpy(s)[None]


def embedding(q,x,a,s):
    # Cache the original first affine output using the exact released GA-Q implementation.
    captured=[]
    hook=q.net[1].register_forward_hook(lambda module,args,out:captured.append(out))
    try:
        score=q.values(x,a,s)
    finally:
        hook.remove()
    return captured[0],score


def observed(o):
    pts=next((e['points'] for e in o['entities'] if e['id']==o['goal']['target_id']),[])
    center=angle=None
    if len(pts)>=2:
        pts=np.asarray(pts);center=pts.mean(0)
        eigen,axes=np.linalg.eigh(np.cov(pts.T))
        if eigen[-1]>.0025 and eigen[-1]>2*eigen[0]:
            angle=math.atan2(axes[1,-1],axes[0,-1])
    return center,angle


def effects(o,n,periods):
    c,angle=observed(o);t,b=observed(n)
    dx=da=np.nan
    if c is not None and t is not None:
        goal=np.array(o['goal']['xy'])-c
        dx=float((t-c)@goal/max(1e-8,np.linalg.norm(goal)))
    if angle is not None and b is not None:
        da=(b-angle+math.pi/2)%math.pi-math.pi/2
    contacts=[float(bool(r['contacting_object']) and n['active_target']==n['goal']['target_id']) for r in n['robots']]
    return [dx,da,*contacts,periods]


def environment(fixture):
    return W.Environment(fixture,W.EventConfig(mode='duration'),fast_feedback=True,
                         history_factory=W.StreamHistory,record_details=False)


def step(env,action):
    return env.step(action,1800-env.env.total_periods)


def terminal(env):
    return env.env.termination is not None or env.env.total_periods>=1800 or env.decision_count>=300


def at_prefix(fixture,actions):
    e=environment(fixture)
    try:
        for a in actions:
            step(e,a)
        return e
    except BaseException:
        e.close_reads()
        raise


def collect(case):
    cid=case['case_id'];dest=OUT/'samples'/(cid+'.npz')
    if dest.exists():return
    assert old.sha(case['path'])==case['sha']
    with gzip.open(case['path'],'rt') as stream:log=json.load(stream)
    n=len(log['events'])
    rng=np.random.default_rng(int(order(cid)[:12],16))
    parts=np.array_split(np.arange(1,n),3)
    depths=sorted([0]+[int(rng.choice(p)) for p in parts if len(p)])
    actions=[r['requested'] for r in log['events']]
    library=W.LIBRARY
    library.reset()
    main=environment(case['fixture'])
    rows=[]; mismatch=-1; verified=0
    try:
        for i in range(depths[-1]+1):
            if main.history.sha256!=log['public_rows'][i]['history_hash']:
                mismatch=i
                break
            packet=library.advance(main.history)
            if i in depths:
                verified+=1
            if i in depths and len(packet['pool'])>=2:
                prefix_hash=main.history.sha256; o=current_event(main.history)['observation']
                with torch.no_grad():
                    u,q=embedding(old.NETWORKS['q'],*tensors(packet))
                # Branches replay the same physical prefix. Expert memories keep only public inputs.
                snapshot=copy.deepcopy((library.memory,library.geometry))
                for slot,action in enumerate(packet['pool']):
                    branch=at_prefix(case['fixture'],actions[:i])
                    try:
                        assert branch.history.sha256==prefix_hash
                        ev=step(branch,action)
                        new=current_event(branch.history)['observation']
                        if terminal(branch):
                            label=float(branch.env.success); top=-1
                        else:
                            library.memory,library.geometry=copy.deepcopy(snapshot)
                            library.episode=branch.history.episode;library.index=i
                            library.geometry.episode=branch.history.episode
                            nxt=library.advance(branch.history)
                            with torch.no_grad():
                                scores=all_v(old.NETWORKS['v'],torch.from_numpy(flatten(nxt))[None])[0]
                            legal=np.flatnonzero(nxt['legal'])
                            top=int(legal[scores.numpy()[legal].argmax()]) if len(legal) else -1
                            label=float(scores[top]) if top>=0 else 0.
                        rows.append((i,slot,u[0,slot].numpy().copy(),float(q[0,slot]),label,
                                     effects(o,new,ev['periods']),top))
                    finally:branch.close_reads()
                library.memory,library.geometry=snapshot
                library.episode=main.history.episode;library.index=i;library.packet=packet
            if i<depths[-1]:step(main,actions[i])
    finally:
        main.close_reads();fast.ACTIVE=None
    dest.parent.mkdir(exist_ok=True)
    temp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(temp,u=np.array([r[2] for r in rows],dtype='f').reshape(-1,64),
                        q=np.array([r[3] for r in rows],'f'),y=np.array([r[4] for r in rows],'f'),
                        e=np.array([r[5] for r in rows],'f').reshape(-1,5),index=[r[0] for r in rows],
                        slot=[r[1] for r in rows],top=[r[6] for r in rows],mismatch=mismatch,verified=verified,planned=len(depths))
    temp.replace(dest)


def collect_hold(case):
    path=OUT/'held'/(case['case_id']+'.npz')
    if path.exists():return
    archive=read(old.OUT/'cases'/(case['case_id']+'.json'))
    a,b=archive['branches']['ga'];f=a['fork']
    W.LIBRARY.reset()
    main=environment(case['fixture']);actions=[]
    try:
        for i in range(f['index']+1):
            p=W.LIBRARY.advance(main.history)
            with torch.no_grad():u,q=embedding(old.NETWORKS['q'],*tensors(p))
            if i==f['index']:break
            j=int(q[0,:len(p['pool'])].argmax())
            actions.append(p['pool'][j]);step(main,actions[-1])
        assert main.history.sha256==f['hash']
        o=current_event(main.history)['observation']
        slots=[p['pool'].index(command) for command in f['candidates']]
        eff=[]
        for j,row in enumerate((a,b)):
            e=at_prefix(case['fixture'],actions)
            try:
                event=step(e,f['candidates'][j])
                assert e.history.sha256==row['successor']['hash']
                eff.append(effects(o,current_event(e.history)['observation'],event['periods']))
            finally:e.close_reads()
        y=[r['successor']['max_v'] for r in (a,b)]
        np.testing.assert_allclose(q[0,slots].numpy(),f['q'],rtol=1e-6,atol=1e-7)
        path.parent.mkdir(exist_ok=True)
        tmp=path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,u=u[0,slots].numpy(),e=np.array(eff,'f'),q=f['q'],y=y,z=[a['success'],b['success']])
        tmp.replace(path)
    finally:main.close_reads();fast.ACTIVE=None


def parallel(fn,items,task,stage):
    with ProcessPoolExecutor(10,mp_context=mp.get_context('spawn'),initializer=old.initialize) as pool:
        futures=[pool.submit(fn,c) for c in items]
        for i,f in enumerate(as_completed(futures),1):
            f.result();publish(stage,task=task,count=i)


def main():
    train=plan()
    publish('采集开发轨迹多动作分叉',task=0,count=0)
    parallel(collect,train,0,'采集开发轨迹多动作分叉')
    held=[c for c in read(old.OUT/'panel.json') if read(old.OUT/'cases'/(c['case_id']+'.json'))['eligible']]
    assert len(held)==196
    parallel(collect_hold,held,1,'重建固定 GA-Q 续跑的留出分叉')
    save(OUT/'data_complete.json',dict(train=len(train),held=len(held)))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        publish('采集异常',str(exc));raise
