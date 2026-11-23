"""Frozen data and causal features for action/geometry actor ablations."""
import gzip
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tools import cap_actor_replan as prior
from experiments.planar_tasks.semantic_rl.relational_transfer_features import action_encoding

run = prior.run
read, save = run.ref.read, run.ref.save
OUT = run.campaign.OUT/'actor_inputs_20261121_s17'
DOC = prior.DOC.parent/'2026-11-21_actor_inputs'
ARMS = ('action','action_ln','geometry_action','geometry_action_ln','router','priority')
SCRATCH = Path('/dev/shm/coi_actor_inputs_20261121_s17')
GW = 11304
FRAME = GW//9
# Entity block ends each frame, immediately preceded by the previous command.
gkeep = np.flatnonzero(np.tile(np.r_[np.ones(FRAME-108-69,bool),np.zeros(69,bool),np.ones(108,bool)],9))
GMEAN = GSCALE = CMEAN = CSCALE = None


def init():
    global GMEAN, GSCALE, CMEAN, CSCALE
    old, held, members = run.initialize()
    base = run.ref.scale.original.BASE
    with np.load(base/'normalizers.npz') as n:
        GMEAN, GSCALE = n['mean'][gkeep], n['scale'][gkeep]
        # Only use the old statistics for the target geometry (world-frame A has new axes).
        CMEAN, CSCALE = n['candidate_mean'][69:], n['candidate_scale'][69:]
    assert read(base/'dataset.json')['layout']['geometry_width']==GW
    return old,held,members


def encode(x,mean,scale):
    v=np.isfinite(x)
    return np.concatenate((np.clip(np.where(v,(x-mean)/scale,0.),-10,10),v.astype(np.float32)),axis=-1)


def features(lib, packet):
    x,c,s,valid,pool,_=packet
    # Scores cannot use the policy iteration order as a tie breaker.
    order=sorted(range(len(pool)),key=lambda i:(pool[i]['a0'],pool[i]['a1'],pool[i]['ticks'],pool[i]['target_id'] or ''))
    actions=[pool[i] for i in order]
    explicit=np.zeros((5,69),dtype=np.float32)
    target=np.zeros((5,32),dtype=np.float32)
    supporters=np.zeros((5,5),dtype=bool)
    for j,i in enumerate(order):
        explicit[j]=action_encoding(pool[i],lib.geometry.goal_id,0.)
        target[j]=encode(c[i,69:],CMEAN,CSCALE)
        supporters[j]=s[i]
    return dict(g=encode(x[gkeep],GMEAN,GSCALE),a=explicit,t=target,s=supporters,
                valid=valid.copy()),actions


def getjobs():
    old,held,members=init()
    scale=run.ref.scale
    teachers=read(scale.OUT/'teachers.json')
    rooms=old|read(scale.OUT/'natural.json')|read(scale.OUT/'variants.json')
    extra=run.ref.sets.unzip(scale.OUT/'new_teacher_actions.json.gz')
    jobs=[]
    for fold,fixtures in [('train',rooms),('held',held)]:
        for key,fixture in sorted(fixtures.items()):
            if fold=='train':
                if key not in teachers: continue
                ref=teachers[key]
                sid=ref['sid']
            else:
                successes=[(r['ticks'],sid,r) for sid in members
                           if (r:=read(run.campaign.OUT/'runs'/sid/(key+'.json')))['success']]
                if not successes: continue
                _,sid,ref=min(successes)
            if key in extra:
                seq=extra[key]['steps']
                assert extra[key]['sid']==sid
            else:
                with gzip.open(run.campaign.OUT/'runs'/sid/(key+'.json.gz'),'rt') as f:
                    seq=[{'a':r['a'],'tick':r['tick']} for r in json.load(f)]
            jobs.append((fold,key,fixture,members.index(sid),seq,ref['history']))
    order=read(run.ref.fit.OUT/'priority_reference.json')['priority']
    assert [j[0] for j in jobs].count('train')==491 and len(jobs)==635
    design=dict(members=members,priority=order,train=491,held=144,seed=17,batch=64,updates=24000,lr=.0003,
                current_panel=read(prior.OUT/'protocol.json')['geometry'],arms=list(ARMS),
                inputs='raw command axes, source-blind candidate scores, canonical command tie break; geometry history excludes prior action fields; unlearned LayerNorm at first fused preactivation',
                router='5 logits, hard fastest teacher CAP identity, mask invalid CAPs, score action by highest supporting logit',
                normalizers=run.campaign.sha(scale.original.BASE/'normalizers.npz'),
                teachers=run.campaign.sha(scale.OUT/'teachers.json'))
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json')==design
    save(OUT/'protocol.json',design)
    return jobs


def replay(job):
    fold,key,f,teacher,steps,expected=job
    path=SCRATCH/fold/(key+'.npz')
    if path.exists(): return
    rows=[]
    with run.runtime.world(f) as (env,lib):
        for step in steps:
            packet=lib.panel(env.history)
            frame,actions=features(lib,packet)
            index=actions.index(step['a'])
            assert frame['s'][index,teacher]
            rows.append(dict(frame,y=index,teacher=teacher))
            run.runtime.advance(env,step['a'])
            assert env.env.total_periods==step['tick']
        assert run.terminal(env)==1 and env.history.sha256==expected
    tmp=path.with_suffix('.tmp.npz')
    np.savez(tmp,**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    os.replace(tmp,path)


def main():
    OUT.mkdir(exist_ok=True)
    jobs=getjobs()
    for fold in ('train','held'): (SCRATCH/fold).mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(16,mp_context=mp.get_context('fork')) as pool:
        for i,f in enumerate(as_completed([pool.submit(replay,j) for j in jobs]),1):
            f.result()
            if i%5==0: save(OUT/'extraction.json',dict(done=i,total=len(jobs)))
    for fold,count in [('train',15601),('held',4420)]:
        files=sorted((SCRATCH/fold).glob('*.npz'))
        pos=0
        arrays={}
        for i,p in enumerate(files):
            with np.load(p) as frame:
                length=len(frame['y'])
                if not arrays:
                    for k in frame.files:
                        v=frame[k]
                        arrays[k]=np.lib.format.open_memmap(SCRATCH/fold/(k+'.npy'),mode='w+',dtype=v.dtype,shape=(count,*v.shape[1:]))
                    arrays['room']=np.lib.format.open_memmap(SCRATCH/fold/'room.npy',mode='w+',dtype='int32',shape=(count,))
                for k in frame.files: arrays[k][pos:pos+length]=frame[k]
                arrays['room'][pos:pos+length]=i
                pos+=length
        assert pos==count
        for a in arrays.values(): a.flush()
        for p in files: p.unlink()
    save(OUT/'data_ready.json',dict(train=15601,held=4420))


if __name__=='__main__': main()
