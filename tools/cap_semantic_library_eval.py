"""Freeze the two authored libraries, fit matched boundary actors, run paired closed loops."""
import gzip
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.planar_tasks.semantic_rl.corridor_semantics import corridor_semantics
from tools import cap_semantic_library_pilot as study
from tools import cap_actor_rollout_value as runtime
from tools import cap_regional_directq as dq

OUT,WORK,ARMS,base=study.OUT,study.WORK,study.ARMS,study.base
read,save=study.read,study.save
DOC=base.ROOT/'docs/research/key_findings/2026-11-23_semantic_library'
SEED=66231141
MODELS={}
MODULES=[]
RULES=[]


def freeze_panel():
    dev=read(OUT/'development.json')
    ex=base.OUT.parent
    used=set(read(ex/'cap_inverse_scale_20261119_01/forbidden.json'))
    used.update(r['content_sha256'] for r in read(base.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json'))
    def visit(x):
        if isinstance(x,dict):
            if {'object','goal','robots','room','movable_objects'}<=x.keys():
                used.add(study.previous.input_hash(x))
            else:
                for v in x.values():visit(v)
        elif isinstance(x,list):
            for v in x:visit(v)
    for name in ('panel.json','panels.json','held.json','natural.json','variants.json','development.json','families.json','extra.json','evaluation_panel.json'):
        for path in ex.glob('**/'+name):
            if OUT not in path.parents:visit(read(path))
    visit(dev)
    held={f'sem23_{v["id"]}':v['fixture'] for v in base.rooms.generate(SEED)['cases']}
    hashes={study.previous.input_hash(v) for v in held.values()}
    assert len(hashes)==len(held)==200 and not hashes&used
    protocol=dict(held=base.digest(held),dev=base.digest(dev),seed_rooms=SEED,
        author='two independent 6.2-sol high, same s025 plus 3 new members per arm; 3 slots x 2 x 32 room cap per arm',
        budget='single pilot; approximately 12 minutes active development each; report usage and wall time',
        semantic='author freezes 2-3 public predicates before first candidate; same frozen bits for both banks',
        score='shared 64x64 candidate MLP; 9 public geometry features, action parameters, source membership, budget; append frozen bits or equal-width zeros',
        teacher='fastest successful member per development room, earliest member tie; one joint-action target per multi-proposal teacher boundary',
        fit='seed17 8000 Adam updates batch64, lr=.0003, uniform multi-candidate boundaries; fixed last checkpoint',
        evaluation='each frozen single CAP and both learned action scorers; room matched two-by-two interaction; no selection on held')
    for name,v in [('test',held),('protocol',protocol)]:
        p=OUT/(name+'.json')
        if p.exists():assert read(p)==v
        else:save(p,v)


def seal(arm):
    c=base.host.RoomCampaign(OUT/arm)
    manifest=OUT/f'{arm}_library.json'
    if manifest.exists():
        c.verify_frozen()
        return read(manifest)
    final=read(WORK/arm/'FINAL.json')['members']
    assert sorted(v['slot'] for v in final)==[1,2,3]
    members=[dict(id='basis',path=str(study.previous.CAM/'semantics/s025.py'))]
    for m in sorted(final,key=lambda v:v['slot']):
        r=read(OUT/arm/f'probe_m{m["slot"]}v{m["revision"]}.json')
        src=OUT/arm/'semantics'/(r['sid']+'.py')
        assert not r['errors'] and base.digest(src.read_text())==r['sha'] and src.read_bytes()==(WORK/arm/f'm{m["slot"]}v{m["revision"]}.py').read_bytes()
        members.append(dict(id=r['sid'],path=str(src),**m))
    assert len(set(base.digest(Path(v['path']).read_text()) for v in members))==4
    for v in members:v['sha']=base.digest(Path(v['path']).read_text())
    c.freeze();c.verify_frozen()
    save(manifest,members)
    return members


def init(sources):
    global MODULES,RULES
    torch.set_num_threads(1)
    MODULES=[dq.load_native(src)[0] for src in sources]
    RULES=read(OUT/'conditions.json')


def command(mod,mem,event,obs):
    mem=mod.update_memory(mem,dq.owned_copy(event))
    ctx=mod.prepare_context(dq.owned_copy(mem),dq.owned_copy(obs))
    action=ctx['proposals'][0] if ctx['proposals'] else None
    if action is not None:
        try:dq.validate_action(action,obs)
        except (ValueError,KeyError,TypeError):action=None
    return mem,action


def panel(env,mem,angle,features):
    h=env.history
    ev=dq.current_event(h);obs=ev['observation']
    event=dict(ev,executed_ticks=None if h.previous_tick is None else obs['tick']-h.previous_tick)
    pool=[];support=[]
    for i,(mod,m) in enumerate(zip(MODULES,mem)):
        mem[i],a=command(mod,m,event,obs)
        if a is None:continue
        if a not in pool:pool.append(a);support.append(np.zeros(len(mem)))
        support[pool.index(a)][i]=1.
    if not pool:pool=[dict(a0=44,a1=44,ticks=1,target_id=None)];support=[np.zeros(len(mem))]
    if not features or len(pool)==1:return pool,None
    g=base.geom(obs)
    if features=='basic':bits=np.zeros(len(RULES))
    else:
        s=corridor_semantics(obs);s['observed']=s['kind']
        bits=np.array([study.matches(rule,s) for rule in RULES])
    x=np.stack([np.r_[g,obs['tick']/1800,env.decision_count/300,dq.action_encoding(a,obs['goal']['target_id'],angle),sup,bits]
                for a,sup in zip(pool,support)]).astype('float32')
    return pool,x


def bank_model(dim):
    return nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))


@torch.inference_mode()
def episode(job):
    mode,k,fixture,path,teacher=job
    if path.exists():return
    mem=[m.initial_memory() for m in MODULES]
    route=[];panels=[];targets=[]
    with runtime.runtime.world(fixture) as (env,_):
        while not runtime.runtime.ended(env):
            if not env.decision_count:
                obs=dq.current_event(env.history)['observation']
                t=obs['goal']['target_id']
                pts=next((e['points'] for e in obs['entities'] if e['id']==t and e['points']),[])
                xy=np.mean(pts,0) if pts else np.mean([[r['own']['x'],r['own']['y']] for r in obs['robots']],0)
                offset=np.asarray(obs['goal']['xy'])-xy
                angle=float(np.arctan2(offset[1],offset[0]))
            pool,x=panel(env,mem,angle,False if mode=='single' else mode)
            if mode=='single':a=pool[0];route.append(a)
            elif mode=='teacher':
                a=teacher['actions'][env.decision_count];idx=pool.index(a)
                if x is not None:
                    pad=np.full((4,x.shape[1]),np.nan,dtype=np.float32);pad[:len(x)]=x
                    panels.append(pad);targets.append(idx)
            elif x is None:a=pool[0]
            else:
                model,mean,sd=MODELS[mode]
                val=model(torch.from_numpy((x-mean)/sd)).flatten().numpy()
                assert np.isfinite(val).all()
                a=pool[np.argmax(val)]
            runtime.runtime.advance(env,a)
        y=runtime.terminal(env)
        res=dict(success=y,ticks=env.env.total_periods,decisions=env.decision_count,history=env.history.sha256)
    path.parent.mkdir(exist_ok=True,parents=True)
    if mode=='teacher':
        assert res['history']==teacher['history'] and y
        np.savez_compressed(path,x=np.array(panels,dtype=np.float32),y=np.array(targets,dtype=np.int64))
    else:
        if mode=='single' and teacher=='dev':res['actions']=route
        save(path,res)


def jobs_run(jobs,sources,stage):
    jobs=[j for j in jobs if not j[3].exists()]
    if not jobs:return
    init(sources)
    with ProcessPoolExecutor(min(int(os.environ.get('SEM23_WORKERS','14')),len(jobs)),mp_context=mp.get_context('fork')) as p:
        for i,_ in enumerate(p.map(episode,jobs,chunksize=1),1):
            if i%10==0 or i==len(jobs):
                progress=dict(stage=stage,done=i,total=len(jobs))
                save(OUT/'progress.json',progress)
                save(OUT/(stage.split()[0]+'_progress.json'),progress)


def singles(arm,members,dev,test):
    paths=[]
    for n,m in enumerate(members):
        src=Path(m['path']).read_text();assert base.digest(src)==m['sha']
        root=OUT/'singles'/m['sha']
        root.mkdir(exist_ok=True,parents=True)
        jobs=[]
        for name,db in [('dev',dev),('test',test)]:
            for k,f in db.items():
                p=root/(k+'.json')
                if p.exists():continue
                archived=(study.previous.OUT/'runs/s025' if n==0 else OUT/arm/'compact'/m['id'])/(k+'.json')
                if name=='dev' and archived.exists():
                    rec=read(archived);assert rec['complete'] and not rec['error']
                    with gzip.open(archived.with_suffix('.json.gz'),'rt') as stream:actions=[row['a'] for row in json.load(stream)]
                    save(p,dict(success=int(rec['success']),ticks=rec['ticks'],decisions=rec['decisions'],history=rec['history'],actions=actions))
                else:jobs.append(('single',k,f,p,name))
        jobs_run(jobs,[src],arm+f' single {n}')
        paths.append(root)
    return paths


def train(arm,directory):
    chunks=[np.load(p) for p in sorted(directory.glob('*.npz'))]
    x=np.concatenate([c['x'] for c in chunks if len(c['y'])]);y=torch.from_numpy(np.concatenate([c['y'] for c in chunks if len(c['y'])]))
    assert np.isfinite(x[:,0]).all()
    mask=np.isfinite(x[:,:,0])
    valid=torch.from_numpy(mask)
    for mode in ('basic','semantic'):
        v=x.copy()
        if mode=='basic':v[:,:,-len(RULES):]=0
        values=v[mask]
        mu=values.mean(0);sd=np.maximum(.1,values.std(0));z=torch.from_numpy(np.nan_to_num((v-mu)/sd))
        torch.manual_seed(17);model=bank_model(z.shape[-1]);opt=torch.optim.Adam(model.parameters(),lr=.0003)
        for i in range(8000):
            ix=torch.randint(len(y),(64,))
            logits=model(z[ix]).squeeze(-1).masked_fill(~valid[ix],-1e9)
            loss=nn.functional.cross_entropy(logits,y[ix]);opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():acc=(model(z).squeeze(-1).masked_fill(~valid,-1e9).argmax(1)==y).float().mean().item()
        MODELS[mode]=(model.eval(),mu,sd)
        torch.save(dict(state=model.state_dict(),mean=torch.from_numpy(mu),scale=torch.from_numpy(sd)),OUT/f'{arm}_{mode}.pt')
        save(OUT/f'{arm}_{mode}_fit.json',dict(accuracy=acc,boundaries=len(y),rooms=len(chunks)))


def run_arm(arm):
    dev,test=read(OUT/'development.json'),read(OUT/'test.json')
    members=seal(arm);roots=singles(arm,members,dev,test)
    src=[Path(m['path']).read_text() for m in members]
    arrays=OUT/'features'/arm;arrays.mkdir(exist_ok=True,parents=True)
    jobs=[]
    for k,f in dev.items():
        opts=[read(p/(k+'.json')) for p in roots]
        wins=[r for r in opts if r['success']]
        if wins:jobs.append(('teacher',k,f,arrays/(k+'.npz'),min(wins,key=lambda r:r['ticks'])))
    jobs_run(jobs,src,arm+' extract')
    init(src);train(arm,arrays)
    for mode in ('basic','semantic'):
        jobs_run([(mode,k,f,OUT/'evaluation'/arm/mode/(k+'.json'),None) for k,f in test.items()],src,arm+' '+mode)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='panel':freeze_panel()
    elif sys.argv[1]=='basis':
        init([(study.previous.CAM/'semantics/s025.py').read_text()])
        singles('usual',[dict(id='basis',path=str(study.previous.CAM/'semantics/s025.py'),
                              sha=base.digest((study.previous.CAM/'semantics/s025.py').read_text()))],
                read(OUT/'development.json'),read(OUT/'test.json'))
    else:
        import fcntl
        arm=sys.argv[1]
        assert arm in ARMS
        with (OUT/(arm+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            done=OUT/(arm+'_evaluated.json')
            if not done.exists():
                run_arm(arm)
                save(done,dict(finished=True))
