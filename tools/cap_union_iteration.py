"""Paired one-command improvement of the frozen semantic actor on its development union."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from torch import nn
from tools import cap_router_distill as d

e=d.e
OUT=d.OUT.parent/'cap_union_iteration_20261123_01'
DOC=d.DOC.parent/'2026-11-23_union_iteration'
TEST_SEED=662316693
STAGES=(1,2)


def dev():
    known=e.read(d.OUT/'protocol.json')['rooms']
    pools=(e.dev(),d.rooms(d.SEEDS[0]))
    assert all({k:e.s.phase.input_hash(f) for k,f in p.items()}==known[i] for i,p in enumerate(pools))
    return {f'{i}_{k}':f for i,p in enumerate(pools) for k,f in p.items()}


def screen_one(job):
    k,f=job
    path=OUT/'screen'/(k+'.json')
    if path.exists():
        return
    archive=e.OUT/'train'/(k[2:]+'.json')
    if k.startswith('0_') and archive.is_file():
        row=e.read(archive)
        v={m:row[m]['result']['success'] for m in ('A','B')}
    else:
        v={m:e.simulate(f,m,'stall')['result']['success'] for m in ('A','B')}
    e.save(path,v)


def prepare():
    d.init(profile=e.perception.LEGACY)
    d.load('semantic')
    assert e.read(d.OUT/'lock.json')['semantic']['sha']==e.read(d.OUT/'fit/semantic.json')['sha']
    e.save(OUT/'protocol.json',dict(checkpoint=e.read(d.OUT/'lock.json')['semantic'],test_seed=TEST_SEED,
            round_count=2,train_pool='two original distillation development pools, filtered on A or B success',
            sampling='all union rooms; up to 3 different command pairs sampled across early, middle and late thirds of on-policy disagreements each round',
            target='one alternate command, actual terminal success with current actor for both suffixes; ties anchor only',
            fit='warm start all original semantic actor weights; 300 Adam steps 0.0003 per round; preference BCE plus current actor soft-label BCE on same round union states, coefficient 1; bounded deterministic sampling seed',
            primary='round 2 vs original actor, paired natural fresh rooms, no held checkpoint selection',
            secondary='union and complement; baseline singles and teacher on same panel; round 1 on development only'))
    e.pool(screen_one,dev().items(),18)
    DOC.mkdir(exist_ok=True,parents=True)


if __name__=='__main__' and sys.argv[1]=='prepare':
    prepare()


def load_stage(stage):
    if stage==0:
        d.load('semantic')
    else:
        path=OUT/f'actor{stage}.npz'
        assert d.source.graph.file_sha(path)==e.read(OUT/f'fit{stage}.json')['sha']
        with np.load(path) as archive:
            d.MODELS['semantic']=[archive[k] for k in archive.files]


def run_fork(f,reference=None,forced=None,*,comparison=None):
    history=d.PublicHistory()
    points=[]
    prefix=hashlib.sha256()
    matched=False
    def choose(unused,obs,a,b,c):
        nonlocal matched
        x=history.vector(obs,a,b,c)
        boundary=c['n']
        prefix.update(x.tobytes())
        prefix.update(json.dumps([a,b],sort_keys=True).encode())
        who='A'
        if c['triggered'] and a!=b:
            who='B' if d.logits(x,'semantic')>0 else 'A'
            if reference is None or boundary==forced:
                fingerprint=prefix.copy()
                fingerprint.update(json.dumps(obs,sort_keys=True).encode())
                key=fingerprint.hexdigest()
                if reference is None:
                    point=(boundary,x,int(who=='B'),key)
                    if comparison is None:
                        points.append(point)
                    else:
                        from experiments.planar_tasks.semantic_rl.sparse_branches import alternatives
                        for cmd, names in alternatives({'A':a,'B':b},who,comparison).items():
                            points.append((*point,cmd,names[0]))
                else:
                    assert key==reference[3] and np.array_equal(x,reference[1]) and int(who=='B')==reference[2]
                    if comparison is None:
                        who='A' if who=='B' else 'B'
                    else:
                        who=reference[5]
                        cmd={'A':a,'B':b}[who]
                        assert tuple(cmd[k] for k in ('target_id','a0','a1','ticks'))==tuple(reference[4])
                    matched=True
        prefix.update(who.encode())
        history.executed(who,a,b,c)
        return who
    d.source.ROUTE=choose
    row=d.source.play(f)
    assert reference is None or matched
    return (row['result'] if comparison is not None else row['result']['success']),points


def fork_one(job):
    stage,k,f=job
    path=OUT/f'round{stage}'/(k+'.npz')
    if path.exists():return
    baseline,points=run_fork(f)
    rng=np.random.default_rng(int(e.key(f'fork/{stage}/{k}')[:16],16))
    indices=np.asarray([int(rng.choice(block)) for block in np.array_split(np.arange(len(points)),3) if len(block)],int)
    alt=[]
    for idx in indices:
        ref=points[idx]
        other,_=run_fork(f,ref,ref[0])
        alt.append(other)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending.npz')
    np.savez_compressed(pending,x=np.array([p[1] for p in points],np.float32),b=np.array([p[2] for p in points],int),
                        index=indices,baseline=baseline,alternate=np.array(alt,int),boundaries=[points[i][0] for i in indices],
                        prefixes=[points[i][3] for i in indices])
    pending.replace(path)


def counts(paths):
    anchors,features,targets=[],[],[]
    totals=dict(rooms=0,baseline_success=0,forks=0,rescue=0,protect=0,ties=0,rooms_with_rescue=0,rooms_with_loss=0)
    for path in paths:
        with np.load(path) as q:
            x=q['x']
            if len(x):
                anchors.append(x)
            base=int(q['baseline'])
            totals['rooms']+=1
            totals['baseline_success']+=base
            totals['rooms_with_rescue']+=int(any(q['alternate']>base))
            totals['rooms_with_loss']+=int(any(q['alternate']<base))
            for idx,alt in zip(q['index'],q['alternate']):
                totals['forks']+=1
                if alt==base:
                    totals['ties']+=1
                    continue
                rescue=alt>base
                totals['rescue' if rescue else 'protect']+=1
                y=int(q['b'][idx])
                features.append(x[idx])
                targets.append(1-y if rescue else y)
    return np.concatenate(anchors),np.asarray(features),np.asarray(targets,np.float32),totals


def fit(stage,paths,*,retention_weights=None):
    anchors,x,y,totals=counts(paths)
    fit_preferences(stage,anchors,x,y,totals,retention_weights=retention_weights)


def fit_preferences(stage,anchors,x,y,totals,*,retention_weights=None,seed=17):
    current=d.MODELS['semantic']
    net=nn.Sequential(nn.Linear(anchors.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    with torch.no_grad():
        for param,v in zip(net.parameters(),current):
            param.copy_(torch.from_numpy(v.T.copy()))
    torch.manual_seed(seed+stage)
    optimizer=torch.optim.Adam(net.parameters(),lr=.0003)
    anchor=torch.tensor(anchors)
    with torch.no_grad():
        if retention_weights is None:
            original=net(anchor).flatten().sigmoid()
            assert np.all((original.numpy()>.5)==(d.logits(anchors,'semantic')>0))
        else:
            from copy import deepcopy
            reference=deepcopy(net)
            for param,value in zip(reference.parameters(),retention_weights,strict=True):
                param.copy_(torch.from_numpy(value.T.copy()))
            original=reference(anchor).flatten().sigmoid()
    if len(y):
        xx=torch.tensor(x)
        yy=torch.tensor(y)
        for _ in range(300):
            i=torch.randint(len(x),(64,))
            j=torch.randint(len(anchor),(128,))
            loss=nn.functional.binary_cross_entropy_with_logits(net(xx[i]).flatten(),yy[i])
            retention=nn.functional.binary_cross_entropy_with_logits(net(anchor[j]).flatten(),original[j])
            optimizer.zero_grad()
            (loss+retention).backward()
            optimizer.step()
    new=[p.detach().numpy().T.copy() for p in net.parameters()]
    d.MODELS['semantic']=new
    path=OUT/f'actor{stage}.npz'
    np.savez(path,*new)
    e.save(OUT/f'fit{stage}.json',dict(**totals,sha=d.source.graph.file_sha(path),anchors=len(anchor),informative=len(y),
                                      fit_correct=int(sum((d.logits(x,'semantic')>0)==y)) if len(y) else 0))


def iterate():
    d.init(profile=e.perception.LEGACY)
    torch.set_num_threads(1)
    rooms=dev()
    union={k:f for k,f in rooms.items() if any(e.read(OUT/'screen'/(k+'.json')).values())}
    e.save(OUT/'union.json',list(sorted(union)))
    for stage in STAGES:
        load_stage(stage-1)
        if (OUT/f'fit{stage}.json').exists():
            load_stage(stage)
            continue
        e.pool(fork_one,[(stage,k,f) for k,f in union.items()],18)
        paths=[OUT/f'round{stage}'/(k+'.npz') for k in sorted(union)]
        fit(stage,paths)
    e.save(OUT/'lock.json',dict(parent=e.read(d.OUT/'lock.json')['semantic']['sha'],final=e.read(OUT/'fit2.json')['sha']))


def dev_eval(job):
    k,f=job
    path=OUT/'dev_eval'/(k+'.json')
    if not path.exists():
        value=d.episode(f,'semantic',label=False)[0]['success']
        e.save(path,value)


def test_one(job):
    k,f=job
    path=OUT/'test'/(k+'.json')
    if path.exists():return
    result={m:e.simulate(f,m,'stall')['result']['success'] for m in ('A','B')}
    result['teacher']=d.episode(f)[0]['success']
    for arm in ('original','final'):
        result[arm]=d.episode(f,arm,label=False)[0]['success']
    e.save(path,result)


def evaluate():
    d.init(profile=e.perception.LEGACY)
    load_stage(2)
    assert e.read(OUT/'lock.json')['final']==d.source.graph.file_sha(OUT/'actor2.npz')
    fixtures=dev()
    union=e.read(OUT/'union.json')
    e.pool(dev_eval,[(k,fixtures[k]) for k in union],18)
    d.MODELS['final']=d.MODELS['semantic']
    d.load('semantic')
    d.MODELS['original']=d.MODELS['semantic']
    fresh=d.rooms(TEST_SEED)
    ids={k:e.s.phase.input_hash(f) for k,f in fresh.items()}
    known=e.s.collect_identities()
    for p in e.read(d.OUT/'protocol.json')['rooms']:
        known.update(p.values())
    for path in (d.source.OUT/'confirmation_ids.json',e.OUT/'confirmation_ids.json',e.old.OUT/'fresh_panel.json'):
        known.update(e.read(path).values())
    for p in e.read(e.old.previous.OUT/'panels.json')['rooms'].values():
        known.update(p.values())
    for name in ('development','held'):
        known.update(e.s.phase.input_hash(f) for f in e.read(e.s.OUT/(name+'.json')).values())
    for seed in (662315862,662315879,e.g.prep.SEED):
        known.update(e.s.phase.input_hash(r['fixture']) for r in e.s.phase.cap.rooms.generate(seed)['cases'])
    assert len(set(ids.values()))==200 and not known.intersection(ids.values())
    e.save(OUT/'fresh_ids.json',ids)
    e.pool(test_one,fresh.items(),18)


if __name__=='__main__' and sys.argv[1] in ('iterate','evaluate'):
    {'iterate':iterate,'evaluate':evaluate}[sys.argv[1]]()
