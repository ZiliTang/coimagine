"""Compare public inputs for imitation of the frozen two-CAP router."""
from pathlib import Path
import csv
import json
import os
import sys

import numpy as np
import torch
from torch import nn

from tools import cap_authored_router as source
from tools.cap_option_handoff_opportunity_report import paired
from experiments.planar_tasks.relational_search.motion import ACTIONS

e = source.e
OUT = e.OUT.parent/'cap_router_distill_20261123_01'
DOC = e.DOC.parent/'2026-11-23_router_distill'
ARMS = ('current','history','semantic','relabel')
SEEDS = (662316481,662316517)
UPDATES = 4000
MODELS = {}


def init(*, profile=e.perception.DEFAULT):
    global TEACHER
    version = source.initialize(profile=profile)
    freeze = e.read(source.OUT/'frozen.json')
    assert e.base.host.RoomCampaign(source.CAM).verify_frozen() == freeze['seal']
    path = source.CAM/f"router_v{freeze['selection']['revision']}.py"
    assert source.graph.file_sha(path) == freeze['router_sha']
    TEACHER = source.load_router(path.read_text())
    return version


CODE = np.eye(len(ACTIONS),dtype=np.float32)


class PublicHistory:
    def __init__(self):
        self.mode = 'A'
        self.started = self.advanced = 0
        self.trigger = None
        self.mark = None

    def vector(self, obs, a, b, c):
        tick, gap = c['tick'], c['gap']
        if c['triggered'] and self.trigger is None:
            self.trigger = tick
        if gap >= 0 and (self.mark is None or gap < self.mark-.14):
            self.mark,self.advanced = gap,tick
        geom = [tick/1800, gap/3, c['touch'], *np.ravel(c['xy'])/5,
                *np.asarray(obs['goal']['xy'])/5,
                *(int(r['contacting_static']) for r in obs['robots'])]
        target = obs['goal']['target_id']
        pair_a = (a['a0'],a['a1'])
        pair_b = (b['a0'],b['a1'])
        commands = [x for cmd in (a,b) for x in (cmd['ticks']/50,cmd['target_id'] == target,cmd['target_id'] is None)]
        base = np.concatenate([geom,commands,*(CODE[i] for i in (*pair_a,*pair_b))])
        history = [float(c['previous']=='B'),float(self.mode=='B'),(tick-self.started)/500,
                   (tick-self.advanced)/500,(tick-self.trigger)/500 if self.trigger is not None else 0,
                   c['progress'] is not None,c['progress'] or 0]
        same_drive = any(x<32 and y<32 and x//4==y//4 for x,y in zip(pair_a,pair_b))
        semantic = [pair_a==pair_b,b['ticks']<a['ticks'],(b['ticks']-a['ticks'])/50,
                    min(pair_a)<32,min(pair_b)<32,same_drive,44 in pair_a,46 in pair_b,
                    same_drive and 44 in pair_a and 46 in pair_b]
        return np.asarray([*base,*history,*semantic],np.float32)

    def executed(self,chosen,a,b,c):
        # Infer sustained control from the last actually selected change of command pair.
        if (a['a0'],a['a1'])!=(b['a0'],b['a1']) and chosen!=self.mode:
            self.mode=chosen
            self.started=self.advanced=c['tick']
            self.mark=c['gap'] if c['gap']>=0 else None


def project(x,arm):
    v=x.copy()
    if arm=='current':
        v[..., -16:]=0
    elif arm=='history':
        v[..., -9:]=0
    return v


def logits(x,arm):
    w1,b1,w2,b2,w3,b3=MODELS[arm]
    return (np.maximum(0,np.maximum(0,project(x,arm)@w1+b1)@w2+b2)@w3+b3)[...,0]


def episode(f, actor=None, label=True):
    shadow={}
    hist=PublicHistory()
    xs,ys=[],[]
    def choose(unused,obs,a,b,c):
        target=TEACHER(shadow,obs,a,b,c) if label else None
        x=hist.vector(obs,a,b,c)
        if c['triggered'] and a!=b:
            who=target if actor is None else ('B' if logits(x,actor)>0 else 'A')
            if label:
                xs.append(x)
                ys.append(int(target=='B'))
        else:
            # Canonical equal-command execution avoids leaking the teacher's expert ID.
            who='A'
        hist.executed(who,a,b,c)
        return who
    source.ROUTE=choose
    row=source.play(f)
    return row['result'],xs,ys


def rooms(seed):
    return {r['id']:r['fixture'] for r in e.s.phase.cap.rooms.generate(seed)['cases']}


def prepare():
    init(profile=e.perception.LEGACY)
    development=e.dev()
    pools=[development,*[rooms(seed) for seed in SEEDS]]
    known=e.s.collect_identities()
    for path in (source.OUT/'confirmation_ids.json',e.OUT/'confirmation_ids.json',e.old.OUT/'fresh_panel.json'):
        known.update(e.read(path).values())
    for old in e.read(e.old.previous.OUT/'panels.json')['rooms'].values():
        known.update(old.values())
    for name in ('development','held'):
        known.update(e.s.phase.input_hash(f) for f in e.read(e.s.OUT/(name+'.json')).values())
    for seed in (662315862,662315879,e.g.prep.SEED):
        known.update(e.s.phase.input_hash(r['fixture']) for r in e.s.phase.cap.rooms.generate(seed)['cases'])
    identities=[]
    for idx,pool in enumerate(pools):
        ids={k:e.s.phase.input_hash(f) for k,f in pool.items()}
        assert len(set(ids.values()))==200
        if idx:
            assert not known.intersection(ids.values())
        known.update(ids.values())
        identities.append(ids)
    e.save(OUT/'protocol.json',dict(rooms=identities,seeds=SEEDS,arms=ARMS,updates=UPDATES,seed=17,
             teacher=e.read(source.OUT/'frozen.json'),primary='semantic vs teacher; report paired interval, no prespecified noninferiority margin',
             labels='after public trigger, physically distinct commands only',intervention='pilot semantic visits same second pool; teacher labels observed prefixes; no reward update'))
    DOC.mkdir(exist_ok=True,parents=True)
    e.pool(collect,[(fold,k,f) for fold,pool in zip(('base','extra'),pools[:2]) for k,f in pool.items()],18)


def collect(job):
    fold,k,f=job
    path=OUT/fold/(k+'.npz')
    if path.exists():
        return
    result,x,y=episode(f,'pilot' if fold=='relabel' else None)
    if fold=='base' and (source.OUT/'probes/v3'/(k+'.json')).exists():
        ref=e.read(source.OUT/'probes/v3'/(k+'.json'))['result']
        assert result['history']==ref['history']
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pending.npz')
    np.savez_compressed(temp,x=np.asarray(x,np.float32),y=np.asarray(y,np.int64),success=result['success'])
    os.replace(temp,path)


def dataset(folds):
    xs,ys=[],[]
    for fold in folds:
        files=sorted((OUT/fold).glob('*.npz'))
        assert len(files)==200
        for path in files:
            with np.load(path) as d:
                if len(d['y']):
                    xs.append(d['x'])
                    ys.append(d['y'])
    return np.concatenate(xs),np.concatenate(ys)


def train(arm,folds):
    x,y=dataset(folds)
    x=torch.from_numpy(project(x,arm))
    labels=torch.from_numpy(y.astype(np.float32))
    torch.manual_seed(17)
    net=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    optim=torch.optim.Adam(net.parameters(),lr=.001)
    for _ in range(UPDATES):
        batch=torch.randint(len(labels),(128,))
        loss=nn.functional.binary_cross_entropy_with_logits(net(x[batch]).flatten(),labels[batch])
        optim.zero_grad()
        loss.backward()
        optim.step()
    weights=[p.detach().cpu().numpy().T.copy() for p in net.parameters()]
    MODELS[arm]=weights
    dest=OUT/'fit'
    dest.mkdir(exist_ok=True,parents=True)
    np.savez(dest/(arm+'.npz'),*weights)
    truth=logits(x.numpy(),arm)>0
    e.save(dest/(arm+'.json'),dict(boundaries=len(y),positives=int(y.sum()),correct=int(sum(truth==y)),
                                  sha=source.graph.file_sha(dest/(arm+'.npz'))))


def load(arm):
    path=OUT/'fit'/(arm+'.npz')
    assert source.graph.file_sha(path)==e.read(path.with_suffix('.json'))['sha']
    with np.load(path) as d:
        MODELS[arm]=[d[k] for k in d.files]


def fit():
    init(profile=e.perception.LEGACY)
    torch.set_num_threads(1)
    train('pilot',['base'])
    e.pool(collect,[('relabel',k,f) for k,f in rooms(SEEDS[0]).items()],18)
    for arm in ARMS:
        train(arm,['base','relabel' if arm=='relabel' else 'extra'])
    e.save(OUT/'lock.json',{arm:e.read(OUT/'fit'/(arm+'.json')) for arm in ARMS})


def test(job):
    k,f=job
    path=OUT/'test'/(k+'.json')
    if path.exists():return
    results={}
    for arm in ('A','B'):
        results[arm]=e.simulate(f,arm,'stall')['result']['success']
    teacher,x,y=episode(f)
    results['teacher']=teacher['success']
    labels=np.asarray(y)
    matches={}
    for arm in ARMS:
        correct=int(np.sum((logits(np.stack(x),arm)>0)==labels)) if x else 0
        matches[arm]=correct
        actual=episode(f,arm,label=False)[0]
        if correct==len(y):
            assert actual['history']==teacher['history']
        results[arm]=actual['success']
    e.save(path,dict(outcomes=results,n=len(y),correct=matches))


def evaluate():
    init(profile=e.perception.LEGACY)
    for arm in ARMS:
        load(arm)
    pool=rooms(SEEDS[1])
    assert e.read(OUT/'protocol.json')['rooms'][2]=={k:e.s.phase.input_hash(f) for k,f in pool.items()}
    e.pool(test,pool.items(),18)


if __name__=='__main__':
    {'collect':prepare,'fit':fit,'test':evaluate}[sys.argv[1]]()
