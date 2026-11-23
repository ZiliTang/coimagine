"""Replay frozen expanding-pool labels from matched warm and cold initializations."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import multiprocessing as mp
import sys

import numpy as np
import torch
from torch import nn
from tools import cap_expanding_pool as exp

p,d,e=exp.p,exp.d,exp.e
OUT=exp.OUT.parent/'cap_cold_start_20261123_01'
DOC=exp.DOC.parent/'2026-11-23_cold_start'
WEIGHTS={}


def weights(path):
    with np.load(path) as data:
        return [data[k] for k in data.files]


def fit_all():
    d.load('semantic')
    warm=d.MODELS['semantic']
    torch.manual_seed(17)
    dim=warm[0].shape[0]
    random_actor=nn.Sequential(nn.Linear(dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    cold=[v.detach().numpy().T.copy() for v in random_actor.parameters()]
    files={i:sorted((exp.stage_dir(i)/'fit_data').glob('*.npz')) for i in range(1,5)}
    for i,paths in files.items():
        assert [f.stem for f in paths]==sorted(v['room'] for v in e.read(exp.stage_dir(i)/'visits.json'))
    digest=hashlib.sha256()
    for i,paths in files.items():
        for path in paths:
            digest.update(f'{i}/{path.name}/{d.source.graph.file_sha(path)}\n'.encode())
    e.save(OUT/'protocol.json',dict(data_sha=digest.hexdigest(), seed=17,primary='cold self retention vs own start and warm final on same 200',
           warm_anchor_control='same cold weights, frozen warm pre-round soft labels',warm_start=e.read(exp.OUT/'protocol.json')['parent'],
           shape=[dim,64,64,1],stages=4,updates=300,hard_batch=64,anchor_batch=128))
    np.savez(OUT/'cold0.npz',*cold)
    WEIGHTS['cold0']=cold
    for mode in ('warm_replay','self','warm_anchor'):
        p.OUT=OUT/mode
        p.OUT.mkdir(exist_ok=True)
        d.MODELS['semantic']=warm if mode=='warm_replay' else cold
        for i,paths in files.items():
            anchor=None
            if mode=='warm_anchor':
                anchor=warm if i==1 else weights(exp.OUT/f'actor{i-1}.npz')
            p.fit(i,paths,retention_weights=anchor)
            new=d.MODELS['semantic']
            if mode=='warm_replay':
                expected=weights(exp.OUT/f'actor{i}.npz')
                assert all(np.array_equal(a,b) for a,b in zip(new,expected,strict=True))
            else:
                WEIGHTS[f'{mode}{i}']=new


def score(job):
    panel,k,models=job
    for model in models:
        path=OUT/panel/model/(k+'.json')
        if path.exists():
            continue
        d.MODELS['semantic']=WEIGHTS[model]
        r=d.episode(exp.ROOMS[k],'semantic',label=False)[0]
        e.save(path,dict(success=r['success'],ticks=r['ticks']))


def run():
    exp.setup()
    fit_all()
    endpoints=['cold0','self4','warm_anchor4']
    jobs=[]
    with (exp.DOC/'diagnostic64.csv').open() as f:
        diag=list(csv.DictReader(f))
    previous=d.rooms(exp.trend.FRESH)
    for row in diag:
        k=row['room_hash']
        assert k not in exp.ROOMS and e.s.phase.input_hash(previous[row['previous_room']])==k
        exp.ROOMS[k]=previous[row['previous_room']]
        jobs.append(('diagnostic',k,['cold0',*(f'{m}{i}' for i in range(1,5) for m in ('self','warm_anchor'))]))
    for panel,keys in [('held',exp.WAVES[5]),('train',sorted(exp.old_visits(5)))]:
        jobs.extend((panel,k,endpoints) for k in keys)
    with ProcessPoolExecutor(max_workers=exp.WORKERS,mp_context=mp.get_context('fork')) as pool:
        for _ in pool.map(score,jobs):
            pass


if __name__=='__main__':
    run()
