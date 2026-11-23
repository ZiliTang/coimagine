"""Matched ablation of 1/N success targets for the regional expert V."""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tools import cap_regional_extension_run as study
from tools import cap_regional_unique_v as diagnostic
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import Store
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import all_v,load_checkpoint

OUT=study.OUT.parent/'cap_regional_inverse_v_20261120_01'
DOCS=study.DOCS.parent/'2026-11-20_regional_inverse_v'
BASE=study.OUT/'directq'
MODEL=OUT/'models/GA_s17'
read,save=study.read,study.save


def rows_at(path):
    with path.open() as f:return list(csv.DictReader(f))


def fit():
    OUT.mkdir(parents=True,exist_ok=True)
    a_store=Store(BASE/'arrays/train');a=a_store.arrays
    roots=a['root'];ids=np.flatnonzero(a['boundary']==0)
    labels=a['y'];sids=read(study.OUT/'frozen.json')['sids']
    matrix={r['case_id']:r for r in rows_at(study.DOCS/'per_scene.csv') if r['panel'] in ('dev','aug')}
    scaled=np.empty_like(labels)
    distribution=Counter()
    for cid,r in matrix.items():
        root=int(cid.split('_')[1])*100+int(cid.split('_')[2])
        selected=roots==root;indices=ids[selected[ids]]
        indices=indices[np.argsort(a['cap'][indices])]
        truth=np.array([int(r[s]) for s in sids])
        np.testing.assert_array_equal(a['cap'][indices],np.arange(len(sids)))
        np.testing.assert_array_equal(labels[selected],truth[a['cap'][selected]])
        n=int(truth.sum());distribution[n]+=1
        scaled[selected]=labels[selected]/max(1,n)
    assert len(matrix)==len(np.unique(roots))==215 and sum(distribution.values())==215
    spec=dict(model='ten regional experts; GA; seed17',v_success='1/N by training episode initial room, N successful experts among 10',
              v_failure=0,q_terminal='observed success 1 or failure 0',q_nonterminal='max legal scaled V at actual successor',
              topology='64x64',batch=64,v_updates=16000,q_updates=8000,lr=.0003,sampling='uniform decision boundaries',
              training_boundaries=len(a_store),histogram={str(k):v for k,v in sorted(distribution.items())},
              base_q=study.sha(BASE/'models/GA_s17/offline.pt'),base_v=study.sha(BASE/'models/GA_s17/V.pt'),
              frozen_panels=read(study.OUT/'frozen.json')['held'],primary='singleton fresh: initial V top-1 and directQ closed-loop successes',
              secondary='train, development variants, held residual variants; overall performance; top3 rank; unique winner first-action agreement')
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==spec
    else:save(OUT/'protocol.json',spec)
    for fn in ['normalizers.npz','dataset.json']:
        if not (OUT/fn).exists():(OUT/fn).symlink_to(BASE/fn)
    save(OUT/'stage.json',dict(stage='fit'))
    study.dq.full_learner.train_offline(OUT,'GA',17,train_store=a_store,evaluate_validation=False,v_targets=scaled)
    new=np.load(MODEL/'Q_init_targets.npy',mmap_mode='r')
    np.testing.assert_array_equal(new[a['done']],a['reward'][a['done']])


def evaluate():
    dq=study.dq;ext=study.ext
    frozen=read(study.OUT/'frozen.json')
    dev=read(study.OUT/ext.DEV);held=read(study.OUT/'held.json')
    assert ext.old.digest(dev)==frozen['development'] and ext.old.digest(held)==frozen['held']
    fixtures=dev|held
    dq.OUT=OUT;study.init_models(frozen['sids'])
    dq.MODEL=load_checkpoint(MODEL/'offline.pt')[0]
    for p in ('dev','aug','fresh','near'):
        group=[(k,f,'inverse_v',None,None,p) for k,f in fixtures.items() if k.startswith(p+'_')]
        save(OUT/'stage.json',dict(stage='evaluate',panel=p))
        dq.batch(group,16,'evaluate_'+p)
    return fixtures


if __name__=='__main__':
    torch.set_num_threads(1)
    try:
        fit()
        fixtures=evaluate()
        from tools.cap_regional_inverse_v_report import report
        report(fixtures)
    except Exception as exc:
        save(OUT/'stage.json',dict(stage='error',error=repr(exc)));raise
