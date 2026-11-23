"""Continue the pre-existing two rounds and confirm the fixed sixth checkpoint."""
from pathlib import Path
import sys
import numpy as np
from tools import cap_union_iteration as p

d,e=p.d,p.e
PARENT=p.OUT
OUT=PARENT.parent/'cap_union_trend_20261123_01'
DOC=p.DOC.parent/'2026-11-23_union_trend'
FRESH=662316887


def load(stage):
    p.OUT=PARENT if stage<=2 else OUT
    p.load_stage(stage)
    p.OUT=OUT


def train():
    d.init(profile=e.perception.LEGACY)
    p.torch.set_num_threads(1)
    cases=p.dev()
    union=e.read(PARENT/'union.json')
    assert len(union)==271
    assert e.read(PARENT/'lock.json')['final']==d.source.graph.file_sha(PARENT/'actor2.npz')
    spec=dict(parent=e.read(PARENT/'lock.json'),development=union,main_comparison=[6,2],secondary=[6,0],
              progress_panel=p.TEST_SEED,confirmation_seed=FRESH,rounds=[3,4,5,6],train_script=d.source.graph.file_sha(p.__file__),
              interventions='unchanged; original rooms, sampling, optimizer and budgets; no symmetry augmentation or model selection')
    path=OUT/'protocol.json'
    if path.exists():assert e.read(path)==spec
    else:e.save(path,spec)
    for stage in range(3,7):
        load(stage-1)
        if (OUT/f'fit{stage}.json').exists():
            load(stage)
            continue
        e.pool(p.fork_one,[(stage,k,cases[k]) for k in union],18)
        p.fit(stage,[OUT/f'round{stage}'/(k+'.npz') for k in sorted(union)])
    e.save(OUT/'lock.json',{i:d.source.graph.file_sha(OUT/f'actor{i}.npz') for i in range(3,7)})


def run_eval(job):
    fold,k,f=job
    path=OUT/fold/(k+'.json')
    if path.exists():return
    if fold=='diagnostic':
        old=e.read(PARENT/'test'/(k+'.json'))
        row=dict(A=old['A'],B=old['B'],r0=old['original'],r2=old['final'])
        stages=(1,3,4,5,6)
    else:
        row={m:e.simulate(f,m,'stall')['result']['success'] for m in ('A','B')}
        stages=(0,2,6)
    for stage in stages:
        key=f'r{stage}'
        row[key]=d.episode(f,key,label=False)[0]['success']
    e.save(path,row)


def evaluate():
    d.init(profile=e.perception.LEGACY)
    assert all(d.source.graph.file_sha(OUT/f'actor{i}.npz')==v for i,v in e.read(OUT/'lock.json').items())
    cases=p.dev()
    for stage in range(7):
        load(stage)
        d.MODELS[f'r{stage}']=d.MODELS['semantic']
    e.pool(p.dev_eval,[(k,cases[k]) for k in e.read(PARENT/'union.json')],18)
    diag=d.rooms(p.TEST_SEED)
    ids=lambda pool:{k:e.s.phase.input_hash(f) for k,f in pool.items()}
    assert ids(diag)==e.read(PARENT/'fresh_ids.json')
    e.pool(run_eval,[('diagnostic',k,f) for k,f in diag.items()],18)
    fresh=d.rooms(FRESH)
    known=e.s.collect_identities()
    for caseset in e.read(d.OUT/'protocol.json')['rooms']:
        known.update(caseset.values())
    for path in (PARENT/'fresh_ids.json',d.source.OUT/'confirmation_ids.json',e.OUT/'confirmation_ids.json',e.old.OUT/'fresh_panel.json'):
        known.update(e.read(path).values())
    for caseset in e.read(e.old.previous.OUT/'panels.json')['rooms'].values():
        known.update(caseset.values())
    for name in ('development','held'):
        known.update(e.s.phase.input_hash(f) for f in e.read(e.s.OUT/(name+'.json')).values())
    for seed in (662315862,662315879,e.g.prep.SEED):
        known.update(ids(d.rooms(seed)).values())
    fresh_ids=ids(fresh)
    assert len(set(fresh_ids.values()))==200 and not known.intersection(fresh_ids.values())
    e.save(OUT/'confirmation_ids.json',fresh_ids)
    e.pool(run_eval,[('confirmation',k,f) for k,f in fresh.items()],18)


if __name__=='__main__':
    {'train':train,'evaluate':evaluate}[sys.argv[1]]()
