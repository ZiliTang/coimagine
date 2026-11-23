"""Confirm the frozen opening semantic router without seeing the new room outcomes."""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from tools import cap_initial_specialization_probe as probe

EXP=probe.study
OUT=EXP.OUT/'initial_confirmation'
DOC=EXP.DOC/'initial_confirmation'
SEEDS=(662314126,662314127)
FEATURES=('basic','evidence')
PLANS=('single',*FEATURES)


def freeze():
    protocol=EXP.read(EXP.OUT/'protocol.json')
    EXP.RULES=protocol['rules']
    dev=EXP.read(EXP.OUT/'development.json')
    assert EXP.phase.cap.digest(dev)==protocol['dev']
    known=EXP.collect_identities()
    known.update(EXP.phase.input_hash(f) for p in ('held','development') for f in EXP.read(EXP.OUT/(p+'.json')).values())
    pace=EXP.OUT/'pacing'/'rooms.json'
    known.update(EXP.phase.input_hash(f) for f in EXP.read(pace).values())
    held={}
    for i,seed in enumerate(SEEDS):
        for r in EXP.phase.cap.rooms.generate(seed)['cases']:
            room=f'confirm{i}_{r["id"]}'
            assert EXP.phase.input_hash(r['fixture']) not in known
            known.add(EXP.phase.input_hash(r['fixture']))
            held[room]=r['fixture']
    assert len(held)==400
    ids={p:sorted(a) for p,a in [('dev',dev),('held',held)]}
    data={p:[probe.features(rows[k]) for k in ids[p]] for p,rows in [('dev',dev),('held',held)]}
    x={p:{a:np.stack([r[a] for r in d]) for a in FEATURES} for p,d in data.items()}
    selections={}
    for ex in EXP.CONTROLLERS:
        selections[ex]={}
        for bank,members in protocol['libraries'].items():
            y=np.stack([[EXP.read(EXP.singles_path(ex,m['sha'],k,'development'))['success'] for k in ids['dev']]
                        for m in members],axis=1)
            best=int(y.sum(0).argmax())
            vals={'single':np.full(len(held),best)}
            for arm in FEATURES:
                vals[arm]=probe.fit_linear(x['dev'][arm],y,x['held'][arm]).argmax(1)
            selections[ex][bank]={a:{k:members[int(j)]['sha'] for k,j in zip(ids['held'],values)} for a,values in vals.items()}
    lock=dict(seeds=SEEDS,held=EXP.phase.cap.digest(held),frozen_development=protocol['dev'],
              classifier=hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
              candidate_rule='development fixed best; same linear probabilities for basic and full evidence as first panel',
              primary='mean evidence gain in specialized libraries, equal controllers and paired by room',
              comparison='specialized versus usual difference in evidence gain; new labels not used for selection',
              selection_sha=EXP.phase.cap.digest(selections))
    for name,data in [('protocol',lock),('held',held),('selections',selections)]:
        path=OUT/(name+'.json')
        if path.exists():assert EXP.phase.cap.digest(EXP.read(path))==EXP.phase.cap.digest(data)
        else:EXP.save(path,data)
    return held,selections


def path(ex,member,room):
    return OUT/'runs'/ex/member/(room+'.json')


def summarize(held,selections):
    ids=sorted(held)
    panels=np.array([k[:8] for k in ids])
    types=np.array([f'{p}/{len(held[k]["movable_objects"])+1}' for p,k in zip(panels,ids)])
    result={}
    evidence={}
    rows=[]
    con=lambda d:probe.report.contrast(d,types,panels)
    for ex,banks in selections.items():
        result[ex]={}
        for bank,arms in banks.items():
            cells={}
            for arm,assign in arms.items():
                rr=[EXP.read(path(ex,assign[k],k)) for k in ids]
                assert all(0<r['ticks']<=1800 and 0<r['decisions']<=300 for r in rr)
                cells[arm]=np.array([r['success'] for r in rr])
                rows.extend(dict(room=k,executor=ex,bank=bank,feature=arm,member=assign[k],success=r['success'],ticks=r['ticks'])
                            for k,r in zip(ids,rr))
            d=cells['evidence']-cells['basic']
            evidence[ex,bank]=d
            result[ex][bank]=dict(wins={k:int(y.sum()) for k,y in cells.items()},
                                  evidence_vs_basic=con(d),evidence_vs_single=con(cells['evidence']-cells['single']),
                                  rescue=int(sum(d>0)),loss=int(sum(d<0)),
                                  revised_choices=sum(arms['evidence'][k]!=arms['basic'][k] for k in ids))
    a,b=EXP.BANKS
    common={bank:np.stack([evidence[ex,bank] for ex in EXP.CONTROLLERS]).mean(0) for bank in EXP.BANKS}
    result['averages']={bank:con(d) for bank,d in common.items()}
    result['library_interaction']=con(common[b]-common[a])
    DOC.mkdir(exist_ok=True,parents=True)
    with (DOC/'rooms.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    EXP.save(DOC/'results.json',result)
    EXP.save(OUT/'FINISHED.json',result)
    print(result,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=5)
    args=p.parse_args()
    torch.set_num_threads(1)
    held,selected=freeze()
    EXP.initialize()
    by={}
    for ex,banks in selected.items():
        for arms in banks.values():
            for plans in arms.values():
                for k,sid in plans.items():
                    by[ex,sid,k]=(ex,sid,k,held[k],'confirmation',path(ex,sid,k))
    original=EXP.save
    def save(p,data):
        if p.name=='progress.json':p=OUT/'progress.json'
        original(p,data)
    EXP.save=save
    EXP.execute(EXP.single,list(by.values()),'independent opening-route confirmation',args.workers)
    summarize(held,selected)


if __name__=='__main__':main()
