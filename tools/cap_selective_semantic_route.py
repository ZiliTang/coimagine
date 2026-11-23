"""Development-calibrated semantic override of the strongest specialized CAP."""
import argparse
from collections import defaultdict

import numpy as np
import torch

from tools import cap_initial_specialization_confirm as confirm
from tools import cap_initial_specialization_probe as probe
from tools import cap_option_semantic_factorial_report as report

EXP=confirm.EXP
OUT=EXP.OUT/'selective_semantics'
DOC=EXP.DOC/'selective_semantics'
GRID=(0.,.025,.05,.075,.10,.15,.20,.30,1.)
SEEDS=(662314128,662314129)


def prepare():
    protocol=EXP.read(EXP.OUT/'protocol.json')
    EXP.RULES=protocol['rules']
    dev=EXP.read(EXP.OUT/'development.json')
    proof=EXP.read(confirm.OUT/'held.json')
    used=EXP.collect_identities()
    for panel in (dev,proof,EXP.read(EXP.OUT/'held.json'),EXP.read(EXP.OUT/'pacing/rooms.json')):
        used.update(EXP.phase.input_hash(f) for f in panel.values())
    fresh={}
    for j,seed in enumerate(SEEDS):
        for r in EXP.phase.cap.rooms.generate(seed)['cases']:
            fixture=r['fixture']
            signature=EXP.phase.input_hash(fixture)
            assert signature not in used
            used.add(signature)
            fresh[f'select{j}_{r["id"]}']=fixture
    keys=sorted(dev)
    x=np.stack([probe.features(dev[k])['evidence'] for k in keys])
    target=np.stack([probe.features(fresh[k])['evidence'] for k in sorted(fresh)])
    rng=np.random.default_rng(29)
    folds=np.empty(len(keys),int)
    groups=defaultdict(list)
    for i,k in enumerate(keys):
        groups[len(dev[k]['movable_objects'])].append(i)
    for ids in groups.values():
        for i,row in enumerate(rng.permutation(ids)):
            folds[row]=i%5
    banks=protocol['libraries']['specialized']
    scores={};routes={}
    for ex in EXP.CONTROLLERS:
        y=np.stack([[EXP.read(EXP.singles_path(ex,m['sha'],k,'development'))['success'] for k in keys] for m in banks],1)
        estimate=np.empty_like(y,dtype=float)
        best=np.empty(len(keys),int)
        for fold in range(5):
            train=folds!=fold
            estimate[~train]=probe.fit_linear(x[train],y[train],x[~train])
            best[~train]=int(y[train].sum(0).argmax())
        i=np.arange(len(y))
        high=estimate.argmax(1)
        gap=estimate[i,high]-estimate[i,best]
        candidates={t:np.where(gap>=t,high,best) for t in GRID}
        threshold=max(GRID,key=lambda t:(int(y[i,candidates[t]].sum()),-int(np.sum(candidates[t]!=best)),t))
        best_final=int(y.sum(0).argmax())
        all_p=probe.fit_linear(x,y,target)
        high_final=all_p.argmax(1)
        best_final_vec=np.full(len(fresh),best_final)
        p_gap=all_p[np.arange(len(fresh)),high_final]-all_p[:,best_final]
        conditional=np.where(p_gap>=threshold,high_final,best_final_vec)
        scores[ex]=dict(threshold=threshold,oof_baseline=int(y[i,best].sum()),oof_selected=int(y[i,candidates[threshold]].sum()),
                        validation=[dict(threshold=t,wins=int(y[i,candidates[t]].sum()),
                        overrides=int(np.sum(candidates[t]!=best))) for t in GRID])
        routes[ex]={a:{k:banks[int(j)]['sha'] for k,j in zip(sorted(fresh),selection)} for a,selection in
                    [('single',best_final_vec),('always',high_final),('selective',conditional)]}
    manifest=dict(seeds=SEEDS,held=EXP.phase.cap.digest(fresh),routing_sha=EXP.phase.cap.digest(routes),
                  design='one expert for the full room; margins chosen in five development folds, ties preserve base',
                  planned='selective vs fixed and selective vs always applied evidence on untouched rooms',selection=scores)
    for name,obj in [('protocol',manifest),('held',fresh),('selections',routes)]:
        p=OUT/(name+'.json')
        if p.exists():assert EXP.phase.cap.digest(EXP.read(p))==EXP.phase.cap.digest(obj)
        else:EXP.save(p,obj)
    return fresh,routes,manifest


def run(held,routes,workers):
    EXP.initialize()
    jobs={}
    for ex,arms in routes.items():
        for selected in arms.values():
            for k,model in selected.items():
                path=OUT/'runs'/ex/model/(k+'.json')
                jobs[ex,k,model]=(ex,model,k,held[k],'selective',path)
    save=EXP.save
    def progress(p,obj):
        if p.name=='progress.json':p=OUT/'progress.json'
        save(p,obj)
    EXP.save=progress
    EXP.execute(EXP.single,list(jobs.values()),'selective semantic opening route',workers)


def report_out(held,routes,manifest):
    ids=sorted(held)
    panels=np.array([k[:7] for k in ids])
    strata=np.array([f'{g}/{len(held[k]["movable_objects"])+1}' for g,k in zip(panels,ids)])
    result={'development_selection':manifest['selection']}
    total={}
    for ex,cells in routes.items():
        vals={}
        for arm,selected in cells.items():
            vals[arm]=np.array([EXP.read(OUT/'runs'/ex/selected[k]/(k+'.json'))['success'] for k in ids])
        total[ex]=vals
        result[ex]=dict(wins={a:int(r.sum()) for a,r in vals.items()},
                         switch_count={a:sum(cells[a][k]!=cells['single'][k] for k in ids) for a in ('always','selective')},
                         vs_best=report.contrast(vals['selective']-vals['single'],strata,panels),
                         always_vs_best=report.contrast(vals['always']-vals['single'],strata,panels),
                         vs_always=report.contrast(vals['selective']-vals['always'],strata,panels))
    difference=np.mean([r['selective']-r['single'] for r in total.values()],0)
    result['selective_vs_best']=report.contrast(difference,strata,panels)
    DOC.mkdir(exist_ok=True,parents=True)
    EXP.save(DOC/'results.json',result)
    EXP.save(OUT/'FINISHED.json',result)
    print(result,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=5)
    a=p.parse_args()
    torch.set_num_threads(1)
    held,routes,proof=prepare()
    run(held,routes,a.workers)
    report_out(held,routes,proof)


if __name__=='__main__':main()
