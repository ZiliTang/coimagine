"""A room-level test of whether the authored semantics identify expert advantages."""
import csv
from functools import lru_cache

import numpy as np
import torch

from tools import cap_option_semantic_factorial as study
from tools import cap_option_semantic_factorial_report as report

DOC=study.DOC/'initial_routing'
OUT=study.OUT/'initial_routing'


def features(fixture):
    obs=study.old.base.public(fixture)
    c=study.corridor_semantics(obs)
    c['observed']=c['kind']
    tags=[study.lib.matches(r,c) for r in study.RULES]
    compact=study.old.base.geom(obs)
    # Keep the tag set identical to what the authors received.
    corridor=np.r_[compact,tags]
    evidence=np.r_[corridor,*[c[k] for k in ('target_estimated','static_hit','movable_hit',
                                             'swept_visibility','observed_clear','unknown',
                                             'visible_other_movables','goal_wall_distance_m')]]
    return dict(basic=compact,corridor=corridor,evidence=evidence)


def fit_linear(a,y,b):
    mu=a.mean(0)
    sd=np.maximum(.05,a.std(0))
    x=torch.tensor((a-mu)/sd,dtype=torch.float64)
    z=torch.tensor(y,dtype=torch.float64)
    w=torch.zeros((x.shape[1],y.shape[-1]),requires_grad=True,dtype=torch.float64)
    intercept=torch.zeros(y.shape[-1],requires_grad=True,dtype=torch.float64)
    opt=torch.optim.LBFGS([w,intercept],max_iter=100,tolerance_grad=1e-9,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(x@w+intercept,z)+.05*w.square().mean()
        loss.backward()
        return loss
    opt.step(closure)
    return (torch.tensor((b-mu)/sd,dtype=torch.float64)@w.detach()+intercept.detach()).sigmoid().numpy()


def main():
    torch.set_num_threads(1)
    p=study.read(study.OUT/'protocol.json')
    study.RULES=p['rules']
    sets={k:study.read(study.OUT/(k+'.json')) for k in ('development','held')}
    labels={k:sorted(x) for k,x in sets.items()}
    inputs={}
    for k,db in sets.items():
        extracted=[features(db[j]) for j in labels[k]]
        inputs[k]={arm:np.stack([r[arm] for r in extracted]) for arm in extracted[0]}
    held=labels['held']
    panels=np.array([k[:6] for k in held])
    strata=np.array([f'{k[:6]}/{1+len(sets["held"][k]["movable_objects"])}' for k in held])
    result={}
    rows=[]
    for ex in study.CONTROLLERS:
        result[ex]={}
        for bank,models in p['libraries'].items():
            groups={}
            for fold,rooms in labels.items():
                groups[fold]=np.array([[r['success'] for r in report.read_rows(
                           study.OUT/'singles'/fold/ex/m['sha'],rooms)] for m in models]).T
            y,z=groups['development'],groups['held']
            best=int(y.sum(0).argmax())
            grand=z[:,best]
            summary=dict(best=best,fixed=int(grand.sum()),union=int(z.any(1).sum()),arms={})
            selected={}
            for arm in inputs['development']:
                score=fit_linear(inputs['development'][arm],y,inputs['held'][arm])
                choice=score.argmax(1)
                a=z[np.arange(len(z)),choice]
                selected[arm]=a
                auc=[]
                for i in range(z.shape[1]):
                    win=z[:,i]==1
                    d=score[win,i,None]-score[None,~win,i]
                    auc.append(float(np.mean((d>0)+.5*(d==0))))
                # Among selectable differences only, score ordering of winning vs losing experts.
                pairs=[np.array([score[j,i]-score[j,k] for i in range(z.shape[1]) for k in range(z.shape[1])
                                 if z[j,i]>z[j,k]]) for j in range(len(z))]
                accuracies=[float(np.mean((d>0)+.5*(d==0))) for d in pairs if len(d)]
                summary['arms'][arm]=dict(wins=int(a.sum()),auc=auc,brier=float(np.mean((score-z)**2)),
                       mixed_label_rooms=len(accuracies),mean_pair_accuracy=float(np.mean(accuracies)),
                       vs_single=report.contrast(a-grand,strata,panels))
                if arm!='basic':summary['arms'][arm]['vs_basic']=report.contrast(a-selected['basic'],strata,panels)
                rows.extend(dict(room=k,executor=ex,bank=bank,feature=arm,choice=int(i),success=int(r))
                            for k,i,r in zip(held,choice,a))
            result[ex][bank]=summary
    DOC.mkdir(parents=True,exist_ok=True)
    with (DOC/'rooms.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    study.save(DOC/'results.json',result)
    study.save(OUT/'FINISHED.json',result)
    for ex,r in result.items():
        print(ex,{b:(v['fixed'],v['union'],{a:x['wins'] for a,x in v['arms'].items()}) for b,v in r.items()},flush=True)


if __name__=='__main__':main()
