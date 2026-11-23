"""Replay matched optimizer steps and split their Adam first moments by loss."""
import csv
import itertools

import numpy as np
import torch
from tools import cap_success_replay as run

NAMES=("pref","soft","margin")
PAIRS=tuple(itertools.combinations(range(3),2))


def flat(parts):
    return torch.cat([p.detach().reshape(-1) for p in parts])


class GradientMeter:
    def __init__(self):
        self.rows=[]
        self.moments=None

    def before(self,losses,model,optim,margin_active):
        self.margin_active=margin_active
        params=list(model.parameters())
        self.previous=flat(params)
        vectors=[]
        for loss in losses:
            vectors.append(flat(torch.autograd.grad(loss,params,retain_graph=True))
                           if loss is not None else torch.zeros_like(self.previous))
        self.g=torch.stack(vectors)
        if self.moments is None:
            self.moments=torch.zeros_like(self.g)
        beta1=optim.param_groups[0]["betas"][0]
        self.moments.lerp_(self.g,1-beta1)
        self.losses=[float(x.detach()) if x is not None else 0 for x in losses]

    def after(self,model,optim):
        params=list(model.parameters())
        t=len(self.rows)+1
        group=optim.param_groups[0]
        beta1,beta2=group["betas"]
        assert not group["weight_decay"] and not group["amsgrad"]
        v=flat([optim.state[p]["exp_avg_sq"] for p in params])
        scale=(-group["lr"]/(1-beta1**t))/(v.sqrt()/np.sqrt(1-beta2**t)+group["eps"])
        steps=self.moments*scale
        step=steps.sum(0)
        actual=flat(params)-self.previous
        step2=float(step.square().sum())
        relative_error=float((actual-step).norm()/actual.norm())
        assert relative_error<.005, (t,relative_error)
        norms=self.g.norm(dim=1)
        row=dict(step=t,step_sq=step2,relative_error=relative_error,margin_active=self.margin_active)
        for i,name in enumerate(NAMES):
            row[name+"_loss"]=self.losses[i]
            row[name+"_norm"]=float(norms[i])
            row[name+"_step_dot"]=float(steps[i]@step)
            row[name+"_adam_share"]=row[name+"_step_dot"]/step2
        assert abs(sum(row[n+"_adam_share"] for n in NAMES)-1)<.0001
        for i,j in PAIRS:
            den=float(norms[i]*norms[j])
            row[NAMES[i]+"_"+NAMES[j]+"_cos"]=float(self.g[i]@self.g[j])/den if den>1e-12 else None
        self.rows.append(row)


def aggregate(rows):
    stats={"steps":len(rows),"reconstruction_max_relerr":max(r["relative_error"] for r in rows)}
    active=[r["margin_active"] for r in rows if r["margin_active"] is not None]
    stats["margin_active"]=float(np.mean(active)) if active else None
    energy=sum(r["step_sq"] for r in rows)
    for name in NAMES:
        stats[name]=dict(mean_norm=float(np.mean([r[name+"_norm"] for r in rows])),
                         max_norm=float(max(r[name+"_norm"] for r in rows)),
                         share=sum(r[name+"_step_dot"] for r in rows)/energy)
    for i,j in PAIRS:
        key=NAMES[i]+"_"+NAMES[j]+"_cos"
        v=[r[key] for r in rows if r[key] is not None]
        stats[key]=dict(mean=float(np.mean(v)) if v else None, negative=float(np.mean(np.array(v)<0)) if v else None,n=len(v))
    return stats


def main():
    torch.set_num_threads(1)
    plan=run.e.read(run.OUT/"protocol.json")
    lock=run.e.read(run.OUT/"locked.json")
    all_rows=[]
    results={}
    for mode in run.MODES[1:]:
        pooled=[]
        for seed in run.SEEDS:
            name=f"s{seed}_{mode}"
            assert lock[name]==run.d.source.graph.file_sha(run.OUT/"models"/(name+".npz"))
            data=run.training_data(seed,plan)
            meter=GradientMeter()
            run.fit(seed,mode,plan,data,meter=meter)
            pooled.extend(meter.rows)
            all_rows.extend(dict(seed=seed,arm=mode,**r) for r in meter.rows)
            results[name]=aggregate(meter.rows)
        results[mode]={period:aggregate([r for r in pooled if lo<=r["step"]<=hi]) for period,lo,hi in
                       (("all",1,300),("early",1,50),("middle",51,250),("late",251,300))}
    run.DOC.mkdir(exist_ok=True,parents=True)
    with (run.DOC/"gradients.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    run.e.save(run.DOC/"gradient_summary.json",results)


if __name__=="__main__":
    main()
