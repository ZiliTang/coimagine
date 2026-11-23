"""Parent-held-out probes and clustered uncertainty for the paired neighborhood experiment."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools import cap_regional_advantage as run

OUT = run.OUT
DOC = run.study.DOCS.parent/'2026-11-20_regional_advantage'
RADII = (0,1,2,5)
STAGES = ('initial','divergence','contact')
PARENTS = sorted(run.study.read(OUT/'panel.json'),key=lambda p:p['parent'])
PARENT_ID = {p['parent']:i for i,p in enumerate(PARENTS)}
RNG = np.random.default_rng(6820421)
BOOT = np.hstack([RNG.choice(group,(10000,len(group))) for n in range(2,6)
                  for group in [[i for i,p in enumerate(PARENTS) if p['objects']==n]]])


def dump_csv(filename,rows):
    with (DOC/filename).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ci(rows, gains):
    i=np.array([PARENT_ID[r['parent']] for r in rows])
    w=np.bincount(i,minlength=len(PARENTS))
    v=np.bincount(i,weights=gains,minlength=len(PARENTS))
    return (100*np.quantile(v[BOOT].sum(1)/w[BOOT].sum(1),[.025,.975])).round(2).tolist()


def probe(x,y,fold):
    """Regress measured signed advantage. Imputation and scale fitted inside each fold."""
    p=np.full(len(y),np.nan)
    base=np.zeros(len(y))
    for f in range(6):
        train=fold!=f
        test=~train
        x0=x[train]
        valid=np.isfinite(x0)
        mean=np.nansum(x0,axis=0)/np.maximum(1,valid.sum(axis=0))
        centered=np.where(np.isfinite(x),x-mean,0)
        sd=np.sqrt((centered[train]**2).sum(0)/np.maximum(1,valid.sum(0)))
        z=np.column_stack([np.clip(centered/np.maximum(sd,1e-6),-5,5),np.isfinite(x)])
        z-=z[train].mean(0)
        yt=y[train].mean()
        weights=np.linalg.solve(z[train].T@z[train]+train.sum()*np.eye(z.shape[1]),
                                z[train].T@(y[train]-yt))
        p[test]=yt+z[test]@weights
        base[test]=yt
    return p,base


def auc(y,s):
    win=s[y>0]
    lose=s[y<0]
    if not len(win) or not len(lose):
        return None
    return round(float(((win[:,None]>lose).mean()+.5*(win[:,None]==lose).mean())),4)


def auc_draws(rows,y,score):
    positive=np.flatnonzero(y>0)
    negative=np.flatnonzero(y<0)
    family=np.array([PARENT_ID[r['parent']] for r in rows])
    # Number and rank sums of win-loss pairs by the parents contributing each side.
    cmp=score[positive,None]-score[negative]
    n=len(PARENTS)
    ids=(family[positive,None]*n+family[negative]).ravel()
    kernel=np.bincount(ids,weights=((cmp>0)+.5*(cmp==0)).ravel(),minlength=n*n).reshape(n,n)
    weights=np.stack([(BOOT==i).sum(1) for i in range(n)],axis=1)
    denom=(weights @ np.bincount(family[positive],minlength=n))*(weights @ np.bincount(family[negative],minlength=n))
    return np.divide(((weights@kernel)*weights).sum(1),denom,out=np.full(len(denom),np.nan),where=denom>0)


def stage_metrics(rows, score, base_score, early_score):
    a=np.array([r['reference'] for r in rows])
    b=np.array([r['alternate'] for r in rows])
    y=b-a
    outcome=np.where(score>0,b,a)
    default=np.where(base_score>0,b,a)
    early=np.where(early_score>0,b,a)
    diff=y!=0
    draws=auc_draws(rows,y,score)
    early_draws=auc_draws(rows,y,early_score)
    return dict(auc_ci=np.nanquantile(draws,[.025,.975]).round(4).tolist(),
                paired_auc_delta_ci=np.nanquantile(draws-early_draws,[.025,.975]).round(4).tolist(),
                n=len(y),families=len(set(r['parent'] for r in rows)),first=int(a.sum()),second=int(b.sum()),
                ties=int(sum(~diff)),either=int(np.maximum(a,b).sum()),exclusive_second=int(sum(y>0)),
                choices_alt=int(sum(score>0)),hits=int(outcome[diff].sum()),discord=int(diff.sum()),
                auc=auc(y,score),initial_auc=auc(y,early_score),pred=int(outcome.sum()),base=int(default.sum()),
                early=int(early.sum()),gain_vs_const_ci=ci(rows,outcome-default),
                gain_vs_early_ci=ci(rows,outcome-early),gain_vs_first_ci=ci(rows,outcome-a),
                median_tick=float(np.median([r['tick'] for r in rows])))


def analyze(arm,directory):
    data=[run.study.read(f) for f in sorted((directory/'cases').glob('*.json'))]
    assert len(data)==384 and len({(r['parent'],r['cm']) for r in data})==384
    bykey={(r['parent'],r['cm']):r for r in data}
    radii=[]
    for cm in RADII:
        vals=[]
        for p in PARENTS:
            root=bykey[p['parent'],0]
            sub=bykey[p['parent'],cm]
            first,second=root['baseline']['success'],root['labels']['initial']['success']
            a,b=sub['baseline']['success'],sub['labels']['initial']['success']
            vals.append([first,second,a,b])
        first,second,a,b=np.array(vals).T
        root_pref=second-first
        pref=b-a
        active=root_pref!=0
        flip=(a!=first).astype(int)-(b!=second).astype(int)
        radii.append(dict(cm=cm,n=len(vals),first=int(a.sum()),second=int(b.sum()),first_flip=int(sum(a!=first)),
                          second_flip=int(sum(b!=second)),flip_diff_ci=ci(PARENTS,flip),
                          roots_discord=int(active.sum()),preserve=int(sum(active&(pref==root_pref))),
                          reverse=int(sum(active&(pref==-root_pref))),tie=int(sum(active&(pref==0))),
                          new_discord=int(sum(~active&(pref!=0))),prefer_first=int(sum(pref<0)),
                          prefer_second=int(sum(pref>0))))
    allrows=[]
    stages={}
    for name in STAGES:
        files=[r for r in data if name in r['stages']]
        rows=[]
        for r in files:
            st=r['stages'][name]
            rows.append(dict(pair=arm,parent=r['parent'],cm=r['cm'],fold=r['fold'],stage=name,
                             step=st['step'],tick=st['tick'],prefix=st['prefix'],
                             reference=r['baseline']['success'],alternate=r['labels'][name]['success']))
        fold=np.array([r['fold'] for r in rows])
        y=np.array([r['alternate']-r['reference'] for r in rows])
        scores,base=probe(np.array([r['stages'][name]['features'] for r in files],dtype=float),y,fold)
        initial,_=probe(np.array([r['stages']['initial']['features'] for r in files],dtype=float),y,fold)
        stages[name]=stage_metrics(rows,scores,base,initial)
        roots=[i for i,r in enumerate(rows) if r['cm']==0]
        stages[name]['origin_only']=stage_metrics([rows[i] for i in roots],scores[roots],base[roots],initial[roots])
        for r,p,b,q in zip(rows,scores,base,initial):
            r.update(pred=float(p),constant=float(b),initial_features=float(q))
        allrows.extend(rows)
    runs=sum(r['runs'] for r in data)
    return dict(radii=radii,stages=stages,rollouts=runs),allrows


def figures(results):
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for i,(arm,r) in enumerate(results.items()):
        left,right=axes[i]
        for field,label in [('first_flip','ID actor' if arm=='actors' else 'Primary CAP'),
                            ('second_flip','Action actor' if arm=='actors' else 'Complementary CAP')]:
            left.plot(RADII,[100*d[field]/d['n'] for d in r['radii']],marker='o',label=label)
        left.set(xlabel='Robot 0 displacement (cm)',ylabel='Outcome flips vs original (%)',xticks=RADII)
        left.legend(frameon=False)
        names=['Opening','First disagreement','First contact']
        for j,(target,label) in enumerate([('auc','Current prefix'),('initial_auc','Opening on same cases')]):
            right.bar(np.arange(3)+.34*(j-.5),[r['stages'][k][target] for k in STAGES],width=.33,label=label)
        right.axhline(.5,color='black',ls=':',lw=1)
        right.set(ylim=(0,1),xticks=range(3),xticklabels=names,ylabel='Winner AUC on discordant outcomes')
        right.legend(frameon=False,fontsize=8)
    fig.savefig(DOC/'diagnostic.png',dpi=165)
    plt.close(fig)


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    results={}
    output=[]
    for arm,folder in [('actors',OUT),('experts',OUT/'experts')]:
        results[arm],rows=analyze(arm,folder)
        output+=rows
    dump_csv('predictions.csv',output)
    dump_csv('radii.csv',[dict(pair=k,**v) for k,r in results.items() for v in r['radii']])
    run.study.save(DOC/'metrics.json',results)
    figures(results)
    protocol=dict(collection=run.study.read(OUT/'protocol.json'),experts=run.study.read(OUT/'experts/protocol.json')['pair'],
                  report_sha=run.study.sha(__file__),expert_source=run.study.sha(Path(__file__).with_name('cap_regional_advantage_experts.py')))
    run.study.save(DOC/'protocol.json',protocol)


if __name__=='__main__':
    main()
