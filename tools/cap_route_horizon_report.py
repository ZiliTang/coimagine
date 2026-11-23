"""Room-paired forward route selection and delayed prediction diagnostic."""
import hashlib
import numpy as np

from tools import cap_route_horizon as exp

THRESHOLDS = (0., .10, .25, 1.e6)


def ridge(x,y,z):
    mean=x.mean(0)
    sd=np.maximum(.1,x.std(0))
    centered=(x-mean)/sd
    weights=np.linalg.solve(centered.T@centered+len(x)*.15*np.eye(x.shape[1]),centered.T@(y-y.mean()))
    return (z-mean)/sd@weights+y.mean()


def data(records,h):
    a=np.array([r['baseline']['success'] for r in records])
    b=np.array([r['horizons'][str(h)]['success'] for r in records])
    cuts=[r['horizons'][str(h)]['cut'] for r in records]
    ready=np.array([bool(c and c['eligible']) for c in cuts])
    dim=next(len(c['x']) for c in cuts if c and c['eligible'])
    x=np.stack([c['x'] if flag else np.zeros(dim) for flag,c in zip(ready,cuts)])
    return a,b,x,ready


def folds(records):
    order=sorted(range(len(records)),key=lambda i:hashlib.sha256(('cv/'+records[i]['room']).encode()).digest())
    idx=np.empty(len(order),int)
    idx[order]=np.arange(len(order))%5
    return idx


def estimate(dev,test,h,use_original=False,only_geometry=False):
    a,b,x,mask=data(dev,h)
    c,d,z,ready=data(test,h)
    if use_original:
        b=data(dev,0)[1]
    if only_geometry:
        # First nine columns are the original public geometric state.
        x,z=x[:,:9],z[:,:9]
    y=b-a
    score=np.full(len(test),-np.inf)
    score[ready]=ridge(x[mask],y[mask],z[ready])
    cv=folds(dev)
    oof=np.full(len(dev),-np.inf)
    for k in range(5):
        train=mask & (cv!=k)
        val=mask & (cv==k)
        if val.any():
            oof[val]=ridge(x[train],y[train],x[val])
    return oof,score


def auc(y,s):
    pos=s[y==1]
    neg=s[y==-1]
    if not len(pos) or not len(neg):
        return None
    diff=pos[:,None]-neg[None,:]
    return float(np.mean((diff>0)+.5*(diff==0)))


def median(values):
    return float(np.median(values)) if len(values) else None


def paired(x,y,bootstrap):
    delta=x-y
    return dict(rescued=int(sum(delta>0)),lost=int(sum(delta<0)),difference_pp=float(100*delta.mean()),
                ci95_pp=(100*np.quantile(delta[bootstrap].mean(1),[.025,.975])).tolist())


def main():
    plan=exp.study.read(exp.OUT/'protocol.json')
    dev,test=[[exp.study.read(exp.OUT/'runs'/f/(r+'.json')) for r in plan['rooms'][f]] for f in exp.FOLDS]
    strata=np.array([r['room'][:6] for r in test])
    rng=np.random.default_rng(20261123)
    bootstrap=np.concatenate([rng.choice(ix,(10000,len(ix))) for g in np.unique(strata)
                              if len(ix:=np.flatnonzero(strata==g))],axis=1)
    a,b0,_,_=data(test,0)
    discord=(a!=b0)
    first=[r['baseline']['first'] for r in test]
    fidx=np.array([r['tick'] if r else -1 for r in first])
    fail_ticks=np.array([r['baseline']['ticks'] if not v else r['horizons']['0']['ticks'] for r,v in zip(test,a)])
    win_ticks=np.array([r['baseline']['ticks'] if v else r['horizons']['0']['ticks'] for r,v in zip(test,a)])
    result=dict(rooms=len(test),fixed_A=int(a.sum()),fixed_B=int(b0.sum()),only_A=int(sum(a>b0)),only_B=int(sum(b0>a)),
                union=int(np.maximum(a,b0).sum()),different_first=int(sum(fidx>=0)),
                first_tick_zero=int(sum(fidx==0)),discord_first_tick_zero=int(sum(discord & (fidx==0))),
                discord_median_first_tick=median(fidx[discord]),discord_median_ticks_to_failure_end=median((fail_ticks-fidx)[discord]),
                discord_median_ticks_to_success=median((win_ticks-fidx)[discord]),horizons={})
    result['discord_loser_budget_end']=sum(
        (r['baseline'] if not v else r['horizons']['0'])['termination']=='task_budget_exhausted'
        for r,v,z in zip(test,a,b0) if v!=z)
    choices=[]
    cases={r['room']:dict(A=int(q),B=int(v),first_tick=int(tick), horizons={}) for r,q,v,tick in zip(test,a,b0,fidx)}
    for h in exp.HORIZONS:
        a,b,x,ready=data(test,h)
        train_a,train_b,_,_=data(dev,h)
        cv,score=estimate(dev,test,h)
        _,geom_score=estimate(dev,test,h,only_geometry=True)
        _,early_score=estimate(dev,test,h,use_original=True)
        statistics=[]
        for threshold in THRESHOLDS:
            choose=cv>threshold
            wins=int(np.where(choose,train_b,train_a).sum())
            statistics.append((wins,-int(choose.sum()),threshold))
        wins,negative_switches,t=max(statistics)
        choices.append((wins,negative_switches,-h,t))
        decisions=score>t
        outcome=np.where(decisions,b,a)
        basic=np.where(score>0,b,a)
        changed=b!=a
        hinfo=dict(available=int(ready.sum()), switch_wins=int(b.sum()),upper_bound=int(np.maximum(a,b).sum()),
                  repairable=int(sum(b>a)),at_risk=int(sum(a>b)),initial_rescues_retained=int(sum((b0>a)&(b>a))),
                  new_rescues_on_initial_double_failure=int(sum((a==0)&(b0==0)&(b==1))),
                  rank_auc=auc((b-a)[ready],score[ready]),geometry_auc=auc((b-a)[ready],geom_score[ready]),
                  rank_accuracy_on_discordant=float(np.mean((score[changed]>0)==(b[changed]>a[changed]))) if changed.any() else None,
                  initial_winner_auc_read_later=auc((b0-a)[ready],early_score[ready]),
                  initial_winner_common_n=int(sum(discord&ready)),
                  threshold=t,cv_wins=wins,thresholded_wins=int(outcome.sum()),switches=int(decisions.sum()),
                  vs_A=paired(outcome,a,bootstrap),zero_threshold_wins=int(basic.sum()),zero_threshold_vs_A=paired(basic,a,bootstrap),
                  off_policy_fallback_steps=sum(r['horizons'][str(h)]['invalid'] for r in test),
                  panels={str(p):dict(wins=int(outcome[strata==p].sum()),A=int(a[strata==p].sum())) for p in np.unique(strata)})
        samples=[]
        for ix in bootstrap[:2500]:
            ix=ix[ready[ix]]
            u=auc((b-a)[ix],score[ix])
            v=auc((b-a)[ix],geom_score[ix])
            if u is not None and v is not None:samples.append([u,u-v])
        hinfo['auc_ci95'],hinfo['auc_over_geometry_ci95']=np.quantile(samples,[.025,.975],axis=0).T.tolist()
        ticks=[r['horizons'][str(h)]['cut']['tick']-r['baseline']['first']['tick'] for r in test
               if r['horizons'][str(h)]['cut'] is not None]
        hinfo['median_elapsed_ticks_where_reached']=median(ticks)
        for i,r in enumerate(test):
            item=r['horizons'][str(h)]
            cases[r['room']]['horizons'][str(h)]=dict(cut_tick=item['cut']['tick'] if item['cut'] else None,
                 handoff=int(b[i]),eligible=bool(ready[i]),score=float(score[i]) if ready[i] else None,chosen=bool(decisions[i]))
        result['horizons'][str(h)]=hinfo
    _,_,negative_h,_=max(choices)
    union=np.maximum(a,b0)
    for h in exp.HORIZONS:
        union=np.maximum(union,data(test,h)[1])
    result['all_schedules_union']=int(union.sum())
    result['rescued_initial_double_failures_distinct']=int(sum((a==0)&(b0==0)&(union==1)))
    result['selected_horizon']=-negative_h
    exp.study.save(exp.DOC/'results.json',result)
    exp.study.save(exp.DOC/'rooms.json',cases)
    print(result)


if __name__=='__main__':
    main()
