"""Offline observability diagnostic using actual short prefixes from both possible routes."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np

from tools import cap_route_horizon as exp
from tools import cap_route_horizon_report as report

STEPS=(0,3,10,30)
study=exp.study


def route(fixture,pair,which):
    policies=[study.phase.FrozenCAP(study.MODULES[e]) for e in pair]
    result={}
    counts=Counter()
    first=None
    n=0
    with study.phase.world(fixture) as (w,_):
        def update(history):
            for policy in policies:policy.update(history)
        update(w.history)
        while not study.ended(w):
            obs=study.observe(w)
            cmd=policies[which].propose(obs)
            if first is None and exp.try_action(policies[1-which],obs)!=cmd:
                # Prior to divergence both are on their ordinary legal trajectory.
                alternative=exp.try_action(policies[1-which],obs)
                if alternative is not None:
                    first=(n,obs['tick'])
                    counts.clear()
                    start=study.old.base.geom(obs)
                    previous=start
                    best=start[0] if start[3] else None
                    result['origin']=w.history.sha256
            if first is not None:
                h=n-first[0]
                geom=study.old.base.geom(obs)
                if geom[3]:
                    best=geom[0] if best is None else min(best,geom[0])
                if h in STEPS:
                    c=study.corridor_semantics(obs)
                    result[str(h)]=np.r_[geom,geom-start,geom-previous,
                        (obs['tick']-first[1])/1800,-1 if best is None else best,
                        [float(r['contacting_object']) for r in obs['robots']],
                        [float(c[k]) for k in ('target_estimated','static_hit','movable_hit','swept_visibility','goal_wall_distance_m')],
                        [counts[k]/(1+h) for k in ('contact_ready','contact_contact_timeout','retimed')]].tolist()
                if h>=max(STEPS):break
                previous=geom
            study.tick_action(w,cmd,exp.EXECUTOR,update,counts)
            n+=1
    return result


def replay(job,pair):
    fold,room,f=job
    path=exp.OUT/'shadow'/fold/(room+'.json')
    if path.exists():return
    rows=[route(f,pair,i) for i in (0,1)]
    assert rows[0].get('origin')==rows[1].get('origin')
    study.save(path,rows)


def get(fold,plan):
    rows=[]
    for room in plan['rooms'][fold]:
        r=study.read(exp.OUT/'runs'/fold/(room+'.json'))
        paths=study.read(exp.OUT/'shadow'/fold/(room+'.json'))
        cut=r['horizons']['0']['cut']
        assert paths[0].get('origin')==(cut['history'] if cut else None)
        if cut is None:continue
        rows.append((room,cut['x'],paths,r['baseline']['success'],r['horizons']['0']['success']))
    return rows


def evaluate(plan):
    train,test=[get(f,plan) for f in exp.FOLDS]
    output={}
    for h in STEPS:
        key=str(h)
        def prepare(data,arm):
            x,y,ids=[],[],[]
            for room,base,paths,a,b in data:
                if any(key not in p for p in paths):continue
                obs=[np.array(p[key]) for p in paths]
                x.append(base if arm=='first' else np.r_[base,obs[0]] if arm=='A' else np.r_[base,obs[0],obs[1]-obs[0]])
                y.append(b-a)
                ids.append(room)
            return np.array(x),np.array(y),ids
        out={}
        scores={}
        for arm in ('first','A','both'):
            x,y,_=prepare(train,arm)
            z,d,ids=prepare(test,arm)
            s=report.ridge(x,y,z)
            idx=d!=0
            scores[arm]=s
            out[arm]=dict(auc=report.auc(d,s),correct=int(sum((s[idx]>0)==(d[idx]>0))),
                          advantages=int(sum(idx)),rooms=len(d),net_over_A=int(d[s>0].sum()))
        rng=np.random.default_rng(771723+h)
        samples=[]
        for _ in range(2500):
            sample=rng.integers(len(d),size=len(d))
            values=[report.auc(d[sample],scores[k][sample]) for k in ('first','A','both')]
            if None not in values:
                samples.append([values[2]-values[0],values[2]-values[1]])
        out['auc_difference_ci95'] = dict(zip(('both_minus_first','both_minus_A'),
                                              np.quantile(samples,[.025,.975],axis=0).T.tolist()))
        output[key]=out
    study.save(exp.DOC/'shadow.json',output)
    print(output,flush=True)


if __name__=='__main__':
    import sys
    plan=study.read(exp.OUT/'protocol.json')
    if '--evaluate' in sys.argv:
        evaluate(plan)
    else:
        study.initialize()
        jobs=[]
        for fold in exp.FOLDS:
            db=study.read(study.OUT/(fold+'.json'))
            jobs.extend((fold,r,db[r]) for r in plan['rooms'][fold])
        with ProcessPoolExecutor(5) as pool:
            for _ in pool.map(partial(replay,pair=plan['pair']),jobs):pass
