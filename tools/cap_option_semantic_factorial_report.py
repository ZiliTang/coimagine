"""Room-paired estimates for the held option and routing-feature comparisons."""
import csv
from collections import Counter
import itertools

import numpy as np

from tools import cap_option_semantic_factorial as study

OUT,DOC=study.OUT,study.DOC
CONTROLLERS,BANKS,FEATURES,SEEDS=study.CONTROLLERS,study.BANKS,study.FEATURES,study.SEEDS


def read_rows(root,rooms):
    rows=[study.read(root/(k+'.json')) for k in rooms]
    assert all(0<r['ticks']<=1800 and 0<r['decisions']<=300 for r in rows)
    assert all(r['termination'] is not None or r['ticks']==1800 or r['decisions']==300 for r in rows)
    return rows


def bootstrap(values,strata):
    values=np.asarray(values,dtype=float)
    draw=np.zeros(10000)
    rng=np.random.default_rng(17)
    for group in sorted(set(strata)):
        a=values[strata==group]
        sample=rng.integers(len(a),size=(len(draw),len(a)))
        draw+=a[sample].sum(1)
    return (100*np.quantile(draw,[.025,.975])/len(values)).tolist()


def contrast(d,strata,panels,*,per_seed=None):
    d=np.asarray(d)
    data=dict(difference_pp=float(100*d.mean()),ci=bootstrap(d,strata),
              panels={p:float(100*d[panels==p].mean()) for p in sorted(set(panels))})
    if per_seed is not None:
        data['seeds']={str(s):float(100*d.mean()) for s,d in zip(SEEDS,per_seed)}
    return data


def makes_plot(result):
    import matplotlib.pyplot as plt

    arms=['basic','corridor','contact','both']
    labels=['Basic','Corridor','Contact','Both']
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.8),sharex=True,sharey=True,layout='constrained')
    for i,executor in enumerate(CONTROLLERS):
        for j,bank in enumerate(BANKS):
            ax=axes[i,j]
            r=result['mix'][executor][bank]
            y=[r[arm]['avg'] for arm in arms]
            ax.plot(range(len(y)),y,color='#176666',marker='o', lw=2)
            for seed in SEEDS:
                points=[r[a]['per_seed'][str(seed)]['wins']/4 for a in arms]
                ax.scatter(np.arange(len(arms)),points,c='#b05d44',alpha=.5,s=17)
            baseline=result['single'][executor][bank]
            ax.axhline(baseline['held_dev_best']/4,ls='--',color='#555e70',lw=1)
            ax.set_title(f'{executor.replace("_"," ")} · {bank}')
            ax.set_xticks(range(4),labels)
            ax.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('Success on 400 rooms (%)')
    axes[1,0].set_ylabel('Success on 400 rooms (%)')
    fig.suptitle('Seed averages and individual runs · dashed lines are dev-selected single CAPs')
    fig.savefig(DOC/'matrix.png',dpi=160)
    plt.close(fig)


def case_patterns(fixtures,rules):
    values={}
    for room,fixture in fixtures.items():
        obs=study.old.base.public(fixture)
        sem=study.corridor_semantics(obs)
        sem['observed']=sem['kind']
        values[room]=tuple(study.lib.matches(rule,sem) for rule in rules)
    return values


def initial_choice(train,held,dev_pattern,held_pattern,best):
    picks={}
    for pattern in set(dev_pattern):
        mask=np.array([p==pattern for p in dev_pattern])
        if mask.sum()>=12:
            picks[pattern]=int(train[:,mask].sum(1).argmax())
    selection=np.array([picks.get(p,best) for p in held_pattern])
    return held[selection,np.arange(held.shape[1])]


def main():
    DOC.mkdir(exist_ok=True,parents=True)
    protocol=study.read(OUT/'protocol.json')
    cases=study.read(OUT/'held.json')
    dev=study.read(OUT/'development.json')
    assert study.phase.cap.digest(cases)==protocol['held']
    rooms=sorted(cases)
    panels=np.array([k[:6] for k in rooms])
    clutter=np.array([1+len(cases[k]['movable_objects']) for k in rooms])
    strata=np.array([f'{p}/{n}' for p,n in zip(panels,clutter)])
    patterns=case_patterns(dev|cases,protocol['rules'])
    dev_pattern=[patterns[k] for k in sorted(dev)]
    held_pattern=[patterns[k] for k in rooms]
    result=dict(single={},mix={},effects={},interactions={})
    scores={}
    audit=[]
    for ex in CONTROLLERS:
        result['single'][ex]={}
        result['mix'][ex]={}
        result['effects'][ex]={}
        for bank in BANKS:
            bank_ids=[m['sha'] for m in protocol['libraries'][bank]]
            held=np.array([[r['success'] for r in read_rows(OUT/'singles'/'held'/ex/m,rooms)] for m in bank_ids])
            train=np.array([[r['success'] for r in read_rows(OUT/'singles'/'development'/ex/m,sorted(dev))] for m in bank_ids])
            best=int(train.sum(1).argmax())
            union=held.any(0)
            result['single'][ex][bank]=dict(dev= train.sum(1).tolist(),held=held.sum(1).tolist(),dev_best=best,
                held_dev_best=int(held[best].sum()),held_union=int(union.sum()),dev_union=int(train.any(0).sum()))
            selected=initial_choice(train,held,dev_pattern,held_pattern,best)
            details=result['single'][ex][bank]
            details['tag_route']=dict(wins=int(selected.sum()),vs_single=contrast(selected-held[best],strata,panels))
            details['by_condition']={}
            for i,rule in enumerate(protocol['rules']):
                mask=np.array([p[i] for p in held_pattern])
                sample=held[:,mask]
                details['by_condition'][rule['name']]=dict(n=int(mask.sum()),success=sample.sum(1).tolist(),
                                                            union=int(sample.any(0).sum()))
            audit.extend(dict(room=k,controller=ex,bank=bank,method='tag_route',seed='',success=int(y),ticks='',decisions='')
                         for k,y in zip(rooms,selected))
            for i,values in enumerate(held):
                audit.extend(dict(room=k,controller=ex,bank=bank,method=f'single{i}',seed='',success=int(y),ticks='',decisions='')
                             for k,y in zip(rooms,values))
            modes={}
            for arm in FEATURES:
                by_seed={}
                values=[]
                for seed in SEEDS:
                    rows=read_rows(OUT/'eval'/ex/bank/arm/f's{seed}',rooms)
                    y=np.array([r['success'] for r in rows])
                    values.append(y)
                    by_seed[str(seed)]=dict(wins=int(y.sum()),inside=int(y[union].sum()),outside=int(y[~union].sum()),
                                          rescued=int(np.sum(y>held[best])),lost=int(np.sum(y<held[best])),
                                          mean_ticks=float(np.mean([r['ticks'] for r in rows])),
                                          fit=study.read(OUT/'models'/f'{ex}_{bank}_{arm}_{seed}.json'))
                    audit.extend(dict(room=k,controller=ex,bank=bank,method=arm,seed=seed,
                                     success=r['success'],ticks=r['ticks'],decisions=r['decisions']) for k,r in zip(rooms,rows))
                z=np.stack(values)
                scores[ex,bank,arm]=z
                modes[arm]=dict(avg=float(100*z.mean()),per_seed=by_seed,
                    vs_single=contrast(z.mean(axis=0)-held[best],strata,panels,per_seed=z-held[best]))
            result['mix'][ex][bank]=modes
            effects={}
            for a,b in [('basic','corridor'),('basic','contact'),('basic','both'),('corridor','both'),('contact','both')]:
                d=scores[ex,bank,b]-scores[ex,bank,a]
                effects[b+'_vs_'+a]=contrast(d.mean(0),strata,panels,per_seed=d)
            result['effects'][ex][bank]=effects
    coupled,phased=CONTROLLERS
    usual,spec=BANKS
    changes={}
    for bank,arm in itertools.product(BANKS,FEATURES):
        d=scores[phased,bank,arm]-scores[coupled,bank,arm]
        changes[bank+'_'+arm]=contrast(d.mean(0),strata,panels,per_seed=d)
    result['executor_effect']=changes
    ix=result['interactions']
    for bank,arm in itertools.product(BANKS,FEATURES[1:]):
        d=(scores[phased,bank,arm]-scores[phased,bank,'basic'])-(scores[coupled,bank,arm]-scores[coupled,bank,'basic'])
        ix['controller_x_'+arm+'_'+bank]=contrast(d.mean(0),strata,panels,per_seed=d)
    for ex,arm in itertools.product(CONTROLLERS,FEATURES[1:]):
        d=(scores[ex,spec,arm]-scores[ex,spec,'basic'])-(scores[ex,usual,arm]-scores[ex,usual,'basic'])
        ix['bank_x_'+arm+'_'+ex]=contrast(d.mean(0),strata,panels,per_seed=d)
    result['counts']=dict(held_rooms=len(rooms),dev_rooms=len(dev),fit_models=len(SEEDS)*len(FEATURES)*len(BANKS)*len(CONTROLLERS),
                          held_mixed_runs=len(rooms)*len(SEEDS)*len(FEATURES)*len(BANKS)*len(CONTROLLERS),
                          unique_members=len({m['sha'] for v in protocol['libraries'].values() for m in v}))
    with (DOC/'rooms.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    makes_plot(result)
    study.save(DOC/'results.json',result)
    study.save(OUT/'FINISHED.json',result)
    study.save(OUT/'progress.json',dict(stage='complete',done=result['counts']['held_mixed_runs'],total=result['counts']['held_mixed_runs']))
    print(result['counts'],flush=True)


if __name__=='__main__':
    main()
