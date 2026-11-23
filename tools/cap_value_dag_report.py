"""Room-paired prediction, risk calibration and replayed route selection."""
import csv
import itertools
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools import cap_value_dag as fit

DOC,OUT=fit.DOC,fit.OUT
M='public_history/root'
PAIRS=list(itertools.combinations(range(4),2))


def stat(x,ids):
    x=np.asarray(x,float)
    ids=np.broadcast_to(ids.reshape((len(ids),)+(1,)*(x.ndim-1)),x.shape)
    valid=np.isfinite(x)
    if not valid.any():return None
    _,idx=np.unique(ids[valid],return_inverse=True)
    sums=np.bincount(idx,weights=x[valid]); counts=np.bincount(idx)
    draws=np.random.default_rng(907).integers(len(sums),size=(2500,len(sums)))
    ci=np.quantile(sums[draws].sum(1)/counts[draws].sum(1),[.025,.975])
    return dict(mean=float(sums.sum()/counts.sum()),ci=ci.tolist(),n=int(valid.sum()),rooms=len(counts))


def ranking(y,p,dim,margin):
    vals=[]
    for a,b in PAIRS:
        d=y[:,a,dim]-y[:,b,dim]
        q=np.round(p[:,a,dim]-p[:,b,dim],12)
        vals.append(np.where((np.abs(d)>margin)&np.isfinite(q), .5+.5*np.sign(q*d), np.nan))
    return np.stack(vals,axis=1)


def metrics(y,p):
    return dict(mae=abs(p[:,:,0]-y[:,:,0]),brier=(p[:,:,2]-y[:,:,2])**2,
                tail_brier=(p[:,:,3]-y[:,:,3])**2,near_brier=(p[:,:,4]-y[:,:,4])**2,
                rank_distance=ranking(y,p,0,.02),rank_success=ranking(y,p,2,0))


def choices(y,p,reference,select,ids):
    k=0 if select=='distance' else 2
    val=p[:,:,k]*(-1 if k==0 else 1)
    # At unreachable starts all real continuations are identical. Choose AA for accounting.
    best=np.argmax(np.nan_to_num(val,nan=-1e8).round(12),axis=1)
    actual=y[np.arange(len(y)),best]
    result=dict(success=int(actual[:,2].sum()),distance=float(actual[:,0].mean()),
                choices=dict(Counter(fit.ROUTES[i] for i in best)))
    for name,j in [('AA',0),('dev_fixed',reference)]:
        d=actual[:,2]-y[:,j,2]
        result[name]=dict(gain=int(sum(d>0)),loss=int(sum(d<0)),effect=stat(d,ids),
                          gap_change=stat(actual[:,0]-y[:,j,0],ids))
    return result


def returns_on_revisit(cases):
    counts=Counter()
    rooms=set()
    seen=set()
    for c in cases:
        for route,b in c['branches'].items():
            for i,(u,v) in enumerate(zip(b['states'],b['states'][1:])):
                if u[0]!=v[0]:continue
                if i!=1:continue # continuation is fixed; one actual action/suffix per recorded history
                key=(c['room'],b['hashes'][i],route[i])
                if key in seen:continue
                seen.add(key)
                counts['self']+=1
                win=int(b['y'][2])
                counts['win']+=win
                counts['net_advance_2cm']+=int(b['positions'][i]-b['positions'][i+1]>=.02)
                if win:rooms.add(c['room'])
                counts['n_'+v[4]]+=1
                counts['win_'+v[4]]+=win
    return dict(**counts,winning_rooms=len(rooms))


def summarize(fold):
    cases,vs=fit.evaluate(fold)
    ids=np.array([c['room'] for c in cases])
    y=np.array([[c['branches'][r]['y'] for r in fit.ROUTES] for c in cases])
    lengths=np.array([[len(c['branches'][r]['states']) for r in fit.ROUTES] for c in cases])
    valid=lengths[:,0]>0
    assert np.all(y[~valid]==y[~valid,:1])
    references=fit.read(OUT/'frozen.json')['reference']
    summary=dict(rooms=len(set(ids)),assigned=len(cases),reached=int(sum(valid)),complete_routes=int((lengths==3).sum()),
                 fixed={r:dict(success=int(y[:,i,2].sum()),distance=float(y[:,i,0].mean())) for i,r in enumerate(fit.ROUTES)},
                 hindsight_success=int(y[:,:,2].max(1).sum()),hindsight_distance=float(y[:,:,0].min(1).mean()),
                 revisits=returns_on_revisit(cases),models={},comparisons={},mid_switch={})
    all_metrics={}
    for m,p in vs.items():
        mvals=metrics(y,p)
        all_metrics[m]=mvals
        total=fit.variance(p)
        delta=(p[:,:,0]-y[:,:,0])**2-total
        report=dict(**{k:stat(z,ids) for k,z in mvals.items()},
                    risk=dict(squared_error=stat((p[:,:,0]-y[:,:,0])**2,ids),var=stat(total,ids),
                              residual_minus_var=stat(delta,ids)))
        if m in (M,'public_history/direct','public_history/observed_mid','public_history/observed_leaf'):
            ranked=mvals['rank_distance']
            report['rank_change_first_only']=stat(ranked[:,[PAIRS.index((0,2)),PAIRS.index((1,3))]],ids)
            report['rank_change_second_only']=stat(ranked[:,[PAIRS.index((0,1)),PAIRS.index((2,3))]],ids)
        if m.endswith('/root'):
            report['decomposition']=[stat(z,ids)['mean'] for z in (p[:,:,5],p[:,:,6]-p[:,:,5],total-p[:,:,6])]
            order=np.argsort(total[np.isfinite(total)])
            xv=total[np.isfinite(total)]; err=(y[:,:,0]-p[:,:,0])[np.isfinite(total)]**2
            report['risk']['quartiles']=[[int(len(i)),float(xv[i].mean()),float(err[i].mean())] for i in np.array_split(order,4)]
        if 'observed_' not in m:
            report['choice']={s:choices(y,p,fit.ROUTES.index(references[s]),s,ids) for s in references}
        else:
            hold=lengths==3
            report['both_windows_nonterminal']={k:stat(np.where(hold,z,np.nan),ids) for k,z in mvals.items() if not k.startswith('rank')}
            if m.endswith('observed_mid'):
                use=(lengths[:,0]>=2)&(p[:,1,2].round(12)>p[:,0,2].round(12))
                d=use*(y[:,1,2]-y[:,0,2])
                report_mid=dict(eligible=int(sum(lengths[:,0]>=2)),switch=int(sum(use)),gain=int(sum(d>0)),loss=int(sum(d<0)),
                               success=int(sum(y[:,0,2]+d)),effect=stat(d,ids))
                summary['mid_switch'][m]=report_mid
        summary['models'][m]=report
    for lhs,rhs in [('physical_history/root','physical/root'),(M,'public/root'),(M,'public_history/direct'),
                    (M,'constant/direct'),(M,'public_history/one'),('physical/root','budget/root'),
                    (M,'physical_history/root'),('public_history/observed_mid',M),
                    ('public_history/observed_leaf','public_history/observed_mid')]:
        key=lhs+' minus '+rhs
        summary['comparisons'][key]={k:stat(all_metrics[lhs][k]-v,ids) for k,v in all_metrics[rhs].items()}
        if 'observed_' in lhs:
            summary['comparisons'][key]['nonterminal_mae']=stat(np.where(lengths==3,all_metrics[lhs]['mae']-all_metrics[rhs]['mae'],np.nan),ids)
    for space in ('public','physical'):
        lhs=space+'_history/observed_leaf'
        rhs=space+'/observed_leaf'
        summary['comparisons'][lhs+' minus '+rhs]={
            k:stat(np.where(lengths==3,all_metrics[lhs][k]-all_metrics[rhs][k],np.nan),ids)
            for k in ('mae','brier','tail_brier')}
    predicted=[]
    for i,c in enumerate(cases):
        if not valid[i]:continue
        for j,r in enumerate(fit.ROUTES):
            b=c['branches'][r];state=b['states'][0]
            row=dict(panel=fold,room=c['room'],depth=c['depth'],route=r,root_history=b['hashes'][0],
                     root_tick=b['ticks'][0],root_physical=state[0],root_public=state[1],windows_reached=int(lengths[i,j])-1,
                     final_distance_m=b['y'][0],success=int(b['y'][2]))
            for m in vs:
                a=vs[m][i,j]
                row.update({m+'_'+suffix:round(float(v),7) for suffix,v in zip(('mean_m','var_m2','success','tail_gt1m','near_le20cm'),
                                                                                (a[0],fit.variance(a),*a[2:5]))})
            predicted.append(row)
    return summary,predicted


def figure(results):
    labels=['Route mean','Budget','Physical','Physical + history','Public','Public + history','Direct public + history']
    keys=['constant/direct','budget/root','physical/root','physical_history/root','public/root',M,'public_history/direct']
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    for n,(metric,title) in enumerate([('mae','Terminal distance MAE (m)'),('rank_distance','Pairwise distance ranking'),('brier','Task success Brier')]):
        ax=axes[n]
        for off,fold,color in [(-.12,'diagnostic','#829baa'),(.12,'confirm','#006c68')]:
            v=[results[fold]['models'][k][metric] for k in keys]
            means=np.array([s['mean'] for s in v])
            bounds=np.array([s['ci'] for s in v])
            ax.errorbar(means,np.arange(len(keys))+off,xerr=[means-bounds[:,0],bounds[:,1]-means],fmt='o',color=color,capsize=2,label=fold)
        if metric=='rank_distance':ax.axvline(.5,color='0.7',lw=1)
        ax.set_yticks(range(len(keys)),labels if n==0 else ['']*len(labels));ax.invert_yaxis();ax.set_title(title,fontsize=11)
        ax.grid(axis='x',alpha=.2)
    axes[2].legend(fontsize=9)
    fig.savefig(DOC/'value.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    largest=0.
    for fold,color in [('diagnostic','#728d9f'),('confirm','#00776d')]:
        pts=np.asarray(results[fold]['models'][M]['risk']['quartiles'])
        axes[0].plot(pts[:,1],pts[:,2],'o-',color=color,label=fold)
        largest=max(largest,float(pts[:,1:].max()))
    axes[0].plot([0,largest],[0,largest],':',color='0.5')
    axes[0].set(xlabel='Predicted variance (m²)',ylabel='Mean squared residual (m²)',title='Terminal distance by risk quartile')
    axes[0].legend()
    for offset,fold,color in [(-.18,'diagnostic','#93a9b4'),(.18,'confirm','#15746e')]:
        s=results[fold]
        vals=[s['fixed']['AA']['success'],s['models'][M]['choice']['success']['success'],
              s['models']['public_history/direct']['choice']['success']['success'],s['hindsight_success']]
        bars=axes[1].bar(np.arange(4)+offset,vals,.36,color=color,label=fold)
        axes[1].bar_label(bars,fontsize=8)
    axes[1].set(xticks=range(4),xticklabels=['Fixed AA','2-step DAG','Direct','Hindsight'],ylabel='Successes / 400 assigned trials',title='Matched route selection')
    axes[1].legend(fontsize=8)
    fig.savefig(DOC/'risk.png',dpi=150)
    plt.close(fig)


def main():
    result={}
    all_rows=[]
    for fold in ('diagnostic','confirm'):
        result[fold],rows=summarize(fold)
        all_rows.extend(rows)
    with (DOC/'predictions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    fit.save(DOC/'metrics.json',result)
    figure(result)


if __name__=='__main__':
    main()
