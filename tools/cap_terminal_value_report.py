"""Per-room progress curves, all-CAP initial dispatch, and causal perturbation audit."""
from collections import defaultdict
import csv

import numpy as np
import torch

from tools import cap_terminal_value_fit as fit
from tools import cap_route_horizon_report as scoring

study=fit.study
DOC=study.DOC
F=study.FRACTIONS
CLOCK=(0,300,600,900,1200,1500)
COLS=('room','expert','axis','landmark','tick','end','d','y','dhat','phat','prior_d','prior_p','clock_d','clock_p','persist')


def at(t,tick):
    return max(0,np.searchsorted(t,tick,side='right')-1)


def prior_fit(rows):
    ds=np.array([float(r['d']) for r in rows]);ys=np.array([int(r['y']) for r in rows])
    table=[]
    for lo in range(0,1800,150):
        weights=np.array([np.mean((r['t']>=lo)&(r['t']<lo+150)) for r in rows])
        table.append([np.average(ds,weights=weights),np.average(ys,weights=weights)] if weights.any() else [ds.mean(),ys.mean()])
    return ds.mean(),ys.mean(),table


def observations(r,e,priors):
    md,mp,byclock=priors
    out=[]
    t=r['t'];end=int(r['end'])
    last_seen=r['x'][:,1].cumsum()>0
    landmarks=[('fraction',f,end*f) for f in F]+[('clock',k,k) for k in CLOCK if end>k]
    for axis,k,tick in landmarks:
        j=at(t,tick)
        clock_d,clock_p=byclock[min(11,int(t[j])//150)]
        out.append(dict(zip(COLS,(r['room'],e,axis,k,int(t[j]),end,float(r['d']),int(r['y']),float(r['pred_d'][j]),
                  float(r['pred_p'][j]),md,mp,clock_d,clock_p,float(r['x'][j,0]) if last_seen[j] else md))))
    return out


def stats(rows):
    d=np.array([r['d'] for r in rows])
    y=np.array([r['y'] for r in rows]);p=np.array([r['phat'] for r in rows])
    pred=np.array([r['dhat'] for r in rows])
    def mae(key):
        return float(np.mean(np.abs(d-np.array([r[key] for r in rows]))))
    def brier(key):
        return float(np.mean((y-np.array([r[key] for r in rows]))**2))
    return dict(n=len(y),positives=int(y.sum()),median_ticks_remaining=float(np.median([r['end']-r['tick'] for r in rows])),mae=mae('dhat'),rmse=float(np.mean((d-pred)**2)**.5),
                mean_actual_d=float(d.mean()),mean_pred_d=float(pred.mean()),prior_mae=mae('prior_d'),
                clock_mae=mae('clock_d'),persist_mae=mae('persist'),brier=brier('phat'),prior_brier=brier('prior_p'),
                clock_brier=brier('clock_p'),auc=scoring.auc(2*y-1,p),distance_as_success_auc=scoring.auc(2*y-1,-pred),
                last_visible_as_success_auc=scoring.auc(2*y-1,-np.array([r['persist'] for r in rows])),
                within_025m=float(np.mean(abs(d-pred)<=.25)),abs_error_p90=float(np.quantile(abs(d-pred),.9)),
                failure_mae=float(abs(d[y==0]-pred[y==0]).mean()) if (y==0).any() else None,
                success_mae=float(abs(d[y==1]-pred[y==1]).mean()) if (y==1).any() else None,
                confident_win_n=int(sum(p>=.8)),confident_win_fail=int(sum((p>=.8)&(y==0))),
                confident_loss_n=int(sum(p<=.2)),confident_loss_win=int(sum((p<=.2)&(y==1))))


def volatility(r):
    ds=r['pred_d'];ps=r['pred_p'];d=np.abs(np.diff(ds));p=np.abs(np.diff(ps))
    landmarks=np.array([at(r['t'],r['end']*f) for f in F])
    errors=np.abs(ds[landmarks]-float(r['d']))
    middle=(r['t']>=r['end']*.25)&(r['t']<r['end']*.75)
    return dict(room=r['room'],n=len(d),y=int(r['y']),strictly_nonincreasing=bool(np.all(np.diff(errors)<=1e-6)),
                final_beats_early=bool(errors[-1]<errors[0]),mid_confident_win=bool(np.any(ps[middle]>=.8)),
                middle_p_range=float(np.ptp(ps[middle])) if middle.any() else 0.,mean_change_d=float(d.mean()) if len(d) else 0.,
                max_change_d=float(d.max()) if len(d) else 0.,max_change_p=float(p.max()) if len(p) else 0.,
                large_d=int(sum(d>.5)),large_p=int(sum(p>.3)))


def compare(original,branch,e):
    j=int(original['cut'])
    assert j==branch['cut'] and original['origin']==branch['origin']
    if j<0:
        assert original['final_hash']==branch['final_hash']
        return None
    assert np.array_equal(original['before'],branch['before'])
    assert np.array_equal(original['x'][:j],branch['x'][:j])
    a,b=original['after'],branch['after']
    shift=np.linalg.norm(np.array([a[:2]-b[:2],a[3:5]-b[3:5],a[5:]-b[5:]]),axis=1)
    dd=float(branch['d']-original['d'])
    ds=lambda r,k:float(r['pred_d'][k])
    ps=lambda r,k:float(r['pred_p'][k])
    r=dict(room=original['room'],expert=e,tick=int(original['t'][j]),actual_tick_delta=int(branch['post_tick']-original['post_tick']),
        max_xy_shift_m=float(shift.max()),object_shift_m=float(shift[0]),angle_delta_rad=abs(float(a[2]-b[2])),
        d0=float(original['d']),d1=float(branch['d']),delta_d=dd,y0=int(original['y']),y1=int(branch['y']),
        d_fork0=ds(original,j),d_fork1=ds(branch,j),p_fork0=ps(original,j),p_fork1=ps(branch,j))
    if j+1<len(original['t']) and j+1<len(branch['t']):
        r.update(d_next0=ds(original,j+1),d_next1=ds(branch,j+1),p_next0=ps(original,j+1),p_next1=ps(branch,j+1))
    for delay in (0,1,3,10):
        i=j+delay
        if i<len(original['t']) and i<len(branch['t']):
            r[f'pred_distance_delta_{delay}']=ds(branch,i)-ds(original,i)
            r[f'pred_success_delta_{delay}']=ps(branch,i)-ps(original,i)
    return r


def branch_forecasts(rows):
    result={}
    for delay in (0,1,3,10):
        group=[r for r in rows if f'pred_distance_delta_{delay}' in r]
        flips=[r for r in group if r['y0']!=r['y1']]
        diff=np.array([r['delta_d'] for r in group])
        delta=np.array([r[f'pred_distance_delta_{delay}'] for r in group])
        nontrivial=abs(diff)>.25
        result[str(delay)]=dict(n=len(group),flips=len(flips),n_delta_gt_025m=int(nontrivial.sum()),
            mean_abs_actual=float(abs(diff).mean()),mean_abs_prediction=float(abs(delta).mean()),
            direction_accurate_on_gt_025m=float(np.mean(np.sign(delta[nontrivial])==np.sign(diff[nontrivial]))) if nontrivial.any() else None,
            flip_direction_correct=sum(np.sign(r[f'pred_success_delta_{delay}'])==np.sign(r['y1']-r['y0']) for r in flips).item() if flips else 0,
            flip_mean_abs_pred_success_delta=float(np.mean([abs(r[f'pred_success_delta_{delay}']) for r in flips])) if flips else None)
    return result


def branch_stats(rows):
    if not rows:return dict(n=0)
    y,z=np.array([[r['y0'],r['y1']] for r in rows]).T
    dd=np.abs([r['delta_d'] for r in rows])
    shift=np.array([r['max_xy_shift_m'] for r in rows])
    return dict(n=len(rows),success0=int(y.sum()),success1=int(z.sum()),rescued=int(sum(z>y)),lost=int(sum(z<y)),
                flips=int(sum(z!=y)),flips_delta_d_le_005m=int(sum((z!=y)&(dd<=.05))),
                flips_delta_d_gt_05m=int(sum((z!=y)&(dd>.5))),both_failed_delta_d_gt_05m=int(sum((y==0)&(z==0)&(dd>.5))),
                median_delta_d_on_flips=float(np.median(dd[y!=z])) if (y!=z).any() else None,flip_rate=float(np.mean(z!=y)),delta_d_median=float(np.median(dd)),delta_d_mean=float(dd.mean()),
                delta_d_p90=float(np.quantile(dd,.9)),delta_d_gt_50cm=int(sum(dd>.5)),post_shift_median_m=float(np.median(shift)),
                post_shift_p90_m=float(np.quantile(shift,.9)),physical_ticks_changed=sum(r['actual_tick_delta']!=0 for r in rows),
                nonzero_xy=sum(r['max_xy_shift_m']>1e-7 for r in rows))


def boot_room(rows,func,seed=333):
    groups=defaultdict(list)
    for r in rows:groups[r['room']].append(r)
    # Each component: numerator and denominator, aggregated across all selected experts within a room.
    parts=np.array([func(rs) for rs in groups.values()])
    rng=np.random.default_rng(seed)
    draws=rng.integers(len(parts),size=(5000,len(parts)))
    s=parts[draws].sum(1)
    return (np.quantile(s[:,0]/s[:,1],[.025,.975])).tolist()


def csv_out(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def routing(start,priors):
    experts=sorted(priors)
    fixed_d=min(experts,key=lambda e:priors[e][0])
    fixed_p=max(experts,key=lambda e:priors[e][1])
    outcomes=defaultdict(list)
    for r in start:outcomes[r['room']].append(r)
    selected=defaultdict(list)
    for group in outcomes.values():
        assert len(group)==len(experts)
        options={r['expert']:r for r in group}
        for key,r in [('dev_best_d',options[fixed_d]),('dev_best_success',options[fixed_p]),
                      ('pred_distance',min(group,key=lambda x:x['dhat'])),('pred_success',max(group,key=lambda x:x['phat'])),
                      ('truth_distance',min(group,key=lambda x:x['d'])),('truth_success',max(group,key=lambda x:x['y']))]:
            selected[key].append(r)
    y0=np.array([r['y'] for r in selected['dev_best_success']])
    ix=np.random.default_rng(771).integers(len(y0),size=(5000,len(y0)))
    stats={}
    for k,rs in selected.items():
        y=np.array([r['y'] for r in rs])
        stats[k]=dict(wins=int(y.sum()),mean_terminal_m=float(np.mean([r['d'] for r in rs])),
                      vs_fixed_success=scoring.paired(y,y0,ix))
    return dict(fixed_distance= fixed_d,fixed_success=fixed_p,results=stats)


def report():
    torch.set_num_threads(1)
    DOC.mkdir(parents=True,exist_ok=True)
    experts=study.study.read(study.OUT/'protocol.json')['experts']
    data=[];paired=[];vs=[];priors={};branch_n={};examples={}
    for e in experts:
        priors[e]=prior_fit(fit.read('development',e))
        models=fit.load_models(e)
        test=fit.predict(models,fit.read('fresh',e))
        treated=fit.predict(models,fit.read('branch',e))
        assert len(test)==200 and len(treated)==56
        base={r['room']:r for r in test}
        for r in test:
            data.extend(observations(r,e,priors[e]));vs.append(dict(expert=e,**volatility(r)))
        pairs=[q for r in treated if (q:=compare(base[r['room']],r,e)) is not None]
        paired+=pairs;branch_n[e]=branch_stats(pairs)
        # Descriptive example of strongest change among the binary-loss pairs.
        opposite=[r for r in pairs if r['y0']>r['y1']]
        if opposite:
            example=max(opposite,key=lambda r:abs(r['delta_d']))
            alt=next(r for r in treated if r['room']==example['room'])
            examples[e]=dict(case=example,curves=[dict(t=r['t'].tolist(),pred_d=r['pred_d'].tolist(),pred_p=r['pred_p'].tolist())
                for r in (base[example['room']],alt)])
    fractions={str(f):stats([r for r in data if r['axis']=='fraction' and r['landmark']==f]) for f in F}
    expert_curve={e:{str(f):stats([r for r in data if r['axis']=='fraction' and r['landmark']==f and r['expert']==e]) for f in F} for e in experts}
    clocks={str(k):stats([r for r in data if r['axis']=='clock' and r['landmark']==k]) for k in CLOCK}
    survivors={str(k):stats([r for r in data if r['axis']=='clock' and r['landmark']==k and r['end']>1200]) for k in CLOCK if k<=1200}
    start=[r for r in data if r['axis']=='fraction' and r['landmark']==0]
    initially={(r['expert'],r['room']):r for r in start}
    delta_ci={}
    for f in F[1:]:
        group=[r for r in data if r['axis']=='fraction' and r['landmark']==f]
        delta_ci[str(f)]=boot_room(group,lambda rs: (sum(abs(r['d']-r['dhat'])-abs(r['d']-initially[r['expert'],r['room']]['dhat']) for r in rs),len(rs)))
    piece=[r for r in paired if 'd_next0' in r]
    for r in piece:r['next_pred_delta']=r['d_next1']-r['d_next0']
    physical=[r for r in paired if r['max_xy_shift_m']>1e-7]
    tiny=[r for r in physical if r['max_xy_shift_m']<=.02 and r['angle_delta_rad']<=.05]
    terminal=np.array([(r['d'],r['y']) for r in start])
    summary=dict(terminal_labels=dict(episodes=len(start),failures=int(sum(terminal[:,1]==0)),
             failures_inside_goal=int(sum((terminal[:,1]==0)&(terminal[:,0]<=.2))),
             failure_distance_median=float(np.median(terminal[terminal[:,1]==0,0])),
             success_distance_median=float(np.median(terminal[terminal[:,1]==1,0]))),fractions=fractions,per_cap=expert_curve,clock=clocks,same_survivors_through_1200=survivors,
            fraction_mae_delta_ci=delta_ci,routing=routing(start,priors),branch_per_cap=branch_n,
            branches=branch_stats(paired),branch_forecasts=branch_forecasts(paired),branches_changed_geometry=branch_stats(physical),tiny_nonzero_pose_shift=branch_stats(tiny),
            branch_flip_ci=boot_room(paired,lambda rs:(sum(r['y0']!=r['y1'] for r in rs),len(rs))),
            branch_next=dict(n=len(piece),mean_abs_pred_delta=float(np.mean([abs(r['next_pred_delta']) for r in piece])),
                mean_abs_actual_delta=float(np.mean([abs(r['delta_d']) for r in piece])),
                flip_only_mean_abs_pred_delta=float(np.mean([abs(r['next_pred_delta']) for r in piece if r['y0']!=r['y1']])),
                flip_only_mean_abs_actual_delta=float(np.mean([abs(r['delta_d']) for r in piece if r['y0']!=r['y1']]))),
            volatility=dict(nonincreasing_distance_errors=sum(r['strictly_nonincreasing'] for r in vs),
                final_beats_initial=sum(r['final_beats_early'] for r in vs),n_episodes=len(vs),
                mid_confident_n=sum(r['mid_confident_win'] for r in vs),
                mid_confident_failures=sum(r['mid_confident_win'] and not r['y'] for r in vs),
                median_mid_p_range=float(np.median([r['middle_p_range'] for r in vs])),
                median_episode_max_d=float(np.median([r['max_change_d'] for r in vs])),
                median_episode_max_p=float(np.median([r['max_change_p'] for r in vs])),
                change_gt_05m=sum(r['large_d'] for r in vs),change_gt_03prob=sum(r['large_p'] for r in vs),
                adjacent_pairs=sum(r['n'] for r in vs)))
    study.study.save(DOC/'results.json',summary)
    study.study.save(DOC/'examples.json',examples)
    csv_out(study.OUT/'predictions.csv',data)
    csv_out(DOC/'branches.csv',paired)
    draw(summary,examples)
    study.study.save(study.OUT/'FINISHED.json',dict(experts=len(experts),natural_runs=7*600,branch_attempts=392,
                                              branch_applied=len(paired),results=str(DOC/'results.json')))
    print(summary['branches'],summary['routing'],flush=True)


def draw(s,examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.2))
    for e,rs in s['per_cap'].items():
        axs[0].plot(F,[rs[str(f)]['mae'] for f in F],marker='.',label=e)
        axs[1].plot(F,[rs[str(f)]['brier'] for f in F],marker='.')
    for ax,title in zip(axs,['Terminal distance MAE (m)','Task success Brier score']):
        ax.set_title(title);ax.set_xlabel('Fraction of actual episode elapsed');ax.grid(alpha=.2)
    axs[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(DOC/'forecast.png',dpi=155);plt.close(fig)
    if not examples:return
    e=max(examples,key=lambda k:abs(examples[k]['case']['delta_d']))
    item=examples[e];c=item['case']
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for i,(curve,name) in enumerate(zip(item['curves'],('Original','One tick shorter'))):
        line,=axs[0].plot(curve['t'],curve['pred_d'],label=name)
        axs[0].axhline(c['d'+str(i)],color=line.get_color(),ls=':',alpha=.8)
        axs[1].plot(curve['t'],curve['pred_p'],label=name)
    for ax in axs:ax.axvline(c['tick'],color='grey',ls='--');ax.set_xlabel('Physics tick');ax.legend(fontsize=8)
    axs[0].set_ylabel('Predicted terminal distance (m)');axs[1].set_ylabel('Predicted success probability')
    fig.suptitle(f"{e} / {c['room']} — largest terminal distance change among observed losses")
    fig.tight_layout();fig.savefig(DOC/'paired_forecast.png',dpi=150);plt.close(fig)


if __name__=='__main__':report()
