"""Post-hoc window length control using identical training and test rooms at 10 and 30."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import torch
from tools import cap_route_information_fit as fit

OUT=fit.data.OUT/'window_models'


def distance_score(row,h,align_time=False):
    final=[]
    idx=fit.data.HORIZONS.index(h)
    shared_tick=min(float(row[k][row['ends'][i][idx]-1,11]) for i,k in enumerate(('A','B')))
    for i,key in enumerate(('A','B')):
        seq=torch.cat((row['prefix'],row[key][:row['ends'][i][fit.data.HORIZONS.index(h)]]))
        visible=seq[:,3]>0
        if align_time:visible &= seq[:,11]<=shared_tick
        if not visible.any():return 0.
        final.append(float(seq[visible][-1,0]))
    return final[0]-final[1]


def main():
    torch.set_num_threads(1)
    fit.TRAIN=fit.load('development')
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(partial(fit.fit,cohort=30,directory=OUT),[(10,'future',s) for s in fit.data.SEEDS]))
    result={}
    for fold in ('held','fresh'):
        rows=[r for r in fit.load(fold) if fit.eligible(r,30)]
        y=np.array([r['labels'][1]-r['labels'][0] for r in rows])
        preds={}
        for h,directory in ((10,OUT),(30,fit.MODELS)):
            preds[str(h)]=np.stack([fit.score(m,rows,'future',h) for m in fit.ensemble(h,'future',directory)]).mean(0)
            preds['visible_gap_'+str(h)]=np.array([distance_score(r,h) for r in rows])
            preds['time_aligned_gap_'+str(h)]=np.array([distance_score(r,h,True) for r in rows])
        aucs={k:fit.old_report.auc(y,s) for k,s in preds.items()}
        rng=np.random.default_rng(423)
        diffs=[]
        gap_diffs=[]
        for _ in range(4000):
            ix=rng.integers(len(y),size=len(y))
            a,b=[fit.old_report.auc(y[ix],preds[h][ix]) for h in ('10','30')]
            if a is not None and b is not None:
                diffs.append(b-a)
                early,late=[fit.old_report.auc(y[ix],preds['visible_gap_'+h][ix]) for h in ('10','30')]
                gap_diffs.append([late,late-early,late-b])
        gap_select={}
        for h in fit.data.HORIZONS:
            allrooms=[r for r in fit.load(fold) if fit.eligible(r,h)]
            v=np.array([distance_score(r,h) for r in allrooms])
            m=fit.statistics(allrooms,v[None,:])
            gap_select[str(h)]={k:m[k] for k in ('n','only_A','only_B','auc','correct','wins','fixed_A','fixed_B','rescued','lost')}
        result[fold]=dict(rooms=len(y),disagreements=int(sum(y!=0)),auc=aucs,
                          delta_ci95=np.quantile(diffs,[.025,.975]).tolist(),gap_ci95=dict(zip(('auc30','vs10','vs_gru30'),np.quantile(gap_diffs,[.025,.975],axis=0).T.tolist())),gap_selection=gap_select)
    fit.data.study.save(fit.data.DOC/'window_control.json',result)
    print(result)


if __name__=='__main__':main()
