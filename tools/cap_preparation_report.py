"""Summarize the fixed preparation comparisons with the room as the paired unit."""
import csv
from pathlib import Path

import numpy as np

from tools import cap_preparation_pairs as experiment

OUT = experiment.OUT
DOC = experiment.study.phase.cap.ROOT/'docs/research/key_findings/2026-11-23_cap_preparation_pairs'
ARMS = ('A','B','A3','A10','C10','wait10','event')
FIELDS = ('stable','target_push','progress','push_along_m','repeated_stall')
read,save = experiment.study.read,experiment.study.save


def estimate(x):
    x=np.asarray(x,dtype=float)
    if not len(x):
        return None
    rng=np.random.default_rng(713)
    bounds=np.quantile(x[rng.integers(len(x),size=(5000,len(x)))].mean(1),[.025,.975])
    return dict(n=len(x),mean=float(x.mean()),ci=bounds.tolist())


def paired(rows, a, b):
    diff=[r[a]['success']-r[b]['success'] for r in rows]
    return dict(effect=estimate(diff),gain=diff.count(1),loss=diff.count(-1))


def local(rows,a,b):
    keep=[r for r in rows if r[a]['window'] and r[b]['window']]
    if b in ('wait10','C10'):
        assert all(r[a]['hand']['tick']==r[b]['hand']['tick'] for r in keep)
    out = {f:dict(a=estimate([r[a]['window'][f] for r in keep]),
                  b=estimate([r[b]['window'][f] for r in keep]),
                  diff=estimate([r[a]['window'][f]-r[b]['window'][f] for r in keep])) for f in FIELDS}
    for name,test in [('local_advance_2cm',lambda w:w['progress']>=.02),
                      ('contact_and_advance',lambda w:w['stable'] and w['progress']>=.02)]:
        diffs=[int(test(r[a]['window']))-int(test(r[b]['window'])) for r in keep]
        out[name]=dict(a=sum(test(r[a]['window']) for r in keep),b=sum(test(r[b]['window']) for r in keep),
                       gain=diffs.count(1),loss=diffs.count(-1),effect=estimate(diffs))
    # Is the branch that makes more progress also the branch that wins?
    flips=[r for r in keep if r[a]['success']!=r[b]['success']]
    margin=.02
    aligned=[(r[a]['window']['progress']-r[b]['window']['progress'])*(r[a]['success']-r[b]['success']) for r in flips]
    out['flips_progress']=dict(n=len(flips),correct=sum(v>margin for v in aligned),wrong=sum(v< -margin for v in aligned),
                               within=sum(abs(v)<=margin for v in aligned))
    still = [r for r in keep if r['B']['window'] and
             all(abs(r[k]['hand']['true_gap_m']-r['B']['hand']['true_gap_m'])<=.02 for k in (a,b))]
    out['both_handoffs_within_2cm_of_fork_goal_gap']=dict(n=len(still),terminal=paired(still,a,b),
        progress_diff=estimate([r[a]['window']['progress']-r[b]['window']['progress'] for r in still]))
    return out


def fold_result(data):
    rows=list(data.values())
    both_fail=[r for r in rows if not r['A']['success'] and not r['B']['success']]
    result=dict(n=len(rows),double_fail=len(both_fail),fork=sum(bool(r['A']['first']) for r in rows),
                eligible10=sum('10' in r['A']['times'] for r in rows),arms={},local={})
    for a in ARMS:
        result['arms'][a]=dict(success=sum(r[a]['success'] for r in rows),
                               new=sum(r[a]['success'] for r in both_fail),vsA=paired(rows,a,'A'),
                               hand=sum(bool(r[a]['hand'] and r[a]['hand']['legal']) for r in rows))
    result['fixed3or10_new']=sum(r['A3']['success'] or r['A10']['success'] for r in both_fail)
    result['unique_prep']=sum(r['A10']['success'] and not any(r[a]['success'] for a in ('C10','wait10')) for r in both_fail)
    result['time_matched']={a:paired(rows,'A10',a) for a in ('C10','wait10','B')}
    result['event_vs_fixed']={a:paired(rows,'event',a) for a in ('A3','A10')}
    result['local']={a:local(rows,'A10',a) for a in ('C10','wait10','B')}
    result['event']={}
    for category in ('station','new_push'):
        keep=[r for r in rows if r['A']['event'] and r['A']['event']['reason']==category]
        result['event'][category]=dict(n=len(keep),vsA=paired(keep,'event','A'))
    result['handoff_gap_diff_cm']={a:estimate([100*(r['A10']['hand']['true_gap_m']-r[a]['hand']['true_gap_m']) for r in rows
                                              if r['A10']['window'] and r[a]['window']]) for a in ('C10','wait10','B')}
    return result


def local_readout(train, test):
    # One A10 checkpoint per room, same four public inputs for all three targets.
    def extract(data):
        out=[]
        for r in data.values():
            m=r['A10']
            if not m['window']:
                continue
            pub=m['hand']['public']
            d=pub['distance']
            x=[float(d is not None), min(d,3.) if d is not None else 0.,
               min(pub['clearance'],1.) if d is not None else 0.,float(pub['push'])]
            out.append((x,m['window']['stable'],m['window']['progress']>=.02,
                        m['success'],m['success']-r['A']['success']))
        return out
    train,test=extract(train),extract(test)
    result={}
    for j,name in enumerate(('stable','advance_2cm','success','handoff_advantage'),1):
        u,v=train,test
        if j==4:
            u=[r for r in u if r[j]!=0]
            v=[r for r in v if r[j]!=0]
        x,xx=(np.array([r[0] for r in subset]) for subset in (u,v))
        y,yy=(np.array([r[j]>0 for r in subset],float) for subset in (u,v))
        push_only=xx[yy==1,3,None]-xx[yy==0,3]
        push_auc=float(((push_only>0)+.5*(push_only==0)).mean())
        mean,scale=x.mean(0),np.maximum(x.std(0),.05)
        x,xx=(np.column_stack([np.ones(len(a)),(a-mean)/scale]) for a in (x,xx))
        beta=np.zeros(x.shape[1])
        ridge=np.diag([0.]+[1.]*(len(beta)-1))
        for _ in range(30):
            p=1/(1+np.exp(-x@beta))
            step=np.linalg.solve(x.T@((p*(1-p))[:,None]*x)+ridge,x.T@(p-y)+ridge@beta)
            beta-=step
            if abs(step).max()<1e-9:break
        pred=1/(1+np.exp(-xx@beta))
        pair=pred[yy==1,None]-pred[yy==0]
        result[name]=dict(train=len(y),test=len(yy),positives=int(yy.sum()),
                          auc=float(((pair>0)+.5*(pair==0)).mean()),push_flag_auc=push_auc,
                          brier=float(((pred-yy)**2).mean()),constant_brier=float(((y.mean()-yy)**2).mean()),
                          brier_gain=estimate((y.mean()-yy)**2-(pred-yy)**2))
        if name=='advance_2cm':
            switch=pred>=.5
            delta=np.asarray([r[4] for r in v])*switch
            result[name]['exploratory_gate']=dict(switched=int(switch.sum()),gain=int(sum(delta>0)),
                                                    loss=int(sum(delta<0)),net=int(sum(delta)))
    return result


def plot(summary):
    from matplotlib import pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    x=np.arange(7)
    for offset,fold,color in [(-.18,'diagnostic','#90a7b7'),(.18,'fresh','#146b64')]:
        res=summary[fold]
        vals=[res['arms'][arm]['success'] for arm in ARMS]
        bars=axes[0].bar(x+offset,vals,width=.36,color=color,label=fold)
        axes[0].bar_label(bars,fontsize=8)
    axes[0].set(xticks=x,xticklabels=['A','B','A3','A10','C10','Brake','Event'],
                ylabel='Rooms solved / 200',title='Closed-loop outcomes')
    axes[0].set_ylim(0,135)
    axes[0].legend(frameon=False)
    for dy,fold,color in [(-.1,'diagnostic','#738ea2'),(.1,'fresh','#146b64')]:
        for i,arm in enumerate(('wait10','C10','B')):
            m=summary[fold]['local'][arm]['progress']['diff']
            value=100*m['mean']
            low,high=np.array(m['ci'])*100
            axes[1].errorbar(value,i+dy,xerr=[[value-low],[high-value]],fmt='o',color=color,
                             capsize=3,label=fold if i==0 else None)
    axes[1].axvline(0,color='.6',lw=1)
    axes[1].set(yticks=[0,1,2],yticklabels=['A10 vs brake','A10 vs C10','A10 vs B now'],
                xlabel='Paired difference in goal progress (cm)',title='First 120 ticks after B takes over')
    axes[1].legend(frameon=False)
    axes[1].set_ylim(-.5,2.7)
    fig.savefig(DOC/'paired_outcomes.png',dpi=150)
    plt.close(fig)


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    summary={}
    panels={}
    table=[]
    examples={}
    for fold in ('diagnostic','fresh'):
        data={p.stem:read(p) for p in sorted((OUT/'runs'/fold).glob('*.json'))}
        panels[fold]=data
        assert len(data)==200
        for room,r in data.items():
            for arm,m in r.items():
                hand=m['hand'] or {}
                w=m['window'] or {}
                pub=hand.get('public',{})
                if w:
                    assert w['ticks']==min(m['ticks']-hand['tick'],experiment.WINDOW),(room,arm)
                table.append(dict(panel=fold,room=room,arm=arm,success=m['success'],ticks=m['ticks'],decisions=m['decisions'],
                                  fork_tick=(m['first'] or {}).get('tick'),handoff_tick=hand.get('tick'),
                                  hand_legal=hand.get('legal'),true_gap_at_handoff=hand.get('true_gap_m'),
                                  station_m=pub.get('distance'),clearance_m=pub.get('clearance'),proposed_target_push=pub.get('push'),
                                  trigger=(r['A']['event'] or {}).get('reason'),window_ticks=w.get('ticks'),
                                  **{f:w.get(f) for f in FIELDS}))
        summary[fold]=fold_result(data)
        keys = [k for k,r in data.items() if r['A10']['success'] and not any(r[a]['success'] for a in ('A','B','C10','wait10'))]
        losses=[k for k,r in data.items() if not r['A10']['success'] and r['C10']['success'] and r['B']['success']]
        selected=keys[:2]+losses[:1]+(['test_a_dev_2_05'] if fold=='diagnostic' else [])
        examples[fold]={k:data[k] for k in selected}
    with (DOC/'rooms.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    summary['public_readout']=local_readout(panels['diagnostic'],panels['fresh'])
    save(DOC/'results.json',summary)
    plot(summary)
    save(DOC/'examples.json',examples)
    save(DOC/'protocol.json',read(OUT/'protocol.json'))
    save(OUT/'FINISHED.json',dict(rooms=400,conditions=len(table),results=str(DOC/'results.json')))


if __name__=='__main__':
    main()
