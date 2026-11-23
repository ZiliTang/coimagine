"""Paired closed-loop report for explicit-contact CAPs."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from tools import cap_push_phase_comparison as study

OUT,DOC=study.OUT,study.DOC
CAPS,ARMS=study.CAPS,study.ARMS
read,save=study.cap.read,study.cap.save


def load(panel):
    fixtures=read(OUT/(panel+'.json'))
    ids=sorted(fixtures)
    counts=np.array([1+len(fixtures[room]['movable_objects']) for room in ids])
    data={}
    for sid in CAPS:
        data[sid]={
            arm:[read(OUT/'runs'/panel/sid/arm/(room+'.json')) for room in ids]
            for arm in ARMS}
    return ids,counts,data


def outcomes(rows):
    return np.array([int(r['success']) for r in rows])


def paired_mean(a,b,counts):
    delta=np.mean(b-a,axis=0)
    rng=np.random.default_rng(31)
    draws=np.zeros(10000)
    for n in sorted(set(counts)):
        values=delta[counts==n]
        draws+=rng.choice(values,(len(draws),len(values))).sum(axis=1)
    return dict(first=float(100*a.mean()),second=float(100*b.mean()),delta=float(100*delta.mean()),
                ci=(100*np.quantile(draws,[.025,.975])/len(counts)).tolist(),rooms=len(counts))


def summary(panel,ids,counts,data):
    result={}
    ys={}
    room_rows=[]
    changes=[]
    def comparison(before,after):
        return study.paired(before,after,counts)
    for sid,rows in data.items():
        reference=rows['automatic']
        ys[sid]={k:outcomes(r) for k,r in rows.items()}
        raw=ys[sid]
        result[sid]={}
        for arm,v in rows.items():
            cnt=Counter()
            for r in v:
                assert 0<r['ticks']<=study.MAX_TICKS and 0<r['decisions']<=study.MAX_DECISIONS
                assert r['termination'] is not None or r['ticks']==study.MAX_TICKS or r['decisions']==study.MAX_DECISIONS
                cnt.update(r['counts'])
            changed=sum(bool(r['changes']) for r in v)
            if arm.startswith('feedback'):
                for r,base in zip(v,reference):
                    if not r['changes']:
                        fields=('history','success','ticks','decisions')
                        assert tuple(r[k] for k in fields)==tuple(base[k] for k in fields),(panel,sid,arm,r,base)
            result[sid][arm]=dict(wins=int(raw[arm].sum()),counts=dict(cnt),exposed=changed,
                 mean_ticks=float(np.mean([r['ticks'] for r in v])),mean_boundaries=float(np.mean([r['decisions'] for r in v])),
                 vs_coupled=comparison(raw['coupled'],raw[arm]),vs_automatic=comparison(raw['automatic'],raw[arm]))
            for room,objects,r in zip(ids,counts,v):
                for change in r['changes']:
                    changes.append(dict(room=room,cap=sid,arm=arm,**change))
                room_rows.append(dict(room=room,objects=int(objects),cap=sid,arm=arm,ticks=r['ticks'],boundaries=r['decisions'],
                     success=int(r['success']),termination=r['termination'],contact_ready=r['counts'].get('contact_ready',0),
                     contact_timeout=r['counts'].get('contact_contact_timeout',0),contact_ticks=r['counts'].get('contact_ticks',0),
                     push_ticks=r['counts'].get('push_ticks',0),retarget=r['counts'].get('retarget_contact',0),
                     retime=r['counts'].get('retime_push',0)))
    arrays={a:np.stack([ys[s][a] for s in CAPS]) for a in ARMS}
    avg={}
    for prev,now in [('coupled','automatic'),('automatic','feedback'),('automatic','feedback_precision'),
                     ('feedback','feedback_precision'),('coupled','feedback'),('coupled','feedback_precision')]:
        avg[now+'_vs_'+prev]=paired_mean(arrays[prev],arrays[now],counts)
    result['equal_cap_mean']=avg
    result['two_cap_union']={a:int((arr>0).any(0).sum()) for a,arr in arrays.items()}
    path=DOC/f'{panel}_rooms.csv'
    with path.open('w',newline='') as file:
        out=csv.DictWriter(file,fieldnames=list(room_rows[0]))
        out.writeheader()
        out.writerows(room_rows)
    save(DOC/f'{panel}_interventions.json',changes)
    return result


def figures(results):
    import matplotlib.pyplot as plt

    names=['Combined PUSH','Automatic continuation','Retry with contact feedback','Feedback and near-goal pacing']
    fig,ax=plt.subplots(figsize=(8.3,3.8),layout='constrained')
    n=len(ARMS)
    labels=np.arange(n)
    colors=['#427383','#b86342']
    for j,sid in enumerate(CAPS):
        row=results['held'][sid]
        values=[row[a]['wins']/2 for a in ARMS]
        bars=ax.bar(labels+(j-.5)*.31,values,width=.31,color=colors[j],label=sid)
        ax.bar_label(bars,labels=[str(row[a]['wins']) for a in ARMS],padding=3,fontsize=9)
    ax.set_ylabel('Successful rooms (%)')
    ax.set_xticks(labels,names,rotation=9,ha='right')
    ax.set_ylim(0,70)
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False)
    ax.set_title('Fresh evaluation panel · 200 matched rooms per CAP')
    fig.savefig(DOC/'success.png',dpi=150)
    plt.close(fig)


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    result={p:summary(p,*load(p)) for p in ('development','held')}
    figures(result)
    save(DOC/'results.json',result)
    save(OUT/'FINISHED.json',dict(held=result['held']['equal_cap_mean'],dev=result['development']['equal_cap_mean']))
    for panel,s in result.items():
        print(panel,{cap:{a:s[cap][a]['wins'] for a in ARMS} for cap in CAPS},s['equal_cap_mean'],flush=True)


if __name__=='__main__':
    main()
