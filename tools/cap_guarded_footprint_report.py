"""Geometry errors and paired choices for the unchanged and footprint-aware gate."""
from collections import Counter,defaultdict
import csv
import math
from pathlib import Path

import numpy as np
from tools.cap_guarded_footprint import ROOT, ARMS, base, loop

DEST=Path('docs/research/key_findings/2026-11-25_guarded_footprint')


def selection(decisions):
    if decisions[1]=='admit':
        return 1
    if decisions[0]=='admit':
        return 0
    return None


def summarize(packets):
    totals={a:Counter() for a in ARMS}
    groups=Counter()
    changes=defaultdict(Counter)
    sub=defaultdict(lambda:defaultdict(Counter))
    geometry=[]
    rows=[]
    perf=[]
    for p in packets:
        if p['status']!='ready':
            continue
        short,long=p['branches']
        group=('neither','long_only','short_only','both')[2*int(short['safe'])+int(long['safe'])]
        groups[group]+=1
        truth=short['outcome']['start_wall_gap_m']
        visible=p['walls']['visible']
        interval=p['interval']
        if interval:
            lo,hi=interval
            geometry.append([truth,lo,hi])
        row=dict(id=p['id'],group=group,short_safe=short['safe'],long_safe=long['safe'],
                 physical_gap=truth,visible_gap=visible[0] if visible else None,
                 envelope_lo=interval[0] if interval else None,envelope_hi=interval[1] if interval else None)
        outcomes=[]
        for arm in ARMS:
            d=p['decisions'][arm]
            i=selection(d)
            choice='wait' if i is None else ('safe' if p['branches'][i]['safe'] else 'unsafe')
            outcomes.append(choice)
            totals[arm][choice]+=1
            sub[group][arm][choice]+=1
            row.update({arm+'_short':d[0],arm+'_long':d[1],arm+'_choice':choice})
            if d==['admit','reject']:
                sub[group][arm]['admit_short_reject_long']+=1
        changes[group][outcomes[0]+'->'+outcomes[1]]+=1
        perf.append([int(outcomes[1]==k)-int(outcomes[0]==k) for k in ('safe','unsafe')])
        rows.append(row)
    arr=np.array(geometry)
    truth,lo,hi=arr.T
    diff=np.array(perf)
    rng=np.random.default_rng(253027)
    ci=np.quantile(diff[rng.integers(len(diff),size=(8000,len(diff)))].mean(axis=1),[.025,.975],axis=0)
    metrics=dict(inventory=Counter(p['status'] for p in packets),groups=groups,totals=totals,sub=sub,changes=changes,
                 visible_count=sum(r['visible_gap'] is not None for r in rows), interval_count=len(arr),
                 interval_contains=int(((lo<=truth)&(truth<=hi)).sum()),optimistic_lower=int((lo>truth).sum()),
                 median_signed_lower_m=float(np.median(lo-truth)),median_abs_lower_m=float(np.median(abs(lo-truth))),
                 median_abs_visible_m=float(np.median(abs(hi-truth))),median_interval_width_m=float(np.median(hi-lo)),
                 delta=dict(zip(('safe','unsafe'),np.column_stack((diff.mean(axis=0),ci.T)).tolist())))
    return metrics,rows


def run():
    DEST.mkdir(parents=True,exist_ok=True)
    plan=loop.read(ROOT/'plan.json')
    result={}
    with (DEST/'paired.csv').open('w') as f:
        writer=None
        for split in ('review','fresh'):
            packets=[loop.read(ROOT/split/(c['id']+'.json')) for c in plan[split]]
            metrics,rows=summarize(packets)
            result[split]=metrics
            for row in rows:
                row['split']=split
            if writer is None:
                writer=csv.DictWriter(f,rows[0].keys(),lineterminator='\n')
                writer.writeheader()
            writer.writerows(rows)
    loop.seal(DEST/'metrics.json',result)
    # Exploratory audit after the primary comparison. Use development physics
    # only to set the margin; keep it out of the frozen decision experiment.
    errors=[]
    for case in plan['development']:
        layout=loop.read(ROOT/'development'/(case['id']+'.json'))
        if layout['status']=='ready' and layout['interval']:
            original=loop.read(base.ROOT/'development'/(case['id']+'.json'))
            errors.append(layout['interval'][0]-original['branches'][0]['outcome']['start_wall_gap_m'])
    rank=math.ceil(.90*(len(errors)+1))
    margin=max(0.,sorted(errors)[rank-1])
    audit=dict(development_n=len(errors),rank=rank,margin_m=margin)
    for split in ('review','fresh'):
        layouts=[loop.read(ROOT/split/(case['id']+'.json')) for case in plan[split]]
        covered=total=0
        for layout in layouts:
            if layout['status']=='ready' and layout['interval']:
                low,high=layout['interval']
                truth=layout['branches'][0]['outcome']['start_wall_gap_m']
                covered+=int(low-margin<=truth<=high)
                total+=1
        audit[split]=dict(covered=covered,total=total)
    loop.seal(DEST/'margin_audit.json',audit)
    return result


if __name__=='__main__':
    for split,metrics in run().items():
        print(split,metrics)
