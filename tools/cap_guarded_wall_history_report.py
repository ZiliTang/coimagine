"""Summarize paired decisions and physical outcomes in the static-map validation."""
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from tools.cap_guarded_wall_history import ROOT, ARMS, loop

DEST = Path('docs/research/key_findings/2026-11-25_guarded_wall_history')


def report():
    plan = loop.read(ROOT/'plan.json')
    panels = {s: [loop.read(ROOT/s/(c['id']+'.json')) for c in plan[s]] for s in ('validation','development')}
    totals, strata, groups, records = {}, {}, Counter(), []
    for arm in (*ARMS,'fixed_short'):
        total = Counter()
        sub = {}
        for p in panels['validation']:
            if p['status']!='ready':
                continue
            a,b = p['branches']
            group = ('neither','long_only','short_only','both')[2*int(a['safe'])+int(b['safe'])]
            if arm == ARMS[0]:
                groups[group]+=1
            decisions = p['decisions'][arm] if arm in ARMS else ['admit','reject']
            i = 1 if decisions[1]=='admit' else (0 if decisions[0]=='admit' else None)
            choice = 'wait' if i is None else ('safe' if p['branches'][i]['safe'] else 'unsafe')
            total[choice]+=1
            counts = sub.setdefault(group,Counter())
            counts[choice]+=1
            if decisions==['reject','reject']:
                counts['reject_both']+=1
            if decisions==['admit','reject']:
                counts['admit_short_reject_long']+=1
            if decisions==['admit','defer']:
                counts['admit_short_defer_long']+=1
            if i==1:
                counts['long_chosen']+=1
            for branch,d in zip(p['branches'],decisions):
                total['decision_'+d]+=1
                if d=='admit':
                    total['valid_admits' if branch['safe'] else 'invalid_admits']+=1
            records.append(dict(id=p['id'],group=group,method=arm,short=decisions[0],long=decisions[1],
                                selected_dose=None if i is None else p['branches'][i]['dose'],outcome=choice,
                                current_gap=p['walls']['current'][0] if p['walls']['current'] else None,
                                history_gap=p['walls']['history'][0] if p['walls']['history'] else None))
        totals[arm],strata[arm] = total,sub
    current = [r for r in records if r['method']=='current']
    history = [r for r in records if r['method']=='history']
    changes = Counter()
    for a,b in zip(current,history,strict=True):
        assert a['id']==b['id']
        changes[a['outcome']+'->'+b['outcome']]+=1
    rng = np.random.default_rng(765229)
    bootstrap = rng.integers(len(history),size=(10000,len(history)))
    intervals={}
    for kind in ('safe','unsafe'):
        delta=np.array([int(b['outcome']==kind)-int(a['outcome']==kind) for a,b in zip(current,history)])
        intervals[kind]=[float(delta.mean()),*np.quantile(delta[bootstrap].mean(axis=1),[.025,.975]).tolist()]
    inventory={s:dict(status=Counter(p['status'] for p in rows), with_geometry={
        arm:sum(p['walls'][arm] is not None for p in rows if p['status']=='ready') for arm in ARMS})
        for s, rows in panels.items()}
    stats=dict(inventory=inventory, groups=groups, totals=totals, strata=strata, changes=changes,
               paired_difference_and_interval=intervals)
    DEST.mkdir(parents=True,exist_ok=True)
    loop.seal(DEST/'metrics.json',stats)
    with (DEST/'decisions.csv').open('w') as f:
        w=csv.DictWriter(f,records[0].keys(),lineterminator='\n');w.writeheader();w.writerows(records)
    return stats


if __name__=='__main__':
    print(report())
