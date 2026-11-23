"""Aggregate frozen one-step successor decisions at the independent scene level."""
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .cap_successor_value import CAPS, OUT, read, save
from .cap_trace_value_long import sha

KEY = Path('docs/research/key_findings/2026-11-19_successor_value')


def action_name(action):
    return f"{action['target_id']} / {action['a0']},{action['a1']} / {action['ticks']} ticks"


def pair_row(case, suffix):
    a,b=case['branches'][suffix]
    f=a['fork']
    x,y=a['successor'],b['successor']
    metric='max_v' if suffix=='ga' else 'v009'
    score=[x[metric],y[metric]]
    pick=int(score[1]>score[0])
    result=[a['success'],b['success']]
    duration_only=all(f['candidates'][0][k]==f['candidates'][1][k] for k in ('target_id','a0','a1'))
    return dict(case_id=case['case_id'],movable_count=case['movable_count'],suffix=suffix,
                min_boundary=case['min_boundary'],boundary=f['index'],tick=f['tick'],
                prefix_hash=f['hash'],action0=action_name(f['candidates'][0]),action1=action_name(f['candidates'][1]),
                caps0=','.join(CAPS[i] for i in f['sources'][0]),caps1=','.join(CAPS[i] for i in f['sources'][1]),
                period0=x['periods'],period1=y['periods'],successor0=x['hash'],successor1=y['hash'],
                duration_only=duration_only,cap_fallback=a['fallback']+b['fallback'],
                terminal=int(x['terminal'] or y['terminal']),q0=f['q'][0],q1=f['q'][1],
                s0=score[0],s1=score[1],picked=pick,y0=result[0],y1=result[1],
                y_chosen=result[pick],distinct=int(result[0]!=result[1]),
                ticks0=a['ticks'],ticks1=b['ticks'])


def percentile(rows, column):
    if not rows:
        return None
    rng=np.random.default_rng(47)
    columns={k:np.array([column(r) for r in rows if r['movable_count']==k]) for k in sorted({r['movable_count'] for r in rows})}
    draws=sum(rng.choice(a,size=(20000,len(a))).sum(1) for a in columns.values())/len(rows)
    return np.round(100*np.quantile(draws,[.025,.975]),2).tolist()


def results(rows):
    diff=[r for r in rows if r['distinct']]
    gained=sum(r['y_chosen']>r['y0'] for r in rows)
    lost=sum(r['y_chosen']<r['y0'] for r in rows)
    n=gained+lost
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(gained,lost)+1))/2**n)
    return dict(pairs=len(rows),both_win=sum(r['y0']*r['y1'] for r in rows),
                both_lose=sum(not(r['y0'] or r['y1']) for r in rows),
                informative=len(diff),q_correct=sum(r['y0'] for r in diff),
                v_correct=sum(r['y_chosen'] for r in diff),q_success=sum(r['y0'] for r in rows),
                v_success=sum(r['y_chosen'] for r in rows),union=sum(max(r['y0'],r['y1']) for r in rows),
                reversed=sum(r['picked'] for r in rows),rescues=gained,losses=lost,
                identical_q=sum(r['q0']==r['q1'] for r in rows),
                identical_v=sum(r['s0']==r['s1'] for r in rows),paired_p=p,
                success_delta_ci_pp=percentile(rows,lambda r:r['y_chosen']-r['y0']),
                informative_delta_ci_pp=percentile(diff,lambda r:r['y_chosen']-r['y0']))


def report():
    cases=[read(OUT/'cases'/(c['case_id']+'.json')) for c in read(OUT/'panel.json')]
    assert len(cases)==len({c['case_id'] for c in cases})==200
    data=[pair_row(c,s) for c in cases if c['eligible'] for s in ('ga','009')]
    nofork=[c for c in cases if not c['eligible']]
    stats=dict(scenes=len(cases),no_fork=len(nofork),
               no_fork_success=sum(c['branches']['ga'][0]['success'] for c in nofork),
               no_fork_ids=[c['case_id'] for c in nofork],
               executions=sum(sum(map(len,c['branches'].values())) for c in cases))
    for suffix in ('ga','009'):
        all_rows=[r for r in data if r['suffix']==suffix]
        rows=[r for r in all_rows if not r['cap_fallback']]
        stats[suffix]=dict(all=results(rows),excluded_fallback=len(all_rows)-len(rows),
                           nonterminal=results([r for r in rows if not r['terminal']]),
                           duration_only=results([r for r in rows if r['duration_only']]),
                           different_command=results([r for r in rows if not r['duration_only']]))
    votes=Counter()
    legal_counts=Counter()
    for case in cases:
        if case['eligible']:
            for branch in case['branches']['ga']:
                s=branch['successor']
                if not s['terminal']:
                    legal_counts[sum(s['legal'])]+=1
                if not s['terminal'] and any(s['legal']):
                    i=max((j for j in range(7) if s['legal'][j]),key=lambda j:s['v'][j])
                    votes[CAPS[i]]+=1
    stats['next_max_v_cap']=votes
    stats['next_legal_cap_counts']=legal_counts
    # On exactly the same successor pairs, compare whether changing the suffix reverses the winning action.
    by_case={r['case_id']:r for r in data if r['suffix']=='009' and not r['cap_fallback']}
    dual=[(r,by_case[r['case_id']]) for r in data if r['suffix']=='ga' and r['case_id'] in by_case]
    stats['suffix_diagnostic']=dict(both_informative=sum(g['distinct'] and c['distinct'] for g,c in dual),
                                    opposite_preference=sum(g['distinct'] and c['distinct'] and g['y0']!=c['y0'] for g,c in dual))
    KEY.mkdir(exist_ok=True,parents=True)
    with (KEY/'per_pair.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)
    save(KEY/'summary.json',stats)
    save(KEY/'protocol.json',read(OUT/'protocol.json'))
    save(OUT/'FINISHED.json',dict(scenes=len(cases),runs=stats['executions'],summary=sha(KEY/'summary.json'),
                                  per_pair=sha(KEY/'per_pair.csv')))
    return stats



if __name__=='__main__':
    report()
