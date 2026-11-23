"""Audit observable PUSH effect feedback against measured physical displacement."""
from collections import Counter
import csv
import numpy as np
from tools import push_contract_report as report
from tools import push_contract_bench as bench


def main():
    feedback=report.load('stress','split_feedback')
    pairs=report.load('stress','split_near5')
    statuses=Counter()
    rows=[]
    for a,b in zip(feedback,pairs):
        assert (a['id'],a['offset'],a['prefix'],a['status'])==(b['id'],b['offset'],b['prefix'],b['status'])
        if a['status']!='executed':
            continue
        for field in ('ticks','reason','stable','effect','along','lateral','frames'):
            assert a[field]==b[field],(a['id'],a['offset'],field)
        phase=a['phases']
        if not phase['drive_ticks']:
            continue
        n=phase['contact_ticks']
        start=a['frames'][n-1]['xy'] if n else a['start']
        actual=float(np.subtract(a['frames'][-1]['xy'][:2],start[:2])@a['dir'])
        observed=phase['observed_effect']
        statuses[observed['status']]+=1
        rows.append(dict(room=a['case']['room'],kind=a['case']['kind'],offset=a['offset'],
                         actual=actual,estimate=observed.get('signed_along_m'),status=observed['status']))
    measured=[r for r in rows if r['estimate'] is not None]
    diff=np.array([r['estimate']-r['actual'] for r in measured])
    error=dict(mae=float(np.mean(abs(diff))),median_abs=float(np.median(abs(diff))),
               sign_correct=sum(int(r['estimate']>0)==int(r['actual']>0) for r in measured if abs(r['actual'])>=.02),
               sign_n=sum(abs(r['actual'])>=.02 for r in measured),
               progress_correct=sum((r['estimate']>=.02)==(r['actual']>=.02) for r in measured),
               false_positive=sum(r['estimate']>=.02 and r['actual']<.02 for r in measured),
               false_negative=sum(r['estimate']<.02 and r['actual']>=.02 for r in measured))
    bench.save(report.DOC/'effect_feedback.json',dict(statuses=dict(statuses),error=error,
               n=len(rows),observed=len(measured)))
    with (report.DOC/'effect_feedback.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(dict(statuses,**error),flush=True)


if __name__=='__main__':
    main()
