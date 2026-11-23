"""Compare explicit CONTACT and PUSH_FROM_CONTACT against the combined action."""
from collections import Counter
import argparse
import csv

import numpy as np

from tools import push_contract_bench as bench
from tools import push_contract_report as report

OUT,DOC=bench.OUT,report.DOC


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stress',action='store_true')
    args=parser.parse_args()
    data={}
    rows=[]
    panels=[('stress',('split','split_near5'))] if args.stress else [
        ('development',('split','split_near8')),('validation',('split','split_near8','split_near5'))]
    for split,arms in panels:
        baseline=report.load(split,'base')
        by_key={(r['id'],r['offset']):r for r in baseline}
        data[split]={}
        for arm in arms:
            path=OUT/('protocol_stress.json' if split=='stress' else 'protocol.json')
            cases=[c for c in bench.read(path)['cases'] if c['split']==split and c['kind']!='dual']
            paired=[dict(bench.read(OUT/'runs'/arm/off/(c['id']+'.json')),case=c) for c in cases for off in bench.OFFSETS]
            comparison=report.summarize(paired,baseline)
            active=report.execution(paired)
            cert=Counter(r['phases']['contact'] for r in active)
            drive=Counter(r['phases']['drive'] or 'not_started' for r in active)
            confirmed=[r for r in active if r['phases']['drive_ticks']]
            phase_motion=[]
            fixed_window=[]
            for r in confirmed:
                n=r['phases']['contact_ticks']
                at_contact=r['frames'][n-1]['xy'] if n else r['start']
                projection=lambda end,start:float(np.subtract(end[:2],start[:2])@r['dir'])
                phase_motion.append((projection(at_contact,r['start']),projection(r['frames'][-1]['xy'],at_contact)))
                if r['case']['command']['ticks']-n>=12:
                    physical=r['frames'][n:n+12]
                    moved=projection(physical[-1]['xy'],at_contact)
                    kept=len(physical)==12 and all(f['contact'][r['robot']] for f in physical)
                    fixed_window.append((moved,kept,len(physical)==12))
            data[split][arm]=dict(table=comparison,median_contact_along_m=float(np.median([x[0] for x in phase_motion])),
                                  median_push_along_m=float(np.median([x[1] for x in phase_motion])),
                                  fixed_12=dict(requested=len(fixed_window),executed=sum(x[2] for x in fixed_window),
                                                continuous_contact=sum(x[1] for x in fixed_window),
                                                continuous_and_effect=sum(x[1] and x[0]>=.02 for x in fixed_window)),
                                  push_only_reaches_2cm=sum(x[1]>=.02 for x in phase_motion),n=len(active),certification=dict(cert),drive=dict(drive),
                                  confirmed_n=len(confirmed),confirmed_effect=sum(r['effect'] for r in confirmed),
                                  confirmed_stable=sum(r['stable'] is not None for r in confirmed),
                                  coupled_success_on_confirmed=sum(by_key[r['id'],r['offset']]['effect'] for r in confirmed),
                                  baseline_effect=sum(by_key[r['id'],r['offset']]['effect'] for r in active),
                                  effect=sum(r['effect'] for r in active),certified_initial=sum(
                                      r['phases']['contact']=='ready' and not r['phases']['contact_ticks'] for r in active))
            for r in active:
                b=by_key[r['id'],r['offset']]
                rows.append(dict(split=split,arm=arm,room=r['case']['room'],kind=r['case']['kind'],offset=r['offset'],
                                 baseline_effect=b['effect'],effect=r['effect'],baseline_along=b['along'],along=r['along'],
                                 stable_contact=r['stable'],**r['phases']))
    DOC.mkdir(parents=True,exist_ok=True)
    name='phases_stress' if args.stress else 'phases'
    with (DOC/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if args.stress:
        baseline=report.load('stress','base')
        candidate=report.load('stress','near5')
        data['coupled']={arm:report.summarize(rows,baseline) for arm,rows in [('base',baseline),('near5',candidate)]}
    bench.save(DOC/(name+'.json'),data)
    for name,r in data['stress' if args.stress else 'validation'].items():
        print(name,{k:v for k,v in r.items() if k!='table'},flush=True)


if __name__=='__main__':
    main()
