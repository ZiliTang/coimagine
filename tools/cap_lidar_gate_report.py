"""Write paired closed-loop counts and snapshot geometry for the static-gate study."""
from collections import Counter
import csv
import json
from statistics import median
import numpy as np
from tools import cap_lidar_static_gate as g


def pair(rows,arm,base):
    gain=sum(r[arm]['success']>r[base]['success'] for r in rows)
    loss=sum(r[arm]['success']<r[base]['success'] for r in rows)
    return dict(recovered=gain,lost=loss,changed_histories=sum(r[arm]['history']!=r[base]['history'] for r in rows))


def table(name, rows):
    with (g.DOC/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    g.DOC.mkdir(parents=True,exist_ok=True)
    panels = {}
    episodes,centers = [],[]
    for name,n in [('original',32),('mirror_y',32),('natural',64)]:
        panels[name]={p.stem:g.e.read(p) for p in sorted((g.OUT/name).glob('*.json'))}
        assert len(panels[name])==n
        for room,rows in panels[name].items():
            for arm,r in rows.items():
                episodes.append(dict(panel=name,room=room,arm=arm,success=r['success'],ticks=r['ticks'],history=r['history']))
    for room,row in panels['original'].items():
        samples=g.e.read(g.OUT/'centers'/(room+'.json'))
        assert samples['history']==row['local_gated']['history']
        centers.extend(dict(room=room,**s) for s in samples['samples'])
    table('episodes.csv',episodes)
    table('centers.csv',centers)
    panel_rows={name:list(items.values()) for name,items in panels.items()}
    panel_rows['paired']=panel_rows['original']+panel_rows['mirror_y']
    outcomes=[]
    gate={}
    for name,rows in panel_rows.items():
        for center in g.CENTERS:
            for scan in g.SCANS:
                arm=center+'_'+scan
                outcomes.append(dict(panel=name,arm=arm,n=len(rows),success=sum(r[arm]['success'] for r in rows),
                                     **pair(rows,arm,'world_fixed')))
            arm=center+'_gated'
            counts=Counter()
            for r in rows: counts.update(r[arm]['gate'])
            gate[name+'_'+center]=dict(**pair(rows,arm,center+'_stagger'),**counts)
    table('outcomes.csv',outcomes)
    q=[s for s in centers if s['quiet']]
    byroom={}
    for s in q: byroom.setdefault(s['room'],[]).append(s['fusion_cm']-s['current_cm'])
    stats=dict(total=len(centers),quiet=len(q),rooms=len(byroom),better=sum(s['current_cm']-s['fusion_cm']>.001 for s in q),
               worse=sum(s['fusion_cm']-s['current_cm']>.001 for s in q),
               median_room_mean_delta=median(float(np.mean(v)) for v in byroom.values()),
               improved_rooms=sum(float(np.mean(v))<-.001 for v in byroom.values()),
               worsened_rooms=sum(float(np.mean(v))>.001 for v in byroom.values()))
    for field in ('current_cm','fusion_cm'):
        v=[s[field] for s in q]
        stats[field]=dict(median=median(v),mean=float(np.mean(v)),p90=float(np.percentile(v,90)))
    flips={arm:sum(r[arm]['success']!=panels['mirror_y'][k][arm]['success'] for k,r in panels['original'].items())
           for arm in panel_rows['original'][0]}
    result=dict(outcomes=outcomes,gate=gate,centers=stats,flips=flips)
    if (g.OUT/'extension.json').is_file():
        locked=g.e.read(g.OUT/'extension.json')
        remaining={k:g.e.read(g.OUT/'remaining'/(k+'.json')) for k in locked['rooms']}
        assert len(remaining)==136 and not set(remaining).intersection(panels['natural'])
        result['extension']=[]
        for group,rows in [('remaining',list(remaining.values())),('full',list(remaining.values())+panel_rows['natural'])]:
            for center,scan in locked['arms']:
                arm=center+'_'+scan
                result['extension'].append(dict(panel=group,n=len(rows),arm=arm,success=sum(r[arm]['success'] for r in rows),
                                                **pair(rows,arm,'world_fixed')))
        table('extension.csv',[dict(room=k,arm=a,success=r['success'],ticks=r['ticks'],history=r['history'])
                               for k,arms in remaining.items() for a,r in arms.items()])
    g.e.save(g.OUT/'summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
