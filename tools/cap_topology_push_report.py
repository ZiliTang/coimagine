"""Summarize realized single-option effects at the graph's sampled boundaries."""
import json
from collections import defaultdict

import numpy as np

from tools import cap_topology_graph as g
from tools.cap_topology_graph_report import write_csv


def compare(rows,flag):
    rooms=defaultdict(lambda:np.zeros((2,2)))
    for r in rows:
        cell=rooms[r['room']][int(r[flag])]
        cell[0]+=r['effect']
        cell[1]+=1
    v=np.stack(list(rooms.values()))
    counts=v.sum(0)
    rng=np.random.default_rng(230)
    boot=v[rng.integers(len(v),size=(3000,len(v)))].sum(1)
    boot=boot[(boot[:,:,1]>0).all(1)]
    delta=boot[:,1,0]/boot[:,1,1]-boot[:,0,0]/boot[:,0,1]
    return dict(counts=counts.astype(int).tolist(),
                difference=float(counts[1,0]/counts[1,1]-counts[0,0]/counts[0,1]) if (counts[:,1]>0).all() else None,
                ci=np.quantile(delta,[.025,.975]).tolist() if len(delta) else None)


def main():
    table=[]
    data={}
    for fold in ('dev','fresh'):
        runs=g.study.read(g.OUT/('pushes_'+fold+'.json'))
        unique={}
        proposals=0
        for r in runs:
            for p in r['pushes']:
                k=(r['room'],p['history'],json.dumps(p['action'],sort_keys=True))
                proposals+=1
                v=dict(room=r['room'],**p)
                if k in unique:
                    assert all(unique[k][f]==p[f] for f in ('effect','along','lateral','ready','support'))
                    unique[k]['expert']='AB'
                else:unique[k]=v
        records=list(unique.values())
        for r in records:
            v=dict(panel=fold,**r)
            cmd=v.pop('action')
            table.append(dict(**v,**cmd))
        data[fold]=dict(rooms=len(runs),expert_queries=sum(r['attempts'] for r in runs),proposals=proposals,
                        unique=len(records),rooms_with_push=len({r['room'] for r in records}),
                        preconditions={k:compare(records,k) for k in ('touch','support','clear','compatible')})
    write_csv('actual_pushes.csv',table)
    g.study.save(g.DOC/'actual_pushes.json',data)


if __name__=='__main__':
    main()
