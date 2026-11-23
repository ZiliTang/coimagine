"""Check a development-selected bottleneck against the full seven-policy union."""
import csv
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from tools import cap_value_dag as fit
from tools import cap_topology_graph_report as graph

g=fit.data.g


def main():
    panels=fit.read(fit.OUT/'panels.json')
    fixtures={r['id']:r['fixture'] for r in g.study.phase.cap.rooms.generate(fit.data.SEED)['cases']}
    assert {k:g.study.phase.input_hash(f) for k,f in fixtures.items()}==panels['rooms']['confirm']
    g.study.initialize()
    experts=sorted(g.study.MODULES)
    assert len(experts)==7
    jobs=[]
    for k,f in fixtures.items():
        for s in experts:
            path=fit.OUT/'singles'/s/(k+'.json')
            if not path.exists():jobs.append((g.prep.prior.EXECUTOR,s,k,f,'confirm',path))
    with ProcessPoolExecutor(18,mp_context=mp.get_context('fork')) as pool:
        for i,item in enumerate(as_completed([pool.submit(g.study.single,j) for j in jobs]),1):
            item.result()
            if i%200==0:print(i,flush=True)
    paths=sorted((fit.OUT/'data/confirm').glob('*.json'))
    assert len(paths)==200
    rooms=[fit.read(p) for p in paths]
    points=graph.extract(rooms);centres,_=graph.frozen_model()
    labels=dict(zip(points,graph.nearest(graph.matrix(list(points.values()),'y'),centres)))
    flagged={};early={}
    for e in graph.edge_records(rooms):
        if labels[e['source']]==3 and all(graph.node(labels,d)==3 for d in e['dest']):
            flagged[e['room']]=e['source'][1]
            if e['depth']==4 and e['origin']=='root':early[e['room']]=e['source'][1]
    rows=[]
    for r in rooms:
        k=r['room']
        y=[int(fit.read(fit.OUT/'singles'/s/(k+'.json'))['success']) for s in experts]
        rows.append(dict(room=k,early=int(k in early),any_boundary=int(k in flagged),library_fail=int(not any(y)),
                         **dict(zip(experts,y))))
    with (fit.DOC/'confirm_pool.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    stats={}
    for field in ('early','any_boundary'):
        x=np.array([r[field] for r in rows],bool);y=np.array([r['library_fail'] for r in rows])
        idx=np.random.default_rng(931).integers(200,size=(4000,200));bx=x[idx];by=y[idx]
        keep=(bx.sum(1)>0)&(bx.sum(1)<200);bx,by=bx[keep],by[keep]
        dif=(bx*by).sum(1)/bx.sum(1)-((~bx)*by).sum(1)/(~bx).sum(1)
        stats[field]=dict(flagged=int(sum(x)),positive=int(y[x].sum()),unflagged=int(sum(~x)),other_positive=int(y[~x].sum()),
                          delta=float(y[x].mean()-y[~x].mean()),ci=np.quantile(dif,[.025,.975]).tolist(),total_positive=int(sum(y)))
    fit.save(fit.DOC/'confirm_pool.json',stats)


if __name__=='__main__':
    main()
