"""Development-only feasibility screen for a matched future CAP authoring comparison."""
import csv
from collections import Counter, defaultdict

from tools import cap_value_dag as fit
from tools import cap_topology_graph_report as graph


def main():
    rooms=graph.load('dev')
    points=graph.extract(rooms)
    centres,_=graph.frozen_model()
    labels=dict(zip(points,graph.nearest(graph.matrix(list(points.values()),'y'),centres)))
    edges=graph.edge_records(rooms)
    p=fit.data.g.study.read(fit.data.g.study.OUT/'protocol.json')
    experts=sorted({m['sha'] for members in p['libraries'].values() for m in members})
    prefix=fit.data.g.study.OUT/'singles/development'/fit.data.g.prep.prior.EXECUTOR
    hits=defaultdict(list)
    for e in edges:
        if labels[e['source']]==3 and all(graph.node(labels,d)==3 for d in e['dest']):
            hits[e['room']].append(e)
    rows=[]
    counts=Counter()
    for room in rooms:
        k=room['room']
        n=sum(fit.read(prefix/s/(k+'.json'))['success'] for s in experts)
        witnesses=hits[k]
        counts['rooms']+=1
        counts['library_fail']+=int(n==0)
        counts['flagged']+=int(bool(witnesses))
        counts['eligible']+=int(n==0 and bool(witnesses))
        early=any(e['origin']=='root' and e['depth']==4 for e in witnesses)
        counts['early_flagged']+=int(early)
        counts['early_eligible']+=int(early and n==0)
        rows.append(dict(room=k,expert_successes=n,witnesses=len(witnesses),
                         early=early,example=witnesses[0]['source'][1] if witnesses else ''))
    assert len(experts)==7
    with (fit.DOC/'development_pool.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fit.save(fit.DOC/'development_pool.json',dict(counts))


if __name__=='__main__':
    main()
