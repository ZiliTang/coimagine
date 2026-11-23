"""Recover exact boundary budgets and collect an independent frozen graph panel."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from unittest.mock import patch

from tools import cap_topology_graph as g

SEED=662315973
OUT=g.OUT.parent/'cap_value_dag_20261123_01'
DOC=g.DOC.parent/'2026-11-23_cap_value_dag'
read,save=g.study.read,g.study.save


class PrefixVerified(Exception):
    pass


def row_state(s, count):
    return dict(history=s['history'],tick=s['tick'],decisions=count,x=s['x'][:29],y=s['y'],gap=s['gap'])


def collect(job):
    fold,key,fixture,origin,ids=job
    dest=OUT/'data'/fold/(key+'.json')
    if dest.exists():return
    source=None
    if fold!='confirm':
        source=read(g.OUT/'runs'/('dev' if fold=='dev' else 'fresh')/(key+'.json'))
        assert source['origin']==origin
    records=[]
    original=g.sample
    for j,depth in enumerate(g.PREFIX):
        states={}
        branches={}
        for route in g.ROUTES:
            if source:
                rec=source['records'][j]
                branch=rec['branches'][route]
                expected=branch['states']
                if any(h not in states for h in expected):
                    visited=[]
                    def record(w,obs,hist):
                        h=w.history.sha256
                        assert h==expected[len(visited)] and obs['tick']==rec['points'][h]['tick']
                        val=row_state(rec['points'][h],w.decision_count)
                        assert h not in states or states[h]==val
                        states[h]=val
                        visited.append(h)
                        if len(visited)==len(expected):raise PrefixVerified()
                        return val
                    with patch.object(g,'sample',record):
                        try:g.rollout(fixture,ids,origin,depth,route)
                        except PrefixVerified:pass
                    assert visited==expected
            else:
                first=branches.get(route[0]+'A')
                if first and len(first['states'])<2:
                    branches[route]=first
                    continue
                def record(w,obs,hist):
                    return row_state(original(w,obs,hist),w.decision_count)
                with patch.object(g,'sample',record):
                    branch=g.rollout(fixture,ids,origin,depth,route)
                for val in branch['states']:
                    h=val['history']
                    assert h not in states or states[h]==val
                    states[h]=val
                branch['states']=[s['history'] for s in branch['states']]
            branches[route]={k:branch[k] for k in ('states','success','gap','end','endhash')}
            if not branch['states']:
                branches=dict.fromkeys(g.ROUTES,branches[route])
                break
        assert len({tuple(v['states'][:1]) for v in branches.values()})==1
        for a in 'AB':
            assert branches[a+'A']['states'][:2]==branches[a+'B']['states'][:2]
        for b in branches.values():
            vals=[states[h] for h in b['states']]
            for s,t in zip(vals,vals[1:]):
                assert t['tick']-s['tick']==g.WINDOW and t['decisions']>s['decisions']
        records.append(dict(depth=depth,points=states,branches=branches))
    save(dest,dict(room=key,origin=origin,records=records))


def main():
    p=read(g.OUT/'protocol.json')
    existing=g.study.read(g.study.OUT/'development.json')
    casebank=lambda seed:{r['id']:r['fixture'] for r in g.study.phase.cap.rooms.generate(seed)['cases']}
    dev={k:existing[k] for k in p['rooms']['dev']}
    old=casebank(g.SEED)
    new=casebank(SEED)
    panels={'dev':dev,'diagnostic':old,'confirm':new}
    hashes={k:{i:g.study.phase.input_hash(f) for i,f in v.items()} for k,v in panels.items()}
    assert hashes['dev']==p['rooms']['dev'] and hashes['diagnostic']==p['rooms']['fresh']
    seen=g.study.collect_identities() | set(hashes['dev'].values()) | set(hashes['diagnostic'].values())
    seen.update(g.study.phase.input_hash(f) for name in ('development','held') for f in read(g.study.OUT/(name+'.json')).values())
    for seed in (662315862,662315879,g.prep.SEED):
        seen.update(g.study.phase.input_hash(f) for f in casebank(seed).values())
    assert len(set(hashes['confirm'].values()))==200 and not seen.intersection(hashes['confirm'].values())
    save(OUT/'panels.json',dict(rooms=hashes,seed=SEED,pair=p['pair']))
    g.study.initialize()
    jobs=[]
    for fold,fixtures in panels.items():
        keys=sorted(fixtures,key=lambda k:hashlib.sha256(('topology/'+k).encode()).digest())
        jobs.extend((fold,k,fixtures[k],i%2,p['pair']) for i,k in enumerate(keys))
    with ProcessPoolExecutor(18,mp_context=mp.get_context('fork')) as pool:
        for i,f in enumerate(as_completed([pool.submit(collect,j) for j in jobs]),1):
            f.result()
            if i%30==0:print(i,flush=True)


if __name__=='__main__':
    main()
