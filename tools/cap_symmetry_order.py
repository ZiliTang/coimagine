"""Change only candidate enumeration in the proposal routine on every decision."""
from collections import Counter
from tools import cap_symmetry_mechanism as m

s,e=m.s,m.e


def worker(row):
    k,arm,name=row['room'],row['arm'],row['name']
    cap=e.s.phase.FrozenCAP(e.AM if arm=='A' else e.BM)
    path=m.OUT/'order_full'/f'{k}_{arm}_{name}.json'
    if path.exists():return
    remap=[s.option(i,name) for i in range(len(s.ACTIONS))]
    with e.s.phase.world(s.transform(ROOMS[k],name)) as (w,_):
        cap.update(w.history)
        counts=Counter()
        while not e.s.ended(w):
            o=e.s.observe(w)
            assert all(pairs==sorted(pairs) for pairs in o['entity_legal_pairs'].values())
            menus={k:sorted(pairs,key=lambda ab:(remap[ab[0]],remap[ab[1]])) for k,pairs in o['entity_legal_pairs'].items()}
            act=cap.propose(dict(o,entity_legal_pairs=menus))
            e.s.tick_action(w,act,e.g.prep.prior.EXECUTOR,cap.update,counts)
        e.save(path,dict(room=k,name=name,arm=arm,restored=int(w.env.success)==row['original_success'],success=int(w.env.success)))


if __name__=='__main__':
    s.t.d.init(profile=s.e.perception.LEGACY)
    ROOMS=s.t.p.dev()
    data=[r for path in sorted((m.OUT/'rooms').glob('*.json')) for r in e.read(path)]
    e.pool(worker,data,6)
