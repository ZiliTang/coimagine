"""Intervene once at the natural first proposal mismatch, then resume the same expert."""
from collections import Counter
from tools import cap_symmetry_mechanism as m

s,e=m.s,m.e


def worker(row):
    k,arm,name=row['room'],row['arm'],row['name']
    path=m.OUT/'once'/f'{k}_{arm}_{name}.json'
    if path.exists():return
    cap=e.s.phase.FrozenCAP(e.AM if arm=='A' else e.BM)
    first=row['first']
    index=0
    changed=False
    with e.s.phase.world(s.transform(ROOMS[k],name)) as (w,_):
        cap.update(w.history)
        counts=Counter()
        while not e.s.ended(w):
            obs=e.s.observe(w)
            action=cap.propose(obs)
            if index==first['i']:
                assert action==first['actual'] and obs['tick']==first['tick']
                try:
                    action=m.validate_action(first['mapped'],obs)
                except ValueError:
                    e.save(path,dict(room=k,arm=arm,name=name,status='unavailable'))
                    return
                changed=True
            e.s.tick_action(w,action,e.g.prep.prior.EXECUTOR,cap.update,counts)
            index+=1
        assert changed
        e.save(path,dict(room=k,arm=arm,name=name,status='completed',success=int(w.env.success),
                         restored=int(w.env.success)==row['original_success']))


if __name__=='__main__':
    s.t.d.init(profile=s.e.perception.LEGACY)
    ROOMS=s.t.p.dev()
    data=[r for path in sorted((m.OUT/'rooms').glob('*.json')) for r in e.read(path)]
    e.pool(worker,data,5)
