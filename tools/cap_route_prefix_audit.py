"""Audit the actual A prefix in rooms where a delayed B handoff loses an initial rescue."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tools import cap_route_horizon as exp

study=exp.study


def audit(item,pair):
    room,fixture,windows=item
    recorded=study.read(exp.OUT/'runs/held'/(room+'.json'))
    origin=recorded['baseline']['first']
    policies=[study.phase.FrozenCAP(study.MODULES[e]) for e in pair]
    kept={}
    with study.phase.world(fixture) as (w,_):
        def update(history):
            for p in policies:p.update(history)
        update(w.history)
        n=0
        counts=Counter()
        max_move=0.
        max_turn=0.
        push_commands=0
        while not study.ended(w):
            obs=study.observe(w)
            a=policies[0].propose(obs)
            if n==origin['n']:
                assert w.history.sha256==origin['history']
                backend=w.env.env.backend
                start={k:(np.array(v.position),v.angle) for k,v in backend.bodies.items()}
                detector=w.detector.update
                def log(packet):
                    nonlocal max_move,max_turn
                    value=detector(packet)
                    contacts=backend.contacts()
                    counts['collision_ticks']+=any(c['pair'].startswith('robot_') for c in contacts)
                    counts['target_contact_ticks']+=any(c['pair']=='robot_object' and c['entity_id']==obs['goal']['target_id'] for c in contacts)
                    max_move=max(max_move,max(np.linalg.norm(np.array(v.position)-start[k][0]) for k,v in backend.bodies.items()))
                    max_turn=max(max_turn,max(abs(v.angle-start[k][1]) for k,v in backend.bodies.items()))
                    return value
                w.detector.update=log
            h=n-origin['n']
            if h in windows:
                cut=recorded['horizons'][str(h)]['cut']
                assert cut and w.history.sha256==cut['history']
                kept[str(h)]=dict(collision_ticks=counts['collision_ticks'],target_contact_ticks=counts['target_contact_ticks'],
                    target_push_commands=push_commands,max_object_m=float(max_move),max_object_rotation_rad=float(max_turn))
                if h==max(windows):break
            if h>=0:
                push_commands+=int(a['target_id']==obs['goal']['target_id'] and bool(study.translating(a)))
            study.tick_action(w,a,exp.EXECUTOR,update,Counter())
            n+=1
    return room,kept


def main():
    plan=study.read(exp.OUT/'protocol.json')
    db=study.read(study.OUT/'held.json')
    jobs=[]
    missed=Counter()
    for room in plan['rooms']['held']:
        row=study.read(exp.OUT/'runs/held'/(room+'.json'))
        hs=row['horizons']
        if row['baseline']['success'] or not hs['0']['success']:continue
        windows=[h for h in (3,10) if not hs[str(h)]['success']]
        for h in windows:missed[h]+=1
        windows=[h for h in windows if hs[str(h)]['cut'] is not None]
        if windows:jobs.append((room,db[room],windows))
    study.initialize()
    with ProcessPoolExecutor(6) as pool:
        rows=dict(pool.map(partial(audit,pair=plan['pair']),jobs))
    result={}
    for h in (3,10):
        rs=[v[str(h)] for v in rows.values() if str(h) in v]
        result[str(h)]=dict(lost_initial_rescues=missed[h],cut_reached=len(rs),no_target_push_command=sum(r['target_push_commands']==0 for r in rs),
            no_target_contact=sum(r['target_contact_ticks']==0 for r in rs),
            no_robot_collision=sum(r['collision_ticks']==0 for r in rs),
            quiet=sum(r['collision_ticks']==0 and r['max_object_m']<.01 and r['max_object_rotation_rad']<.05 for r in rs))
    study.save(exp.DOC/'prefix_audit.json',dict(summary=result,rooms=rows))
    print(result)


if __name__=='__main__':
    main()
