"""Recheck a frozen handoff condition at every natural option boundary, once per room."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from tools import cap_option_handoff as exp
from tools import cap_option_handoff_report as report

g,s=exp.g,exp.s


def public(w,obs,hist):
    vec=g.public_vector(obs,hist)
    _,node,_=PREDICT([dict(x=vec)])
    return dict(node=int(node[0]),flags=exp.public_flags(vec))


def rollout(fixture,pair,depth,x,y,hand,origin=0):
    policies=[s.phase.FrozenCAP(s.MODULES[e]) for e in pair]
    counts=Counter()
    history=deque(maxlen=4)
    events=None
    pending=None
    with s.phase.world(fixture) as (w,_):
        def update(h):
            for p in policies:p.update(h)
        update(w.history)
        n=0
        while not s.ended(w):
            obs=s.observe(w)
            command=g.prep.prior.try_action(policies[origin],obs)
            picked=False
            if events is None:
                history.append(g.live_frame(obs))
                if n>=depth:
                    other=g.prep.prior.try_action(policies[1-origin],obs)
                    distinct=command is not None and other is not None and command!=other
                    if distinct:
                        now=public(w,obs,history)
                        if pending is not None and report.has(now,y):
                            events=dict(source=pending,mid=w.history.sha256,n=n,tick=obs['tick'])
                            if hand:command=other
                            picked=True
                        pending=w.history.sha256 if report.has(now,x) and not report.has(now,y) else None
                    else:
                        pending=None
            command=command or g.prep.prior.try_action(policies[1-origin],obs) or g.prep.BRAKE
            if picked:
                target=w.env.primary_id
                goal=obs['goal']['xy']
                gap=g.prep.goal_gap(w,target,goal)
            s.tick_action(w,command,g.prep.prior.EXECUTOR,update,counts)
            if picked:
                events.update(gain=gap-g.prep.goal_gap(w,target,goal),end=w.history.sha256,
                              action=command,ticks=w.env.total_periods-obs['tick'])
            n+=1
        return dict(event=events,result=s.output(w,counts))


def case(job):
    key,fixture,pair,x,y=job
    path=exp.OUT/'opportunities'/(key+'.json')
    if path.exists():return
    main=exp.read(exp.OUT/'fresh'/(key+'.json'))
    a=rollout(fixture,pair,main['depth'],x,y,False)
    for k in ('success','ticks','history','decisions'):
        assert a['result'][k]==main['arms']['AA']['result'][k]
    b=rollout(fixture,pair,main['depth'],x,y,True) if a['event'] else a
    if a['event']:
        assert all(a['event'][k]==b['event'][k] for k in ('source','mid','n','tick'))
    own=rollout(fixture,pair,main['depth'],x,y,False,origin=1)
    assert all(own['result'][k]==main['single_b'][k] for k in ('success','ticks','decisions','history'))
    exp.save(path,dict(room=key,a=a,b=b,b_own=own,single_b=main['single_b']['success']))


def main():
    rooms,pair=exp.panel('fresh')
    rule=exp.read(exp.OUT/'frozen.json')['winner']
    assert rule['x']=='any' and rule['y']=='node4'
    global PREDICT
    s.initialize()
    _,PREDICT=exp.graph.frozen_model()
    with ProcessPoolExecutor(18,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(case,(k,f,pair,rule['x'],rule['y'])) for k,f in rooms.items()]
        for i,job in enumerate(as_completed(futures),1):
            job.result()
            if i%25==0:print('opportunities',i,flush=True)


if __name__=='__main__':main()
