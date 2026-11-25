"""Development-only smoke test of prospective module handoffs during PUSH."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing as mp
from unittest.mock import patch

import numpy as np
from experiments.planar_tasks.relational_search.estimator import fit_box
from experiments.planar_tasks.semantic_rl.candidate_semantics import ACTIONS, direction, sweep_hits
from experiments.planar_tasks.semantic_rl.phase_aware_cap import translating
from tools import cap_cluster_module as exp

ROOMS = ('047','048','049','094','030','126')
OUT = exp.OUT.parent/'cap_push_takeover_preflight_20261124_01'


def static_ahead(obs, action, left):
    tid=obs['goal']['target_id']
    if action['target_id']!=tid:
        return False
    points=next((e['points'] for e in obs['entities'] if e['id']==tid and
                 e['last_seen_tick']==obs['tick']),[])
    if len(points)<3:
        return False
    points=np.asarray(points)
    origin=points.mean(axis=0)
    box=fit_box(points-origin,[np.asarray(r['scan']['origin'])-origin for r in obs['robots']])
    if box is None:
        return False
    box['x']+=origin[0]
    box['y']+=origin[1]
    for i in translating(action):
        move=ACTIONS[action[f'a{i}']]
        # Upper-bound command distance is only a geometric hypothesis.
        delta=direction(move)*min(.20,move.speed*left*.05)
        static,_=sweep_hits(box,delta,tid,obs['robots'])
        if static:
            return True
    return False


def probe(job):
    key, fixture, intervene=job
    prefix=exp.prefix
    source=exp.isolated.IsolatedCAP(exp.isolated.BROKER)
    events=[]
    touches=[]
    commands=[]
    original_world=prefix.s.phase.world

    @contextmanager
    def observer(f):
        with original_world(f) as (world,other):
            backend=world.env.env.backend
            target=prefix.s.observe(world)['goal']['target_id']
            original=world.env.joint_step
            touching=False
            def step(*args,**kwargs):
                nonlocal touching
                value=original(*args,**kwargs)
                now=any(c['pair']=='object_wall' and c['entity_id']==target for c in backend.contacts())
                if now and not touching: touches.append(world.env.total_periods)
                touching=now
                return value
            world.env.joint_step=step
            yield world,other

    def at_boundary(world,obs,action,pending):
        commands.append(dict(tick=obs['tick'],action=action,pending=pending))
        return action

    switched=False
    def consult(obs,action,left,reason):
        nonlocal switched
        risk=static_ahead(obs,action,left)
        item=dict(tick=obs['tick'],left=left,reason=reason,pushers=len(translating(action)),risk=risk,
                  command_start=commands[-1]['tick'])
        events.append(item)
        if intervene and risk and not switched and [44,44] in obs['global_legal_pairs']:
            switched=True
            item['test_brake']=True
            return dict(a0=44,a1=44,ticks=4,target_id=None)
        return None

    with patch.object(prefix.s.phase,'world',observer):
        _,row=prefix.run_member((key,fixture,False),source,exp.isolated.preview,at_boundary,push_takeover=consult)
    if not intervene:
        assert row['result']==exp.read(exp.OUT/'snapshots'/(key+'.json'))['result']
    if switched:
        t=next(e['tick'] for e in events if e.get('test_brake'))
        assert any(c['pending'] and c['tick']==t and c['action']==dict(a0=44,a1=44,ticks=4,target_id=None) for c in commands)
    warned=[]
    for t in touches:
        start=max((c['tick'] for c in commands if c['tick']<t),default=-1)
        warnings=[e for e in events if e['risk'] and e['command_start']==start and e['tick']<t]
        if warnings:warned.append(dict(contact=t,earliest=warnings[0]['tick'],lead=t-warnings[0]['tick']))
    return dict(room=key,profile='brake_once' if intervene else 'shadow',result=row['result'],
                events=events,contact_onsets=touches,warned_same_command=warned,switched=switched)


def main():
    fixtures=exp.read(exp.study.OUT/'rooms.json')['rooms']['dev']
    source=exp.source_parent()
    jobs=[('diag_dev_'+k,fixtures['diag_dev_'+k],i) for k in ROOMS for i in (False,True)]
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork'),initializer=exp.isolated.initialize,initargs=(source,)) as pool:
        rows=list(pool.map(probe,jobs))
    exp.save(OUT/'results.json',rows)
    for r in rows:
        if r['profile']=='shadow':
            print(r['room'],len(r['events']),len(r['contact_onsets']),r['warned_same_command'])
    print('brake smoke tests',sum(r['switched'] for r in rows))


if __name__=='__main__':
    main()
