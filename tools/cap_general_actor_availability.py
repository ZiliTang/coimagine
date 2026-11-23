"""Audit legality, proposals and ranking at the first departures of matched one-tick forks."""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import multiprocessing as mp

import numpy as np
import torch

from experiments.option_rl_v19.unknown_shape import visible_faces, direction_matches
from experiments.option_rl_v21 import spec
from experiments.planar_tasks.relational_search.action_scale import CONTROL_DT
from tools import cap_general_actor_forks as ref

dq=ref.dq
OUT=ref.OUT/'availability'


@contextmanager
def world(fixture):
    env=dq.fast_runtime.ScopedEnvironment(fixture,dq.study.CFG,fast_feedback=True,record_details=False,
                                          history_factory=dq.fast_runtime.AuditedCachedHistory)
    try:
        yield env,dq.Proposals()
    finally:
        env.close_reads()
        dq.fast_runtime.ACTIVE=None


def advance(env,action):
    event=env.step(action,remaining_ticks=1800-env.env.total_periods)
    return event['reason']


def ended(env):
    return env.env.termination is not None or env.env.total_periods>=1800 or env.decision_count>=300


def support(env,obs,action):
    target=action['target_id']
    if target is None:
        return None
    robots=[]
    for i in range(2):
        code=action[f'a{i}']
        a=spec.ACTIONS[code]
        rays=obs['robots'][i]['scan']['entity_ids'].count(target)
        tick=obs['tick']
        sensor=env.env._sensor(i,tick,tick*CONTROL_DT,target)
        faces=visible_faces(sensor)
        allowed=bool(spec.mask(sensor)[code])
        translation=a.kind=='push' and a.effect_kind=='translate'
        n_match=sum(bool(direction_matches(f[1],a.effect_direction)) for f in faces) if translation else None
        if translation:
            assert bool(n_match)==allowed
        robots.append(dict(label=a.label,rays=rays,faces=len(faces),matching=n_match,allowed=allowed))
    return robots


def view(env,panel,action,choice,details=True):
    obs=dq.current_event(env.history)['observation']
    try:
        dq.validate_action(action,obs)
        reason=None
    except ValueError as e:
        reason=str(e)
    pool=panel[4]
    target=action['target_id']
    pts=next((len(e['points']) for e in obs['entities'] if e['id']==target and e['last_seen_tick']==obs['tick']),0)
    info=dict(tick=obs['tick'],reason=reason,sources=[s for s,p in zip(ref.SIDS,panel[5]) if p==action],
              pool=len(pool),chosen=choice,offered=action in pool)
    assert not reason or not info['offered']
    if details:
        info.update(target_points=pts,active_target=obs['active_target'],choice_sources=[s for s,p in zip(ref.SIDS,panel[5]) if p==choice],
                    robot_support=support(env,obs,action),target_pair=[action['a0'],action['a1']] in obs['entity_legal_pairs'].get(target,[]),
                    alt_same_parameters=any(all(p[k]==action[k] for k in ('a0','a1','target_id')) for p in pool),
                    requested_target_proposed=any(p['target_id']==target for p in pool))
    return info


def factual(fixture,arm):
    steps=[]
    with world(fixture) as (env,lib):
        while not ended(env):
            panel=lib.panel(env.history)
            a=ref.choose(ref.MODELS[arm],panel)
            v=view(env,panel,a,a)
            assert v['reason'] is None and v['offered']
            v.update(a=a,h=env.history.sha256,menu=panel[4])
            v['end_reason']=advance(env,a)
            steps.append(v)
        return steps,env.history.sha256,int(env.env.success)


def cause(v):
    if v['reason']:
        return 'illegal'
    if not v['offered']:
        return 'proposal'
    return 'rank'


def pair_info(i,cut,env,panel,path,choice):
    refrow=path[i]
    action=refrow['a']
    now=view(env,panel,action,choice)
    return dict(step=i,offset=i-cut,ref_tick=refrow['tick'],action=action,now=now,
                original={k:refrow[k] for k in ('target_points','robot_support','sources','pool','active_target')},
                winner_also_in_old=choice in refrow['menu'],history=env.history.sha256)


def replay(fixture,arm,r,path):
    cut=r['cut']
    micro=r['micro'][arm]
    primary=micro.get('fixed_status')=='unavailable_command'
    result=dict(actor=arm,original=r['actual'][arm],perturbed=micro['success'],primary=primary,cut=cut,
                flip=micro['success']!=r['actual'][arm],first=None,no_proposal=None,rejection=None)
    previous_reason=None
    invalid=False
    with world(fixture) as (env,lib):
        while not ended(env):
            i=env.decision_count
            if i>=len(path):
                break
            panel=lib.panel(env.history)
            a=path[i]['a']
            if i<=cut:
                assert env.history.sha256==path[i]['h']
                if i==cut:
                    assert env.history.sha256==r['prefix']
                    a=dict(a,ticks=a['ticks']+micro['requested_delta'])
            else:
                choice=ref.choose(ref.MODELS[arm],panel)
                if a!=choice:
                    info=pair_info(i,cut,env,panel,path,choice)
                    v=info['now']
                    info['stage']=cause(v)
                    info['preceding_reason']=[path[i-1]['end_reason'],previous_reason]
                    if result['first'] is None:
                        result['first']=info
                    if not v['offered'] and result['no_proposal'] is None:
                        result['no_proposal']=info
                    if v['reason']:
                        result['rejection']=info
                        invalid=True
                        break
            previous_reason=advance(env,a)
        status='unavailable_command' if invalid else 'complete' if ended(env) else 'end_of_recorded_suffix'
        result['fixed_status']=status
        if result['flip']:
            assert status==micro['fixed_status']
        if primary:
            assert invalid and result['first']['offset']-1==micro['next_diff']
            # Original suffix is now illegal. Return control without executing that request.
            advance(env,info['now']['chosen'])
            while not ended(env):
                advance(env,ref.choose(ref.MODELS[arm],lib.panel(env.history)))
            result['resume_at_rejection']=int(env.env.success)
    if primary and result['first']['stage']!='illegal':
        first=result['first']
        with world(fixture) as (env,lib):
            while not ended(env):
                i=env.decision_count
                panel=lib.panel(env.history)
                if i<=first['step']:
                    a=path[i]['a']
                    if i==cut:
                        a=dict(a,ticks=a['ticks']+micro['requested_delta'])
                    if i==first['step']:
                        assert env.history.sha256==first['history'] and first['now']['reason'] is None
                else:
                    a=ref.choose(ref.MODELS[arm],panel)
                advance(env,a)
            result['one_forced_original']=int(env.env.success)
    return result


@torch.inference_mode()
def case(job):
    room,fixture,r,baseline=job
    results=[]
    for arm,key in enumerate(('actor800_history','ln_history')):
        path,h,success=factual(fixture,arm)
        assert h==baseline[key] and success==r['actual'][arm]
        results.append(replay(fixture,arm,r,path))
    return dict(room=room,branches=results)


def main():
    torch.set_num_threads(1)
    _,rooms,ref.SIDS=ref.scale.inputs()
    protocol=ref.read(ref.OUT/'protocol.json')
    assert ref.campaign.base.digest(rooms)==protocol['panel'] and ref.SIDS==protocol['sids']
    assert ref.campaign.sha(ref.scale.OUT/'actor.pt')==protocol['old']
    assert ref.campaign.sha(ref.fit.OUT/'best_train.pt')==protocol['layernorm']
    ref.MODELS=[ref.sets.checkpoint(ref.scale.OUT),ref.fit.load_best()]
    dq.MODULES=[dq.load_native((ref.campaign.CAM/'semantics'/(s+'.py')).read_text())[0] for s in ref.SIDS]
    with (ref.base.OUT/'per_scene.csv').open() as f:
        prior={r['room']:r for r in csv.DictReader(f)}
    dest=OUT/'rooms'
    dest.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for k,f in rooms.items():
        r=ref.read(ref.OUT/'rooms'/(k+'.json'))
        if r['cut'] is not None:
            jobs.append((k,f,r,prior[k]))
    assert len(jobs)==189
    settings=dict(source=ref.campaign.sha(__file__),panel=protocol['panel'],old=protocol['old'],layernorm=protocol['layernorm'])
    if (OUT/'protocol.json').exists():
        assert settings==ref.read(OUT/'protocol.json')
    ref.save(OUT/'protocol.json',settings)
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(case,j) for j in jobs if not (dest/(j[0]+'.json')).exists()]
        for n,future in enumerate(as_completed(futures),1):
            result=future.result()
            ref.save(dest/(result['room']+'.json'),result)
            ref.save(OUT/'progress.json',dict(completed=len(jobs)-len(futures)+n,total=len(jobs)))
    assert len(list(dest.glob('*.json')))==len(jobs)


if __name__=='__main__':
    main()
