"""Paired test of visible-support clearance versus a public footprint clearance range."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter
import math
import random

from tools import cap_guarded_wall_history as hist
from experiments.planar_tasks.relational_search import footprint_gap

base, forks, loop = hist.base, hist.forks, hist.loop
ROOT = hist.ROOT.parent / 'cap_guarded_footprint_v1'
ARMS = ('visible','envelope')


def prefix(fixture):
    record,anchor,*_ = base.setup()
    action=anchor['original_action']
    target=action['target_id']
    walls=hist.static_obstacles.StaticObstacleMap()
    envelope=footprint_gap.FootprintHistory()
    with forks.archive.S.phase.world(fixture) as (world,_):
        def ingest():
            obs=forks.archive.S.observe(world)
            hist.static_map.update_static_map(walls,obs)
            envelope.observe(obs,target)
            return obs
        for row in record['steps'][:anchor['step_index']]:
            for seg in row['segments']:
                obs=ingest()
                if world.env.termination or not forks.legal(seg['requested'],obs):
                    return dict(status='preparation_unavailable')
                world.step(seg['requested'],remaining_ticks=seg['end_tick']-seg['start_tick'])
        obs=ingest()
        robot,axis=base.push_phases.pusher(action)
        if not forks.legal(action,obs) or base.push_phases.contact_state(world,action,robot,axis)!='ready':
            return dict(status='no_ready_contact')
        wall=hist.describe(walls,obs,action)
        bounds=None
        if wall is not None:
            phi=math.atan2(axis[1],axis[0])+wall[1]
            bounds=envelope.gap_range(wall[0],(math.cos(phi),math.sin(phi)))
        return dict(status='ready',public_history=world.history.sha256,features=forks.features(obs,action),
                    interval=bounds,walls=dict(visible=wall,envelope=[bounds[0],wall[1]] if bounds else None))


def collect(args):
    split,spec,models=args
    path=ROOT/split/(spec['id']+'.json')
    if path.exists():
        return loop.read(path)
    fixture,plane,angle=base.room(spec)
    p=dict(id=spec['id'],fixture_hash=forks.archive.content_hash(fixture),**prefix(fixture))
    if p['status']=='ready' and models:
        p['decisions']={arm:[base.risk_decision(models[arm],p['features'],p['walls'][arm],dose,'wall_scan')
                             for dose in base.DOSES] for arm in ARMS}
    if split!='fresh':
        old=loop.read(hist.ROOT/('development' if split=='development' else 'validation')/(p['id']+'.json'))
        if p['fixture_hash']!=old['fixture_hash'] or p.get('public_history')!=old.get('public_history'):
            raise ValueError('archived prefix mismatch')
        if p['status']=='ready':
            if p['walls']['visible']!=old['walls']['history']:
                raise ValueError('changed wall descriptor')
            if old['status']=='ready' and models and old['decisions']['history']!=p['decisions']['visible']:
                raise ValueError('changed baseline decision')
        p['status']=old['status']
        if p['status']=='ready':
            p['branches']=old['branches']
    elif p['status']=='ready':
        record,anchor,_,_,old_model,donors=base.setup()
        p['branches']=[]
        for dose in base.DOSES:
            o=base.bench.branch(fixture,plane,angle,record,anchor,dose,old_model,donors)
            if o['public_history']!=p['public_history']:
                raise ValueError('branch prefix changed')
            if o['status']!='measured':
                p['status']=o['status']
                break
            p['branches'].append(dict(dose=dose,safe=o['useful'] and o['collision_free_through_contact_check'],outcome=o))
    loop.seal(path,p)
    return p


def run():
    prior=loop.read(hist.ROOT/'plan.json')
    seed=915047
    rng=random.Random(seed)
    fresh=[dict(id=f'outline-{i:03}',gap=rng.uniform(.065,.575),normal_offset=rng.uniform(20,70),
                lateral=rng.uniform(-.15,.15),width=rng.uniform(.65,.86),
                dx=rng.uniform(-.02,.02) if i%2 else 0.,dy=rng.uniform(-.02,.02) if i%2 else 0.) for i in range(120)]
    plan=dict(development=prior['development'],review=prior['validation'],fresh=fresh,seed=seed,
              safety=prior['safety'],selection=prior['selection'],thresholds=prior['thresholds'],
              estimator='two-dimensional rectangular envelope, lower gap endpoint; unsupported cases defer',
              source_hashes={Path(m.__file__).name:loop.sha(Path(m.__file__)) for m in (base,hist,footprint_gap,hist.static_obstacles,hist.static_map)},
              script_hash=loop.sha(Path(__file__)),collector=forks.frozen_hash())
    loop.seal(ROOT/'plan.json',plan)
    seen={loop.read(f)['fixture_hash'] for folder in (base.ROOT/'validation',hist.ROOT/'development',hist.ROOT/'validation')
          for f in folder.glob('*.json')}
    fresh_ids=[forks.archive.content_hash(base.room(c)[0]) for c in fresh]
    if len(set(fresh_ids))!=len(fresh_ids) or seen & set(fresh_ids):
        raise ValueError('duplicate fixture')
    models={}
    for split in ('development','review','fresh'):
        with ProcessPoolExecutor(8) as pool:
            rows=list(pool.map(collect,[(split,s,models) for s in plan[split]]))
        print(split,Counter(p['status'] for p in rows),flush=True)
        if split=='development':
            for arm in ARMS:
                models[arm]=[dict(id=p['id'],features=p['features'],wall=p['walls'][arm],dose=b['dose'],safe=b['safe'])
                             for p in rows if p['status']=='ready' for b in p['branches']]
            if models['visible']!=loop.read(hist.ROOT/'models.json')['history']:
                raise ValueError('changed training set')
            loop.seal(ROOT/'models.json',models)


if __name__=='__main__':
    run()
