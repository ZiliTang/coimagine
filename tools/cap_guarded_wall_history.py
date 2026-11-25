"""Current versus accumulated static surfaces; frozen gate and paired physical suffixes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random

from coimagine.continuous2d import static_obstacles
from experiments.planar_tasks.relational_search import static_map
from tools import cap_guarded_wall_validation as base

forks, loop = base.forks, base.loop
ROOT = base.ROOT.parent / 'cap_guarded_wall_history_v1'
ARMS = ('current', 'history')
SEED = 341927


def fresh_layouts(n):
    rng = random.Random(SEED)
    return [dict(id=f'fresh-{i:03}', gap=rng.uniform(.065, .575), normal_offset=rng.uniform(20, 70),
                 lateral=rng.uniform(-.15, .15), width=rng.uniform(.65, .86),
                 dx=rng.uniform(-.02, .02) if i % 2 else 0., dy=rng.uniform(-.02, .02) if i % 2 else 0.)
            for i in range(n)]


def describe(obstacle_map, obs, action):
    points = next((e['points'] for e in obs['entities'] if e['id'] == action['target_id']), [])
    _, axis = base.push_phases.pusher(action)
    hit = obstacle_map.forward_surface(points, axis, lateral_margin=.80, min_alignment=.25,
                                       min_gap=-.04, max_gap=1.20)
    return None if hit is None else [hit.normal_gap, hit.normal_angle]


def compared_prefix(fixture):
    record, anchor, *_ = base.setup()
    memory = static_obstacles.StaticObstacleMap()
    with forks.archive.S.phase.world(fixture) as (world, _):
        for row in record['steps'][:anchor['step_index']]:
            for seg in row['segments']:
                obs = forks.archive.S.observe(world)
                if world.env.termination or not forks.legal(seg['requested'], obs):
                    return dict(status='preparation_unavailable')
                static_map.update_static_map(memory, obs)
                world.step(seg['requested'], remaining_ticks=seg['end_tick'] - seg['start_tick'])
        obs = forks.archive.S.observe(world)
        action = anchor['original_action']
        robot, axis = base.push_phases.pusher(action)
        if not forks.legal(action, obs) or base.push_phases.contact_state(world, action, robot, axis) != 'ready':
            return dict(status='no_ready_contact')
        current = static_obstacles.StaticObstacleMap()
        for m in (current, memory):
            static_map.update_static_map(m, obs)
        wall = describe(current, obs, action)
        old = base.scan_descriptor(obs, action)
        if (wall is None) != (old is None) or (wall and any(abs(a - b) > 1e-9 for a, b in zip(wall, old))):
            raise ValueError('current-frame control differs from the frozen descriptor')
        return dict(status='ready', public_history=world.history.sha256, features=forks.features(obs, action),
                    walls=dict(current=wall, history=describe(memory, obs, action)))


def collect(job):
    spec, split, models = job
    path = ROOT / split / (spec['id'] + '.json')
    if path.exists():
        return loop.read(path)
    fixture, plane, angle = base.room(spec)
    p = dict(id=spec['id'], fixture_hash=forks.archive.content_hash(fixture), **compared_prefix(fixture))
    if split == 'development':
        old = loop.read(base.ROOT / split / (spec['id'] + '.json'))
        if old['fixture_hash'] != p['fixture_hash'] or old.get('public_history') != p.get('public_history'):
            raise ValueError('development fixture or preparation changed')
        if p['status'] == 'ready':
            if old['status'] != 'measured':
                raise ValueError('missing development labels')
            p['branches'] = [dict(dose=b['dose'], safe=b['safe']) for b in old['branches']]
        elif old['status'] != p['status']:
            raise ValueError('preparation changed')
    elif p['status'] == 'ready':
        decisions = {arm: [base.risk_decision(models[arm], p['features'], p['walls'][arm], d, 'wall_scan')
                            for d in base.DOSES] for arm in ARMS}
        p['decisions'] = decisions
        record, anchor, _, _, old_model, donors = base.setup()
        branches = []
        for dose in base.DOSES:
            outcome = base.bench.branch(fixture, plane, angle, record, anchor, dose, old_model, donors)
            if outcome['public_history'] != p['public_history']:
                raise ValueError('physical fork diverged before the decision')
            if outcome['status'] != 'measured':
                p['status'] = outcome['status']
                break
            branches.append(dict(dose=dose, safe=outcome['useful'] and outcome['collision_free_through_contact_check'],
                                 outcome=outcome))
        p['branches'] = branches
    loop.seal(path, p)
    return p


def run(workers):
    old_plan = loop.read(base.ROOT / 'plan.json')
    plan = dict(development=old_plan['partitions']['development'], validation=fresh_layouts(120),
                seed=SEED, doses=list(base.DOSES), thresholds=old_plan['thresholds'], source_hashes={
                    Path(m.__file__).name: loop.sha(Path(m.__file__)) for m in (static_obstacles, static_map, base, base.bench)},
                driver_hash=loop.sha(Path(__file__)), collector=forks.frozen_hash(),
                old_model=loop.sha(base.ROOT/'model-72.json'),
                selection='long if admitted, else short if admitted, else wait',
                safety=old_plan['safe'])
    loop.seal(ROOT / 'plan.json', plan)
    old_ids = {loop.read(f)['fixture_hash'] for phase in ('development','validation')
               for f in (base.ROOT / phase).glob('*.json')}
    fresh_ids = [forks.archive.content_hash(base.room(c)[0]) for c in plan['validation']]
    if len(set(fresh_ids)) != len(fresh_ids) or set(fresh_ids) & old_ids:
        raise ValueError('overlapping fixtures')
    models = {}
    for split in ('development','validation'):
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(collect, [(s, split, models) for s in plan[split]]))
        print(split, Counter(r['status'] for r in rows), flush=True)
        if split == 'development':
            for arm in ARMS:
                models[arm] = [dict(id=p['id'], features=p['features'], wall=p['walls'][arm],
                                    dose=b['dose'], safe=b['safe']) for p in rows if p['status']=='ready' for b in p['branches']]
            if models['current'] != loop.read(base.ROOT/'model-72.json'):
                raise ValueError('current-frame training set differs from frozen baseline')
            loop.seal(ROOT/'models.json', models)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    run(parser.parse_args().workers)
