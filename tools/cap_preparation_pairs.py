"""Paired preparation and handoff experiments with two frozen CAP experts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

import numpy as np

from tools import cap_route_horizon as prior

study = prior.study
OUT = study.OUT.parent / 'cap_preparation_pairs_20261123_01'
SEED = 662315904
WINDOW = 120
BRAKE = dict(target_id=None, a0=44, a1=44, ticks=1)


def protocol():
    previous = study.read(prior.OUT / 'protocol.json')
    common = next(m['sha'] for m in study.read(study.OUT / 'protocol.json')['libraries']['usual']
                  if m['id'] == 'basis')
    return dict(pair=previous['pair'], control=common, seed=SEED, window=WINDOW,
                arms=['A', 'B', 'A3', 'A10', 'C10', 'wait10', 'event'],
                gate='After 1..10 A proposals since first disagreement, switch to legal B on new target PUSH, '
                     'or if estimated goal-facing rear station is within 0.45m and robot-to-station scan clearance '
                     '>=0.16m and distance improved by >=0.10m vs fork (or became visible). Otherwise retain A.',
                window_readout='First 120 physical ticks after handoff or episode end; terminal state held fixed. '
                               'Physical contact requires robot with goal object on 3 consecutive ticks; progress '
                               'is reduction in physical goal gap. A/C/wait keep the original total episode budgets.',
                controller=prior.EXECUTOR, old_panel=previous['rooms']['held'])


def public_state(obs, b):
    target = obs['goal']['target_id']
    pushing = bool(b and b['target_id'] == target and study.translating(b))
    scans = [study.endpoints(r) for r in obs['robots']]
    hits = [p[ids == target] for p, _, ids in scans]
    seen = np.concatenate(hits)
    distance = gap = None
    if len(seen) >= 2:
        axis = np.asarray(obs['goal']['xy']) - seen.mean(0)
        axis /= max(1e-10, np.linalg.norm(axis))
        options = []
        for i, points in enumerate(hits):
            if len(points) < 2:
                continue
            rear = points[np.argsort(points @ axis)[:max(1,len(points)//3)]].mean(0)
            station = rear - (study.ROBOT_RADIUS + .15)*axis
            own = np.asarray(study.position(obs, i))
            p, _, ids = scans[i]
            peer = np.asarray([study.position(obs, 1-i)])
            obstacles = np.concatenate((p[ids != target], peer))
            options.append((float(np.linalg.norm(station-own)), study.clearance(own, station, obstacles)))
        if options:
            distance, gap = min(options)
    return dict(push=pushing, distance=distance, clearance=gap)


def gate(start, now):
    if now['push'] and not start['push']:
        return 'new_push'
    d = now['distance']
    if d is not None and d <= .45 and now['clearance'] >= .16:
        if start['distance'] is None or start['distance'] - d >= .10:
            return 'station'
    return None


def goal_gap(w, target, goal):
    return float(np.linalg.norm(np.asarray(w.env.env.backend.bodies[target].position)-goal))


def observe_window(w, obs):
    target = obs['goal']['target_id']
    goal = np.asarray(obs['goal']['xy'])
    initial = goal_gap(w, target, goal)
    start = obs['tick']
    data = dict(ticks=0, stable=False, progress=0., target_push=False, push_along_m=0., repeated_stall=False)
    original = w.detector.update
    streak = [0, 0]
    direction = [None]
    previous_xy = [np.asarray(w.env.env.backend.bodies[target].position)]
    def audit(packet):
        status = original(packet)
        elapsed = w.env.total_periods - start
        if 0 < elapsed <= WINDOW:
            data['ticks'] = elapsed
            touched = {c['robot_id'] for c in w.env.env.backend.contacts()
                       if c['pair'] == 'robot_object' and c['entity_id'] == target}
            for i in (0, 1):
                streak[i] = streak[i]+1 if i in touched else 0
            data['stable'] |= max(streak) >= 3
            xy = np.asarray(w.env.env.backend.bodies[target].position)
            if direction[0] is not None:
                data['push_along_m'] += float((xy-previous_xy[0]) @ direction[0])
            previous_xy[0] = xy
            data['progress'] = initial - goal_gap(w, target, goal)
        return status
    w.detector.update = audit
    return data, direction


def simulate(fixture, ids, arm, schedule=None):
    control = arm in ('C10', 'wait10')
    policies = [study.phase.FrozenCAP(study.MODULES[e]) for e in ids[:3 if arm == 'C10' else 2]]
    n = 0
    first = start_public = hand = window = event = None
    times = {}
    recent = []
    routed = False
    invalid = trimmed = 0
    counts = Counter()
    with study.phase.world(fixture) as (w, _):
        def update(history):
            for p in policies:
                p.update(history)
        update(w.history)
        while not study.ended(w):
            obs = study.observe(w)
            tick = obs['tick']
            a = None if routed else prior.try_action(policies[0], obs)
            b = None
            if routed or first is None or (arm in ('A','event') and n-first['n'] <= 10):
                b = prior.try_action(policies[1], obs)
            if first is None and a is not None and b is not None and a != b:
                first = dict(n=n, tick=tick, history=w.history.sha256)
                start_public = public_state(obs, b)
            h = n-first['n'] if first else -1
            if arm in ('A','event') and 0 <= h <= 10:
                if h in (0, 3, 10):
                    times[str(h)] = dict(tick=tick, history=w.history.sha256)
                if h and b is not None and event is None:
                    signal = gate(start_public, public_state(obs, b))
                    if signal:
                        event = dict(h=h, reason=signal, tick=tick, history=w.history.sha256)
            due = False
            if schedule is not None and hand is None and first is not None:
                due = tick == schedule['tick'] if control else h == schedule['h']
            if arm=='event' and event and h==event['h']:
                due=True
            if due:
                b = prior.try_action(policies[1], obs)
                hand = dict(tick=tick, history=w.history.sha256, public=public_state(obs, b), legal=b is not None)
                if not control and arm != 'event':
                    assert hand['history'] == schedule['history']
                if b is not None:
                    routed = True
                    window, direction = observe_window(w, obs)
                    hand['true_gap_m'] = goal_gap(w,obs['goal']['target_id'],obs['goal']['xy'])
            if routed:
                if b is None:
                    invalid += 1
                    a = prior.try_action(policies[0], obs)
                step = b or a or BRAKE
                if window['ticks'] < WINDOW:
                    signature = (step['target_id'],step['a0'],step['a1'])
                    xy = np.asarray([study.position(obs,i) for i in (0,1)])
                    window['repeated_stall'] |= any(sig==signature and np.linalg.norm(xy-pos,axis=1).max()<.06 for sig,pos in recent[-3:])
                    recent.append((signature,xy))
                    movers = study.translating(step) if step['target_id']==obs['goal']['target_id'] else []
                    window['target_push'] |= bool(movers)
                    direction[0] = np.mean([study.ACTIONS[step[f'a{i}']].effect_direction for i in movers],axis=0) if movers else None
            elif control and schedule and first is not None and tick < schedule['tick']:
                remaining = schedule['tick']-tick
                step = (prior.try_action(policies[2], obs) or a or BRAKE) if arm=='C10' else dict(BRAKE,ticks=40)
                trimmed += int(step['ticks']>remaining)
                step = dict(step,ticks=min(step['ticks'],remaining))
            else:
                step = a or BRAKE
            before = tick
            study.tick_action(w, step, prior.EXECUTOR, update, counts)
            if control and schedule and first and before < schedule['tick']:
                assert w.env.total_periods <= schedule['tick']
            n += 1
        out = study.output(w, counts)
    out.pop('counts')
    return dict(**out, first=first, hand=hand, window=window, event=event, times=times, invalid=invalid, trimmed=trimmed)


def room(job):
    fold, key, fixture, ids = job
    path = OUT/'runs'/fold/(key+'.json')
    if path.exists():
        return
    rows = {}
    rows['A'] = base = simulate(fixture, ids, 'A')
    for h, arm in [(0, 'B'), (3, 'A3'), (10, 'A10')]:
        cut = base['times'].get(str(h))
        rows[arm] = simulate(fixture, ids, arm, dict(**cut,h=h)) if cut else base
    for arm in ('C10','wait10'):
        cut = base['times'].get('10')
        rows[arm] = simulate(fixture,ids,arm,cut) if cut else base
    rows['event'] = simulate(fixture, ids, 'event')
    assert base['event'] == rows['event']['event']
    if not base['event']:
        assert base['history'] == rows['event']['history']
    assert all(v['first']==base['first'] for v in rows.values())
    if fold=='diagnostic':
        archived = study.read(prior.OUT/'runs'/'held'/(key+'.json'))
        for arm, old in [('A',archived['baseline'])]+[(arm,archived['horizons'][h]) for arm,h in [('B','0'),('A3','3'),('A10','10')]]:
            assert all(rows[arm][s] == old[s] for s in ('history','ticks','decisions','success')), (key,arm)
    study.save(path,rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=18)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    p=protocol()
    ids=p['pair']+[p['control']]
    held=study.read(study.OUT/'held.json')
    fresh={r['id']:r['fixture'] for r in study.phase.cap.rooms.generate(SEED)['cases']}
    hashes={study.phase.input_hash(f) for f in fresh.values()}
    used={study.phase.input_hash(f) for k in ['held','development'] for f in study.read(study.OUT/(k+'.json')).values()}
    for path in (prior.OUT/'confirmation/panel.json',study.OUT/'initial_confirmation/held.json',
                 study.OUT/'selective_semantics/held.json'):
        used.update(map(study.phase.input_hash,study.read(path).values()))
    for seed in (662315862,662315879):
        used.update(study.phase.input_hash(r['fixture']) for r in study.phase.cap.rooms.generate(seed)['cases'])
    used.update(r['content_sha256'] for r in study.read(study.phase.cap.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json'))
    assert len(hashes)==200 and not hashes&used
    jobs=[('diagnostic', k, held[k], ids) for k in p['old_panel']]+[('fresh',k,f,ids) for k,f in fresh.items()]
    study.initialize()
    if args.smoke:
        room(next(j for j in jobs if j[1]=='test_a_dev_2_05'))
        return
    path=OUT/'protocol.json'
    frozen=dict(**p, fresh_hashes=sorted(hashes))
    if path.exists():
        assert study.read(path)==frozen
    else:
        study.save(path,frozen)
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('fork')) as pool:
        tasks=[pool.submit(room,j) for j in jobs]
        for i,f in enumerate(as_completed(tasks),1):
            f.result()
            if i%10==0 or i==1:
                study.save(OUT/'progress.json',dict(rooms=i,total=len(jobs)))
                print(i,len(jobs),flush=True)


if __name__=='__main__':
    main()
