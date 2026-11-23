"""Paired offline branches from the frozen s025 development episodes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import math

import numpy as np

from experiments.option_rl_v19 import unknown_shape as push
from tools import cap_regional_five as cap

SOURCE_DIR = cap.OUT.parent / 'cap_general_specialists_62_20261120_01'
OUTPUT = cap.OUT.parent / 's025_standoff_diagnostic_v1'
ARCHIVE = SOURCE_DIR / 'runs/s025'
ARMS = ('continue', 'same_face', 'return_cap')


def rows(room):
    with gzip.open(ARCHIVE / (room + '.json.gz'), 'rt') as stream:
        return json.load(stream)


def select(room, moving_control=False):
    history = rows(room)
    for a, b, c in zip(history, history[1:], history[2:]):
        request = c['a']
        if request['ticks'] < 16 or b['tick'] - a['tick'] < 20:
            continue
        for i in (0, 1):
            if request[f'a{i}'] >= 32 or request[f'a{1-i}'] < 32:
                continue
            old = b['a']
            if moving_control:
                if (old['target_id'] != request['target_id'] or old[f'a{1-i}'] < 32
                        or old[f'a{i}'] // 4 != request[f'a{i}'] // 4):
                    continue
            elif old != request:
                continue
            if a['robots'][i][2] != 'standoff' or b['robots'][i][2] != 'standoff':
                continue
            travel = math.dist(a['robots'][i][:2], b['robots'][i][:2])
            in_range = .05 < travel <= .40 if moving_control else travel <= .05
            if in_range:
                return dict(room=room, start=b['tick'], tick=b['tick'] + 8, robot=i,
                            request=request, travel=travel,
                            cohort='moving_success' if moving_control else 'stationary')
    return None


def alternate(env, robot, apply):
    """Choose one new point ~10 cm along the same observed finite segment."""
    q = env._commit_q.get(robot)
    anchor = env._commit_anchor.get(robot)
    if q is None or anchor is None:
        return dict(status='no_anchor')
    obs = env.boundary_observations()[robot]
    _, face, _, _ = push.track_contact_q(obs, q, anchor)
    if face is None:
        return dict(status='unmatched_segment')
    tangent = np.asarray(face[2])
    qt = float(tangent @ q)
    lo, hi = float(face[4]) + .02, float(face[5]) - .02
    candidates = [] if lo > hi else [float(np.clip(qt + offset, lo, hi)) for offset in (-.10, .10)]
    candidates = [x for x in candidates if abs(qt - x) >= .04]
    if not candidates:
        return dict(status='short_segment')
    own = (obs.own.x, obs.own.y)
    value = min(candidates, key=lambda x: math.dist(own, push._face_world(face, x)))
    point = tuple(push._face_world(face, value))
    if apply:
        env._commit_q[robot] = point
        env._commit_anchor[robot] = push._face_anchor(face, value)
    return dict(status='available', origin=list(q), point=point, shift=abs(value - qt))


def branch(job):
    case, arm = job
    room, i, trigger = case['room'], case['robot'], case['tick']
    fixture = cap.read(SOURCE_DIR / 'development.json')[room]
    archive = rows(room)
    ref = cap.read(ARCHIVE / (room + '.json'))
    log = dict(case=case, arm=arm, reference_success=ref['success'])
    flags = dict(contact_run=0, first_contact=None, stable_contact=None, window=0)

    def construct(*args, **kwargs):
        kwargs['history_factory'] = cap.fast.AuditedCachedHistory
        world = cap.fast.ScopedEnvironment(*args, **kwargs)
        env = world.env
        detector = world.detector.update
        original_step = world.step
        target = world.env.primary_id

        def position():
            # Offline outcome only. Neither the intervention nor the policy reads this.
            return tuple(env.env.backend.bodies[target].position)

        def observe(packet):
            detected = detector(packet)
            tick = env.total_periods
            robot = packet['robots'][i]
            if tick == trigger:
                current = cap.current_event(world.history)['observation']
                valid = (log.get('command_start') == case['start'] and env.active_target == target and case['request']['target_id'] == target
                         and not robot['contacting_object'] and robot['execution']['stage'] == 'standoff'
                         and detected not in ('contact_lost', 'blocked', 'controller_end'))
                log['triggered'] = valid
                log['prefix'] = cap.digest(dict(history=world.history.sha256, packet=packet,
                                                tick=tick, q=env._commit_q.get(i), object_xy=position()))
                log['goal'] = current['goal']['xy']
                log['xy_start'] = position()
                if valid:
                    log['alternative'] = alternate(env, i, arm == 'same_face')
                    if arm == 'return_cap':
                        return 'controller_end'
            if trigger < tick <= trigger + 80 and log.get('triggered'):
                contact = env.active_target == target and robot['contacting_object']
                flags['contact_run'] = flags['contact_run'] + 1 if contact else 0
                if contact and flags['first_contact'] is None:
                    flags['first_contact'] = tick - trigger
                if flags['contact_run'] >= 3 and flags['stable_contact'] is None:
                    flags['stable_contact'] = tick - trigger
                flags['window'] = tick - trigger
                log['xy_end'] = position()
            return detected

        def step(request, remaining_ticks=None):
            index = world.decision_count
            start = env.total_periods
            log['command_start'] = start
            if start <= case['start']:
                assert request == archive[index]['a'], (room, index, 'request mismatch')
            if start == case['start']:
                assert request == case['request']
            if start >= trigger and 'next_request' not in log:
                log.update(next_request=request.copy(), next_tick=start)
            event = original_step(request, remaining_ticks)
            if start < case['start']:
                assert event['end_tick'] == archive[index]['tick']
            return event

        world.detector.update = observe
        world.step = step
        return world

    try:
        res, _ = cap.rollout(fixture, cap.SOURCE, features=cap.FEATURES, config=cap.CFG,
                             deterministic=True, fast_feedback=True, record_details=False,
                             environment_factory=construct, max_ticks=1800, max_decisions=300)
        summary = res['summary']
        if summary['error'] or summary['stop'] != 'terminal':
            raise RuntimeError((room, arm, summary))
        log.update(success=summary['success'], final_tick=summary['ticks'], **flags)
        if arm == 'continue':
            assert cap.fast.ACTIVE.history.sha256 == ref['history']
        if 'xy_end' in log:
            log['progress_m'] = math.dist(log['xy_start'], log['goal']) - math.dist(log['xy_end'], log['goal'])
        return log
    finally:
        if cap.fast.ACTIVE:
            cap.fast.ACTIVE.close_reads()
            cap.fast.ACTIVE = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rooms = sorted(cap.read(SOURCE_DIR / 'development.json'))
    cases = [c for room in rooms if (c := select(room)) is not None]
    strict_ids = {c['room'] for c in cases}
    controls = [c for room in rooms if room not in strict_ids
                and cap.read(ARCHIVE / (room + '.json'))['success']
                and (c := select(room, moving_control=True)) is not None]
    cases.extend(controls)
    print('selected', len(cases), flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = (SOURCE_DIR / 'author/semantics/s025.py').read_text()
    with ProcessPoolExecutor(args.workers, initializer=cap.start_worker, initargs=(source,)) as pool:
        results = list(pool.map(branch, [(c, a) for c in cases for a in ARMS]))
    for c in cases:
        group = [r for r in results if r['case'] == c]
        assert all(r.get('prefix') for r in group)
        assert len({(r['prefix'], r['triggered']) for r in group}) == 1
        assert all(r.get('alternative') == group[0].get('alternative') for r in group)
    cap.save(OUTPUT / 'branches.json', results)
    print('verified', len(results), flush=True)


if __name__ == '__main__':
    main()
