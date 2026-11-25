"""Paired progress-triggered PUSH and a fixed, physically executed change-of-face test."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.relational_search.motion import ACTIONS
from experiments.planar_tasks.semantic_rl import joint_face_change, push_phases
from experiments.planar_tasks.semantic_rl.phase_aware_cap import translating
from experiments.planar_tasks.semantic_rl.push_takeover import push_checkpoints
from tools import cap_guarded_option_forks as old
from tools import cap_safe_nominal_study as archive


OUT = archive.OUT.parent / 'cap_guarded_calibration_v3'
PROGRESS = (.10, .20)
PUSH_BUDGET = 40
FOLLOW_BUDGET = 120
ORBIT = 40  # canonical REPOSITION_CCW code
BRAKE = 44
MIN_LATE_TICK = 500
ANCHOR_GAP = 120
ROOMS = 16
VERSION = 'change-face-ccw-joint-orbit-v3'


def frozen_hash():
    from experiments.planar_tasks.relational_search import joint_orbit, target_binding
    sources = (__file__, joint_face_change.__file__, joint_orbit.__file__, target_binding.__file__, push_phases.__file__)
    return hashlib.sha256(b''.join(hashlib.sha256(Path(p).read_bytes()).digest() for p in sources)).hexdigest()


@lru_cache(maxsize=1)
def source_lock():
    lock = archive.read(archive.OUT / 'locked.json')
    if archive.sha(archive.OUT / lock['source']) != lock['sha256']:
        raise ValueError('frozen nominal source changed')
    return lock


@lru_cache(maxsize=1)
def development_rooms():
    return archive.dev_rooms()


@lru_cache(maxsize=1)
def panels():
    dev = development_rooms()
    confirm = archive.read(archive.OUT / 'fixtures.json')
    selected = {key: confirm[key] for key in sorted(confirm)[:ROOMS]}
    if len(dev) != ROOMS or len(confirm) != 64 or set(dev) & set(selected):
        raise ValueError('unexpected frozen room inventory')
    if {archive.content_hash(v) for v in dev.values()} & {archive.content_hash(v) for v in confirm.values()}:
        raise ValueError('overlapping room fixtures')
    return {'dev': dev, 'diagnostic': selected}


def trace_path(split, room):
    phase = {'dev': 'dev', 'diagnostic': 'confirm'}[split]
    return archive.OUT / 'runs' / phase / source_lock()['name'] / 'full' / (room + '.json')


@lru_cache(maxsize=1)
def record_for(split, room):
    rooms = development_rooms() if split == 'dev' else panels()[split]
    if room not in rooms:
        raise ValueError('room outside frozen split')
    return archive.read(trace_path(split, room))


def checked_boundary(world, record, i):
    row, ref = record['steps'][i], record['boundaries'][i]
    obs = archive.S.observe(world)
    if (world.history.sha256 != ref['public_history'] or obs['tick'] != row['tick']
            or archive.public_state(obs)['xy'] != row['xy'] or archive.truth(world) != row['truth']):
        raise ValueError(f'archive replay mismatch at step {i}')
    return obs


def changed_action(action, robot, code, ticks):
    return dict(action, **{f'a{robot}': code, 'ticks': ticks})


def new_face(action, robot):
    code = action[f'a{robot}']
    # Eight world directions, four speeds per direction. Keep the pusher and peer.
    return changed_action(action, robot, (((code // 4 + 2) % 8) * 4 + 1), 24)


def legal(command, obs):
    try:
        validate_action(command, obs)
    except ValueError:
        return False
    return True


def xy(robot):
    own = robot['own']
    return own['x'], own['y']


def features(obs, action):
    robot, (ux, uy) = push_phases.pusher(action)
    own = obs['robots'][robot]
    px, py = xy(own)
    qx, qy = xy(obs['robots'][1-robot])
    gx, gy = obs['goal']['xy']
    dx, dy = qx-px, qy-py
    scan = own['scan']
    if len(scan['valid']) != 72 or len(scan['entity_ids']) != 72:
        raise ValueError('requires fixed 72-ray observation')
    hits = sum(valid and label == action['target_id'] for valid, label in zip(scan['valid'], scan['entity_ids']))
    nearest = own['static_nearest']
    static = 1.2 if not nearest or nearest[0] is None else min(float(nearest[0]), 1.2)
    return dict(pusher_static_gap_m=static, peer_gap_m=math.hypot(dx,dy),
                peer_along_m=dx*ux+dy*uy, peer_lateral_m=dy*ux-dx*uy,
                target_ray_fraction=hits/72, push_speed_mps=ACTIONS[action[f'a{robot}']].speed,
                peer_yield=int(action[f'a{1-robot}'] == 46), goal_along_m=(gx-px)*ux+(gy-py)*uy,
                next_face_offered=int(legal(new_face(action, robot), obs)))


def anchors_for(fixture, record):
    found = []
    if len(record['steps']) != len(record['boundaries']):
        raise ValueError('archive boundary count mismatch')
    with archive.S.phase.world(fixture) as (world, _):
        for i, row in enumerate(record['steps']):
            obs = checked_boundary(world, record, i)
            if obs['tick'] + PUSH_BUDGET + FOLLOW_BUDGET > archive.S.MAX_TICKS:
                break
            action = row['action']
            eligible = not found or obs['tick'] >= max(MIN_LATE_TICK, found[0]['tick'] + ANCHOR_GAP)
            if eligible and action['target_id'] == obs['goal']['target_id'] and len(translating(action)) == 1:
                robot, axis = push_phases.pusher(action)
                if legal(action, obs) and push_phases.contact_state(world, action, robot, axis) == 'ready':
                    found.append(dict(anchor_id='late' if found else 'early', tick=obs['tick'], step_index=i,
                                      public_history=world.history.sha256, original_action=action,
                                      features=features(obs, action)))
                    if len(found) == 2:
                        break
            old.restore_step(world, row)
    return found


def followup(world, action, *, execute):
    return joint_face_change.change_face_ccw(world, action, execute=execute, max_decisions=archive.S.MAX_DECISIONS)


def evaluate(fixture, record, anchor, progress_m, *, execute_followup):
    if progress_m not in PROGRESS:
        raise ValueError('unknown progress action')
    with archive.S.phase.world(fixture) as (world, _):
        initial_tick = world.env.total_periods
        idx = anchor['step_index']
        for i in range(idx):
            checked_boundary(world, record, i)
            old.restore_step(world, record['steps'][i])
        obs = checked_boundary(world, record, idx)
        action = record['steps'][idx]['action']
        if world.history.sha256 != anchor['public_history'] or action != anchor['original_action']:
            raise ValueError('selected anchor changed')
        public = features(obs, action)
        if public != anchor['features']:
            raise ValueError('public feature mismatch')
        robot, (ux, uy) = push_phases.pusher(action)
        peer_before = xy(obs['robots'][1-robot])
        obj_before = archive.truth(world)['object']
        request = dict(action, ticks=PUSH_BUDGET)
        def checkpoint(public_obs, running, left, reason):
            if reason == 'progress':
                return dict(target_id=None, a0=BRAKE, a1=BRAKE, ticks=1)
            return None
        review = push_checkpoints(world, request, checkpoint, progress_m=progress_m, max_decisions=archive.S.MAX_DECISIONS)
        phase = push_phases.push_from_contact(world, request, PUSH_BUDGET, review=review)
        if phase.event is None or phase.ticks <= 0:
            raise ValueError('publicly selected contact PUSH did not execute')
        end = archive.S.observe(world)
        end_truth = archive.truth(world)
        dx, dy, da = (v-u for u,v in zip(obj_before, end_truth['object']))
        outcome = dict(prefix_ticks=obs['tick']-initial_tick, forward_m=dx*ux+dy*uy, lateral_m=dy*ux-dx*uy,
                       yaw_rad=math.atan2(math.sin(da), math.cos(da)),
                       peer_motion_m=math.dist(peer_before, xy(end['robots'][1-robot])),
                       object_static=end_truth['object_static'], push_reason=phase.status, push_ticks=phase.ticks,
                       endpoint_terminal=world.env.termination, progress_reached=phase.status=='review_switch')
        outcome.update(followup(world, action, execute=execute_followup))
        return outcome


def simulate_development_particle(room, anchor_id, progress_m):
    """One actual PUSH to endpoint on an identified development particle; no suffix."""
    if room not in development_rooms():
        raise ValueError('particle must belong to frozen development pool')
    packet = archive.read(OUT/'particles/dev'/(room+'.json'))
    if packet['collector_sha256'] != frozen_hash():
        raise ValueError('development particle belongs to another collector')
    anchor = next((a for a in packet['anchors'] if a['anchor_id'] == anchor_id), None)
    if anchor is None:
        raise ValueError('missing development particle anchor')
    return evaluate(development_rooms()[room], record_for('dev',room), anchor, progress_m, execute_followup=False)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    os.replace(temp,path)


def collect_room(job):
    split, room = job
    fixture = panels()[split][room]
    path = OUT/'forks'/split/(room+'.json')
    provenance = dict(protocol=VERSION, collector_sha256=frozen_hash(), policy_sha256=source_lock()['sha256'],
                      content_sha256=archive.content_hash(fixture), trace_sha256=archive.sha(trace_path(split,room)))
    if path.exists():
        current = archive.read(path)
        if not all(current.get(k) == v for k,v in provenance.items()):
            raise ValueError(f'frozen outputs differ for {room}; use a new output root')
        return split,room,len(current['rows'])
    record = record_for(split,room)
    anchors = anchors_for(fixture,record)
    if split == 'dev':
        write_json(OUT/'particles/dev'/(room+'.json'), dict(**provenance, room=room, anchors=anchors))
    rows = []
    for anchor in anchors:
        for p in PROGRESS:
            outcome = evaluate(fixture,record,anchor,p,execute_followup=True)
            rows.append(dict(room=room, anchor_id=anchor['anchor_id'], tick=anchor['tick'],
                             public_history=anchor['public_history'], progress_m=p, features=anchor['features'],
                             outcome=outcome))
    payload = dict(**provenance, split=split, room=room, anchors=anchors, rows=rows)
    write_json(path, payload)
    return split,room,len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev','diagnostic','both'), default='both')
    parser.add_argument('--workers', type=int, choices=range(1,5), default=4)
    args = parser.parse_args()
    phases = panels() if args.split == 'both' else [args.split]
    jobs = [(split,k) for split in phases for k in sorted(panels()[split])]
    with ProcessPoolExecutor(args.workers) as pool:
        for split,room,n in pool.map(collect_room,jobs):
            print(split,room,n,flush=True)


if __name__ == '__main__':
    main()
