"""Replay frozen transport choices with passive command and controller instrumentation."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from unittest.mock import patch

from experiments.option_rl_v19.unknown_shape import visible_faces
from tools import cap_symbolic_transport as study

transport = study.transport


def angle_degrees(unit_vector, direction):
    dot = sum(x*y for x,y in zip(unit_vector, direction))
    return math.degrees(math.acos(max(-1., min(1., dot))))


def physical_face_error(world, direction):
    """Offline audit of the triangulated target; shared edges are internal."""
    shapes = world.env.env.backend.shapes[world.env.primary_id]
    edge_count = Counter()
    errors = {}
    for shape in shapes:
        vertices = [shape.body.local_to_world(v) for v in shape.get_vertices()]
        centre = sum(vertices[1:], vertices[0]) / len(vertices)
        for a,b in zip(vertices,vertices[1:] + vertices[:1]):
            edge = tuple(sorted((tuple(a),tuple(b))))
            n = (b-a).perpendicular().normalized()
            if n.dot(centre-a) < 0:
                n = -n
            edge_count[edge] += 1
            errors[edge] = angle_degrees(n, direction)
    return min(errors[e] for e, count in edge_count.items() if count == 1)


def replay(spec):
    saved = study.loop.read(study.ROOT / 'fresh' / (spec['id'] + '.json'))
    live = saved.get('live', {})
    reason = live.get('followup_reason')
    if reason not in ('budget', 'ready'):
        return []
    key = live['selected']
    chosen = saved['branches'][key]
    rows = []
    # Also run the other heading with identical dose and turn, where available.
    for name, branch in saved['branches'].items():
        if (branch['dose'], branch['route']) != (chosen['dose'], chosen['route']):
            continue
        fixture, _, _ = study.base.room(spec)
        with study.prepared(fixture) as (world, _, _, commands):
            assert world.history.sha256 == saved['public_history'] and branch['cmd'] in commands
            cmd = branch['cmd']
            transport.run_push(world, cmd, branch['dose'])
            step = world.step
            update = world.detector.update
            ticks = Counter()
            boundary = Counter()
            active = {}
            travelled = 0.
            gate_gaps = []
            observed_angles = []
            physical_angles = []

            def observe(packet):
                nonlocal travelled
                robot = active['robot']
                info = world.env.slots[robot].last_info
                own = packet['robots'][robot]['own']
                xy = (own['x'], own['y'])
                if active['orbit']:
                    travelled += math.dist(active['previous'], xy)
                    ticks['orbit_request'] += 1
                    if info.get('kind') == 'reposition':
                        ticks['orbit_controller'] += 1
                        ticks['orbit_' + str(info['stage'])] += 1
                        if info.get('static_route_guard'):
                            ticks['static_sweep_block'] += 1
                            gate_gaps.append(info['static_route_guard']['swept_return_gap'])
                        if info.get('exit_reason'):
                            ticks[info['exit_reason']] += 1
                        ticks['outer_detour'] += bool((info.get('peer_detour') or {}).get('active'))
                else:
                    ticks['approach_request'] += 1
                active['previous'] = xy
                return update(packet)

            def measure(cmd, *args, **kwargs):
                orbit = 40 in (cmd['a0'], cmd['a1']) or 43 in (cmd['a0'], cmd['a1'])
                robot = int(cmd['a0'] == 44)
                tick = world.env.total_periods
                sensor = world.env._sensor(robot, tick, tick * study.transport.push_phases.CONTROL_DT, cmd['target_id'])
                active.update(orbit=orbit, robot=robot, previous=(sensor.own.x,sensor.own.y))
                if orbit:
                    boundary['orbit_requests'] += 1
                    old, _ = transport.push_phases.pusher(branch['cmd'])
                    direction = branch['cmd'][f'a{old}'] // 4
                    new_code = (direction + (2 if branch['route']=='ccw' else -2)) % 8 * 4 + 1
                    normal = transport.push_phases.ACTIONS[new_code].effect_direction
                event = step(cmd, *args, **kwargs)
                if orbit:
                    end = world.env.total_periods
                    sensor = world.env._sensor(robot, end, end * transport.push_phases.CONTROL_DT, cmd['target_id'])
                    follow = dict(cmd, **{f'a{robot}':new_code}, ticks=24)
                    obs = transport.current_event(world.history)['observation']
                    boundary['legal_approach_after_orbit'] += transport.face.legal(follow, obs)
                    observed_angles.extend(angle_degrees(f[1], normal) for f in visible_faces(sensor))
                    physical_angles.append(physical_face_error(world,normal))
                return event

            with patch.object(world, 'step', measure), patch.object(world.detector, 'update', observe):
                result = study.run_face(world, cmd, branch['route'])
            assert result['followup_completed'] == branch['actual']['completed']
            assert ticks['orbit_request'] + ticks['approach_request'] == result['followup_ticks']
            if name == key:
                assert result['followup_reason'] == reason and result['followup_ticks'] == live['followup_ticks']
            rows.append(dict(room=spec['id'], selected=name==key, command=name, reason=result['followup_reason'],
                             **{k:ticks[k] for k in ('orbit_request','approach_request','orbit_controller','orbit_find','orbit_align','orbit_move','static_sweep_block','safety_abort','outer_detour')},
                             orbit_travel_m=round(travelled, 4), **dict(boundary),
                             minimum_blocked_swept_gap_m=min(gate_gaps,default=None),
                             minimum_observed_face_error_deg=round(min(observed_angles),2) if observed_angles else None,
                             minimum_physical_face_error_deg=round(min(physical_angles),2) if physical_angles else None))
    return rows


def main():
    plan = study.loop.read(study.ROOT / 'plan.json')
    with ProcessPoolExecutor(4) as pool:
        rows = [row for room in pool.map(replay,plan['fresh']) for row in room]
    path = study.ROOT / 'orbit_audit.csv'
    with path.open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows), path=str(path))))


if __name__ == '__main__':
    main()
