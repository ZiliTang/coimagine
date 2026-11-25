"""Check two option assumptions against archived commands of the frozen nominal CAP."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median

from experiments.planar_tasks.authoring_v2.guarded_options import compile_policy
from tools import cap_safe_nominal_study as archive


OUT = archive.OUT.parent / 'cap_guarded_option_v1' / 'physical_replay.json'
PEER_MODES = {44: 'BRAKE_HOLD', 46: 'YIELD_CLEARANCE'}
TOL = 1e-6  # Same tolerance as the original physical endpoint audit.


def collect(path):
    data = archive.read(path)
    steps = data['steps']
    if not steps or steps[-1]['end_tick'] != data['terminal']['tick']:
        raise ValueError(f'incomplete trace in {path}')
    rows = []
    for i, step in enumerate(steps):
        after = steps[i + 1] if i + 1 < len(steps) else None
        if after is not None and after['tick'] != step['end_tick']:
            raise ValueError(f'discontinuous boundaries in {path} at {i}')
        action = step['action']
        codes = (action['a0'], action['a1'])
        if action['target_id'] != 'entity-a' or min(codes) >= 32:
            continue
        permit = step['permit']
        if permit['command'] != action:
            raise ValueError(f'permit and issued command disagree in {path} at {i}')
        bounds = permit['followup']['envelope']
        ux, uy = permit['object_axis']
        if not math.isclose(math.hypot(ux, uy), 1.0):
            raise ValueError(f'non-unit permit axis in {path} at {i}')
        start = step['truth']['object']
        end = step['end_truth']['object']
        dx, dy, angle = (b - a for a, b in zip(start, end))
        forward = dx * ux + dy * uy
        side = -dx * uy + dy * ux
        yaw = math.atan2(math.sin(angle), math.cos(angle))
        row = dict(room=path.stem, tick=step['tick'], end_tick=step['end_tick'], command=action,
                   robot_units=permit['robot_units'], robot_unit_m=permit['robot_unit_m'],
                   forward_m=forward, declared_max_forward_m=bounds['max_forward_m'],
                   endpoint_in_declared_envelope=(
                       -TOL <= forward <= bounds['max_forward_m'] + TOL
                       and abs(side) <= bounds['lateral_m'] + TOL and abs(yaw) <= bounds['yaw_rad'] + TOL))
        for j, code in enumerate(codes):
            if code in PEER_MODES:
                row['peer_mode'] = PEER_MODES[code]
                if after is not None:
                    row['peer_endpoint_motion_m'] = math.dist(step['xy'][j], after['xy'][j])
        rows.append(row)
    return rows


def describe(rows, room_ids, development_bounds):
    by_units = {}
    for units, (lo, hi) in development_bounds.items():
        subset = [r for r in rows if r['robot_units'] == units]
        ds = [r['forward_m'] for r in subset]
        if not ds:
            by_units[units] = dict(commands=0)
            continue
        outside = [r for r in subset if not lo-TOL <= r['forward_m'] <= hi+TOL]
        by_units[units] = dict(commands=len(ds), rooms=len({r['room'] for r in subset}),
            nominal_robot_progress_m=sorted({r['robot_unit_m'] * units for r in subset}),
            box_min_m=min(ds), box_median_m=median(ds), box_max_m=max(ds),
            box_advanced_at_least_2cm=sum(d >= .02 for d in ds),
            outside_development_forward_range=len(outside),
            outside_development_examples=[dict(room=r['room'], tick=r['tick'], end_tick=r['end_tick'],
                                               command=r['command'], forward_m=r['forward_m']) for r in outside])
    peers = {}
    for mode in PEER_MODES.values():
        selected = [r for r in rows if r.get('peer_mode') == mode]
        observed = [r for r in selected if 'peer_endpoint_motion_m' in r]
        moved = [r for r in observed if r['peer_endpoint_motion_m'] > .02]
        peers[mode] = dict(commands=len(selected), measurable=len(observed), rooms_with_measurement=len({r['room'] for r in observed}),
            moved_over_2cm=len(moved), rooms_moving_over_2cm=len({r['room'] for r in moved}),
            max_motion_example=max(observed, key=lambda r: r['peer_endpoint_motion_m'], default=None))
    good = Counter(r['room'] for r in rows if r['endpoint_in_declared_envelope'])
    total = Counter(r['room'] for r in rows)
    return dict(rooms=len(room_ids), push_commands=len(rows), envelope_inside=sum(good.values()),
        by_robot_progress_units=by_units, by_peer=peers,
        per_room={k:dict(push_commands=total[k], envelope_inside=good[k]) for k in sorted(room_ids)})


def check_empirical_peer_motion(rows):
    by_panel = {
        phase: [r['peer_endpoint_motion_m'] * 1000 for r in commands
                if r.get('peer_mode') == 'YIELD_CLEARANCE' and 'peer_endpoint_motion_m' in r]
        for phase, commands in rows.items()
    }
    dev = by_panel['dev']
    span = [math.floor(min(dev)), math.ceil(max(dev))]
    metric = 'peer_endpoint_motion_mm'
    option = 'push_with_yield'
    cell = 'issued_target_push_with_yield'
    profile = dict(units={metric: 'mm'}, cells={cell: {}}, options={
        option: dict(requires={}, quality='empirical', outcomes=[
            dict(observe='boundary', change={metric: span}, post={})]),
        'inspect_next_boundary': dict(requires={})})
    checks = {}
    for name, claim in (('declared_development_range', span), ('declared_stationary', [0, 0])):
        program = dict(rules=[dict(name=name, when={}, option=option, expect={metric: claim},
                                   then={'boundary': 'inspect_next_boundary'})])
        checked = compile_policy(program, profile)
        checks[name] = dict(program=program, check=checked.check(), default_dispatch=checked.decide(cell),
                            empirical_opt_in_dispatch=checked.decide(cell, allow_empirical=True))
    held = by_panel['confirm']
    return dict(scope='Check only the physical claim on endpoint displacement of the YIELD peer. '
                      'The continuation has no preconditions. No route or swept-volume safety is tested.',
                host_profile=profile, calibration_rooms_with_observation=len({
                    r['room'] for r in rows['dev'] if r.get('peer_mode') == 'YIELD_CLEARANCE'
                    and 'peer_endpoint_motion_m' in r}),
                calibration_commands=len(dev), calibration_raw_range_mm=[min(dev), max(dev)],
                confirmation_commands=len(held), confirmation_raw_range_mm=[min(held), max(held)],
                confirmation_outside_interval=sum(not span[0] <= v <= span[1] for v in held), checks=checks)


def run():
    lock = archive.read(archive.OUT / 'locked.json')
    if archive.sha(archive.OUT / lock['source']) != lock['sha256']:
        raise ValueError('frozen nominal source hash mismatch')
    panels = {'dev': archive.dev_rooms(), 'confirm': archive.read(archive.OUT / 'fixtures.json')}
    hashes = {phase: {archive.content_hash(v) for v in fixtures.values()} for phase, fixtures in panels.items()}
    if hashes['dev'] & hashes['confirm'] or set(panels['dev']) & set(panels['confirm']):
        raise ValueError('overlapping archived rooms')
    rows = {}
    trace_digests = {}
    for phase, count in (('dev', 16), ('confirm', 64)):
        files = sorted((archive.OUT / 'runs' / phase / lock['name'] / 'full').glob('*.json'))
        if len(hashes[phase]) != count or {p.stem for p in files} != set(panels[phase]):
            raise ValueError(f'wrong or incomplete {phase} panel')
        rows[phase] = [row for path in files for row in collect(path)]
        trace_digests[phase] = {p.stem: archive.sha(p) for p in files}
    expected_units = {r['robot_units'] for r in rows['dev']}
    if {r['robot_units'] for r in rows['confirm']} - expected_units:
        raise ValueError('held-out archive has unseen robot progress settings')
    bounds = {}
    for units in sorted(expected_units):
        ds = [r['forward_m'] for r in rows['dev'] if r['robot_units'] == units]
        bounds[units] = (min(ds), max(ds))
    return dict(schema='guarded_option_physical_replay_v1', archive=str(archive.OUT.relative_to(archive.ROOT)),
        policy=lock['name'], policy_sha256=lock['sha256'], panel_content_overlap=0,
        evidence='Historical issued target PUSH commands. Object truth is an offline endpoint label. '
                 'Peer displacement is public odometry at consecutive boundaries. No next odometry is '
                 'available for the last command. Movement within a window can exceed endpoint motion.',
        inference='Descriptive, room-clustered observations of the frozen policy. Robot progress units '
                  'are requested review distances. Duration, initial state, and peer mode were not randomized. '
                  'Development extrema are an empirical range, not physical guarantees.',
        development_box_forward_range_m=bounds, trace_sha256=trace_digests,
        yield_empirical_contract=check_empirical_peer_motion(rows),
        panels={k: describe(v, panels[k], bounds) for k, v in rows.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    dest = parser.parse_args().out
    report = run()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(dest)
