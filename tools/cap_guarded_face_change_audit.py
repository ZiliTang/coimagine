"""Replay two frozen development rooms to diagnose the CCW follow-up's contact test."""
import math
from collections import Counter
from unittest.mock import patch

from experiments.planar_tasks.semantic_rl import push_phases
from tools.guarded_calibration_v1 import cap_guarded_calibration_forks as forks


REPORT = forks.OUT / 'face_change_dev_audit.json'


def angle(x, y):
    return math.degrees(math.atan2(y, x))


def legal_reason(command, obs):
    try:
        forks.validate_action(command, obs)
    except ValueError as exc:
        return str(exc)
    return 'offered'


def snapshot(world, original, follow, orbit):
    obs = forks.archive.S.observe(world)
    robot, axis = push_phases.pusher(follow)
    current = obs['robots'][robot]
    x, y = forks.xy(current)
    ox, oy, yaw = forks.archive.truth(world)['object']
    _, source_axis = push_phases.pusher(original)
    return dict(tick=obs['tick'], robot_xy=[x, y], box_xy=[ox, oy], box_yaw_deg=math.degrees(yaw),
                robot_polar_deg=angle(x-ox, y-oy), target_range_m=math.hypot(x-ox, y-oy),
                on_source_face=push_phases.contact_state(world, original, robot, source_axis),
                on_requested_face=push_phases.contact_state(world, follow, robot, axis),
                observed_contact=current['contacting_object'], stage=current['execution'].get('stage'),
                ccw_menu=legal_reason(orbit, obs), requested_push_menu=legal_reason(follow,obs))


def replay_suffix(room, anchor, progress):
    original_suffix = forks.followup
    original_step = forks.next_step
    contact_method = push_phases.contact
    details = []
    def observed_suffix(world, action, *, execute):
        robot, _ = push_phases.pusher(action)
        follow = forks.new_face(action, robot)
        orbit = forks.changed_action(action, robot, forks.ORBIT, 12)
        world_step = world.step

        def log_decision(*args):
            decision = original_step(*args)
            row = snapshot(world, action, follow, orbit)
            details.append(dict(**row, decision_if_budget_remains=decision))
            return decision

        def log_command(command, *args, **kwargs):
            event = world_step(command, *args, **kwargs)
            details[-1]['actual_command'] = dict(code=command[f'a{robot}'], ticks=event['periods'],
                reason=event['reason'], executed_segments=event['segments'])
            return event

        def log_contact(*args, **kwargs):
            phase = contact_method(*args, **kwargs)
            details[-1]['actual_command']['contact_phase_status'] = phase.status
            return phase

        with patch.object(forks, 'next_step', log_decision), patch.object(world, 'step', log_command), \
                patch.object(push_phases, 'contact', log_contact):
            return original_suffix(world, action, execute=execute)

    with patch.object(forks, 'followup', observed_suffix):
        result = forks.evaluate(forks.development_rooms()[room], forks.record_for('dev',room),
                                anchor,progress,execute_followup=True)
    unwrapped = 0.
    start = details[0]['tick']
    previous = details[0]['robot_polar_deg']
    for row in details:
        polar = row['robot_polar_deg']
        unwrapped += (polar - previous + 180) % 360 - 180
        row['ccw_rotation_since_endpoint_deg'] = unwrapped
        row['elapsed_suffix_ticks'] = row['tick'] - start
        previous = polar
    return dict(room=room, anchor_id=anchor['anchor_id'], progress_m=progress,
                original_action=anchor['original_action'], outcome=result, timeline=details)


def positive_control(room, anchor, progress):
    """At the measured candidate endpoint, request contact on its original surface once."""
    reading = {}
    def same_face(world, action, *, execute):
        robot, axis = push_phases.pusher(action)
        request = forks.changed_action(action, robot, action[f'a{robot}'], 24)
        reading.update(endpoint_tick=world.env.total_periods,
                       menu=legal_reason(request, forks.archive.S.observe(world)),
                       before=push_phases.contact_state(world, request, robot, axis))
        if reading['menu'] == 'offered':
            result = push_phases.contact(world, request, 24)
            reading.update(phase_result=result.status, ticks=result.ticks,
                           after=push_phases.contact_state(world, request, robot, axis))
        return {}
    with patch.object(forks, 'followup', same_face):
        forks.evaluate(forks.development_rooms()[room], forks.record_for('dev',room),
                       anchor,progress,execute_followup=True)
    return dict(room=room, anchor_id=anchor['anchor_id'], progress_m=progress, **reading)


def main():
    src = forks.OUT/'forks/dev'
    bundles = [forks.archive.read(src/(room+'.json')) for room in sorted(forks.development_rooms())]
    if any(b['collector_sha256'] != forks.frozen_hash() for b in bundles):
        raise ValueError('development archive uses a different collector')
    rows = [r['outcome'] for b in bundles for r in b['rows']]
    totals = dict(candidates=len(rows), followup_completed=sum(r['followup_completed'] for r in rows),
                  entirely_orbit=sum(r['approach_ticks']==0 for r in rows),
                  issued_approach=sum(r['approach_ticks']>0 for r in rows),
                  consumed_whole_budget=sum(r['followup_ticks']==forks.FOLLOW_BUDGET for r in rows),
                  final_reason=dict(Counter(r['followup_reason'] for r in rows)))
    examples, controls = [], []
    for i, bundle in enumerate(bundles[:2]):
        room = bundle['room']
        for anchor in bundle['anchors']:
            progress = .10
            source = next(r['outcome'] for r in bundle['rows']
                          if r['progress_m']==progress and r['anchor_id']==anchor['anchor_id'])
            if i == 0:
                example = replay_suffix(room, anchor, progress)
                if example['outcome'] != source:
                    raise ValueError('audit changed archived execution')
                examples.append(example)
            if anchor['anchor_id'] == 'early':
                controls.append(positive_control(room,anchor,progress))
    packet = dict(protocol='frozen-change-face-development-audit-v1', collector_sha256=forks.frozen_hash(),
                  selection='first two dev IDs in lexical order; short PUSH from earliest ready; first room also later ready',
                  angle_definition='counter-clockwise polar angle of pusher about the true box center, audit only',
                  totals=totals, examples=examples, same_face_positive_controls=controls)
    forks.write_json(REPORT,packet)
    print(REPORT,totals)


if __name__ == '__main__':
    main()
