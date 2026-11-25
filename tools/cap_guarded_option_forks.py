"""Replay six archived public prefixes and stop the same PUSH at two progress events."""
import json
import math

from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.semantic_rl import push_phases
from experiments.planar_tasks.semantic_rl.phase_aware_cap import translating
from experiments.planar_tasks.semantic_rl.push_takeover import push_checkpoints
from tools import cap_safe_nominal_study as archive


OUT = archive.OUT.parent / 'cap_guarded_option_v1' / 'paired.json'
THRESHOLDS = (.10, .20)
MAX_TICKS = 40


def restore_step(world, row):
    # Recorded controls predate the joint-orbit resolver. Restore that contract
    # for the prefix only; the candidate suffix runs under its own version.
    binding = world.env.binding
    enabled = binding.joint_orbits
    binding.joint_orbits = False
    try:
        for piece in row['segments']:
            if world.env.total_periods != piece['start_tick']:
                raise ValueError('archived segment starts at a different tick')
            ticks = piece['end_tick'] - piece['start_tick']
            event = world.step(piece['requested'], remaining_ticks=ticks)
            if event['segments'] != piece['segments'] or event['end_tick'] != piece['end_tick']:
                raise ValueError('archived physical controls did not replay')
        if archive.truth(world) != row['end_truth']:
            raise ValueError('archived object truth did not replay')
    finally:
        binding.joint_orbits = enabled


def branch(fixture, record, threshold):
    with archive.S.phase.world(fixture) as (world, _):
        for row, boundary in zip(record['steps'], record['boundaries'], strict=True):
            obs = archive.S.observe(world)
            if (world.history.sha256 != boundary['public_history'] or obs['tick'] != row['tick']
                    or archive.public_state(obs)['xy'] != row['xy'] or archive.truth(world) != row['truth']):
                raise ValueError('archived prefix did not replay')
            action = row['action']
            movers = translating(action)
            if action['target_id'] == obs['goal']['target_id'] and len(movers) == 1:
                robot, axis = push_phases.pusher(action)
                if push_phases.contact_state(world, action, robot, axis) == 'ready':
                    break
            restore_step(world, row)
        else:
            return None
        validate_action(action, obs)
        # All six frozen permits retain the same pusher, direction and peer command.
        follow = row['permit']['followup']
        if (follow['kind'] != 'continue_contact' or follow['robot'] != robot
                or follow['direction_index'] != action['a'+str(robot)] // 4):
            raise ValueError('recorded continuation differs from the intended maneuver')
        continuation = dict(action, ticks=1)
        prefix = dict(tick=obs['tick'], public_history=world.history.sha256,
                      object=archive.truth(world)['object'], original_action=action, continuation=continuation)
        request = dict(action, ticks=MAX_TICKS)
        progress_events = []

        def checkpoint(public, running, left, reason):
            if reason == 'progress':
                progress_events.append(public['tick'])
                # A valid replacement ends the window; the replacement is never executed.
                return dict(target_id=None, a0=44, a1=44, ticks=1)
            return None

        review = push_checkpoints(world, request, checkpoint, progress_m=threshold,
                                  max_decisions=archive.S.MAX_DECISIONS)
        result = push_phases.push_from_contact(world, request, MAX_TICKS, review=review)
        end = archive.S.observe(world)
        if result.event is None or result.event['end_tick'] != end['tick']:
            raise ValueError('accepted PUSH did not execute')
        try:
            validate_action(continuation, end)
            offered = True
        except ValueError:
            offered = False
        terminal = world.env.termination
        ready = push_phases.contact_state(world, continuation, robot, axis)
        truth = archive.truth(world)
        obj = truth['object']
        dx, dy = (obj[k] - prefix['object'][k] for k in range(2))
        before, after = (o['robots'][robot]['own'] for o in (obs, end))
        robot_progress = sum((after[k] - before[k]) * u for k, u in zip(('x','y'), axis))
        return dict(prefix=prefix, requested_contact_progress_m=threshold, end_tick=end['tick'],
                    elapsed_ticks=result.ticks, runtime_stop=result.status,
                    progress_event=bool(progress_events), terminal=terminal, object=obj,
                    object_forward_m=dx*axis[0]+dy*axis[1], object_displacement_m=math.hypot(dx, dy),
                    pusher_forward_from_boundary_m=robot_progress, object_touching_static=truth['object_static'],
                    same_command_in_next_public_menu=offered,
                    next_contact_state=ready, next_command_available=offered and terminal is None)


def run():
    lock = archive.read(archive.OUT / 'locked.json')
    if archive.sha(archive.OUT / lock['source']) != lock['sha256']:
        raise ValueError('archived policy source mismatch')
    members = archive.dev_rooms()
    rooms = {}
    for room in sorted(members)[:6]:
        path = archive.OUT / 'runs/dev' / lock['name'] / 'full' / (room + '.json')
        record = archive.read(path)
        pair = [branch(members[room], record, d) for d in THRESHOLDS]
        if pair[0] is None and pair[1] is None:
            rooms[room] = dict(status='no_eligible_anchor')
            continue
        if any(v is None for v in pair) or pair[0]['prefix'] != pair[1]['prefix']:
            raise ValueError(f'branch prefixes disagree in {room}')
        prefix = pair[0]['prefix']
        for row in pair:
            del row['prefix']
        rooms[room] = dict(status='paired', content_sha256=archive.content_hash(members[room]),
                           trace_sha256=archive.sha(path), prefix=prefix, short=pair[0], long=pair[1])
    paired = [row for row in rooms.values() if row['status'] == 'paired']
    return dict(protocol='guarded_option_contact_progress_pair_v1', policy_sha256=lock['sha256'],
        selection='First six sorted development rooms. First publicly ready single target PUSH on archived '
                  'outer boundaries. Public history digest, issued controls and physical object pose checked.',
        treatment='Same target, robot codes and 40-tick cap. The existing in-contact public progress callback '
                  'stops at 0.10 or 0.20 m of directed robot travel. Contact loss and normal early ends retained. '
                  'At a progress event BRAKE is proposed only to stop; it is not executed.',
        continuation='Freeze the contact-preserving nominal proposal from the prefix, using its exact target, '
                     'robot codes and one tick. Check endpoint public menu and tactile face readiness separately. '
                     'No subsequent command or full episode is executed.',
        selected=len(rooms), matched=len(paired), no_anchor=len(rooms)-len(paired),
        thresholds_reached={k:sum(r[k]['progress_event'] for r in paired) for k in ('short','long')},
        menu_available={k:sum(r[k]['next_command_available'] for r in paired) for k in ('short','long')},
        contact_ready={k:sum(r[k]['next_contact_state']=='ready' for r in paired) for k in ('short','long')},
        rooms=rooms)


if __name__ == '__main__':
    report = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(OUT)
