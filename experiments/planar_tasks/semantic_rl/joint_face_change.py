"""Joint CCW change of contact face; the execution layer owns assignment through completion."""
from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.relational_search.joint_orbit import face_owner
from .memory_flow import current_event
from . import push_phases

ORBIT = 40
FOLLOW_BUDGET = 120

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


def next_step(world, robot, follow, orbit, do_orbit, max_decisions):
    """Select by the actual current public menu, tactile contact, and program phase."""
    if world.env.termination or world.decision_count >= max_decisions:
        return 'terminated'
    obs = current_event(world.history)['observation']
    _, axis = push_phases.pusher(follow)
    if push_phases.contact_state(world, follow, robot, axis) == 'ready':
        return 'ready'
    if not do_orbit and legal(follow, obs):
        return 'approach'
    if legal(orbit, obs):
        return 'orbit'
    return 'unavailable'


def change_face_ccw(world, action, *, execute=True, max_decisions=300):
    robot, _ = push_phases.pusher(action)
    follow = new_face(action, robot)
    orbit = changed_action(action, robot, ORBIT, 12)
    original = robot
    if not world.env.termination and world.decision_count < max_decisions:
        obs = current_event(world.history)['observation']
        _, normal = push_phases.pusher(follow)
        if push_phases.contact_state(world, follow, robot, normal) != 'ready':
            robot = face_owner(orbit, original, obs)
            if robot != original:
                follow = dict(follow, a0=follow['a1'], a1=follow['a0'])
                orbit = dict(orbit, a0=orbit['a1'], a1=orbit['a0'])
    start = world.env.total_periods
    orbit_next = True
    counts = {'orbit_ticks': 0, 'approach_ticks': 0}
    step = next_step(world, robot, follow, orbit, orbit_next, max_decisions)
    available = step in ('ready', 'approach', 'orbit')
    if not execute:
        return {'endpoint_followup_available': available, 'followup_ticks': 0}
    while step in ('approach', 'orbit'):
        elapsed = world.env.total_periods - start
        budget = FOLLOW_BUDGET - elapsed
        if budget <= 0:
            step = 'budget'
            break
        if step == 'orbit':
            event = world.step(orbit, remaining_ticks=min(budget, orbit['ticks']))
            ticks = event['periods']
            orbit_next = False
        else:
            result = push_phases.contact_on_new_face(world, follow, min(budget, follow['ticks']))
            ticks = result.ticks
            orbit_next = result.status != 'contact_timeout'
        counts[step + '_ticks'] += ticks
        if ticks == 0:
            raise ValueError('non-progressing follow-up command')
        step = next_step(world, robot, follow, orbit, orbit_next, max_decisions)
    return dict(endpoint_followup_available=available, followup_completed=step == 'ready',
                followup_reason=step, requested_pusher=original, contact_robot=robot, followup_ticks=world.env.total_periods-start, **counts)
