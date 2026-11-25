"""Resolve local orbit conflicts using public, target-labelled rays and odometry."""
import math

from coimagine.random_start.spec import CONTROL_DT, ROBOT_RADIUS
from .motion import ACTIONS

ORBITS = (40, 43)
WAIT, YIELD = 44, 46
# Two discs plus the executor's ordinary swept-path clearance.
MIN_CENTRE_DISTANCE = 2 * ROBOT_RADIUS + .04


def _velocity(robot, code, target):
    own = robot['own']
    if code not in ORBITS:
        return own['vx'], own['vy']
    scan = robot['scan']
    hits = [(r, theta) for r, theta, ok, label in zip(
        scan['distances'], scan['ray_angles'], scan['valid'], scan.get('entity_ids', ()))
        if ok and label == target]
    if not hits:
        return None
    # The nearest return points inwards. Rotate its opposite to get the orbit tangent.
    _, angle = min(hits)
    speed = ACTIONS[code].speed * (1 if code == 40 else -1)
    return speed * math.sin(angle), -speed * math.cos(angle)


def closest_approach(command, observation):
    pair = command['a0'], command['a1']
    robots = observation['robots']
    velocity = [_velocity(r, c, command['target_id']) for r, c in zip(robots, pair)]
    if None in velocity:
        return None
    p = [robots[1]['own'][k] - robots[0]['own'][k] for k in ('x','y')]
    v = [b-a for a,b in zip(*velocity)]
    closing = -sum(x*y for x,y in zip(p,v))
    if closing <= 0:
        return None
    horizon = min(1., command['ticks'] * CONTROL_DT)
    t = min(horizon, closing / sum(x*x for x in v))
    return math.hypot(p[0]+t*v[0], p[1]+t*v[1])


def offered(command, observation):
    menu = observation.get('entity_legal_pairs', {}).get(command['target_id'], ())
    return [command['a0'], command['a1']] in menu


def resolve_pair(command, observation):
    """Return actual codes for this command window. A WAIT slot stays stationary."""
    pair = command['a0'], command['a1']
    if not any(c in ORBITS for c in pair):
        return command, None
    if not all(c in (*ORBITS, WAIT, YIELD) for c in pair):
        raise ValueError('orbit requires an orbit, YIELD or WAIT partner')
    if WAIT in pair:
        return command, None  # Named robot follows its existing outer-arc controller.
    if YIELD in pair and observation['robots'][pair.index(YIELD)]['execution'].get('stage') != 'yield_blocked':
        return command, None
    gap = closest_approach(command, observation)
    if gap is None or gap >= MIN_CENTRE_DISTANCE:
        return command, None
    if YIELD in pair:
        mover = 1 - pair.index(YIELD)
        alternate = dict(command, **{f'a{mover}': WAIT})
        reason = 'hold_for_blocked_yield'
    else:
        # Keep one circling while the other uses the existing clearance controller.
        # Fixed priority makes repeated identical requests stable on ties.
        alternate = dict(command, a1=YIELD)
        reason = 'stagger_dual_orbit'
    if not offered(alternate, observation):
        return command, None
    return alternate, dict(reason=reason, predicted_centre_distance_m=gap)


def face_owner(orbit, original, observation):
    """An interchangeable face goal can be reassigned once, before orbiting begins."""
    partner = 1-original
    assistant = orbit[f'a{partner}']
    if assistant == YIELD and observation['robots'][partner]['execution'].get('stage') != 'yield_blocked':
        return original
    if assistant not in (WAIT, YIELD):
        return original
    gap = closest_approach(orbit, observation)
    if gap is None or gap >= MIN_CENTRE_DISTANCE:
        return original
    swapped = dict(orbit, a0=orbit['a1'], a1=orbit['a0'])
    if not offered(swapped, observation):
        return original
    other_gap = closest_approach(swapped, observation)
    return partner if other_gap is None or other_gap >= MIN_CENTRE_DISTANCE else original
