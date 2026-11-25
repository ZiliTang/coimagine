"""Target-free eight-direction velocity Options; versioned outside frozen V2."""
from dataclasses import asdict, replace
import hashlib
import json
import math
from functools import lru_cache

from coimagine.random_start import spec as R
from coimagine.random_start.skills import Skill, HoldBrake
from experiments.option_rl_v21 import spec as V2
from experiments.option_rl_v21.system import YieldSlot

LEGACY_VERSION = 'relational-options-move-v2'
VERSION = ('relational-options-local-push22.5-v3' if R.GEOMETRY_PROFILE == 'legacy36'
           else 'relational-options-local-push45-compact20-v4')
DIRECTIONS = ('E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE')
SPEEDS = (.10, .20, .30, .40)
MOVE_ACTIONS = tuple(
    R.RobotAction(kind='move', direction=direction, speed=speed, scale=speed,
                  scale_kind='persistent', type_index=15+d, scale_index=s,
                  index=len(V2.ACTIONS)+d*len(SPEEDS)+s,
                  label=f'MOVE_{direction}[v={speed:.2f}m/s]')
    for d, direction in enumerate(DIRECTIONS) for s, speed in enumerate(SPEEDS))
ACTIONS = V2.ACTIONS + MOVE_ACTIONS
ACTIVE_ACTIONS = V2.ACTIVE_ACTIONS + MOVE_ACTIONS
MOVE_IDS = frozenset(a.index for a in MOVE_ACTIONS)
CODEBOOK_HASH = hashlib.sha256(json.dumps([asdict(a) for a in ACTIONS], sort_keys=True).encode()).hexdigest()
SPEC_BY_INDEX = dict(V2.SPEC_BY_INDEX)
for action in MOVE_ACTIONS:
    SPEC_BY_INDEX[action.index] = replace(V2.SPEC_BY_INDEX[V2.STOP],
        key=f'MOVE:{action.index}', family='MOVE', action_index=action.index,
        label=action.label, action=action, params=(), observations=('own', 'scan', 'arena'),
        states=('reference',), stages=('move', 'move_blocked'), joint_requirements=(), source=__file__)


def canonical_index(index):
    if type(index) is not int or not 0 <= index < len(ACTIONS):
        raise ValueError('relational action index out of range')
    return V2.canonical_index(index) if index < len(V2.ACTIONS) else index


def move_id(direction, speed=.30):
    if direction not in DIRECTIONS or type(speed) not in (int, float) or speed not in SPEEDS:
        raise ValueError('MOVE requires an eight-way direction and speed in .10/.20/.30/.40 m/s')
    return len(V2.ACTIONS)+DIRECTIONS.index(direction)*len(SPEEDS)+SPEEDS.index(speed)


def global_pairs():
    # Return a fresh list; the cached protocol template contains immutable pairs.
    return list(_global_pair_template())


@lru_cache(maxsize=1)
def _global_pair_template():
    """Single-robot and aligned moves first; every independent combination follows."""
    order = [move_id(d, speed) for speed in (.30, .10, .20, .40) for d in DIRECTIONS]
    pairs = [(a, V2.STOP) for a in order] + [(V2.STOP, a) for a in order]
    pairs += [(a, a) for a in order] + [(V2.STOP, V2.STOP)]
    ids = [V2.STOP] + order
    pairs += [(a, b) for a in ids for b in ids]
    return tuple(dict.fromkeys(pairs))


def is_global_pair(a0, a1):
    return all(a == V2.STOP or a in MOVE_IDS for a in (a0, a1))


class MoveSkill(Skill):
    """World-frame velocity tracking with scan-based slowing, no target query.

    The corridor guard treats every measured surface as occupied. It is a local
    stopping heuristic, not collision-free navigation or an information reward.
    """
    kind = 'move'

    def __init__(self, action):
        super().__init__(action)
        self.time_budget = None
        self.stage = 'move'

    def step(self, observation, dt):
        self.skill_time += dt
        self._note_position(observation)
        angle = DIRECTIONS.index(self.action.direction)*math.pi/4
        direction = (math.cos(angle), math.sin(angle))
        clearance = R.LIDAR_RANGE - R.ROBOT_RADIUS - R.SAFETY_GAP
        radius = R.ROBOT_RADIUS + R.SAFETY_GAP
        for angle, distance, valid in zip(observation.scan.ray_angles,
                                         observation.scan.distances, observation.scan.valid):
            if not valid:
                continue
            x, y = distance*math.cos(angle), distance*math.sin(angle)
            forward = x*direction[0]+y*direction[1]
            lateral = abs(x*direction[1]-y*direction[0])
            if forward > 0 and lateral <= radius:
                clearance = min(clearance, forward-math.sqrt(max(0., radius**2-lateral**2)))
        # Public arena bounds supplement rays; no hidden obstacle geometry.
        for position, extent, component in zip((observation.own.x, observation.own.y),
                                               observation.arena, direction):
            if abs(component) > 1e-9:
                room = extent-radius-position if component > 0 else position-radius
                clearance = min(clearance, room/abs(component))
        # Allow one control period plus a fixed measured-gap margin for braking.
        speed_limit = math.sqrt(2*R.A_REF_MAX*max(0., clearance-.03-observation.own.speed*dt))
        speed = min(self.action.speed, speed_limit)
        raw = (speed*direction[0], speed*direction[1])
        reference = self.reference.update(raw, dt)
        force, saturated = self._force_from_reference(reference, observation)
        self.stage = 'move_blocked' if speed <= 1e-9 else 'move'
        return force, dict(skill=self.action.label, kind='move', stage=self.stage,
            done=False, exit_reason=None, desired_velocity=raw, reference_velocity=reference,
            force=force, force_saturated=saturated, direction_valid=True,
            direction_unavailable=speed <= 1e-9, desired_heading=direction,
            measured_corridor_clearance=clearance, commanded_speed=self.action.speed)


def make_skill(action, overrides=None, *, estimator=None):
    return MoveSkill(action) if action.kind == 'move' else V2.make_skill(action, overrides, estimator=estimator)


def mask(observation):
    import numpy as np
    return np.concatenate((V2.mask(observation), np.ones(len(MOVE_ACTIONS), dtype=bool)))


class MotionSlot(YieldSlot):
    def step(self, obs):
        # Perception/peer pauses in the legacy slot can bypass the push controller.
        if (self.pending is None and getattr(self.action, 'effect_kind', None) == 'translate'
                and not self.skill.done and (obs.patch is None or self.pause_reason is not None)):
            from experiments.option_rl_v19.unknown_shape import local_push, compatible_faces
            if local_push(obs) and not compatible_faces(obs, self.action):
                self.skill.stop('push_no_legal_face')
        return super().step(obs)

    def select(self, action):
        if action.kind != 'move' and self.action.kind != 'move' and not (
                self.pending is not None and self.pending.kind == 'move'):
            return super().select(action)
        if self.pending is not None:
            self.pending = action
            return
        if self.action.kind == action.kind == 'move' and not self.skill.done:
            # Direction/speed changes share the same acceleration-limited servo.
            self.action = action
            self.skill.action = action
            return
        previous = (self.brake or self.pause_brake or self.skill).reference.value
        self.pending = action
        self.brake = HoldBrake(R.MAX_FORCE)
        self.brake.reference.value = previous
        self.pause_reason = None
        self.pause_brake = None
        self.events.append('mode_change_requested')
