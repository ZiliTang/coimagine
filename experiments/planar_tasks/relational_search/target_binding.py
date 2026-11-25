"""Observation-only target handoff. No body changes or hidden controller reads.

This state machine emits explicit brake actions before authorizing a new target.
The Option adapter resets target-specific allocation/estimation on commit=True.
Target-free MOVE/BRAKE cancels pending handoffs without changing ownership.
"""
from copy import deepcopy
import math
from .contract import validate_action
from .joint_orbit import resolve_pair


class TargetBinding:
    def __init__(self, *, joint_orbits=True):
        self.joint_orbits = joint_orbits
        self.current = None
        self.pending = None
        self.slow_samples = 0
        self.last_tick = None

    def select(self, requested, observation):
        validate_action(requested, observation)
        if observation['tick'] == self.last_tick:
            raise ValueError('one binding update per boundary')
        if self.last_tick is not None and observation['tick'] < self.last_tick:
            raise ValueError('binding clock moved backwards')
        self.last_tick = observation['tick']
        if requested['target_id'] is None:
            self.pending = None
            self.slow_samples = 0
            phase = 'brake' if requested['a0'] == requested['a1'] == 44 else 'move'
            return dict(command=deepcopy(requested), commit=False, target=self.current, phase=phase)
        target = requested['target_id']
        if target == self.current:
            self.pending = None
            self.slow_samples = 0
            return self._bound(requested, observation, target, 'continue')
        if self.pending != target:
            self.pending = target
            self.slow_samples = 0
        robots = observation['robots']
        if len(robots) != 2:
            raise ValueError('two robot feedback packets required')
        velocities = [(r['own']['vx'], r['own']['vy']) for r in robots]
        if not all(math.isfinite(v) for pair in velocities for v in pair):
            raise ValueError('finite odometry velocities required')
        references=[r.get('execution',{}).get('reference_velocity') for r in robots]
        if any(v is not None and (len(v)!=2 or not all(math.isfinite(x) for x in v)) for v in references):
            raise ValueError('finite public reference velocities required')
        slow = (all(math.hypot(*v) <= .025 for v in velocities) and
                all(v is None or math.hypot(*v)<=1e-6 for v in references))
        self.slow_samples = self.slow_samples + 1 if slow else 0
        if self.slow_samples >= 2:
            self.current = target
            self.pending = None
            self.slow_samples = 0
            return self._bound(requested, observation, target, 'commit')
        return dict(command=dict(target_id=None, a0=44, a1=44, ticks=requested['ticks']),
                    commit=False, target=self.current, phase='switch_brake')

    def _bound(self, requested, observation, target, phase):
        command, decision = resolve_pair(requested, observation) if self.joint_orbits else (requested, None)
        result = dict(command=deepcopy(command), commit=phase=='commit', target=target, phase=phase)
        if decision is not None:
            result['orbit_resolution'] = decision
        return result
