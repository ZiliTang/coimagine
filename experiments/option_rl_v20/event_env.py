"""Progress-triggered joint decisions over the current v19 controller.

Progress uses only own odometry, contact sensor and commanded direction; it
is robot travel, not privileged object displacement. Rewards remain those of
the underlying environment and are accumulated over actual control ticks.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from coimagine.random_start import spec as R
from experiments.option_rl_v19 import spec as S
from experiments.option_rl_v19.system import OptionSystemEnv
from experiments.option_rl_v19.unknown_shape import UnknownShapeEnvMixin

VERSION = 'event-progress-v1'
REASONS = ('start', 'contact', 'progress', 'contact_lost', 'failure',
           'acquisition_timeout', 'progress_timeout', 'wait', 'terminal')
OBS_EXTRA = ('elapsed_s', 'progress_0', 'progress_1', 'contact_0', 'contact_1',
             *(f'event_{reason}' for reason in REASONS))


def family(action):
    if action.kind == 'push' and getattr(action, 'effect_kind', None) == 'translate':
        return 'push'
    if action.kind == 'reposition':
        return 'reposition'
    if action.kind == 'wait':
        return 'wait'
    return 'excluded'


def legal_pair(a, b):
    x, y = family(a), family(b)
    return x != 'excluded' and y != 'excluded' and (x == y or 'wait' in (x, y))


def action_masks(observations):
    result = []
    for obs in observations:
        mask = S.mask(obs)
        for a in S.ACTIONS:
            if family(a) == 'excluded':
                mask[a.index] = False
        result.append(mask)
    return tuple(result)


@dataclass(frozen=True)
class EventConfig:
    distance_m: float = .20
    contact_hold_s: float = .15
    contact_loss_s: float = .25
    acquisition_timeout_s: float = 12.
    progress_timeout_s: float = 8.
    wait_timeout_s: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (
                self.distance_m, self.contact_hold_s, self.contact_loss_s,
                self.acquisition_timeout_s, self.progress_timeout_s, self.wait_timeout_s)):
            raise ValueError('event thresholds must be positive and finite')


class EventEnv:
    def __init__(self, instance, config=EventConfig()):
        self.config = config
        self.env = OptionSystemEnv(instance)
        UnknownShapeEnvMixin.install_prior_free(self.env)
        # Underlying joint_step then advances exactly one physical control tick.
        self.env.decision_ticks = 1
        self.signature = None
        self.contact_emitted = False
        self.contact_lost_emitted = False
        self.contact_ticks = 0
        self.loss_ticks = 0
        self.last_event = dict(reason='start', elapsed_s=0., progress=[0., 0.],
                               contact=[False, False])

    def observe(self):
        obs = self.env.boundary_observations()
        base = np.asarray(self.env.observation_vector(*obs), dtype=np.float32)
        e = self.last_event
        extra = np.asarray([e['elapsed_s']/10., *(p/self.config.distance_m for p in e['progress']),
                            *(float(x) for x in e['contact']),
                            *(float(e['reason'] == reason) for reason in REASONS)], dtype=np.float32)
        return np.concatenate((base, extra)), *action_masks(obs)

    def step(self, a0, a1, *, record=False):
        ids = (int(a0), int(a1))
        if any(i < 0 or i >= len(S.ACTIONS) or i in S.ACTION_ALIASES for i in ids):
            raise ValueError('invalid or legacy action')
        actions = tuple(S.ACTIONS[i] for i in ids)
        if not legal_pair(*actions):
            raise ValueError('illegal joint action')
        kinds = tuple(family(a) for a in actions)
        signature = tuple(i for i, k in enumerate(kinds) if k == 'push')
        if signature != self.signature:
            self.signature = signature
            self.contact_emitted = False
            self.contact_lost_emitted = False
            self.contact_ticks = self.loss_ticks = 0
        before = self.env.boundary_observations()
        start_tick = self.env.total_periods
        start_skills = self.env.skills_used
        progress = [0., 0.]
        total = 0.
        components = {k: 0. for k in ('progress', 'distance_cost', 'time_cost', 'success_bonus', 'failure_penalty')}
        contact = [bool(o.contacting_object) for o in before]
        failed_at_start = {id(s.skill) for s in self.env.slots if s.skill.done}
        reason = None
        while reason is None:
            old = before
            old_skills = [s.skill for s in self.env.slots]
            old_signed = [getattr(s, 'signed_progress', 0.) for s in old_skills]
            boundary = self.env.joint_step(*ids, record=record)
            # A call is one physical tick; preserve the public actor-decision count.
            self.env.skills_used = start_skills
            if boundary['periods'] != 1:
                raise RuntimeError('unexpected zero or multi-tick transition')
            total += boundary['reward']['total']
            for key in components:
                components[key] += boundary['reward'][key]
            before = self.env.boundary_observations()
            contact = [bool(o.contacting_object) for o in before]
            for i, kind in enumerate(kinds):
                skill = self.env.slots[i].skill
                committed = self.env.slots[i].pending is None and self.env.slots[i].action.index == ids[i]
                if not committed:
                    continue
                if kind == 'push' and old[i].contacting_object and before[i].contacting_object:
                    d = actions[i].effect_direction
                    dx = before[i].own.x-old[i].own.x
                    dy = before[i].own.y-old[i].own.y
                    progress[i] += dx*d[0]+dy*d[1]
                elif kind == 'reposition' and skill is old_skills[i]:
                    progress[i] += getattr(skill, 'signed_progress', 0.)-old_signed[i]
            active = [i for i, k in enumerate(kinds) if k == 'push']
            all_contact = bool(active) and all(contact[i] and self.env.slots[i].pending is None
                                                and self.env.slots[i].action.index == ids[i] for i in active)
            self.contact_ticks = self.contact_ticks+1 if all_contact else 0
            self.loss_ticks = 0 if all_contact else self.loss_ticks+1
            elapsed = (self.env.total_periods-start_tick)*R.CONTROL_DT
            new_failure = any(s.skill.done and id(s.skill) not in failed_at_start for s in self.env.slots
                              if family(s.action) != 'wait' and s.pending is None)
            if boundary['termination'] is not None:
                reason = 'terminal'
            elif new_failure:
                reason = 'failure'
            elif active and not self.contact_emitted and self.contact_ticks*R.CONTROL_DT >= self.config.contact_hold_s-1e-9:
                self.contact_emitted = True
                self.contact_lost_emitted = False
                progress = [0., 0.]  # zero is the stable contact event
                reason = 'contact'
            elif active and self.contact_emitted and not self.contact_lost_emitted and self.loss_ticks*R.CONTROL_DT >= self.config.contact_loss_s-1e-9:
                self.contact_lost_emitted = True
                reason = 'contact_lost'
            elif max(progress) >= self.config.distance_m-1e-9 and (not active or self.contact_emitted):
                reason = 'progress'
            elif kinds == ('wait', 'wait') and elapsed >= self.config.wait_timeout_s-1e-9:
                reason = 'wait'
            elif active and not self.contact_emitted and elapsed >= self.config.acquisition_timeout_s-1e-9:
                reason = 'acquisition_timeout'
            elif (not active or self.contact_emitted) and elapsed >= self.config.progress_timeout_s-1e-9:
                reason = 'progress_timeout'
            if self.contact_lost_emitted and all_contact and self.contact_ticks*R.CONTROL_DT >= self.config.contact_hold_s-1e-9:
                self.contact_lost_emitted = False
        self.env.skills_used = start_skills+1
        self.last_event = dict(reason=reason, elapsed_s=elapsed, progress=progress, contact=contact)
        return dict(**self.last_event, periods=self.env.total_periods-start_tick,
                    start_tick=start_tick, end_tick=self.env.total_periods,
                    actions=list(ids), reward=total, components=components,
                    termination=boundary['termination'], success=boundary['success'],
                    skill_ends=boundary['skill_ends'])
