"""D004 pair gate: live pair phase + force/stage invariants on top of v14.

Only the coordinator flags and push-stage fallback are changed.  Approach,
ROTATE, reward, observation and action tables are untouched.  The v13/v14
sources are not edited.
"""
from __future__ import annotations
import math
from types import SimpleNamespace
from experiments.option_rl_v14.mode_env import LearningEnv
from experiments.option_rl_v13.push_controller import ContinuousPush
from experiments.option_rl_v18.state_schema import PAIR_PHASES, INVARIANTS


class PairGateEnv(LearningEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_phase = 'IDLE'
        self.gate_rows = []
        self.pair_start_tick = None
        self.engage_tick = None
        self.push_tick = None
        self.contact_loss_latch = False
        self.stage_fallback_reason = None
        self.force_gated = None
        self.one_side_normal_count = 0
        self.gate_last_phase = 'IDLE'
        self.gate_enabled = True

    def set_gate_enabled(self, enabled):
        self.gate_enabled = bool(enabled)

    def _gated_step_control(self, forces, headings, **kwargs):
        if (not self.gate_enabled) or self.pair_phase in ('IDLE', 'APPROACH', 'PUSH'):
            return self._orig_step_control(forces, headings, **kwargs)
        if self.pair_phase in ('RECOVER', 'EXIT'):
            self.force_gated = self.pair_phase.lower() + '_zero'
            return self._orig_step_control(((0., 0.), (0., 0.)), headings, **kwargs)
        # READY/ENGAGE: preserve approach/tangential motion but equalize the
        # normal component of both sides at the smaller value.  A unilateral
        # full normal force is therefore impossible; the joint-engage contact
        # acquisition remains symmetric.
        if self.pair_phase in ('READY', 'ENGAGE'):
            normals = []
            for slot in self.slots:
                face = getattr(slot.skill, 'face', None)
                normals.append(None if face is None else tuple(face.normal))
            if all(n is not None for n in normals):
                f0 = (float(forces[0][0]), float(forces[0][1]))
                f1 = (float(forces[1][0]), float(forces[1][1]))
                n0, n1 = normals
                a0 = f0[0] * n0[0] + f0[1] * n0[1]
                a1 = f1[0] * n1[0] + f1[1] * n1[1]
                m = min(a0, a1)
                f0 = (f0[0] + (m - a0) * n0[0], f0[1] + (m - a0) * n0[1])
                f1 = (f1[0] + (m - a1) * n1[0], f1[1] + (m - a1) * n1[1])
                self.force_gated = f'{self.pair_phase.lower()}_normal_equalized'
                if self.gate_rows:
                    self.gate_rows[-1]['final_forces'] = [[f0[0], f0[1]], [f1[0], f1[1]]]
                    self.gate_rows[-1]['force_gated'] = self.force_gated
                return self._orig_step_control((f0, f1), headings, **kwargs)
        return self._orig_step_control(forces, headings, **kwargs)

    def _coordinate(self, now, observations=None):
        super()._coordinate(now, observations)
        if self.gate_enabled:
            self._apply_pair_gate(now)

    def _apply_pair_gate(self, now):
        slots = self.slots
        skills = [s.skill for s in slots]
        paired = (all(isinstance(s, ContinuousPush) for s in skills)
                  and all(s.pending is None for s in slots))
        if not paired:
            self.pair_phase = 'IDLE'
            self.pair_start_tick = self.engage_tick = self.push_tick = None
            self.contact_loss_latch = False
            self.stage_fallback_reason = None
            self.gate_last_phase = 'IDLE'
            self.gate_rows.append(dict(tick=int(self.total_periods), phase='IDLE', paired=False,
                                       both_ready=False, both_contact=False,
                                       release=False, joint_active=False,
                                       force_enable=[False, False], stage_fallback=None,
                                       contact_loss=False))
            return
        done = any(s.done for s in skills)
        # PeerClearancePrefix.finished is a completion latch, not an active
        # fault.  Only an in-progress clearance (started and not finished) or
        # an explicit failure may gate the pair.
        clearance_active = ((self.clearance.started and not self.clearance.finished)
                            or self.clearance.failed)
        fault = (self.recovery_started is not None or
                 any(s.recovery_requested or s.recovery_ready for s in skills) or
                 clearance_active or
                 any(slot.pause_reason for slot in slots))
        both_ready = all(s.ready and not s.done for s in skills)
        both_contact = all(s.contact_ready and not s.done for s in skills)
        any_contact = any(s.contact_ready and not s.done for s in skills)
        if done:
            phase = 'EXIT'
        elif fault:
            phase = 'RECOVER'
        elif not both_ready:
            phase = 'APPROACH'
        elif not any_contact:
            phase = 'READY'
        elif not both_contact:
            phase = 'ENGAGE'
        else:
            phase = 'PUSH'
        allow_release = both_ready and phase not in ('RECOVER', 'EXIT')
        allow_active = allow_release and both_contact and phase == 'PUSH'
        fallback = None
        for slot, skill in zip(slots, skills):
            # live flags: override the one-way latches in ContinuousModeEnv
            skill.release = bool(allow_release)
            skill.joint_active = bool(allow_active)
            if not allow_release and skill.stage == 'push':
                # Do not erase the per-side ready flag: it is the side's
                # approach-complete state and must persist while the peer
                # catches up.  Only the unilateral push stage is reverted.
                fallback = 'stage_push_to_approach'
                skill.stage = 'approach'
            # A push-stage skill in READY/ENGAGE may keep the joint-engage
            # contact acquisition, but cannot emit a normal push because
            # joint_active is false.  The raw P3 force is checked downstream.
        if both_ready and self.pair_start_tick is None:
            self.pair_start_tick = int(self.total_periods)
        if any_contact and self.engage_tick is None:
            self.engage_tick = int(self.total_periods)
        if both_contact and self.push_tick is None:
            self.push_tick = int(self.total_periods)
        contact_loss = self.gate_last_phase == 'PUSH' and not both_contact
        if contact_loss:
            self.contact_loss_latch = True
        self.stage_fallback_reason = fallback
        self.force_gated = None if phase == 'PUSH' else phase.lower() + '_normal_disabled'
        self.pair_phase = phase
        self.gate_last_phase = phase
        self.gate_rows.append(dict(
            tick=int(self.total_periods), phase=phase, paired=True,
            both_ready=bool(both_ready), both_contact=bool(both_contact),
            release=bool(allow_release), joint_active=bool(allow_active),
            force_enable=[bool(allow_active), bool(allow_active)],
            stage_fallback=fallback, contact_loss=bool(contact_loss),
            both_active=bool(allow_active), force_gated=self.force_gated,
            pair_start_tick=self.pair_start_tick, engage_tick=self.engage_tick,
            push_tick=self.push_tick))


def make_pair_env(instance, context='extended'):
    from experiments.option_rl_v14.scenes import geometry_audit
    if not geometry_audit(instance)['legal']:
        raise ValueError('illegal initial geometry')
    return PairGateEnv(instance, context=context)
