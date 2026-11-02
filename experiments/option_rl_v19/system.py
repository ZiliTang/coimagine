"""D005 central joint coordinator + option env.

Options keep their v13 implementations.  The coordinator owns the joint
lifecycle/phase, the live release/joint_active flags, the stage fallback and
the force gate; options only report their own stage/contact state.
"""
from __future__ import annotations
from experiments.option_rl_v13.mode_env import Slot
from experiments.option_rl_v14.mode_env import LearningEnv
from experiments.option_rl_v19 import spec as S


class OptionSlot(Slot):
    """Slot whose skill factory can carry parameter overrides."""

    def __init__(self, factory):
        self.factory = factory
        super().__init__()
        self.action = S.ACTIONS[S.STOP]
        self.skill = factory(self.action)

    def _commit(self):
        self.action = self.pending
        self.pending = None
        self.skill = self.factory(self.action)
        self.brake = None
        self.pause_brake = None
        self.pause_reason = None
        self.sensor_missing_seconds = 0.
        self.events.append('mode_change_committed')


class JointCoordinator:
    def __init__(self):
        self.phase = 'IDLE'
        self.last_phase = 'IDLE'
        self.rows = []
        self.pair_start_tick = self.engage_tick = self.push_tick = None
        self.contact_loss_latch = False
        self.stage_fallback = None
        self.force_gated = None
        self.enabled = True

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def pair_specs(self, slots):
        if any(s.pending is not None for s in slots):
            return None
        specs = [S.SPEC_BY_INDEX[s.action.index] for s in slots]
        if not all('same_family' in sp.joint_requirements for sp in specs):
            return None
        if specs[0].family != specs[1].family:
            return None
        return specs

    def update(self, env, now):
        slots = env.slots
        specs = self.pair_specs(slots) if self.enabled else None
        if specs is None:
            self.phase = 'IDLE'
            self.pair_start_tick = self.engage_tick = self.push_tick = None
            self.contact_loss_latch = False
            self.stage_fallback = None
            self.force_gated = None
            self.last_phase = 'IDLE'
            self._row(env, pairs=False)
            return
        skills = [s.skill for s in slots]
        done = any(s.done for s in skills)
        clearance_active = ((env.clearance.started and not env.clearance.finished)
                            or env.clearance.failed)
        fault = (env.recovery_started is not None or
                 any(s.recovery_requested or s.recovery_ready for s in skills) or
                 clearance_active or
                 any(s.pause_reason for s in slots))
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
        release = both_ready and phase not in ('RECOVER', 'EXIT')
        active = release and both_contact and phase == 'PUSH'
        fallback = None
        for slot, skill in zip(slots, skills):
            skill.release = bool(release)
            skill.joint_active = bool(active)
            if not release and skill.stage == 'push':
                fallback = 'stage_push_to_approach'
                skill.stage = 'approach'
        if both_ready and self.pair_start_tick is None:
            self.pair_start_tick = int(env.total_periods)
        if any_contact and self.engage_tick is None:
            self.engage_tick = int(env.total_periods)
        if both_contact and self.push_tick is None:
            self.push_tick = int(env.total_periods)
        contact_loss = self.last_phase == 'PUSH' and not both_contact
        if contact_loss:
            self.contact_loss_latch = True
        self.phase = phase
        self.last_phase = phase
        self.stage_fallback = fallback
        self.force_gated = None if phase == 'PUSH' else phase.lower() + '_normal_disabled'
        self._row(env, pairs=True, both_ready=both_ready, both_contact=both_contact,
                  release=release, active=active, contact_loss=contact_loss,
                  fallback=fallback, specs=specs)

    def _row(self, env, pairs, both_ready=False, both_contact=False, release=False,
             active=False, contact_loss=False, fallback=None, specs=None):
        self.rows.append(dict(
            tick=int(env.total_periods), phase=self.phase, paired=bool(pairs),
            both_ready=bool(both_ready), both_contact=bool(both_contact),
            release=bool(release), joint_active=bool(active),
            force_enable=[bool(active), bool(active)], stage_fallback=fallback,
            contact_loss=bool(contact_loss), force_gated=self.force_gated,
            pair_start_tick=self.pair_start_tick, engage_tick=self.engage_tick,
            push_tick=self.push_tick,
            families=[] if specs is None else [sp.family for sp in specs]))

    def gate(self, env, forces, apply):
        """Apply the joint force policy; the env supplies the actual Pymunk call."""
        phase = self.phase
        if (not self.enabled) or phase in ('IDLE', 'APPROACH', 'PUSH'):
            return apply(forces)
        if phase in ('RECOVER', 'EXIT'):
            # Recovery/clearance and final braking own the force channel.  Block
            # ordinary PUSH here, but do not zero the separation/recovery/brake
            # force that is supposed to resolve the condition.
            clearance = getattr(env, 'clearance', None)
            clearance_force = bool(clearance is not None and clearance.started
                                   and not clearance.finished)
            recovery_force = (env.recovery_started is not None
                              or any(getattr(s, 'recovery_ready', False)
                                     for s in env.slots for s in [s.skill]))
            if phase == 'EXIT' or clearance_force or recovery_force:
                self.force_gated = phase.lower() + '_recovery_force_enabled'
                return apply(forces)
            self.force_gated = phase.lower() + '_zero'
            return apply(((0., 0.), (0., 0.)))
        return apply(forces)


class OptionSystemEnv(LearningEnv):
    def __init__(self, instance, *, context='extended', overrides=None):
        super().__init__(instance, context=context)
        self.actions = S.ACTIONS
        self.action_version = S.VERSION
        self.action_codebook_hash = S.CODEBOOK_HASH
        self.overrides = overrides or {}
        self.factory = lambda action: S.make_skill(action, self.overrides)
        self.slots = [OptionSlot(self.factory), OptionSlot(self.factory)]
        self.coordinator = JointCoordinator()
        self._orig_step_control = self.env.step_control
        self.env.step_control = self._gated_step_control

    def joint_step(self, a0, a1, record=False):
        return super().joint_step(S.canonical_index(a0), S.canonical_index(a1), record=record)

    def action_masks(self):
        return tuple(S.mask(o) for o in self.boundary_observations())

    def set_gate_enabled(self, enabled):
        self.coordinator.set_enabled(enabled)

    def _gated_step_control(self, forces, headings, **kwargs):
        self.gate_pre = [[float(forces[0][0]), float(forces[0][1])],
                         [float(forces[1][0]), float(forces[1][1])]]
        def _apply(gated):
            self.gate_post = [[float(gated[0][0]), float(gated[0][1])],
                              [float(gated[1][0]), float(gated[1][1])]]
            return self._orig_step_control(gated, headings, **kwargs)
        return self.coordinator.gate(self, forces, _apply)

    def _coordinate(self, now, observations=None):
        super()._coordinate(now, observations)
        self.coordinator.update(self, now)
