"""V2 environment: explicit YIELD lifecycle and observable execution feedback."""
import numpy as np
from experiments.option_rl_v13.mode_env import ContinuousModeEnv
from experiments.option_rl_v19.system import OptionSystemEnv as OldEnv, OptionSlot, JointCoordinator
from experiments.option_rl_v21 import spec as S
from experiments.option_rl_v19.unknown_shape import UnknownShapeEnvMixin

YIELD_FIELDS = ('requested', 'executing', 'active', 'blocked', 'gap', 'direction_unavailable')


class YieldSlot(OptionSlot):
    def select(self, action):
        if action.kind == 'yield' and (self.pending is not None or self.action.kind != 'yield'):
            previous = (self.brake or self.pause_brake or self.skill).reference.value
            self.pending = action
            self._commit()
            # Preserve acceleration continuity, but do not wait for a pushed
            # robot to reach zero speed before it is allowed to escape.
            self.skill.reference.value = previous
            self.events.append('yield_immediate_commit')
        else:
            super().select(action)


class OptionSystemEnv(OldEnv):
    def __init__(self, instance, *, overrides=None):
        super().__init__(instance, overrides=overrides)
        self.option_spec = S
        self.actions = S.ACTIONS
        self.action_version = S.VERSION
        self.action_codebook_hash = S.CODEBOOK_HASH
        self.factory = lambda action: S.make_skill(action, self.overrides)
        self.slots = [YieldSlot(self.factory), YieldSlot(self.factory)]
        self.coordinator = JointCoordinator(action_spec=S)
        UnknownShapeEnvMixin.install_prior_free(self)

    def joint_step(self, a0, a1, record=False):
        return ContinuousModeEnv.joint_step(self, S.canonical_index(a0), S.canonical_index(a1), record=record)

    def action_masks(self):
        return tuple(S.mask(o) for o in self.boundary_observations())

    def observation_vector(self, *args, **kwargs):
        base = super().observation_vector(*args, **kwargs)
        values = []
        for slot in self.slots:
            info = slot.last_info; debug = info.get('yield_debug') or {}
            gap = debug.get('min_gap')
            values.extend((float((slot.pending or slot.action).kind == 'yield'),
                           float(slot.pending is None and slot.action.kind == 'yield'),
                           float(debug.get('active', False)), float(debug.get('blocked', False)),
                           float(np.clip(gap, -1., 5.)) if gap is not None else 0.,
                           float(info.get('direction_unavailable', False))))
        return np.concatenate((base, np.asarray(values, np.float32)))

    def observation_fields(self):
        return super().observation_fields()+[f'r{i}_yield_{f}' for i in (0, 1) for f in YIELD_FIELDS]
