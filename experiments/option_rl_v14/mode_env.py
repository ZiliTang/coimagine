"""Legal controller feedback and dense observations, with unchanged v13 physics.

Sampling a mode once then holding it produces ONE actor transition. The 5 Hz
observations inside that transition are retained for sequence modelling only.
"""
import numpy as np
from experiments.option_rl_v13.mode_env import ContinuousModeEnv, CONTEXT_DIM, CONTEXT_PER_ROBOT

PHASES=('find','approach','ready','push','align','move','follow','brake','blocked',
        'switching','peer_separate','recover','peer_wait','perception_wait')
FAILURES=('effect_face_lost','patch_lost','endpoint_unobservable','anchor_unobservable',
          'prepare_stalled','engagement_stalled','translation_stalled','rotation_stalled',
          'blocked_no_progress','blocked_static_route','blocked_peer_clearance',
          'joint_recovery_timeout','safety_abort')
EXTRA_FIELDS=([f'executing_type_{i}' for i in range(14)]+[f'phase_{p}' for p in PHASES]
    +[f'failure_{r}' for r in FAILURES]+['failure_other','peer_wait','perception_wait',
      'sensor_missing_fraction','guard_elapsed_fraction','guard_valid','geometry_age',
      'geometry_valid','endpoint_observed_fraction','endpoint_valid','tangent_error',
      'gap','gap_valid','ready','contact_ready','joint_active','recovery_requested',
      'target_dx','target_dy','target_valid'])
EXTRA_PER_ROBOT=len(EXTRA_FIELDS)


def finite(value, scale=1.):
    return float(np.clip(value/scale,-5.,5.)) if value is not None and np.isfinite(value) else 0.


class LearningEnv(ContinuousModeEnv):
    def __init__(self, instance, *, context='extended'):
        if context not in ('legacy','extended'):raise ValueError('unknown context schema')
        self.context=context
        super().__init__(instance)

    def feedback_vector(self):
        values=[]
        for slot in self.slots:
            s=slot.skill;info=slot.last_info
            effect=info.get('effect') or {};endpoint=effect.get('endpoint') or {}
            phase=('peer_separate' if self.clearance.started and not self.clearance.finished else
                   'switching' if slot.pending else 'blocked' if s.done else
                   'perception_wait' if slot.pause_reason=='perception_wait' else
                   'peer_wait' if slot.pause_reason else s.stage)
            reason=s.exit_reason if s.done else None
            guard=getattr(s,'phase_progress',None);g=guard.info if guard is not None else None
            target=effect.get('contact_target');own=info.get('own')
            # Contact target is computed from this robot's own scan. Subtract its
            # own odometry; never use the object's simulator pose or velocity.
            robot_id=len(values)//EXTRA_PER_ROBOT
            state=self.robot_states()[robot_id]
            dx,dy=(target[0]-state.x,target[1]-state.y) if target is not None else (0.,0.)
            values += [float(slot.action.type_index==i) for i in range(14)]
            values += [float(phase==p) for p in PHASES]
            values += [float(reason==r) for r in FAILURES]
            values += [float(reason is not None and reason not in FAILURES),
                float(bool(slot.pause_reason) and slot.pause_reason!='perception_wait'),
                float(slot.pause_reason=='perception_wait'),finite(slot.sensor_missing_seconds,.25),
                finite(g.get('elapsed',0.)/max(g.get('window',3.),1e-6)) if g else 0.,float(g is not None),
                finite(effect.get('geometry_age'),.25),float(bool(effect)),
                finite(endpoint.get('observed_count'),2.),float(bool(endpoint)),
                finite(effect.get('tangential_error'),.5),finite(info.get('gap'),.5),float(info.get('gap') is not None),
                float(getattr(s,'ready',False)),float(getattr(s,'contact_ready',False)),
                float(getattr(s,'joint_active',False)),float(getattr(s,'recovery_requested',False)),
                finite(dx,.5),finite(dy,.5),float(target is not None)]
        return np.asarray(values,dtype=np.float32)

    def observation_vector(self,*args,**kwargs):
        base=super().observation_vector(*args,**kwargs)
        return np.concatenate((base,self.feedback_vector())) if self.context=='extended' else base

    def observation_fields(self):
        base=super().observation_fields()
        return base+[f'r{i}_feedback_{f}' for i in (0,1) for f in EXTRA_FIELDS] if self.context=='extended' else base
