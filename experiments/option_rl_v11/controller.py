"""Frozen fast single-PUSH candidate, legacy rotation motion, bound FOLLOW and 62 actions."""
from dataclasses import replace
from functools import partial
import math
from coimagine.random_start import efficient_profile as EP,env as E
from coimagine.random_start.effect_push import make_effect_skill
ACTIONS=EP.ACTIONS
VERSION='effect-options-v12-fast-solo-normal-taper'
CONFIG=replace(EP.CONFIG,speed=.32,tangential_gain=4.8,tangential_speed=.16*.32/.24,
 point_rate=.16,contact_tangent_damping=.7,approach_tangential_speed=.3,approach_normal_speed=.3,
 approach_tangential_gain=4.5,approach_velocity_damping=1.2,approach_brake_accel=.7,
 standalone_follow_speed=.32,follow_speed=.32)
ROTATION_CONFIG=EP.ROTATION_CONFIG

def skill_factory(action,force_limit):
 config=ROTATION_CONFIG if getattr(action,'effect_kind',None)=='rotate' else CONFIG
 s=make_effect_skill(action,force_limit,config=config)
 if action.kind=='push' and getattr(action,'effect_kind',None)=='translate':
  original=s.reference.update
  def update(raw,dt):
   if s.stage=='push' and s.face is not None and not s.joint_pair:
    n=s.face.normal;radial=sum(raw[i]*n[i] for i in (0,1));remaining=max(0.,action.distance-s.normal_travel)
    cap=max(.025,math.sqrt(2*.25*remaining))
    if radial>cap:raw=tuple(raw[i]-(radial-cap)*n[i] for i in (0,1))
   length=math.hypot(*raw)
   if length>.5:raw=tuple(v*.5/length for v in raw)
   return original(raw,dt)
  s.reference.update=update
 return s

def make_env(instance):
 from experiments.option_rl_v11.scenes import geometry_audit
 audit=geometry_audit(instance)
 if not audit['legal']:raise ValueError(f'illegal initial geometry: {audit}')
 return E.RandomStartEnv(instance,E.EnvConfig(peer_clearance_recovery='boundary_v1',lidar_backend='numpy_batch'),skill_factory=skill_factory,action_table=ACTIONS,action_version=VERSION)
