"""46 persistent closed-loop modes. Intensities are contact-normal speed, not travel."""
from dataclasses import replace,asdict
import math
import numpy as np
from coimagine.random_start import spec,efficient_profile as EP
from coimagine.random_start.effect_push import EffectAction,DIRECTIONS,DIRECTION_NAMES,action_mask
from coimagine.random_start.skills import RepositionSkill,SupportFollowSkill,HoldBrake
from experiments.option_rl_v12.push_controller import ContinuousPush
VERSION='continuous-modes-v1-5hz'
CONFIG=replace(EP.CONFIG,post_push='contact_follow',speed=.32,tangential_gain=4.8,tangential_speed=.213333333333,
 point_rate=.16,contact_tangent_damping=.7,approach_normal_speed=.3,approach_tangential_speed=.3,
 approach_tangential_gain=4.5,approach_velocity_damping=1.2,approach_brake_accel=.7,
 follow_completion='peer_push',standalone_follow_speed=.32)
ROTATION_CONFIG=EP.ROTATION_CONFIG

def table():
 out=[]
 for i,d in enumerate(DIRECTIONS):
  for j,v in enumerate((.04,.08,.16,.32)):
   out.append(EffectAction(kind='push',effect_direction=d,speed=v,scale=v,scale_kind='normal_speed',type_index=i,scale_index=j,index=len(out),label=f'PUSH_{DIRECTION_NAMES[i]}[v={v:.2f}m/s]'))
 for sign,name,typ in [(1,'CCW',8),(-1,'CW',9)]:
  for j,v in enumerate((.02,.04,.06)):
   out.append(EffectAction(kind='push',effect_kind='rotate',rotation_sign=sign,speed=v,scale=v,scale_kind='normal_speed',type_index=typ,scale_index=j,index=len(out),label=f'ROTATE_{name}[normal_v={v:.2f}m/s]'))
 for name,typ in [('CCW',10),('CW',11)]:
  for j,v in enumerate((.06,.12,.18)):
   out.append(spec.RobotAction(kind='reposition',direction=name,speed=v,scale=v,scale_kind='tangent_speed',type_index=typ,scale_index=j,index=len(out),label=f'REPOSITION_{name}[v={v:.2f}m/s]'))
 out.append(spec.RobotAction(kind='wait',type_index=12,index=len(out),label='BRAKE_HOLD',scale_kind='persistent'))
 out.append(spec.RobotAction(kind='follow',type_index=13,index=len(out),label='SUPPORT_FOLLOW_PUSH',scale_kind='persistent'))
 return tuple(out)
ACTIONS=table();STOP=44;FOLLOW=45

def mode_key(action):
 if action.kind=='push':return action.kind,action.effect_kind,action.rotation_sign
 return (action.kind,)

def make_skill(action,force_limit=spec.MAX_FORCE):
 if action.kind=='push':
  config=ROTATION_CONFIG if action.effect_kind=='rotate' else CONFIG
  return ContinuousPush(action,force_limit,replace(config,speed=action.speed))
 if action.kind=='reposition':
  s=RepositionSkill(action,force_limit,boundary_mode='rounded_scan',peer_detour='outer_arc',progress_guard=True)
  s.time_budget=spec.TIME_BUDGET_SECONDS+spec.CONTROL_DT
  original=s.reference.update
  def update(raw,dt):
   if s.stage=='move' and s.tangent is not None:
    t=s.tangent;v=sum(raw[i]*t[i] for i in (0,1));ratio=s.action.speed/spec.REPOSITION_SPEED_MAX
    raw=tuple(raw[i]+(ratio-1.)*v*t[i] for i in (0,1))
   return original(raw,dt)
  s.reference.update=update
  return s
 if action.kind=='follow':
  s=SupportFollowSkill(action,force_limit,completion_clock='peer_push',target_gap=.02,peer_guard='scan_projection_v1',speed_limit=.32)
  s.time_budget=spec.TIME_BUDGET_SECONDS+spec.CONTROL_DT
  return s
 return HoldBrake(force_limit)

def make_env(instance):
 from experiments.option_rl_v12.mode_env import ContinuousModeEnv
 from experiments.option_rl_v12.scenes import geometry_audit
 if not geometry_audit(instance)['legal']:raise ValueError('illegal initial geometry')
 return ContinuousModeEnv(instance)

def mask(obs):
 push=tuple(a for a in ACTIONS if a.kind=='push');local=action_mask(obs,CONFIG,actions=push)
 result=np.zeros(len(ACTIONS),dtype=bool)
 for a,valid in zip(push,local):result[a.index]=valid
 for a in ACTIONS:
  if a.kind=='reposition':result[a.index]=obs.patch is not None
  elif a.kind in ('wait','follow'):result[a.index]=True
 return result
