"""Fixed-period decisions over persistent controllers; one unchanged physical world.

Targets/nominal travel do not terminate modes. The policy requests a new mode or
BRAKE; failures latch until a different mode is selected. No world/velocity reset.
"""
from __future__ import annotations
from dataclasses import replace
import math
import numpy as np
from coimagine.random_start import env as E,spec
from coimagine.random_start.skills import HoldBrake
from coimagine.random_start.peer_clearance import PeerClearancePrefix,public_gap,ENTER
from experiments.option_rl_v12 import controller as C
from experiments.option_rl_v12.push_controller import ContinuousPush
PHASES=('find','approach','ready','push','align','move','follow','brake','blocked','switching','peer_separate')
CONTEXT_PER_ROBOT=14+len(PHASES)+6
CONTEXT_DIM=2*CONTEXT_PER_ROBOT

class Slot:
 def __init__(self):
  self.action=C.ACTIONS[C.STOP];self.skill=C.make_skill(self.action);self.pending=None;self.brake=None
  self.last_info={};self.events=[]
 def select(self,action):
  if self.pending is not None:
   self.pending=action;return
  if C.mode_key(action)==C.mode_key(self.action):
   self.action=action;self.skill.action=action
   if action.kind=='reposition':
    self.skill.cw=action.direction=='CW';self.skill.tangent_direction=action.direction
   if isinstance(self.skill,ContinuousPush):
    self.skill.config=replace(self.skill.config,speed=action.speed);self.skill.speed_command=action.speed
   return
  self.pending=action;self.brake=HoldBrake(spec.MAX_FORCE);self.brake.reference.value=self.skill.reference.value
  self.events.append('mode_change_requested')
 def prepare(self,obs):
  if self.pending is not None and obs.own.speed<=spec.BRAKE_SPEED and math.hypot(*self.brake.reference.value)<=1e-9:
   self.action=self.pending;self.pending=None;self.skill=C.make_skill(self.action);self.brake=None;self.events.append('mode_change_committed')
 def step(self,obs):
  if self.pending is not None:
   if obs.own.speed<=spec.BRAKE_SPEED and math.hypot(*self.brake.reference.value)<=1e-9:
    self.action=self.pending;self.pending=None;self.skill=C.make_skill(self.action);self.brake=None;self.events.append('mode_change_committed')
   else:
    f,info=self.brake.step(obs,spec.CONTROL_DT);info.update(stage='switching',requested_kind=self.pending.kind);self.last_info=info;return f,info
  if self.skill.done:
   if self.brake is None:
    self.brake=HoldBrake(spec.MAX_FORCE);self.brake.reference.value=self.skill.reference.value
   f,info=self.brake.step(obs,spec.CONTROL_DT);info.update(stage='blocked',exit_reason=self.skill.exit_reason,done=True)
  else:f,info=self.skill.step(obs,spec.CONTROL_DT)
  info.update(skill=self.action.label,requested_kind=self.action.kind,continuous_mode=True)
  effect=info.get('effect')
  if effect:
   effect=dict(effect);effect.update(target=None,unit='continuous',normal_travel_target=None);info['effect']=effect
  info['progress_target']=None;info['progress_target_distance']=None
  self.last_info=info;return f,info

class ContinuousModeEnv(E.RandomStartEnv):
 def __init__(self,instance,*,decision_hz=5,collect_dense=False):
  if decision_hz not in (5,10):raise ValueError('prototype supports 5 or 10 Hz decisions')
  ticks=1./(decision_hz*spec.CONTROL_DT)
  if not (decision_hz>0 and abs(ticks-round(ticks))<1e-8):raise ValueError('decision period must be integral control ticks')
  super().__init__(instance,E.EnvConfig(peer_clearance_recovery='boundary_v1',lidar_backend='numpy_batch'),skill_factory=C.make_skill,action_table=C.ACTIONS,action_version=C.VERSION.replace('-5hz',f'-{decision_hz}hz'))
  self.decision_ticks=int(round(ticks));self.slots=[Slot(),Slot()];self.clearance=PeerClearancePrefix()
  self.collect_dense=collect_dense;self.dense_samples=[];self.recovery_time=0.;self.recovery_started=None;self.recovery_attempts=0
 def controller_vector(self):
  values=[]
  for slot in self.slots:
   a=slot.pending or slot.action;phase='peer_separate' if self.clearance.started and not self.clearance.finished else 'switching' if slot.pending else 'blocked' if slot.skill.done else slot.skill.stage
   values += [float(i==a.type_index) for i in range(14)]+[float(p==phase) for p in PHASES]
   ref=(slot.brake if slot.brake is not None else slot.skill).reference.value
   values += [a.speed/.5,min(1.,slot.skill.skill_time/90),ref[0]/.5,ref[1]/.5,float(slot.skill.done),float(slot.pending is not None)]
  return np.asarray(values,dtype=np.float32)
 def observation_vector(self,*args,**kwargs):return np.concatenate([super().observation_vector(*args,**kwargs),self.controller_vector()])
 def observation_fields(self):
  fields=super().observation_fields()
  for i in (0,1):fields += [f'r{i}_mode_{k}' for k in range(14)]+[f'r{i}_phase_{p}' for p in PHASES]+[f'r{i}_{k}' for k in ['mode_speed','mode_age','reference_vx','reference_vy','controller_failed','switching']]
  return fields
 def export_branch_state(self):raise NotImplementedError('Persistent controller snapshots not implemented; use full-episode command replay')
 def restore_branch_state(self,*a,**kw):raise NotImplementedError('Persistent controller snapshots not implemented; use full-episode command replay')
 def _coordinate(self,now):
  ss=[slot.skill for slot in self.slots];paired=all(isinstance(s,ContinuousPush) for s in ss) and all(slot.pending is None for slot in self.slots)
  for s in ss:
   if isinstance(s,ContinuousPush):
    s.cooperative=paired;s.joint_pair=paired;s.recovery_enabled=paired
    if not paired:s.joint_active=False
  if paired:
   for s in ss:
    if s.stage=='push':s.ready=True
   if all(s.ready and not s.done for s in ss):
    for s in ss:s.release=True
   if all(s.contact_ready and not s.done for s in ss):
    for s in ss:s.joint_active=True
   if any(s.done for s in ss):
    for s in ss:
     if not s.done:s.pending_exit_reason='peer_mode_failed'
   elif self.recovery_started is None and any(s.recovery_requested for s in ss):
    self.recovery_attempts+=1
    if self.recovery_attempts>ss[0].config.recovery_max_attempts:
     for s in ss:s.pending_exit_reason='joint_recovery_attempt_limit'
    else:
     for s in ss:s.begin_recovery()
     self.recovery_started=now;self.recovery_time=0.
   elif self.recovery_started is not None:
    self.recovery_time=self.recovery_time+spec.CONTROL_DT if all(s.recovery_ready for s in ss) else 0.
    if self.recovery_time>=ss[0].config.recovery_ready_seconds-1e-9:
     for s in ss:s.resume_recovery()
     self.recovery_started=None
    elif now-self.recovery_started>=ss[0].config.recovery_timeout:
     for s in ss:s.pending_exit_reason='joint_recovery_timeout'
     self.recovery_started=None
  else:self.recovery_started=None;self.recovery_attempts=0
  for i,s in enumerate(ss):
   if s.kind=='follow' and self.slots[i].pending is None:
    peer=ss[1-i]
    if s.bound_push is None:s.bind_push(1-i,self.config.time_budget+spec.CONTROL_DT)
    s.bound_push_phase=self.slots[1-i].last_info.get('stage','prepare')
    if peer.done:s.bound_push_stop='peer_push_failed'
 def joint_step(self,a0,a1,record=False):
  if self.termination is not None:raise RuntimeError('episode already terminated')
  self._joint_action_index(a0,a1);actions=[self.actions[a0],self.actions[a1]]
  if any(a.kind=='follow' and actions[1-i].kind!='push' for i,a in enumerate(actions)):raise ValueError('FOLLOW requires a PUSH peer')
  dense_before=self.boundary_vector().copy() if self.collect_dense else None
  if self.clearance.failed and any(C.mode_key(a)!=C.mode_key(slot.pending or slot.action) for slot,a in zip(self.slots,actions)):self.clearance=PeerClearancePrefix()
  for slot,a in zip(self.slots,actions):slot.select(a)
  start=self.total_periods;distances=[self.object_distance()];anomaly=None;success=False;periods=0;events=[];dense_forces=[]
  for _ in range(self.decision_ticks):
   now=self.total_periods*spec.CONTROL_DT;contacts=self.contacts();observations={i:self.observe(i,self.total_periods,now,contacts) for i in (0,1)}
   for i,slot in enumerate(self.slots):slot.prepare(observations[i])
   self._coordinate(now)
   gap,rate=public_gap(observations[0])
   if self.clearance.finished and not self.clearance.failed and (gap<=ENTER or (gap<.10 and gap+.2*rate<=ENTER)):self.clearance=PeerClearancePrefix()
   if not self.clearance.started and not self.clearance.finished:
    for i,slot in enumerate(self.slots):self.clearance.reference[i].value=(slot.brake if slot.brake is not None else slot.skill).reference.value
   separation=self.clearance.step(observations,spec.CONTROL_DT)
   forces=[];infos={}
   for i,slot in enumerate(self.slots):
    if separation is not None:
     force,info=separation[i]
     slot.skill.skill_time+=spec.CONTROL_DT;slot.skill.previous_position=(observations[i].own.x,observations[i].own.y)
     slot.skill.reference.value=info['reference_velocity']
     if slot.brake is not None:slot.brake.reference.value=info['reference_velocity']
     info.update(skill=actions[i].label,requested_kind=actions[i].kind)
     if self.clearance.failed:slot.skill._finish('blocked_peer_clearance',observations[i])
    else:force,info=slot.step(observations[i])
    slot.last_info=info;forces.append(force);infos[i]=info
    target=info.get('desired_heading')
    if self.orient_target and target is not None:
     error=E._wrap_angle(math.atan2(target[1],target[0])-self.heading[i]);omega=max(-self.config.max_omega,min(self.config.max_omega,self.config.align_gain*error));self.heading_rate[i]=omega;self.heading[i]=E._wrap_angle(self.heading[i]+omega*spec.CONTROL_DT)
    else:self.heading_rate[i]=0.
    if record:info.update(heading=self.heading[i],heading_rate=self.heading_rate[i],force_scale=1.,force_gated=list(force))
    self._last_progress[i]=(0.,0.)
   _,d=self.env.step_control((forces[0],forces[1]),((self.heading[0],self.heading_rate[0]),(self.heading[1],self.heading_rate[1])),minimal=True)
   self.total_periods+=1;self.world_advances+=1;periods+=1
   c=self.success_conditions();distances.append(float(c['distance']));self.hold_seconds=self.hold_seconds+spec.CONTROL_DT if c['distance_ok'] and c['linear_ok'] and c['angular_ok'] else 0.
   success=self.hold_seconds>=self.config.success_hold
   anomaly=E.TERMINATION_NUMERIC if not d['finite'] else E.TERMINATION_PENETRATION if d['penetration']>self.config.penetration_limit else E.TERMINATION_OUT_OF_BOUNDS if not d['in_bounds'] else None
   terminal=success or anomaly is not None or self.total_periods*spec.CONTROL_DT>=self.config.time_budget
   if record:
    row=self._row(self.skills_used+1,self.total_periods,actions[0],actions[1],infos,observations,forces,c,d['penetration']);row['mode_context']=self.controller_vector().tolist();self.rows.append(row)
   if self.collect_dense:
    dense_forces.append([list(f) for f in forces])
    if self.total_periods%2==0 or terminal:
     after=self.boundary_vector().copy();self.dense_samples.append(dict(observation=dense_before.tolist(),next_observation=after.tolist(),actions=[a0,a1],forces=dense_forces,dt=len(dense_forces)*spec.CONTROL_DT,terminal=bool(terminal)));dense_before=after;dense_forces=[]
   if terminal:break
  truncated=not success and anomaly is None and self.total_periods*spec.CONTROL_DT>=self.config.time_budget
  if success:self.success=True;self.termination=E.TERMINATION_SUCCESS
  elif anomaly:self.termination=anomaly
  elif truncated:self.termination=E.TERMINATION_BUDGET
  terms=self._reward_terms(distances,success=success,budget=truncated);self.reward_total+=terms['total']
  for k in self.reward_components:self.reward_components[k]+=terms[k]
  self.skills_used+=1;ends={}
  for i,slot in enumerate(self.slots):
   ends[i]=dict(exit_reason=slot.skill.exit_reason or 'continuing',stage='switching' if slot.pending else slot.skill.stage,time=self.total_periods*spec.CONTROL_DT,period=self.total_periods,speed_at_exit=math.hypot(self.robot_states()[i].vx,self.robot_states()[i].vy),residual_motion=False)
   events+=slot.events;slot.events=[]
  boundary=dict(action_version=self.action_version,action_codebook_hash=self.action_codebook_hash,action_index=[a0,a1],action=[a.label for a in actions],kind=[a.kind for a in actions],periods=periods,start_period=start,start_time=start*spec.CONTROL_DT,end_time=self.total_periods*spec.CONTROL_DT,skill_ends=ends,reward=terms,success=success,truncated=truncated,anomaly=anomaly,termination=self.termination,distance_before=distances[0],distance_after=distances[-1],goal_distance_after=self.object_distance(),execution={'protocol':'persistent-fixed-time-v1','decision_ticks':self.decision_ticks,'events':events},peer_clearance={'entered':self.clearance.started,'failed':self.clearance.failed})
  if record:self.boundaries.append(boundary)
  return boundary
