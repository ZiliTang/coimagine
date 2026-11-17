"""Editable development fixture and multi-object sequence/replay adapter.

Scene geometry is privileged editor/replay data, never a policy observation.
Every execution command still goes through the partial-observation contract.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import math
import time

from coimagine.continuous2d.scenes import Pose, Box
from experiments.planar_tasks.tasks import task_catalog
from .environment import MultiObjectOptionEnv, MovableBox
from . import motion as M

SCENE_ID = 'relational_clear_transport'
PRESET_ID = 'relational_clear_transport_demo_v1'
CONTROLLER = 'relational-v2'
PRIMARY_ID = 'entity-a'


def scene():
    return replace(task_catalog()['planar_open_anchor'].instance,
        name=SCENE_ID, map='relational-development', arena=(6.,4.8),
        object_pose=Pose(2.,2.,0), object_polygon=((- .125,-.5),(.125,-.5),(.125,.5),(-.125,.5)),
        robot_poses=(Pose(1.55,1.75,0),Pose(3.,1.25,math.pi/2)),
        goal_centre=(4.5,2.), goal_radius=.20,
        obstacles=(Box(4.2,.55,1.1,.15),Box(4.2,4.25,1.1,.15)))


def default_boxes():
    return [dict(label='entity-b',x=3.,y=2.,width=.8,height=.8,angle=0.,mass=3.)]


def default_sequence():
    def segment(target,a0,a1,seconds):
        return dict(target_id=target,a0=a0,a1=a1,decisions=round(seconds/.2))
    return [segment('entity-b',44,10,8.), segment(None,44,44,1.),
            segment(None,44,M.move_id('W'),6.), segment(None,44,44,1.),
            segment(PRIMARY_ID,2,2,15.2), segment(None,44,44,2.)]


def parse_boxes(raw):
    if not isinstance(raw,list) or len(raw)>8:
        raise ValueError('movable_objects must be a list of at most eight boxes')
    boxes=[];labels={PRIMARY_ID}
    for value in raw:
        if not isinstance(value,dict) or set(value)!={'label','x','y','width','height','angle','mass'}:
            raise ValueError('box requires label/x/y/width/height/angle/mass')
        if not isinstance(value['label'],str) or not value['label'] or value['label'] in labels:
            raise ValueError('movable object IDs must be unique and distinct from transport target')
        if any(type(value[k]) not in (int,float) or not math.isfinite(value[k]) for k in value if k!='label'):
            raise ValueError('finite box geometry and mass required')
        if min(value[k] for k in ('width','height','mass'))<=0:
            raise ValueError('positive box dimensions and mass required')
        boxes.append(MovableBox(**value));labels.add(value['label'])
    return tuple(boxes)


def validate_geometry(instance, boxes):
    # Offline scene-editor validation, not candidate feasibility filtering.
    env=MultiObjectOptionEnv(instance,boxes)
    backend=env.env.backend
    if not backend.finite() or not backend.in_bounds():
        raise ValueError('scene bodies must be finite and inside the arena')
    penetrations=backend.penetration_by_pair()
    if max(penetrations.values())>.001:
        raise ValueError('initial bodies overlap: '+str(penetrations))
    return env


def normalize_sequence(sequence, labels):
    if not isinstance(sequence,list) or not 1<=len(sequence)<=200:
        raise ValueError('sequence requires 1..200 segments')
    out=[]
    for s in sequence:
        if s.get('budget_m') is not None or s.get('time_budget_s') is not None:
            raise ValueError('multi-object editor currently uses timed segments only')
        if any(type(s.get(k)) is not int for k in ('a0','a1','decisions')):
            raise ValueError('integer actions and decisions required')
        a0,a1=M.canonical_index(s['a0']),M.canonical_index(s['a1'])
        if not 1<=s['decisions']<=200:raise ValueError('segment duration outside 0.2..40 seconds')
        target=s.get('target_id',PRIMARY_ID)
        if target is None:
            if not M.is_global_pair(a0,a1):raise ValueError('target-free segments support only MOVE/BRAKE')
        elif target not in labels or a0 in M.MOVE_IDS or a1 in M.MOVE_IDS:
            raise ValueError('unknown target or target-bound MOVE')
        out.append(dict(target_id=target,a0=a0,a1=a1,decisions=s['decisions']))
    return out


class ReplayEnv(MultiObjectOptionEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.frames=[];self.strip=[];self.segment_index=0

    def joint_step(self,a0,a1,record=False):
        result=super().joint_step(a0,a1,record=record)
        tick=self.total_periods;now=tick*.05
        audit=self.offline_audit()
        scans=[];mb=[];contact=[]
        for i,slot in enumerate(self.slots):
            scan,labels=self._rays(i,tick,now)
            scans.append(dict(asdict(scan),entity_ids=labels))
            obs=self.observe(i,tick,now);info=deepcopy(slot.last_info);eff=info.get('effect') or {}
            contact.append(bool(obs.contacting_object))
            mb.append(dict(eff,action=info.get('skill'),stage=info.get('stage'),
                desired=info.get('desired_velocity'),reference=info.get('reference_velocity'),force=info.get('force'),
                contact=info.get('contact_ready',False),direction_valid=info.get('direction_valid'),
                direction_unavailable=info.get('direction_unavailable'),exit_reason=info.get('exit_reason'),
                yield_debug=info.get('yield_debug'),requested_direction=info.get('requested_direction'),
                candidate=eff.get('candidate'),face_points=eff.get('face_points'),
                face_normal=eff.get('face_normal'),selector_debug=info.get('selector_debug',eff.get('selector_debug'))))
        stage=[s.last_info.get('stage','brake') for s in self.slots]
        self.frames.append(dict(t=now,tick=tick,object=audit['entities'][self.primary_id],robots=audit['robots'],
            entities=audit['entities'],active_target=self.active_target,segment_index=self.segment_index,
            stage=stage,phase=self.coordinator.phase,scan=scans,mb=mb,target=[None,None],ref_centre=[None,None]))
        self.strip.append(dict(t=now,tick=tick,stage=stage,phase=self.coordinator.phase,
            contact=contact,physical_contact=contact,joint_active=any(getattr(s.skill,'joint_active',False) for s in self.slots),
            forces=[s.last_info.get('force',(0.,0.)) for s in self.slots]))
        return result


def run_sequence(instance, boxes, sequence, overrides=None):
    started=time.perf_counter()
    env=ReplayEnv(instance,boxes,overrides=overrides)
    events=[];error=None
    # UI duration is in 0.2-second decisions, each one a real command boundary.
    # Visibility checks and target-handoff braking are charged within that time.
    for index,segment in enumerate(sequence):
        env.segment_index=index
        for _ in range(segment['decisions']):
            if env.termination is not None:break
            if time.perf_counter()-started>120:
                error=dict(segment=index,message='testbed wall limit exceeded');break
            command={k:segment[k] for k in ('target_id','a0','a1')};command['ticks']=4
            try:
                result=env.step_command(command)
                events.append(dict(segment=index,**result))
            except ValueError as exc:
                error=dict(segment=index,tick=env.total_periods,command=command,message=str(exc));break
        if error or env.termination is not None:break
    elapsed=time.perf_counter()-started
    return dict(controller_version=CONTROLLER,action_version=M.VERSION,action_codebook_hash=M.CODEBOOK_HASH,
        action_labels=[a.label for a in M.ACTIONS],sequence=sequence,frames=env.frames,strip=env.strip,
        events=events,execution_error=error,success=bool(env.success),termination=env.termination,
        periods=env.total_periods,reward=float(env.reward_total),primary_id=PRIMARY_ID,
        observation_scope='centralized local scans; editor/replay geometry is privileged',
        frame_semantics='post-step geometry, scans and physical contact; control feedback from that tick',
        final_audit=env.offline_audit(),
        timing=dict(execution_and_recording_seconds=elapsed,frame_build_seconds=0.,session_seconds=elapsed,
                    simulated_seconds=env.total_periods*.05))
