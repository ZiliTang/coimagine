"""Bounded pre-option separation from shared peer pose and EACH robot's own scan.

This is not a global path planner. It never uses object truth, the peer's scan,
nominal action targets, or task reward. No nominal travel is credited here.
"""
from __future__ import annotations
import math
import numpy as np
from ..local_contact_option.control import ReferenceVelocity,velocity_feedback_force
from .spec import ROBOT_RADIUS,A_REF_MAX,KV
from .lidar import CATEGORY_OBJECT,CATEGORY_STATIC

VERSION='peer-clearance-prefix-v1'
ENTER=.03
RESUME=.06
MAX_SPEED=.10
LOOKAHEAD=.4
TIMEOUT=2.
FOLLOW_GUARD_VERSION='follow-peer-projection-v1'


def guard_follow_velocity(observation, nominal):
    """Project FOLLOW away from a closing peer; own returns constrain alternatives.

    This is a local velocity guard, not a guarantee against an independently moving
    peer. The ordinary safety exit remains authoritative. No teammate scan is read.
    """
    gap,rate=public_gap(observation)
    dx=observation.own.x-observation.teammate.x;dy=observation.own.y-observation.teammate.y
    distance=math.hypot(dx,dy)
    info=dict(version=FOLLOW_GUARD_VERSION,active=False,public_gap=gap,public_gap_rate=rate,
              target_gap=RESUME,feasible=True)
    if distance<1e-9:
        return (0.,0.),dict(info,active=True,feasible=False,reason='coincident_public_centres')
    away=np.array((dx/distance,dy/distance));mate=np.array((observation.teammate.vx,observation.teammate.vy))
    nominal=np.asarray(nominal,dtype=float);requested_rate=float((nominal-mate)@away)
    minimum_rate=min(MAX_SPEED,2.*(RESUME-gap))
    info.update(requested_gap_rate=requested_rate,minimum_gap_rate=minimum_rate)
    if gap>=.12 and gap+.4*min(rate,requested_rate)>=RESUME:
        return tuple(nominal),info
    if requested_rate>=minimum_rate:
        return tuple(nominal),info
    preferred=nominal+away*(minimum_rate-requested_rate)
    if np.linalg.norm(preferred)>.18:preferred*=.18/np.linalg.norm(preferred)
    candidates=np.array([tuple(preferred),(0.,0.)]+[
        (speed*math.cos(j*math.pi/16),speed*math.sin(j*math.pi/16))
        for speed in (.025,.05,.10,.18) for j in range(32)])
    feasible=(candidates-mate)@away>=minimum_rate-1e-9
    offsets=[]
    for valid,category,d,angle in zip(observation.scan.valid,observation.scan.category,
                                      observation.scan.distances,observation.scan.ray_angles):
        if valid and category in (CATEGORY_OBJECT,CATEGORY_STATIC):
            offsets.append((d*math.cos(angle),d*math.sin(angle)))
    if observation.contact_point_robot is not None:
        x,y=observation.contact_point_robot;c,s=math.cos(observation.own.angle),math.sin(observation.own.angle)
        offsets.append((c*x-s*y,s*x+c*y))
    if offsets:
        points=np.asarray(offsets);segment=candidates*LOOKAHEAD
        length2=np.sum(segment*segment,axis=1)
        u=np.clip((segment@points.T)/np.maximum(length2[:,None],1e-12),0.,1.)
        clearance=np.linalg.norm(points[None,:,:]-u[:,:,None]*segment[:,None,:],axis=2)
        required=np.minimum(np.linalg.norm(points,axis=1),ROBOT_RADIUS+.005)
        feasible&=(clearance>=required[None,:]-1e-7).all(axis=1)
    if not feasible.any():
        return (0.,0.),dict(info,active=True,feasible=False,reason='no_observed_safe_peer_velocity')
    cost=np.sum((candidates-preferred)**2,axis=1);cost[~feasible]=np.inf
    result=tuple(float(x) for x in candidates[int(np.argmin(cost))])
    return result,dict(info,active=True,commanded_gap_rate=float((np.asarray(result)-mate)@away),
                       reason='projected_away_from_peer')


def public_gap(observation):
    dx=observation.own.x-observation.teammate.x;dy=observation.own.y-observation.teammate.y
    d=math.hypot(dx,dy)
    rate=(dx*(observation.own.vx-observation.teammate.vx)+dy*(observation.own.vy-observation.teammate.vy))/max(d,1e-9)
    return d-2*ROBOT_RADIUS,rate


def local_escape(observation):
    """Choose a short non-approaching swept disc using ONLY this side's returns."""
    dx=observation.own.x-observation.teammate.x;dy=observation.own.y-observation.teammate.y
    d=math.hypot(dx,dy)
    if d<1e-9:return (0.,0.),{'feasible':False,'reason':'coincident_public_centres'}
    away=np.array((dx/d,dy/d))
    angles=[math.atan2(away[1],away[0])]+[j*math.pi/16 for j in range(32)]
    candidates=np.array([(speed*math.cos(a),speed*math.sin(a)) for speed in (MAX_SPEED,.05,.025) for a in angles])
    # Public peer separation must not decrease under the requested local motion.
    feasible=candidates@away>=-1e-9
    offsets=[]
    scan=observation.scan
    for valid,category,distance,angle in zip(scan.valid,scan.category,scan.distances,scan.ray_angles):
        if valid and category in (CATEGORY_OBJECT,CATEGORY_STATIC):
            offsets.append((distance*math.cos(angle),distance*math.sin(angle)))
    # Actual object contact is an additional local observation, not a truth query.
    if observation.contact_point_robot is not None:
        x,y=observation.contact_point_robot;c,s=math.cos(observation.own.angle),math.sin(observation.own.angle)
        offsets.append((c*x-s*y,s*x+c*y))
    if offsets:
        points=np.asarray(offsets);segment=candidates*LOOKAHEAD
        length2=np.sum(segment*segment,axis=1)
        u=np.clip((segment@points.T)/length2[:,None],0.,1.)
        difference=points[None,:,:]-u[:,:,None]*segment[:,None,:]
        distances=np.linalg.norm(difference,axis=2)
        initial=np.linalg.norm(points,axis=1)
        # Existing sub-margin contacts may move apart, never deeper. Other hits
        # keep a 5 mm buffer. Swept points do not certify unseen free space.
        required=np.minimum(initial,ROBOT_RADIUS+.005)
        feasible&=(distances>=required[None,:]-1e-7).all(axis=1)
    if not feasible.any():return (0.,0.),{'feasible':False,'reason':'no_observed_escape_direction','returns':len(offsets)}
    projected=candidates@away
    # Prefer increasing peer distance; deterministic tangent tie break, no private exchange.
    tangent=np.array((-away[1],away[0]))
    score=projected+.001*(candidates@tangent)
    score[~feasible]=-np.inf;index=int(np.argmax(score))
    velocity=tuple(float(v) for v in candidates[index])
    return velocity,{'feasible':True,'peer_outward':away.tolist(),'returns':len(offsets),
                     'separation_speed':float(projected[index]),'candidate_count':int(feasible.sum())}


class PeerClearancePrefix:
    def __init__(self):
        self.finished=False;self.started=False;self.failed=False;self.elapsed=0.;self.ready_time=0.
        self.reference=[ReferenceVelocity(A_REF_MAX),ReferenceVelocity(A_REF_MAX)]
        self.path=[0.,0.];self.previous=[None,None]

    def step(self,observations,dt):
        if self.finished:return None
        gap,rate=public_gap(observations[0])
        if not self.started:
            if not (gap<=ENTER or (gap<.10 and gap+.2*rate<=ENTER)):
                self.finished=True;return None
            self.started=True
        self.elapsed+=dt
        if gap>=RESUME:
            # Brake before handoff; public relative speed prevents release while closing.
            self.ready_time=self.ready_time+dt if abs(rate)<=.03 else 0.
        else:self.ready_time=0.
        released=self.ready_time>=.1-1e-9
        self.failed=self.elapsed>=TIMEOUT-1e-9 and not released
        self.finished=released or self.failed
        values=[]
        for i in (0,1):
            obs=observations[i];position=(obs.own.x,obs.own.y)
            if self.previous[i] is not None:self.path[i]+=math.dist(position,self.previous[i])
            self.previous[i]=position
            raw,evidence=local_escape(obs) if gap<RESUME and not self.failed else ((0.,0.),{'feasible':True,'reason':'brake_handoff'})
            reference=self.reference[i].update(raw,dt)
            force=velocity_feedback_force(reference,(obs.own.vx,obs.own.vy),KV,obs.force_limit)
            info=dict(kind='safety_recovery',stage='peer_separate',done=False,desired_heading=None,
                desired_velocity=raw,reference_velocity=reference,force=force,force_norm=math.hypot(*force),
                force_saturated=math.hypot(*force)>=obs.force_limit-1e-9,progress=0.,signed_progress=0.,
                normal_valid=bool(obs.patch and obs.patch.valid),patch_valid=bool(obs.patch and obs.patch.valid),
                peer_clearance=dict(version=VERSION,phase='failed' if self.failed else 'released' if released else 'separate',
                    public_gap=gap,public_gap_rate=rate,elapsed=self.elapsed,path_m=self.path[i],
                    target_gap=RESUME,entry_gap=ENTER,nominal_progress_credit=0.,**evidence))
            values.append((force,info))
        return values
