"""Experimental public-only anchor recovery for a stalled single PUSH."""
from dataclasses import dataclass, field

import numpy as np

from coimagine.random_start.lidar import CATEGORY_OBJECT
from coimagine.random_start.spec import OBSTACLE_AVOID_GAIN, ROBOT_RADIUS
from experiments.option_rl_v19 import unknown_shape as push
from experiments.planar_tasks.relational_search.motion import ACTIONS


def clearance(start, end, returns):
    """Distance from the measured returns to a swept robot centre segment."""
    delta = end-start
    fraction = np.clip(((returns-start)@delta)/max(delta@delta,1e-12),0,1)
    return float(np.linalg.norm(returns-start-fraction[:,None]*delta,axis=1).min(initial=5.))


@dataclass
class ContactRecovery:
    wait: int
    strategy: str
    reach: float = .20
    samples: list = field(default_factory=list)
    applied: list = field(default_factory=list)
    status: str = 'watching'
    temporary: tuple | None = None

    def close(self):
        if self.temporary is not None:
            skill,name,previous=self.temporary
            setattr(skill,name,previous)
            self.temporary=None

    def tick(self, env, packet, request):
        pushers = [i for i in (0,1) if getattr(ACTIONS[request[f'a{i}']], 'effect_kind', None)=='translate']
        if len(pushers)!=1 or env.active_target!=request['target_id'] or self.applied:
            return
        i=pushers[0]
        robot=packet['robots'][i]
        if self.strategy=='heading':
            if robot['contacting_object'] and robot['execution']['stage']=='push':
                self.correct_drive_direction(env.slots[i].skill)
                self.status='contact_direction_adjusted'
                self.applied.append(dict(tick=env.total_periods,robot=i,lateral_limit=self.reach))
            return
        point=np.array([robot['own']['x'],robot['own']['y']])
        if robot['contacting_object'] or robot['execution']['stage']!='standoff':
            self.samples.clear()
            return
        self.samples.append(point)
        self.samples=self.samples[-self.wait:]
        if len(self.samples)<self.wait or np.ptp(self.samples,axis=0).max()>.018:
            return
        self.status='stalled'
        obs=env.boundary_observations()[i]
        if self.strategy=='standoff':
            self.reduce_standoff(env.slots[i].skill)
            self.status='short_standoff'
            self.applied.append(dict(tick=env.total_periods,robot=i,standoff=self.reach))
            return
        if self.strategy=='margin':
            surfaces=[x for x in (obs.static_nearest,obs.teammate_nearest) if x is not None]
            if not surfaces or min(x[0]-ROBOT_RADIUS for x in surfaces)>.12:
                self.status='no_nearby_return'
                return
            self.reduce_approach_margin(env.slots[i].skill)
            self.status='approach_margin'
            self.applied.append(dict(tick=env.total_periods,robot=i,margin=self.reach))
            return
        old=env._commit_q.get(i)
        anchor=env._commit_anchor.get(i)
        if old is None or anchor is None:
            self.status='anchor_unavailable'
            return
        _,face,_,_=push.track_contact_q(obs,old,anchor)
        if face is None or not push.direction_matches(face[1],ACTIONS[request[f'a{i}']].effect_direction, obs):
            self.status='face_unavailable'
            return
        normal,tangent=face[1:3]
        scalar=float(np.asarray(old)@tangent)
        lo=max(face[4]+.025,scalar-self.reach)
        hi=min(face[5]-.025,scalar+self.reach)
        if hi<lo:
            self.status='short_support'
            return
        choices=np.linspace(lo,hi,9)
        choices=choices[np.abs(choices-scalar)>=.045]
        if not len(choices):
            self.status='short_support'
            return
        points=np.stack([push._face_world(face,q) for q in choices])
        origin=np.array([obs.own.x,obs.own.y])
        if self.strategy=='near':
            selected=int(np.argmin(np.linalg.norm(points-origin,axis=1)))
        else:
            scan=obs.scan
            angle=np.asarray(scan.ray_angles)
            rays=np.column_stack((np.cos(angle),np.sin(angle)))
            occupied=np.asarray(scan.valid)&(np.asarray(scan.category)!=CATEGORY_OBJECT)
            returns=origin+rays[occupied]*np.asarray(scan.distances)[occupied,None]
            mate=np.array([obs.teammate.x,obs.teammate.y])
            start_pose=lambda q:q-normal*(ROBOT_RADIUS+.15)
            def score(q):
                station=start_pose(q)
                contact=q-normal*ROBOT_RADIUS
                if min(*station,*(np.asarray(obs.arena)-station),*contact,*(np.asarray(obs.arena)-contact))<ROBOT_RADIUS+.02:
                    return -1.
                path=min(clearance(origin,station,returns),clearance(station,contact,returns))
                mate_gap=min(clearance(origin,station,mate[None,:]),clearance(station,contact,mate[None,:]))-ROBOT_RADIUS
                gap=min(path,mate_gap)
                return min(gap,.45)-.22*np.linalg.norm(station-origin)
            scores=np.array([score(q) for q in points])
            selected=int(np.argmax(scores))
            if scores[selected]<max(.06,score(np.asarray(old))+.025):
                self.status='no_clearer_anchor'
                return
        env._commit_q[i]=tuple(points[selected])
        env._commit_anchor[i]=push._face_anchor(face,float(choices[selected]))
        self.status='anchor_shifted'
        self.applied.append(dict(tick=env.total_periods,robot=i,old=list(old),new=points[selected].tolist()))


    def reduce_approach_margin(self, skill):
        previous=skill._obstacle_avoidance

        def approach_only(observation, raw, include_teammate=False):
            if skill.stage!='standoff' or observation.contacting_object:
                return previous(observation,raw,include_teammate)
            result=np.array(raw,dtype=float)
            skill.avoidance_active=False
            skill.avoidance_direction=None
            surfaces=[observation.static_nearest]
            if include_teammate:
                surfaces.append(observation.teammate_nearest)
            for item in surfaces:
                if item is None:
                    continue
                distance,point=item
                gap=distance-ROBOT_RADIUS
                if gap>self.reach:
                    continue
                away=np.subtract((observation.own.x,observation.own.y),point)
                away/=max(np.linalg.norm(away),1e-9)
                result+=OBSTACLE_AVOID_GAIN*(self.reach-gap)*away
                skill.avoidance_active=True
                skill.avoidance_direction=tuple(away)
            return tuple(result)

        self.temporary=(skill,'_obstacle_avoidance',previous)
        skill._obstacle_avoidance=approach_only


    def reduce_standoff(self, skill):
        previous=skill._allocated_navigate

        def navigate(observation, dt, patch, normal, tangent, cp):
            q=np.asarray(skill.allocated_q)
            own=np.array([observation.own.x,observation.own.y])
            error=float((q-own)@tangent)
            if observation.contacting_object or abs(error)<=.05:
                return previous(observation,dt,patch,normal,tangent,cp)
            target=q-normal*(ROBOT_RADIUS+self.reach)
            skill.contact_phase=skill.stage='standoff'
            skill.debug_ref=tuple(target)
            skill.candidate_err_t=error
            raw=6*(target-own)
            raw*=min(1.,.30/max(np.linalg.norm(raw),1e-9))
            skill.face=patch
            skill.progress=skill.effective_contact
            raw=skill._obstacle_avoidance(observation,tuple(raw),include_teammate=True)
            ref=skill.reference.update(raw,dt)
            force,saturated=skill._force_from_reference(ref,observation)
            return force,skill._info_unknown(observation,force,saturated,ref,raw,patch,cp)

        self.temporary=(skill,'_allocated_navigate',previous)
        skill._allocated_navigate=navigate


    def correct_drive_direction(self, skill):
        previous=skill._obstacle_avoidance

        def steer(observation, raw, include_teammate=False):
            if skill.stage!='push' or not observation.contacting_object or skill.last_n is None:
                return previous(observation,raw,include_teammate)
            normal=np.asarray(skill.last_n)
            tangent=np.array([-normal[1],normal[0]])
            demand=np.asarray(skill.action.effect_direction)
            tangent_error=float(demand@tangent)
            correction=float(np.clip(skill.action.speed*tangent_error,-self.reach,self.reach))
            return previous(observation,tuple(raw+correction*tangent),include_teammate)

        self.temporary=(skill,'_obstacle_avoidance',previous)
        skill._obstacle_avoidance=steer
