"""An explicit-contact CAP adapter. Inputs are owned legal observations and phase reports."""
from dataclasses import dataclass, field

import numpy as np

from coimagine.random_start.spec import ROBOT_RADIUS
from experiments.option_rl_v19.unknown_shape import direction_matches
from experiments.planar_tasks.relational_search.motion import ACTIONS
from .candidate_semantics import endpoints, fitted
from .push_contact_recovery import clearance

BRAKE = 44
YIELD = 46
RECOVERY_TICKS = 120


def translating(action):
    return [i for i in (0,1) if getattr(ACTIONS[action[f'a{i}']], 'effect_kind', None)=='translate']


def signature(action):
    (robot,) = translating(action)
    return (action['target_id'],robot,tuple(ACTIONS[action[f'a{robot}']].effect_direction))


def position(observation,robot):
    own=observation['robots'][robot]['own']
    return np.array([own['x'],own['y']])


def recovery_options(obs, command, blocked):
    """Legal single-pusher alternatives retaining a positive component of the intended direction."""
    target,_,intent=signature(command)
    if obs['active_target']!=target:
        return []
    options=[]
    for robot in (0,1):
        view=obs['robots'][robot]
        own=position(obs,robot)
        returns,_,ids=endpoints(view)
        returns=returns[ids!=target]
        arena=np.asarray(view['arena'])
        faces=[p for p in (view['patches'] or []) if p['valid'] and len(p['points_world'])>=3]
        for pair in obs['entity_legal_pairs'].get(target,[]):
            code=pair[robot]
            option=ACTIONS[code]
            if getattr(option,'effect_kind',None)!='translate' or pair[1-robot] not in (BRAKE,YIELD):
                continue
            axis=tuple(option.effect_direction)
            key=(target,robot,axis)
            alignment=float(np.dot(axis,intent))
            if key in blocked or alignment<.65 or option.speed!=.20:
                continue
            for face in faces:
                n=np.asarray(face['normal_world'])
                if not direction_matches(n,axis):
                    continue
                q=np.mean(face['points_world'],axis=0)
                station=q-n*(ROBOT_RADIUS+.15)
                if min(*station,*(arena-station))<ROBOT_RADIUS+.02:
                    continue
                gap=clearance(own,station,returns)
                if gap<ROBOT_RADIUS-.015:
                    continue
                distance=float(np.linalg.norm(station-own))
                priority=.38*alignment-.55*distance+.25*min(gap,.40)
                changed=dict(target_id=target,a0=pair[0],a1=pair[1],ticks=command['ticks'])
                options.append((priority,changed))
    return options


@dataclass
class ContactAwareCAP:
    precision: bool = False
    stalled: dict = field(default_factory=dict)
    blocked_until: dict = field(default_factory=dict)

    def finish_contact(self,command,before,after,status,ticks):
        key=signature(command)
        if status=='ready':
            self.stalled.pop(key,None)
            self.blocked_until.pop(key,None)
            return
        _,robot,_=key
        motion=np.linalg.norm(position(after,robot)-position(before,robot))
        if status not in ('contact_timeout','blocked','controller_end','contact_on_other_surface') or ticks<12 or motion>=.05:
            self.stalled.pop(key,None)
            return
        tick=after['tick']
        previous=self.stalled.get(key)
        self.stalled[key]=tick
        if previous is not None and tick-previous<=RECOVERY_TICKS:
            self.blocked_until[key]=tick+RECOVERY_TICKS

    def choose_contact(self, observation, proposal):
        if len(translating(proposal))!=1:
            return proposal
        tick=observation['tick']
        self.blocked_until={key:expiry for key,expiry in self.blocked_until.items() if expiry>tick}
        if signature(proposal) not in self.blocked_until:
            return proposal
        options=recovery_options(observation,proposal,self.blocked_until)
        return max(options,key=lambda opt:opt[0])[1] if options else proposal

    def choose_push(self, observation, command, remaining):
        cap=min(command['ticks'],remaining)
        action=dict(command,ticks=cap)
        if not self.precision:
            return action
        _,robot,axis=signature(command)
        if command['target_id']!=observation['goal']['target_id']:
            return action
        box=fitted(observation,command['target_id'])
        if box is None:
            return action
        delta=np.subtract(observation['goal']['xy'],[box['x'],box['y']])
        gap=float(np.linalg.norm(delta))
        if gap>.48 or delta@axis<.82*gap:
            return action
        code=command[f'a{robot}']
        slower=code-code%4+1
        pair=[action['a0'],action['a1']]
        pair[robot]=slower
        if ACTIONS[code].speed>.20 and pair in observation['entity_legal_pairs'].get(command['target_id'],[]):
            action[f'a{robot}']=slower
        # Keep the original command time if slower speed would increase the horizon.
        action['ticks']=min(cap,12 if gap<=.29 else 20)
        return action
