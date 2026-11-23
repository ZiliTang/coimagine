"""Experimental CONTACT(normal) and PUSH_FROM_CONTACT on the duration runtime.

Input is a legal joint command with one translating pusher. Its target and discrete
direction encode the requested support normal. The original duration runtime
continues to own physics, target binding, and legality.
"""
from dataclasses import dataclass
from math import cos,sin

import numpy as np

from experiments.option_rl_v19.unknown_shape import direction_matches, visible_faces
from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.relational_search.motion import ACTIONS
from coimagine.random_start.spec import CONTROL_DT

from .memory_flow import current_event


@dataclass(frozen=True)
class PhaseResult:
    status: str
    event: dict | None = None
    observed_effect: dict | None = None

    @property
    def ticks(self):
        return self.event['periods'] if self.event else 0


def pusher(request):
    ids=[i for i in (0,1) if getattr(ACTIONS[request[f'a{i}']], 'effect_kind', None)=='translate']
    if len(ids)!=1:
        raise ValueError('single translating pusher required')
    i=ids[0]
    return i,ACTIONS[request[f'a{i}']].effect_direction


def contact_state(world, request, robot, normal):
    env=world.env
    target=request['target_id']
    if target!=env.active_target:
        return 'target_unbound'
    tick=env.total_periods
    obs=env._sensor(robot,tick,tick*CONTROL_DT,target)
    if not obs.contacting_object:
        return 'no_contact'
    faces=[face for face in visible_faces(obs) if direction_matches(face[1],normal)]
    if not faces:
        return 'support_unobservable'
    # A nearby parallel face is insufficient if the tactile point lies elsewhere.
    if obs.contact_point_robot is None:
        return 'contact_point_unobservable'
    dx,dy=obs.contact_point_robot
    angle=obs.own.angle
    point=np.array([obs.own.x+cos(angle)*dx-sin(angle)*dy,obs.own.y+sin(angle)*dx+cos(angle)*dy])
    for face in faces:
        n,t,rho,lo,hi=face[1:6]
        if abs(point@n-rho)<=.04 and lo-.06<=point@t<=hi+.06:
            return 'ready'
    return 'contact_on_other_surface'


def contact(world,request,budget):
    robot,normal=pusher(request)
    validate_action(request,current_event(world.history)['observation'])
    if contact_state(world,request,robot,normal)=='ready':
        return PhaseResult('ready')
    status='no_contact'
    previous=world.detector.update
    def observe(packet):
        nonlocal status
        reason=previous(packet)
        if world.env.active_target==request['target_id'] and packet['robots'][robot]['contacting_object']:
            status=contact_state(world,request,robot,normal)
            # Stop on actual touch, including a touch that cannot be certified.
            return 'controller_end'
        return reason
    world.detector.update=observe
    try:
        event=world.step(request,remaining_ticks=budget)
    finally:
        world.detector.update=previous
    if status=='no_contact':
        status=event['reason'] if event['reason']!='duration' else 'contact_timeout'
    return PhaseResult(status,event)


def push_from_contact(world,request,budget,measure_effect=False):
    robot,normal=pusher(request)
    status=contact_state(world,request,robot,normal)
    if status!='ready':
        return PhaseResult(status)
    before=current_event(world.history)['observation']
    try:validate_action(request,before)
    except ValueError:return PhaseResult('command_unavailable')
    previous=world.detector.update
    lost=False
    def observe(packet):
        nonlocal lost
        reason=previous(packet)
        if not packet['robots'][robot]['contacting_object']:
            lost=True
            return 'contact_lost'
        return reason
    world.detector.update=observe
    try:
        event=world.step(request,remaining_ticks=budget)
    finally:
        world.detector.update=previous
    effect=None
    if measure_effect:
        after=current_event(world.history)['observation']
        effect=public_effect(before,after,request['target_id'],normal)
    return PhaseResult('contact_lost' if lost else event['reason'],event,effect)


def public_effect(before, after, target, axis):
    """Observed direction and distance, with explicit uncertainty under occlusion."""
    from .candidate_semantics import fitted
    first,last=(fitted(o,target) for o in (before,after))
    if first is None or last is None:
        return {'status':'effect_unobservable'}
    # Large fitted footprint changes imply that the returns sample different
    # faces or very different visible support lengths.
    extents=[np.sort([box['height'],box['width']]) for box in (first,last)]
    if np.any(np.abs(extents[0]-extents[1])>np.maximum(.08,.20*extents[0])):
        return {'status':'support_changed'}
    advance=float(np.subtract([last['x'],last['y']],[first['x'],first['y']])@axis)
    return {'status':'observed','signed_along_m':advance}
