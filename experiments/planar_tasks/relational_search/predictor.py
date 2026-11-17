"""Independent cold-start nominal worlds; consumed prefixes rebuilt and charged."""
from copy import deepcopy
from dataclasses import replace
import math
import time
from coimagine.continuous2d.scenes import Box,Pose
from coimagine.random_start.layout import InstanceSpec
from coimagine.random_start import spec as R
from .contract import History,public_observation,digest,validate_action
from .estimator import estimate
from .environment import MultiObjectOptionEnv,MovableBox
from .evidence import Ledger


def build(hypothesis):
    h=deepcopy(hypothesis); goal=h['goal']; primary=goal['target_id']
    obj=next(e for e in h['entities'] if e['id']==primary)
    w,hgt=obj['width']/2,obj['height']/2
    polygon=((-w,-hgt),(w,-hgt),(w,hgt),(-w,hgt))
    robots=h['robots']
    instance=InstanceSpec(name='nominal-ray-hypothesis',map='nominal',split='synthetic',index=0,seed=0,
        arena=tuple(h['arena']),obstacles=tuple(Box(*b) for b in h['obstacles']),
        object_pose=Pose(obj['x'],obj['y'],obj['angle']),
        robot_poses=tuple(Pose(r['x'],r['y'],r['angle']) for r in robots),
        goal_centre=tuple(goal['xy']),goal_radius=goal['radius'],object_polygon=polygon)
    boxes=[MovableBox(e['id'],e['x'],e['y'],e['width'],e['height'],e['angle']) for e in h['entities'] if e['id']!=primary]
    world=MultiObjectOptionEnv(instance,boxes,primary_id=primary)
    active=h['active_target']
    if active!=primary:
        if active not in world.env.backend.bodies:
            raise ValueError('committed target has no geometric hypothesis')
        world._commit_target(active)
        world.binding.current=active
    # Initializing a new nominal world from robot proprioception is not an
    # execution-state clone. Object velocities remain explicitly unknown/zero.
    for body,r in zip(world.env.backend.robot_bodies,robots):
        body.velocity=(r['vx'],r['vy']);body.angular_velocity=r['omega']
    return world


class NominalPredictor:
    def __init__(self,history,*,ticks=1200,calls=40,seconds=5.):
        if not isinstance(history,History):raise TypeError('public History required, not an execution environment')
        self.history=History()
        for i,event in enumerate(history.events):
            self.history.append(event['observation'],None if i==0 else event['action'])
        self.hypothesis=estimate(self.history)
        self._hypothesis_hash=digest(self.hypothesis)
        self.root_observation=deepcopy(self.history.events[-1]['observation'])
        self.ledger=Ledger(dict(ticks=ticks,calls=calls))
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('positive finite prediction wall budget')
        self.seconds=seconds;self.started=time.monotonic()
        self.paths={0:()};self.live={};self.next_id=1
        self.last_charge=0;self.events=[];self.models_created=0

    @property
    def root(self):return 0

    def cost_ticks(self,handle,action):
        path=self.paths[handle]
        return action['ticks']+(0 if handle in self.live else sum(a['ticks'] for a in path))

    def _check_time(self):
        if time.monotonic()-self.started>=self.seconds:raise TimeoutError('nominal prediction wall budget')

    def predict(self,handle,action):
        self._check_time()
        if digest(self.hypothesis)!=self._hypothesis_hash:
            raise ValueError('hypothesis changed; create an independent predictor')
        path=self.paths[handle]
        quote=self.cost_ticks(handle,action)
        self.ledger.charge(ticks=quote,calls=1)
        self.last_charge=quote
        event=dict(handle=handle,requested_ticks=action['ticks'],reserved_ticks=quote,
                   replay_ticks=0,forward_ticks=0,source='nominal_prediction',status='started')
        self.events.append(event)
        try:
            if handle in self.live:
                world=self.live.pop(handle)
            else:
                world=build(self.hypothesis)
                self.models_created+=1
                remaining=self.root_observation['robots'][0].get('remaining_seconds',90.)
                world.config.time_budget=remaining
                for previous in path:
                    self._check_time()
                    result=world.step_command(previous)
                    event['replay_ticks']+=result['end_tick']-result['start_tick']
            self._check_time()
            if world.termination is not None:raise ValueError('nominal prefix already terminated')
            result=world.step_command(action)
            actual=result['end_tick']-result['start_tick']
            event['forward_ticks']=actual
            self._check_time()
            packet=world.public()
            offset=self.hypothesis['root_tick']
            packet['tick']+=offset
            packet['decision']+=self.root_observation.get('decision',0)
            packet['source']='nominal_prediction'
            for e in packet['entities']:e['last_seen_tick']+=offset
            for r in packet['robots']:
                r['tick']=packet['tick'];r['time']=packet['tick']*R.CONTROL_DT
                r['elapsed_seconds']=r['time']
                r['scan']['tick']=packet['tick'];r['scan']['sample_time']=r['time']
            child=self.next_id;self.next_id+=1
            self.paths[child]=path+(deepcopy(action),)
            self.live[child]=world
            event.update(status='predicted',child=child,nominal_termination=world.termination)
            return dict(handle=child,observation=public_observation(packet),ticks=actual,
                        cost_seconds=actual*R.CONTROL_DT,terminal=world.termination is not None,
                        success=world.termination=='success',source='nominal_prediction',charged_ticks=quote)
        except Exception as exc:
            event.update(status='error',error=f'{type(exc).__name__}: {exc}')
            raise

    def close(self):
        self.live.clear()

    def record(self):
        return dict(hypothesis=deepcopy(self.hypothesis),hypothesis_sha256=digest(self.hypothesis),
                    usage=self.ledger.record(),events=deepcopy(self.events),
                    models_created=self.models_created,
                    reconstruction='controller cold-start at root; exact replay within nominal prefixes')
