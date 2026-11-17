"""Partial-observation multi-object adapter for the persistent V2 Options.

Task success always belongs to the original transport object. Selecting another
entity changes sensor interpretation and controller allocation, never body state.
Other movable bodies are non-target occupied surfaces in the legacy skill API;
the public scan retains their distinct instance labels and movable category.
"""
from dataclasses import dataclass, replace
import math
from types import MethodType
import pymunk
from coimagine.continuous2d.backend import PymunkBackend, _body_state
from coimagine.random_start import spec as R
from coimagine.random_start.lidar import LidarScan, scan_patches
from coimagine.random_start.skills import SkillObservation, RobotKinematics
from coimagine.random_start.peer_clearance import PeerClearancePrefix
from experiments.option_rl_v21 import spec as S
from experiments.option_rl_v21.system import OptionSystemEnv, YieldSlot
from experiments.option_rl_v19.system import JointCoordinator
from experiments.option_rl_v19.unknown_shape import UnknownShapeEnvMixin
from experiments.planar_tasks.interface import policy_observation, legal_pair
from .contract import public_observation, validate_action
from .target_binding import TargetBinding


@dataclass(frozen=True)
class MovableBox:
    label: str
    x: float
    y: float
    width: float = .25
    height: float = 1.
    angle: float = 0.
    mass: float = 3.


class MultiBackend(PymunkBackend):
    def __init__(self, scene, physics, config_hash, primary_id, boxes):
        super().__init__(scene, physics, config_hash)
        self.primary_id = primary_id
        self.bodies = {primary_id: self.object_body}
        self.shapes = {primary_id: tuple(self.object_shapes)}
        for b in boxes:
            if b.label in self.bodies or not isinstance(b.label, str) or not b.label:
                raise ValueError('distinct opaque entity IDs required')
            if not all(math.isfinite(v) for v in (b.x,b.y,b.width,b.height,b.angle,b.mass)) or min(b.width,b.height,b.mass) <= 0:
                raise ValueError('finite positive box dimensions and mass required')
            body = pymunk.Body(b.mass, pymunk.moment_for_box(b.mass, (b.width,b.height)))
            body.position = (b.x,b.y); body.angle = b.angle
            shape = pymunk.Poly.create_box(body, (b.width,b.height))
            shape.friction = physics.friction; shape.elasticity = physics.elasticity
            self.space.add(body,shape)
            self.bodies[b.label] = body; self.shapes[b.label] = (shape,)
        self.labels = {shape: label for label, shapes in self.shapes.items() for shape in shapes}

    def _apply_forces(self, control):
        super()._apply_forces(control)
        for label, body in self.bodies.items():
            if label != self.primary_id:
                body.force = -self.physics.linear_damping * body.velocity
                body.torque = -self.physics.angular_damping * body.angular_velocity

    def contacts(self):
        if not hasattr(self,'shapes'):
            return super().contacts()
        if len(self.shapes)==1:
            return super().contacts()
        result = []
        def add(a,b,pair,robot_id=None,entity_id=None):
            for p in a.shapes_collide(b).points:
                result.append(dict(pair=pair,robot_id=robot_id,entity_id=entity_id,
                                   point=(float(p.point_a.x),float(p.point_a.y)),distance=float(p.distance)))
        for i,r in enumerate(self.robot_shapes):
            for label,shapes in self.shapes.items():
                for s in shapes: add(r,s,'robot_object',i,label)
            for s in self.static_shapes: add(r,s,'robot_wall',i)
        for i in (0,1): add(self.robot_shapes[i],self.robot_shapes[1-i],'robot_robot',i)
        labels=list(self.shapes)
        for n,label in enumerate(labels):
            for a in self.shapes[label]:
                for wall in self.static_shapes: add(a,wall,'object_wall',entity_id=label)
                for other in labels[n+1:]:
                    for b in self.shapes[other]: add(a,b,'object_object',entity_id=label)
        return result

    def penetration_by_pair(self):
        result=dict.fromkeys(('robot_object','robot_robot','robot_wall','object_wall','object_object'),0.)
        for c in self.contacts(): result[c['pair']]=max(result[c['pair']],-c['distance'])
        return result

    def finite(self):
        return super().finite() and all(math.isfinite(v) for b in self.bodies.values() for v in _body_state(b).as_tuple())

    def in_bounds(self):
        if not super().in_bounds(): return False
        w,h=self.scene.arena; m=self.physics.out_of_bounds_margin
        for label,shapes in self.shapes.items():
            if label == self.primary_id: continue
            for shape in shapes:
                for p in shape.get_vertices():
                    x,y=shape.body.local_to_world(p)
                    if not (-m<=x<=w+m and -m<=y<=h+m): return False
        return True

    def export_state(self,*args,**kwargs):
        raise NotImplementedError('multi-object branches require fixture + full command replay')

    def approximate_restore(self,*args,**kwargs):
        raise NotImplementedError('partial pose restore would lose extra bodies and cached contacts')

    def restore_exact(self,*args,**kwargs):
        raise NotImplementedError('use full command replay, including target transitions')


class MultiObjectOptionEnv(OptionSystemEnv):
    def __init__(self, instance, boxes=(), *, primary_id='entity-a', overrides=None):
        self._multi_ready=False
        self.active_target=primary_id
        self.primary_id=primary_id
        self.boxes=tuple(boxes)
        super().__init__(instance,overrides=overrides)
        old=self.env.backend
        self.env.backend=MultiBackend(old.scene,old.physics,old.config_hash,primary_id,self.boxes)
        self.decision_ticks=1
        self.binding=TargetBinding(); self.binding.current=primary_id
        self.command_prefix=[]
        self._multi_ready=True

    def _rays(self,robot_id,tick,time):
        b=self.env.backend
        states=self._legal_robot_states(); origin=states[robot_id][:2]; heading=states[robot_id][2]
        angles=tuple(heading+j*2*math.pi/72 for j in range(72))
        valid=[]; distances=[]; categories=[]; labels=[]
        for a in angles:
            end=(origin[0]+R.LIDAR_RANGE*math.cos(a),origin[1]+R.LIDAR_RANGE*math.sin(a))
            hits=[h for h in b.space.segment_query(origin,end,0.,pymunk.ShapeFilter())
                  if h.shape.body is not b.robot_bodies[robot_id]]
            hit=min(hits,key=lambda h:h.alpha) if hits else None
            label=b.labels.get(hit.shape) if hit else None
            valid.append(hit is not None); labels.append(label)
            distances.append(float(hit.alpha*R.LIDAR_RANGE) if hit else 0.)
            categories.append(1 if label else 3 if hit and hit.shape.body in b.robot_bodies else 2 if hit else 0)
        return LidarScan(robot_id,tick,time,tuple(origin),heading,angles,tuple(distances),tuple(valid),tuple(categories)),labels

    def _target_scan(self,robot_id,tick,time,target):
        scan,labels=self._rays(robot_id,tick,time)
        return replace(scan,category=tuple(1 if label==target and label is not None else 2 if label else category
                                          for label,category in zip(labels,scan.category)))

    def scan_for(self,robot_id,tick,time):
        if not getattr(self,'_multi_ready',False) or not self.boxes:
            return super().scan_for(robot_id,tick,time)
        return self._target_scan(robot_id,tick,time,self.active_target)

    def contacts(self):
        if not getattr(self,'_multi_ready',False) or not self.boxes:
            return super().contacts()
        out={'0':[],'1':[]}
        for c in self.env.backend.contacts():
            i=c['robot_id']
            if i not in (0,1): continue
            if c['pair']=='robot_object': category='object' if c['entity_id']==self.active_target else 'wall'
            elif c['pair']=='robot_wall': category='wall'
            elif c['pair']=='robot_robot': category='robot'
            else: continue
            out[str(i)].append((category,i,*c['point']))
        return out

    def _sensor(self,robot_id,tick,time,target):
        # No centre/extent/mass/velocity truth of a manipulated object enters.
        states=self._legal_robot_states()
        own,mate=(RobotKinematics(*states[i]) for i in (robot_id,1-robot_id))
        scan=self._target_scan(robot_id,tick,time,target)
        patches=tuple(scan_patches(scan,(own.x,own.y)))
        patch=min((p for p in patches if p.valid),key=lambda p:p.distance,default=None)
        touch=False; static=False; peer=False; cp=None
        for c in self.env.backend.contacts():
            if c['robot_id'] != robot_id: continue
            if c['pair']=='robot_object' and c.get('entity_id',self.primary_id)==target:
                touch=True
                dx,dy=c['point'][0]-own.x,c['point'][1]-own.y
                cp=(dx*math.cos(own.angle)+dy*math.sin(own.angle),-dx*math.sin(own.angle)+dy*math.cos(own.angle))
            elif c['pair'] in ('robot_object','robot_wall'): static=True
            elif c['pair']=='robot_robot': peer=True
        sn,tn=scan.nearest(2),scan.nearest(3)
        return SkillObservation(robot_id,tick,time,own,mate,scan,patches,patch,touch,cp,static,peer,
            self.instance.goal_centre,self.instance.goal_radius,self.instance.arena,R.MAX_FORCE,
            max(0.,self.config.time_budget-time),time,None if sn is None else sn[:2],None if tn is None else tn[:2])

    def observe(self,robot_id,tick,time,contacts=None):
        if not getattr(self,'_multi_ready',False) or not self.boxes:
            return super().observe(robot_id,tick,time,contacts)
        return self._sensor(robot_id,tick,time,self.active_target)

    def _pairs(self,observations):
        masks=[S.mask(o) for o in observations]
        return [[a.index,b.index] for a in S.ACTIVE_ACTIONS for b in S.ACTIVE_ACTIONS
                if masks[0][a.index] and masks[1][b.index] and legal_pair(a,b)]

    def public(self):
        observations=self.boundary_observations()
        packet=policy_observation(observations,[s.last_info for s in self.slots],len(self.command_prefix))
        now=self.total_periods; seen={}
        for i,robot in enumerate(packet['robots']):
            scan,labels=self._rays(i,now,now*R.CONTROL_DT)
            # Preserve the same ranges as legacy scanning in the zero-extra-body compatibility mode.
            robot['scan']=dict(robot_id=i,tick=now,sample_time=now*R.CONTROL_DT,origin=list(scan.origin),heading=scan.heading,
                               ray_angles=list(scan.ray_angles),distances=list(scan.distances),valid=list(scan.valid),
                               category=list(scan.category),entity_ids=labels)
            for angle,distance,valid,label in zip(scan.ray_angles,scan.distances,scan.valid,labels):
                if valid and label is not None:
                    seen.setdefault(label,[]).append([scan.origin[0]+distance*math.cos(angle),scan.origin[1]+distance*math.sin(angle)])
        packet.update(protocol='relational-option-partial-v1',tick=now,
                      goal=dict(target_id=self.primary_id,xy=list(self.instance.goal_centre),radius=self.instance.goal_radius),
                      entities=[dict(id=k,points=v,last_seen_tick=now) for k,v in seen.items()])
        per_target={label:self._pairs([self._sensor(i,now,now*R.CONTROL_DT,label) for i in (0,1)]) for label in seen}
        packet['entity_legal_pairs']=per_target
        union={(44,44)}
        for pairs in per_target.values(): union.update(tuple(p) for p in pairs)
        packet['legal_pairs']=[list(p) for p in sorted(union)]
        return public_observation(packet)

    def _commit_target(self,target):
        self.active_target=target
        self.slots=[YieldSlot(self.factory),YieldSlot(self.factory)]
        self.coordinator=JointCoordinator(action_spec=S)
        self.clearance=PeerClearancePrefix()
        self.pair_identity=None; self.recovery_started=None; self.recovery_time=0.; self.recovery_attempts=0
        self._pair_contact_latch=False
        # Replace, never nest, the prior-free wrapper and its estimator/allocation.
        self._coordinate=MethodType(OptionSystemEnv._coordinate,self)
        UnknownShapeEnvMixin.install_prior_free(self)
        self._scan_cache={}; self._patch_cache={}

    def step_command(self,requested,*,record=False):
        observation=self.public()
        validate_action(requested,observation)
        handoff=self.binding.select(requested,observation)
        if handoff['commit']: self._commit_target(handoff['target'])
        command=handoff['command']; start=self.total_periods
        for _ in range(command['ticks']):
            if self.termination is not None: break
            self.joint_step(command['a0'],command['a1'],record=record)
        self.command_prefix.append(dict(requested))
        return dict(start_tick=start,end_tick=self.total_periods,requested=dict(requested),
                    executed=command,handoff=handoff,termination=self.termination)

    def replay_commands(self):
        fresh=MultiObjectOptionEnv(self.instance,self.boxes,primary_id=self.primary_id,overrides=self.overrides)
        for command in self.command_prefix: fresh.step_command(command)
        return fresh

    def offline_audit(self):
        return dict(tick=self.total_periods,active_target=self.active_target,
                    entities={k:_body_state(v).as_tuple() for k,v in self.env.backend.bodies.items()},
                    robots=[s.as_tuple() for s in self.robot_states()],
                    termination=self.termination,penetration=self.env.backend.penetration_by_pair())
