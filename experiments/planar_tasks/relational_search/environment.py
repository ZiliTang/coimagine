"""Partial-observation multi-object adapter for the persistent V2 Options.

Task success always belongs to the original transport object. Selecting another
entity changes sensor interpretation and controller allocation, never body state.
Other movable bodies are non-target occupied surfaces in the legacy skill API;
the public scan retains their distinct instance labels and movable category.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import math
from types import MethodType
import pymunk
from coimagine.continuous2d.backend import PymunkBackend, _body_state
from coimagine.random_start import spec as R
from coimagine.random_start.lidar import scan_patches
from coimagine.random_start.skills import SkillObservation, RobotKinematics
from coimagine.random_start.peer_clearance import PeerClearancePrefix
from experiments.option_rl_v21 import spec as S
from experiments.option_rl_v21.system import OptionSystemEnv
from experiments.option_rl_v13.mode_env import ContinuousModeEnv
from experiments.option_rl_v19.system import JointCoordinator
from experiments.option_rl_v19.unknown_shape import UnknownShapeEnvMixin
from experiments.planar_tasks.interface import policy_observation, legal_pair
from .contract import public_observation, _owned_public_observation, validate_action
from .target_binding import TargetBinding
from . import motion as M


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
    @contextmanager
    def contact_snapshot(self, enabled):
        """Reuse narrow-phase results only inside a synchronous read/step scope.

        Each physical advance invalidates the snapshot. No cache survives the
        scope, so editor/probe pose edits between calls cannot leave stale data.
        """
        previous = getattr(self, '_contact_cache_active', False)
        self._contact_cache_active = previous or enabled
        try:
            yield
        finally:
            self._contact_cache_active = previous
            if not previous:self._contact_result = None

    def apply_and_step(self, *args, **kwargs):
        self._contact_result = None
        lidar = getattr(self, 'lidar', None)
        if lidar is not None: lidar.invalidate()
        try:
            return super().apply_and_step(*args, **kwargs)
        finally:
            self._contact_result = None
            if lidar is not None: lidar.invalidate()

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
        if not getattr(self, '_contact_cache_active', False):
            return self._contacts_uncached()
        if getattr(self, '_contact_result', None) is None:
            self._contact_result = self._contacts_uncached()
        # Consumers may annotate contact dictionaries; never share mutable rows.
        return [dict(c) for c in self._contact_result]

    def _contacts_uncached(self):
        if not hasattr(self,'shapes'):
            return super().contacts()
        if len(self.shapes)==1:
            return super().contacts()
        result = []
        def add(a,b,pair,robot_id=None,entity_id=None):
            # Chipmunk's narrow-phase query can return a spurious circle/poly
            # contact for disjoint shapes exactly aligned with a polygon edge.
            # Match the engine broad phase before querying contact points.
            if not a.bb.intersects(b.bb):
                return
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
        self.env.backend.lidar = self.lidar
        self.decision_ticks=1
        self.binding=TargetBinding(); self.binding.current=primary_id
        self.command_prefix=[]
        self._multi_ready=True
        self.option_spec=M
        self.actions=M.ACTIONS
        self.action_version=M.VERSION
        self.action_codebook_hash=M.CODEBOOK_HASH
        self._commit_target(primary_id)

    def joint_step(self, a0, a1, record=False):
        with self.env.backend.contact_snapshot(getattr(self, 'cache_public_sensors', False)), \
             self.lidar.read_scope(getattr(self, 'cache_public_sensors', False)):
            return ContinuousModeEnv.joint_step(self, M.canonical_index(a0), M.canonical_index(a1), record=record)

    def action_masks(self):
        return tuple(M.mask(o) for o in self.boundary_observations())

    def ray_frame(self, robot_id, tick, time, *, cached=True):
        """Shared physical acquisition for public packets, controllers and UI."""
        b = self.env.backend
        states = self._legal_robot_states()
        return self.lidar.physical(b.space, b.robot_bodies, b.labels, robot_id, tick, time,
                                   states[robot_id][:2], states[robot_id][2],
                                   max_range=R.LIDAR_RANGE,
                                   native=getattr(self, 'cache_public_sensors', False), cached=cached)

    def _rays(self, robot_id, tick, time):
        frame = self.ray_frame(robot_id, tick, time)
        return frame.scan, list(frame.entity_ids)

    def _rays_uncached(self, robot_id, tick, time):
        frame = self.ray_frame(robot_id, tick, time, cached=False)
        return frame.scan, list(frame.entity_ids)

    def _target_scan(self, robot_id, tick, time, target):
        return self.lidar.target_view(self.ray_frame(robot_id, tick, time), target)

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
        return self.lidar.memo(('sensor',robot_id,tick,time,target),
                               lambda: self._sensor_uncached(robot_id,tick,time,target))

    def _sensor_uncached(self,robot_id,tick,time,target):
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
        with self.env.backend.contact_snapshot(getattr(self, 'cache_public_sensors', False)), \
             self.lidar.read_scope(getattr(self, 'cache_public_sensors', False)):
            return self._public_packet()

    def _public_packet(self):
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
        packet.update(protocol='relational-option-partial-move-v2',tick=now,
                      action_version=self.action_version,action_codebook_hash=self.action_codebook_hash,
                      active_target=self.active_target,
                      goal=dict(target_id=self.primary_id,xy=list(self.instance.goal_centre),radius=self.instance.goal_radius),
                      entities=[dict(id=k,points=v,last_seen_tick=now) for k,v in seen.items()])
        per_target={label:self._pairs([self._sensor(i,now,now*R.CONTROL_DT,label) for i in (0,1)]) for label in seen}
        packet['entity_legal_pairs']=per_target
        global_pairs=M.global_pairs()
        packet['global_legal_pairs']=[list(p) for p in global_pairs]
        union=set(global_pairs)
        for pairs in per_target.values(): union.update(tuple(p) for p in pairs)
        packet['legal_pairs']=[list(p) for p in sorted(union)]
        # policy_observation/json_safe owns every nested sensor value; the
        # replacement scans, entity points and menus above are freshly built.
        # Transfer this packet through the SAME whitelist/checks, without a
        # second deep copy. No packet reference survives in the environment.
        return _owned_public_observation(packet)

    def _commit_target(self,target):
        self.active_target=target
        self.slots=[M.MotionSlot(self.factory),M.MotionSlot(self.factory)]
        self.coordinator=JointCoordinator(action_spec=M)
        self.clearance=PeerClearancePrefix()
        self.pair_identity=None; self.recovery_started=None; self.recovery_time=0.; self.recovery_attempts=0
        self._pair_contact_latch=False
        # Replace, never nest, the prior-free wrapper and its estimator/allocation.
        self._coordinate=MethodType(OptionSystemEnv._coordinate,self)
        UnknownShapeEnvMixin.install_prior_free(self)
        self.lidar.invalidate(); self._patch_cache.clear()

    def step_command(self,requested,*,record=False):
        if self.termination is not None:
            raise RuntimeError('episode already terminated')
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
