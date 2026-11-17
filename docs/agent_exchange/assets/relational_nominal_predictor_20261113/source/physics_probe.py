"""M1 low-level integration probe, NOT an Option execution environment.

Two movable rigid bodies share the pinned existing backend. Robots receive
forces only. Public packets contain ray hits and opaque instance labels; audit
poses stay offline. Existing snapshot/contact helpers do not cover the added
body, so they are explicitly not exposed as valid multi-body APIs.
"""
from dataclasses import asdict, dataclass
import math
import pymunk
from coimagine.continuous2d.backend import PymunkBackend, _body_state
from coimagine.continuous2d.config import PhysicsConfig
from coimagine.continuous2d.scenes import SceneSpec, Box, Pose


@dataclass(frozen=True)
class Fixture:
    blocker: bool = True
    blocker_mass: float = 3.
    blocker_x: float = 2.42
    labels: tuple = ('entity-a', 'entity-b')


class ProbeBackend(PymunkBackend):
    def __init__(self, fixture):
        scene = SceneSpec(name='relational-contact-probe', category='probe', variant=0,
                          arena=(6., 4.), obstacles=(), goal=Box(5., 2., .3, .3),
                          object_pose=Pose(2., 2., 0.),
                          robot_poses=(Pose(1.59, 1.75, 0.), Pose(1.59, 2.25, 0.)))
        super().__init__(scene, PhysicsConfig(object_width=.4, object_height=.8), 'probe-only')
        self.extra = None
        self.entity_shapes = {self.object_shape: fixture.labels[0]}
        if fixture.blocker:
            mass = fixture.blocker_mass
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError('positive finite blocker mass')
            self.extra = pymunk.Body(mass, pymunk.moment_for_box(mass, (.4, .8)))
            self.extra.position = (fixture.blocker_x, 2.)
            shape = pymunk.Poly.create_box(self.extra, (.4, .8))
            shape.friction = self.physics.friction
            shape.elasticity = self.physics.elasticity
            self.space.add(self.extra, shape)
            self.entity_shapes[shape] = fixture.labels[1]

    def _apply_forces(self, control):
        super()._apply_forces(control)
        if self.extra is not None:
            self.extra.force = -self.physics.linear_damping * self.extra.velocity
            self.extra.torque = -self.physics.angular_damping * self.extra.angular_velocity

    def export_state(self, *args, **kwargs):
        raise NotImplementedError('single-body snapshots are invalid for this probe; replay fixture and prefix')


class PhysicsProbe:
    def __init__(self, fixture=Fixture()):
        if len(fixture.labels) != 2 or len(set(fixture.labels)) != 2:
            raise ValueError('two distinct opaque labels')
        self.fixture = fixture
        self.backend = ProbeBackend(fixture)
        self.prefix = []

    def step_forces(self, forces):
        if len(forces) != 2 or any(len(f) != 2 or not all(math.isfinite(v) for v in f) for f in forces):
            raise ValueError('two finite force vectors')
        self.backend.apply_and_step(forces)
        self.prefix.append([list(f) for f in forces])

    def observation(self):
        b = self.backend
        robots = []
        points = {}
        for i, robot in enumerate(b.robot_bodies):
            origin = tuple(robot.position)
            distances, ids, valid, categories = [], [], [], []
            angles = [j * 2 * math.pi / 72 for j in range(72)]
            for angle in angles:
                end = (origin[0] + 1.2 * math.cos(angle), origin[1] + 1.2 * math.sin(angle))
                hits = [h for h in b.space.segment_query(origin, end, 0., pymunk.ShapeFilter())
                        if h.shape.body is not robot]
                hit = min(hits, key=lambda h: h.alpha) if hits else None
                label = b.entity_shapes.get(hit.shape) if hit else None
                valid.append(hit is not None)
                distances.append(float(hit.alpha * 1.2) if hit else 0.)
                ids.append(label)
                categories.append(1 if label else 3 if hit and hit.shape.body in b.robot_bodies else 2 if hit else 0)
                if label:
                    points.setdefault(label, []).append([float(hit.point.x), float(hit.point.y)])
            def state(body):
                return dict(zip(('x', 'y', 'angle', 'vx', 'vy', 'omega'), _body_state(body).as_tuple()))
            robots.append(dict(own=state(robot), teammate=state(b.robot_bodies[1-i]),
                               scan=dict(origin=origin, ray_angles=angles, distances=distances,
                                         valid=valid, category=categories, entity_ids=ids),
                               execution={}))
        return dict(protocol='relational-physics-probe-v1', tick=b.tick, decision=len(self.prefix),
                    goal=dict(target_id=self.fixture.labels[0], xy=[5., 2.], radius=.1),
                    robots=robots, legal_pairs=[], entities=[dict(id=k, points=v, last_seen_tick=b.tick)
                                                            for k, v in points.items()])

    def audit_state(self):
        return dict(tick=self.backend.tick, robots=[_body_state(b).as_tuple() for b in self.backend.robot_bodies],
                    objects=[_body_state(self.backend.object_body).as_tuple()] +
                    ([] if self.backend.extra is None else [_body_state(self.backend.extra).as_tuple()]))

    def replay(self):
        other = PhysicsProbe(self.fixture)
        for forces in self.prefix:
            other.step_forces(forces)
        return other

    def configuration(self):
        return dict(fixture=asdict(self.fixture), physics=asdict(self.backend.physics))
