"""Versioned development fixtures. Geometry legality is not a solvability claim.

All variants of a root belong to development. No held-out test is released here.
Scene truth is used for construction, audit and replay, never policy inputs.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

from coimagine.continuous2d.scenes import Box, Pose
from coimagine.random_start import geometry as G, spec as R
from coimagine.random_start.layout import InstanceSpec

VERSION = 'planar-development-v1'


@dataclass(frozen=True)
class Task:
    family: str
    root_id: str
    description: str
    instance: InstanceSpec

    def record(self):
        return dict(taskset_version=VERSION, family=self.family, root_id=self.root_id,
                    description=self.description, split='development',
                    feasibility='unverified', instance=self.instance.as_dict())

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True,
                                         allow_nan=False).encode()).hexdigest()


def task_catalog():
    """Three families, each with an anchor, vertical mirror and translated variant."""
    result = {}
    definitions = (
        ('open', 'Open-space transport; control and interface reference.',
         (2., 2., math.pi/2), ((1.55, 1.75, 0.), (1.55, 2.25, 0.)),
         (3.6, 2.), ()),
        ('wall', 'Near-wall release; rotation is a candidate, not a required solution.',
         (1.71, 2., 0.), ((1.45, 1.55, math.pi/2), (2., 2.45, -math.pi/2)),
         (3.65, 2.), (Box(1.05, 2., .15, 1.45),)),
        ('detour', 'Transport around a fixed obstacle; either route is allowed.',
         (1.65, 2., math.pi/2), ((1.20, 1.75, 0.), (1.20, 2.25, 0.)),
         (3.95, 2.), (Box(2.8, 1.75, .20, .55),)),
    )
    for family, description, obj, robots, goal, obstacles in definitions:
        for variant, mirror, dy in (('anchor', False, 0.), ('mirror', True, 0.),
                                    ('shift', False, .20)):
            def point(x, y):
                return (x, (4.-y if mirror else y)+dy)

            def pose(p):
                return Pose(*point(*p[:2]), -p[2] if mirror else p[2])

            name = f'planar_{family}_{variant}'
            instance = InstanceSpec(name=name, map='planar', split='development',
                index=3000+len(result), seed=20261109, arena=(5., 4.),
                obstacles=tuple(Box(*point(b.cx, b.cy), b.half_w, b.half_h,
                                    -b.angle if mirror else b.angle) for b in obstacles),
                object_pose=pose(obj), robot_poses=tuple(pose(p) for p in robots),
                goal_centre=point(*goal), goal_radius=R.SUCCESS_DISTANCE)
            result[name] = Task(family, f'planar_{family}', description, instance)
    return result


def geometry_audit(instance):
    """Check actual obstacles (not the old L0/L1 map lookup), plus goal placement."""
    obj = instance.object_pose
    if instance.object_polygon is not None:
        raise ValueError('v1 task audit supports the fixed rectangle only')
    poly = G.box_polygon(obj.x, obj.y, obj.angle, R.OBJECT_WIDTH/2, R.OBJECT_HEIGHT/2)
    obstacles = [G.box_polygon(b.cx, b.cy, b.angle, b.half_w, b.half_h)
                 for b in instance.obstacles]
    def polygon_gap(a, b):
        # Vertex containment alone misses cross-shaped rectangle intersections.
        # SAT handles crossing edges and contact, then the existing distance
        # routine measures clearance only for separated polygons.
        separated = False
        for polygon in (a, b):
            for start, end in zip(polygon, polygon[1:]+polygon[:1]):
                normal = (start[1]-end[1], end[0]-start[0])
                pa = [p[0]*normal[0]+p[1]*normal[1] for p in a]
                pb = [p[0]*normal[0]+p[1]*normal[1] for p in b]
                if max(pa) < min(pb) or max(pb) < min(pa):
                    separated = True
        return G.polygon_distance(a, b) if separated else 0.

    gaps = dict(object_boundary=min(G.wall_distance(p, instance.arena) for p in poly),
        robot_boundary=[G.wall_distance((p.x, p.y), instance.arena)-R.ROBOT_RADIUS
                        for p in instance.robot_poses],
        robot_object=[G.circle_polygon_gap((p.x, p.y), R.ROBOT_RADIUS, poly)
                      for p in instance.robot_poses],
        robot_robot=G.circle_circle_gap((instance.robot_poses[0].x, instance.robot_poses[0].y),
            R.ROBOT_RADIUS, (instance.robot_poses[1].x, instance.robot_poses[1].y), R.ROBOT_RADIUS),
        object_obstacle=[polygon_gap(poly, p) for p in obstacles],
        robot_obstacle=[G.circle_polygon_gap((r.x, r.y), R.ROBOT_RADIUS, p)
                        for r in instance.robot_poses for p in obstacles],
        obstacle_boundary=[min(G.wall_distance(p, instance.arena) for p in polygon)
                           for polygon in obstacles])
    goal_poly = G.box_polygon(*instance.goal_centre, obj.angle,
                              R.OBJECT_WIDTH/2, R.OBJECT_HEIGHT/2)
    gaps.update(goal_boundary=min(G.wall_distance(p, instance.arena) for p in goal_poly),
                goal_obstacle=[polygon_gap(goal_poly, p) for p in obstacles],
                goal_region_boundary=G.wall_distance(instance.goal_centre, instance.arena)-instance.goal_radius,
                goal_region_obstacle=[G.circle_polygon_gap(instance.goal_centre, instance.goal_radius, p)
                                      for p in obstacles])
    reasons = []
    for key, value in gaps.items():
        values = value if isinstance(value, list) else [value]
        if any(not math.isfinite(v) or v <= 1e-8 for v in values):
            reasons.append(key)
    if math.dist((obj.x, obj.y), instance.goal_centre) <= R.SUCCESS_DISTANCE:
        reasons.append('initially_at_goal')
    return dict(legal=not reasons, reasons=reasons, clearance_m=gaps,
                feasibility='unverified')


def scene_metadata(instance):
    """Privileged visualization metadata. Never send this to the policy."""
    return dict(id=instance.name, name=instance.name, arena=list(instance.arena),
        goal=dict(x=instance.goal_centre[0], y=instance.goal_centre[1], radius=instance.goal_radius),
        obstacles=[asdict(b) for b in instance.obstacles],
        object_size=[R.OBJECT_WIDTH, R.OBJECT_HEIGHT], robot_radius=R.ROBOT_RADIUS,
        object_polygon=instance.object_polygon,
        initial=dict(object=asdict(instance.object_pose),
                     robots=[asdict(p) for p in instance.robot_poses]))


def testbed_scenes():
    return {name: task.instance for name, task in task_catalog().items()}
