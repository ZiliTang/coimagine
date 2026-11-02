"""Polygon fixtures for synthetic visible-face selector tests (no physics)."""
from __future__ import annotations
import math
from typing import Tuple
import numpy as np

from coimagine.random_start.lidar import LidarScan, SurfacePatch
from coimagine.random_start.skills import RobotKinematics, SkillObservation
from experiments.option_rl_v13 import controller as C

Polygon = Tuple[Tuple[float, float], ...]

SHAPES = {
    'T': ((0., 0.), (1., 0.), (1., .30), (.65, .30), (.65, .85),
          (.35, .85), (.35, .30), (0., .30)),
    'triangle': ((0., 0.), (1., 0.), (.5, .85)),
    'octagon': tuple((.5 + .5*math.cos(2*math.pi*i/8),
                      .5 + .5*math.sin(2*math.pi*i/8)) for i in range(8)),
}


def edge_patch(a, b, robot, scale=1.0):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mid = .5*(a+b)
    d = b-a; span = float(np.linalg.norm(d))
    t = d/max(1e-12, span)
    n = np.array([-t[1], t[0]])
    if float(n @ (mid - np.asarray(robot, float))) < 0:
        n = -n
    pts = tuple(tuple(a + (b-a)*u) for u in (0., .25, .5, .75, 1.))
    return SurfacePatch(points_robot=pts, points_world=pts,
                        distance=float(np.linalg.norm(mid-robot)),
                        centroid_distance=float(np.linalg.norm(mid-robot)),
                        nearest_distance=float(np.linalg.norm(a-robot)),
                        normal_world=(float(n[0]), float(n[1])),
                        tangent_world=(float(t[0]), float(t[1])),
                        confidence=1., arc_rays=len(pts), span_m=span,
                        rms_m=.001, valid=True)


def polygon_observation(shape, side, robot_xy, teammate_xy=(0., 0.)):
    poly = SHAPES[shape] if isinstance(shape, str) else tuple(shape)
    patches = tuple(edge_patch(poly[i], poly[(i+1) % len(poly)], robot_xy)
                    for i in range(len(poly)))
    own = RobotKinematics(float(robot_xy[0]), float(robot_xy[1]), 0., 0., 0., 0.)
    mate = RobotKinematics(float(teammate_xy[0]), float(teammate_xy[1]), 0., 0., 0., 0.)
    scan = LidarScan(0, 0, 0., (0., 0.), 0., (), (), (), ())
    return SkillObservation(robot_id=side, tick=0, time=0., own=own, teammate=mate,
        scan=scan, patches=patches, patch=None, contacting_object=False,
        contact_point_robot=None, contacting_static=False, contacting_teammate=False,
        goal_xy=(.5, .5), goal_radius=.1, arena=(5., 4.), force_limit=40.,
        remaining_seconds=10., elapsed_seconds=0., static_nearest=None,
        teammate_nearest=None)
