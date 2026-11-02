"""The 360 degree range scanner: 72 rays, 1.20 m, one scan per control period.

The scanner is a **simulation**: it is built from the true scene geometry (privileged), and
it returns only

* the nearest hit distance, or "no return" beyond the range,
* the observable semantic class of the hit: ``object`` / ``static`` / ``teammate``,

and nothing else. In particular it never returns a face id, an object centre, a hidden
corner, an un-hit contour, or any identifier of a specific obstacle. Class identity is a
declared sensing assumption of this round; the raw occluder label used to build the scan is
kept on a diagnostic-only field and is never assembled into a controller input.

The visible surface patch is fitted from the returns themselves (least squares on the
returned points), so the normal is an *estimate* with a validity flag and a quality number,
not a read of the engine's true normal.
"""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as G
from .spec import (
    BOUNDARY_RADIUS,
    LIDAR_RANGE,
    LIDAR_RAYS,
    MAX_FIT_RMS,
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    PATCH_GAP_TOLERANCE,
    PATCH_MIN_POINTS,
    PATCH_RELATIVE_FACTOR,
    ROBOT_RADIUS,
)

CATEGORY_NONE = 0
CATEGORY_OBJECT = 1
CATEGORY_STATIC = 2
CATEGORY_TEAMMATE = 3
CATEGORY_NAMES: Dict[int, str] = {0: "none", 1: "object", 2: "static", 3: "teammate"}

Vec = Tuple[float, float]


@dataclass(frozen=True)
class LidarScan:
    """One complete scan in the robot frame plus the world-frame ray angles."""

    robot_id: int
    tick: int
    sample_time: float
    origin: Vec
    heading: float
    ray_angles: Tuple[float, ...]
    distances: Tuple[float, ...]         # 0.0 where there is no return
    valid: Tuple[bool, ...]
    category: Tuple[int, ...]
    #: diagnostic only: the occluder label the simulation used. Never a controller input.
    raw_labels: Tuple[Optional[str], ...] = ()

    def arrays(self) -> Dict[str, np.ndarray]:
        return {"distance": np.asarray(self.distances, dtype=np.float32),
                "valid": np.asarray(self.valid, dtype=np.uint8),
                "category": np.asarray(self.category, dtype=np.uint8)}

    def hit_points_world(self) -> List[Vec]:
        out: List[Vec] = []
        for index, is_valid in enumerate(self.valid):
            if not is_valid:
                continue
            angle = self.ray_angles[index]
            out.append((self.origin[0] + self.distances[index] * math.cos(angle),
                        self.origin[1] + self.distances[index] * math.sin(angle)))
        return out

    def nearest(self, category: int) -> Optional[Tuple[float, Vec, int]]:
        best_index = None
        for index, is_valid in enumerate(self.valid):
            if not is_valid or self.category[index] != category:
                continue
            if best_index is None or self.distances[index] < self.distances[best_index]:
                best_index = index
        if best_index is None:
            return None
        angle = self.ray_angles[best_index]
        distance = self.distances[best_index]
        point = (self.origin[0] + distance * math.cos(angle),
                 self.origin[1] + distance * math.sin(angle))
        return (distance, point, best_index)


@dataclass(frozen=True)
class SurfacePatch:
    """One fitted surface fragment of the object, from the robot's own returns."""

    points_robot: Tuple[Vec, ...]
    points_world: Tuple[Vec, ...]
    #: perpendicular distance from the robot centre to the fitted surface line
    distance: float
    #: distance to the centroid of the returns (longer whenever the surface is seen obliquely)
    centroid_distance: float
    #: distance to the nearest individual return: the physical clearance to the surface, and
    #: the only one of the three that stays correct when the perpendicular foot leaves the
    #: observed segment (i.e. right at a corner)
    nearest_distance: float
    normal_world: Vec          # points from the robot towards the surface
    tangent_world: Vec         # rotate90(normal_world), in the global CCW sense
    confidence: float
    arc_rays: int
    span_m: float
    rms_m: float
    valid: bool

    @property
    def normal_outward(self) -> Vec:
        """Outward normal of the *object* surface at the nearest point."""

        return (-self.normal_world[0], -self.normal_world[1])

    def nearest_point_world(self) -> Vec:
        return self.points_world[0]


def _segment_hit(origin: Vec, direction: Vec, start: Vec, end: Vec) -> Optional[float]:
    return G.ray_segment(origin, direction, start, end)




def _object_polygon_world(object_pose, local_polygon=None):
    if local_polygon is None:
        return G.box_polygon(object_pose[0], object_pose[1], object_pose[2],
                             OBJECT_WIDTH / 2.0, OBJECT_HEIGHT / 2.0)
    c, sn = math.cos(object_pose[2]), math.sin(object_pose[2])
    return tuple((object_pose[0] + c*float(v[0]) - sn*float(v[1]),
                  object_pose[1] + sn*float(v[0]) + c*float(v[1])) for v in local_polygon)

def _occluder_geometry(arena: Tuple[float, float], obstacles: Sequence[G.Poly],
                       object_poly: G.Poly, robots: Sequence[Tuple[float, ...]],
                       robot_id: int):
    """The list a ray is tested against: static first, then the object, then the teammate."""

    entries: List[Tuple[str, int, object]] = []
    width, height = arena
    walls = (((0.0, 0.0), (width, 0.0)), ((width, 0.0), (width, height)),
             ((width, height), (0.0, height)), ((0.0, height), (0.0, 0.0)))
    for start, end in walls:
        entries.append(("static", CATEGORY_STATIC, ("seg", start, end, BOUNDARY_RADIUS)))
    for polygon in obstacles:
        entries.append(("static", CATEGORY_STATIC, ("poly", polygon)))
    entries.append(("object", CATEGORY_OBJECT, ("poly", object_poly)))
    for other_id, state in enumerate(robots):
        if other_id == robot_id:
            continue
        entries.append(("teammate", CATEGORY_TEAMMATE,
                        ("circle", (state[0], state[1]), ROBOT_RADIUS)))
    return entries


def scan_lidar(arena: Tuple[float, float], obstacles: Sequence[G.Poly],
               object_pose: Tuple[float, float, float],
               robots: Sequence[Tuple[float, ...]], robot_id: int, tick: int,
               sample_time: float, scan_range: float = LIDAR_RANGE,
               scan_rays: int = LIDAR_RAYS,
               object_angle_override: Optional[float] = None,
               object_polygon: Optional[tuple] = None) -> LidarScan:
    """One full scan of ``robot_id``. ``object_pose`` and ``robots`` are privileged truth.

    ``object_angle_override`` exists for the information-boundary counterfactual test: it
    changes the hidden object orientation while every legal measurement from the *previous*
    pose is held fixed. It is never used by a controller.
    """

    if object_angle_override is not None:
        object_pose = (object_pose[0], object_pose[1], object_angle_override)
    object_poly = _object_polygon_world(object_pose, object_polygon)
    robot = robots[robot_id]
    origin = (robot[0], robot[1])
    heading = robot[2]
    entries = _occluder_geometry(arena, obstacles, object_poly, robots, robot_id)

    angles: List[float] = []
    distances: List[float] = []
    valid: List[bool] = []
    categories: List[int] = []
    labels: List[Optional[str]] = []
    for index in range(scan_rays):
        angle = heading + 2.0 * math.pi * index / scan_rays
        direction = (math.cos(angle), math.sin(angle))
        best_distance: Optional[float] = None
        best_category = CATEGORY_NONE
        best_label: Optional[str] = None
        for name, category, shape in entries:
            if shape[0] == "seg":
                _, start, end, radius = shape
                # A wall is a capsule of ``radius`` around the segment. Its visible surface is
                # the offset *line* (a valid surface only where the offset point projects onto
                # the segment) plus the two end caps. The offset must be measured **along the
                # ray**: for a ray at incidence cos(theta) to the wall normal, the ray travels
                # radius / sin(angle between ray and segment) to reach the offset line. The
                # inherited module moved the hit back by ``radius`` along the ray, i.e. by the
                # perpendicular offset, which made oblique walls look up to a few centimetres
                # too far away; this scanner measures the crossing itself.
                ex, ey = end[0] - start[0], end[1] - start[1]
                length = math.hypot(ex, ey)
                t = None
                if length > 1e-12:
                    ex, ey = ex / length, ey / length
                    sine = abs(direction[0] * ey - direction[1] * ex)
                    base = _segment_hit(origin, direction, start, end)
                    if base is not None and sine > 1e-9:
                        candidate = base - radius / sine
                        point = (origin[0] + candidate * direction[0],
                                 origin[1] + candidate * direction[1])
                        along = ((point[0] - start[0]) * ex + (point[1] - start[1]) * ey) / length
                        if -1e-9 <= along <= 1.0 + 1e-9 and 0.0 <= candidate <= scan_range:
                            t = candidate
                for centre in (start, end):
                    cap = G.ray_circle(origin, direction, centre, radius)
                    if cap is not None and 0.0 <= cap <= scan_range and (t is None or cap < t):
                        t = cap
                if t is None:
                    continue
            elif shape[0] == "poly":
                hit = None
                for start, end in G.polygon_edges(shape[1]):
                    t = _segment_hit(origin, direction, start, end)
                    if t is not None and 0.0 <= t <= scan_range:
                        hit = t if hit is None else min(hit, t)
                if hit is None:
                    continue
                t = hit
            else:
                _, centre, radius = shape
                t = G.ray_circle(origin, direction, centre, radius)
                if t is None or t < 0.0 or t > scan_range:
                    continue
            if best_distance is None or t < best_distance:
                best_distance, best_category, best_label = t, category, name
        angles.append(angle)
        if best_distance is None:
            distances.append(0.0)
            valid.append(False)
            categories.append(CATEGORY_NONE)
            labels.append(None)
        else:
            distances.append(float(best_distance))
            valid.append(True)
            categories.append(int(best_category))
            labels.append(best_label)
    return LidarScan(robot_id=robot_id, tick=tick, sample_time=sample_time, origin=origin,
                     heading=heading, ray_angles=tuple(angles), distances=tuple(distances),
                     valid=tuple(valid), category=tuple(categories), raw_labels=tuple(labels))


@lru_cache(maxsize=128)
def _ray_directions(heading, scan_rays):
    """Immutable sensor-intrinsic math, not cached scene observations."""
    angles = tuple(heading + 2.*math.pi*i/scan_rays for i in range(scan_rays))
    directions = np.array([(math.cos(a), math.sin(a)) for a in angles])
    directions.setflags(write=False)
    return angles, directions


@lru_cache(maxsize=64)
def _primitive_groups(entry_ids):
    """Columns for each occluder are contiguous by construction."""
    ids = np.asarray(entry_ids)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    owners = ids[starts]
    starts.setflags(write=False)
    owners.setflags(write=False)
    return starts, owners


def scan_lidar_batch(arena, obstacles, object_pose, robots, robot_id, tick, sample_time,
                     scan_range=LIDAR_RANGE, scan_rays=LIDAR_RAYS, object_angle_override=None,
                     object_polygon=None):
    """NumPy batch of the SAME analytic rays, capsules, edges and ordered ties.

    CPU only. Real ray count, visibility, categories and timestamp are unchanged;
    this does not interpolate new evidence or ask the physics engine for normals.
    The scalar scanner is retained as the independent geometry reference.
    """
    angle=object_pose[2] if object_angle_override is None else object_angle_override
    poly=_object_polygon_world((object_pose[0],object_pose[1],angle), object_polygon)
    robot=robots[robot_id];origin=(robot[0],robot[1]);heading=robot[2]
    entries=_occluder_geometry(arena,obstacles,poly,robots,robot_id)
    # Match the scalar angle arithmetic/trig, including body-fixed heading.
    angles,direction=_ray_directions(heading,scan_rays)
    starts=[];ends=[];radii=[];segment_entries=[];circles=[];circle_entries=[]
    for k,(_,_,shape) in enumerate(entries):
        if shape[0]=='seg':
            _,a,b,r=shape;starts.append(a);ends.append(b);radii.append(r);segment_entries.append(k)
            circles.extend([(a,r),(b,r)]);circle_entries.extend([k,k])
        elif shape[0]=='poly':
            for a,b in G.polygon_edges(shape[1]):
                starts.append(a);ends.append(b);radii.append(0.);segment_entries.append(k)
        else:
            _,centre,r=shape;circles.append((centre,r));circle_entries.append(k)
    hits=np.full((scan_rays,len(entries)),np.inf)
    if starts:
        a=np.asarray(starts);b=np.asarray(ends);e=b-a;offset=a-np.asarray(origin)
        dx=direction[:,0,None];dy=direction[:,1,None]
        denom=dx*e[None,:,1]-dy*e[None,:,0]
        safe=np.where(np.abs(denom)>=1e-12,denom,1.)
        t=(offset[None,:,0]*e[None,:,1]-offset[None,:,1]*e[None,:,0])/safe
        u=(offset[None,:,0]*dy-offset[None,:,1]*dx)/safe
        valid=(np.abs(denom)>=1e-12)&(t>=0)&(u>=-1e-9)&(u<=1.+1e-9)
        lengths=np.array([math.hypot(x,y) for x,y in e]);unit=e/np.maximum(lengths[:,None],1e-12)
        sine=np.abs(dx*unit[None,:,1]-dy*unit[None,:,0]);wall=np.asarray(radii)>0
        candidate=t-np.asarray(radii)[None,:]/np.maximum(sine,1e-30)
        px=origin[0]+candidate*dx;py=origin[1]+candidate*dy
        along=((px-a[None,:,0])*unit[None,:,0]+(py-a[None,:,1])*unit[None,:,1])/np.maximum(lengths[None,:],1e-12)
        valid &= (~wall[None,:])|((lengths[None,:]>1e-12)&(sine>1e-9)&(along>=-1e-9)&(along<=1.+1e-9))
        t=np.where(wall[None,:],candidate,t)
        t=np.where(valid&(t>=0)&(t<=scan_range),t,np.inf)
        group_starts,owners=_primitive_groups(tuple(segment_entries))
        hits[:,owners]=np.minimum.reduceat(t,group_starts,axis=1)
    if circles:
        centres=np.asarray([c for c,r in circles]);r=np.asarray([r for c,r in circles])
        f=np.asarray(origin)-centres
        bb=2.*(direction[:,0,None]*f[None,:,0]+direction[:,1,None]*f[None,:,1])
        cc=f[:,0]*f[:,0]+f[:,1]*f[:,1]-r*r
        disc=bb*bb-4.*cc[None,:];root=np.sqrt(np.maximum(disc,0.))
        first=(-bb-root)/2.;second=(-bb+root)/2.
        t=np.where(first>=0,first,second)
        t=np.where((disc>=0)&(t>=0)&(t<=scan_range),t,np.inf)
        group_starts,owners=_primitive_groups(tuple(circle_entries))
        hits[:,owners]=np.minimum(hits[:,owners],np.minimum.reduceat(t,group_starts,axis=1))
    # argmin takes the first entry on ties, matching strict scalar '<'.
    nearest=np.argmin(hits,axis=1);distance=hits[np.arange(scan_rays),nearest];valid=np.isfinite(distance)
    return LidarScan(robot_id=robot_id,tick=tick,sample_time=sample_time,origin=origin,heading=heading,
        ray_angles=angles,distances=tuple(np.where(valid,distance,0.).tolist()),
        valid=tuple(valid.tolist()),category=tuple(entries[k][1] if v else CATEGORY_NONE for k,v in zip(nearest,valid)),
        raw_labels=tuple(entries[k][0] if v else None for k,v in zip(nearest,valid)))


def _fit_line(points: Sequence[Vec]):
    count = len(points)
    cx = sum(p[0] for p in points) / count
    cy = sum(p[1] for p in points) / count
    sxx = sum((p[0] - cx) ** 2 for p in points) / count
    syy = sum((p[1] - cy) ** 2 for p in points) / count
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in points) / count
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    normal = (-math.sin(theta), math.cos(theta))
    residuals = [abs((p[0] - cx) * normal[0] + (p[1] - cy) * normal[1]) for p in points]
    rms = math.sqrt(sum(value * value for value in residuals) / count)
    direction = (math.cos(theta), math.sin(theta))
    projections = [(p[0] - cx) * direction[0] + (p[1] - cy) * direction[1] for p in points]
    span = float(max(projections) - min(projections))
    return cx, cy, theta, rms, residuals, span


def _build_patch(points: Sequence[Vec], cx: float, cy: float, theta: float, rms: float,
                 span: float, heading: float, origin: Vec) -> SurfacePatch:
    direction = (math.cos(theta), math.sin(theta))
    normal_robot = (-direction[1], direction[0])
    if normal_robot[0] * cx + normal_robot[1] * cy < 0.0:
        normal_robot = (-normal_robot[0], -normal_robot[1])
    cos, sin = math.cos(heading), math.sin(heading)
    normal_world = G.unit((normal_robot[0] * cos - normal_robot[1] * sin,
                           normal_robot[0] * sin + normal_robot[1] * cos))
    tangent_world = G.rotate90(normal_world)
    points_world = tuple(
        (origin[0] + point[0] * cos - point[1] * sin,
         origin[1] + point[0] * sin + point[1] * cos)
        for point in points)
    count = len(points)
    confidence = (min(1.0, max(0.0, (count - 2) / 6.0))
                  * min(1.0, 0.3 + span / 0.05)
                  / (1.0 + rms / 0.02))
    return SurfacePatch(points_robot=tuple(points), points_world=points_world,
                        distance=float(abs(cx * normal_robot[0] + cy * normal_robot[1])),
                        centroid_distance=float(math.hypot(cx, cy)),
                        nearest_distance=float(min(math.hypot(p[0], p[1]) for p in points)),
                        normal_world=normal_world,
                        tangent_world=tangent_world, confidence=float(confidence),
                        arc_rays=count, span_m=float(span), rms_m=float(rms),
                        valid=bool(count >= PATCH_MIN_POINTS and rms <= MAX_FIT_RMS))


FittedRun = Tuple[List[Vec], float, float, float, float, float]


def _pack(points: List[Vec], fit) -> FittedRun:
    cx, cy, theta, rms, _residuals, span = fit
    return (points, cx, cy, theta, rms, span)


def _split_runs(points: Sequence[Vec], min_points: int, max_rms: float) -> List[FittedRun]:
    """Piecewise-linear segmentation of one run of returns.

    A single ray run can cover two faces of the box when a convex corner is the nearest point
    (the range then rises on both sides of the corner). The split point is therefore chosen by
    exhaustive search over admissible indices, minimising the worst residual of the two
    pieces, and the search recurses until every piece is straight or too short to split.
    """

    stack: List[List[Vec]] = [list(points)]
    out: List[FittedRun] = []
    while stack:
        current = stack.pop()
        count = len(current)
        if count < min_points:
            continue
        cx, cy, theta, rms, _, span = _fit_line(current)
        if rms <= max_rms or count < 2 * min_points:
            out.append((current, cx, cy, theta, rms, span))
            continue
        best: Optional[Tuple[float, int, object, object]] = None
        for split in range(min_points, count - min_points + 1):
            left = _fit_line(current[:split])
            right = _fit_line(current[split:])
            score = max(left[3], right[3])
            if best is None or score < best[0]:
                best = (score, split, left, right)
        if best is None:
            out.append((current, cx, cy, theta, rms, span))
            continue
        score, split, left, right = best
        if score <= max_rms:
            out.append(_pack(current[:split], left))
            out.append(_pack(current[split:], right))
        else:
            stack.append(current[:split])
            stack.append(current[split:])
    return out


def scan_patches(scan: LidarScan, robot_xy: Vec,
                 min_points: int = PATCH_MIN_POINTS,
                 max_rms: float = MAX_FIT_RMS) -> List[SurfacePatch]:
    """Group the object returns of one scan into fitted surface fragments.

    Grouping is purely geometric: angular adjacency of the ``object`` class plus range
    continuity. The continuity band is relative to the range, because a planar face seen at a
    grazing angle legitimately changes range by ``d * dtheta * tan(phi)`` between neighbouring
    rays; an absolute band would shred such a face. A single linear fit that still bends
    (large residual) is split at its worst point, so a run that spans a corner becomes two
    faces instead of one bent one. No face identity is used anywhere.
    """

    count = len(scan.valid)
    mask = [bool(scan.valid[i]) and scan.category[i] == CATEGORY_OBJECT
            for i in range(count)]

    if not any(mask):
        return []
    # incoming[i] is the same adjacency predicate for (i-1, i). Compute
    # it once instead of repeating it during run traversal.
    incoming = [False] * count
    angular_step = 2.0 * math.pi / count
    for second in range(count):
        first = (second - 1) % count
        if mask[first] and mask[second]:
            average = 0.5 * (scan.distances[first] + scan.distances[second])
            band = PATCH_GAP_TOLERANCE + PATCH_RELATIVE_FACTOR * average * angular_step
            incoming[second] = abs(scan.distances[second] - scan.distances[first]) <= band
    breaks = [index for index in range(count) if not incoming[index]]
    runs: List[List[int]] = []
    if not breaks:
        runs.append(list(range(count)))
    else:
        for start in breaks:
            # An invalid ray is always an isolated one-point run, which the
            # fitter would immediately reject. Avoid building its zero point.
            if not mask[start] and min_points > 1:
                continue
            run = [start]
            offset = 1
            while offset < count and incoming[(start + offset) % count]:
                run.append((start + offset) % count)
                offset += 1
            runs.append(run)

    patches: List[SurfacePatch] = []
    for run in runs:
        points: List[Vec] = []
        for index in run:
            relative = scan.ray_angles[index] - scan.heading
            points.append((scan.distances[index] * math.cos(relative),
                           scan.distances[index] * math.sin(relative)))
        for fitted in _split_runs(points, min_points, max_rms):
            current, cx, cy, theta, rms, span = fitted
            patches.append(_build_patch(current, cx, cy, theta, rms, span,
                                        scan.heading, robot_xy))
    patches.sort(key=lambda patch: patch.distance)
    return patches


def best_patch(patches: Sequence[SurfacePatch]) -> Optional[SurfacePatch]:
    return patches[0] if patches else None
