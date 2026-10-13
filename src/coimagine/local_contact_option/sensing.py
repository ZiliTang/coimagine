"""Ideal local surface sensing and the controller-side observation whitelist.

Two strictly separated layers live here:

* :func:`scan_surface` is the **sensing simulation**. It may read the physics truth to produce
  range returns, because that is what a real sensor would be measuring; it returns only local,
  robot-frame measurements.
* :class:`LocalContactObservation` is the only thing a controller may see. It has no field
  that carries the object's global pose, angle, velocity, corner list, face identity or true
  local contact coordinate. ``tests/test_local_contact_option.py`` mutates the hidden object
  truth while holding this object fixed and checks that the commanded force does not move.

The scan is 72 uniformly spaced rays out to 0.6 m in the robot frame. Boundary walls, static
obstacles, the teammate and the target object all occlude. A surface patch needs at least
three returns before it has a normal; a robot that cannot see a usable patch must stop and
report ``surface_unobserved`` instead of inventing a direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec = Tuple[float, float]


# ---------------------------------------------------------------------------
# geometry primitives (analytic, no engine calls, deterministic)
# ---------------------------------------------------------------------------


def _ray_segment(
    ox: float, oy: float, dx: float, dy: float, ax: float, ay: float, bx: float, by: float
) -> Optional[float]:
    """Smallest ``t >= 0`` with ``o + t*d`` on the segment, or ``None``."""

    ex, ey = bx - ax, by - ay
    denom = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return None
    t = ((ax - ox) * ey - (ay - oy) * ex) / denom
    u = ((ax - ox) * dy - (ay - oy) * dx) / denom
    if t >= 0.0 and -1e-9 <= u <= 1.0 + 1e-9:
        return t
    return None


def _ray_circle(ox: float, oy: float, dx: float, dy: float, cx: float, cy: float, r: float) -> Optional[float]:
    fx, fy = ox - cx, oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    for t in ((-b - root) / 2.0, (-b + root) / 2.0):
        if t >= 0.0:
            return t
    return None


def _box_edges(box) -> List[Tuple[Vec, Vec]]:
    cos, sin = math.cos(box.angle), math.sin(box.angle)
    corners = [
        (
            box.cx + sx * box.half_w * cos - sy * box.half_h * sin,
            box.cy + sx * box.half_w * sin + sy * box.half_h * cos,
        )
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return [(corners[index], corners[(index + 1) % 4]) for index in range(4)]


@dataclass(frozen=True)
class Occluder:
    """One thing a ray can hit. ``target`` marks the single object of this round."""

    label: str
    kind: str            # "boundary" | "box" | "circle"
    target: bool = False
    edges: Tuple[Tuple[Vec, Vec], ...] = ()
    centre: Vec = (0.0, 0.0)
    radius: float = 0.0
    boundary_offset: float = 0.0


def _occluders(scene, object_pose, object_size, robots, robot_id: int) -> List[Occluder]:
    width, height = scene.arena
    boundary_radius = 0.05
    out: List[Occluder] = []
    for index, (start, end) in enumerate((
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    )):
        out.append(Occluder(f"boundary{index}", "boundary", False, ((start, end),),
                            boundary_offset=boundary_radius))
    for index, box in enumerate(scene.obstacles):
        out.append(Occluder(f"obstacle{index}", "box", False, tuple(_box_edges(box))))
    half_w, half_h = object_size[0] / 2.0, object_size[1] / 2.0
    from ..continuous2d.scenes import Box

    object_box = Box(object_pose[0], object_pose[1], half_w, half_h, object_pose[2])
    out.append(Occluder("target_object", "box", True, tuple(_box_edges(object_box))))
    for other_id, state in enumerate(robots):
        if other_id == robot_id:
            continue
        out.append(Occluder(f"teammate{other_id}", "circle", False,
                            centre=(state[0], state[1]), radius=scene.robot_radius))
    return out


def cast_ray(ox: float, oy: float, dx: float, dy: float, limit: float,
             occluders: Sequence[Occluder]) -> Optional[Tuple[float, str, bool]]:
    best: Optional[Tuple[float, str, bool]] = None
    for occluder in occluders:
        if occluder.kind == "boundary":
            for start, end in occluder.edges:
                t = _ray_segment(ox, oy, dx, dy, start[0], start[1], end[0], end[1])
                if t is None:
                    continue
                # the boundary is a capsule: the visible surface is its inner offset
                t = t - occluder.boundary_offset / max(1e-9, math.hypot(dx, dy))
                if t < 0.0 or t > limit:
                    continue
                if best is None or t < best[0]:
                    best = (t, occluder.label, occluder.target)
        elif occluder.kind == "box":
            for start, end in occluder.edges:
                t = _ray_segment(ox, oy, dx, dy, start[0], start[1], end[0], end[1])
                if t is None or t < 0.0 or t > limit:
                    continue
                if best is None or t < best[0]:
                    best = (t, occluder.label, occluder.target)
        elif occluder.kind == "circle":
            t = _ray_circle(ox, oy, dx, dy, occluder.centre[0], occluder.centre[1], occluder.radius)
            if t is None or t < 0.0 or t > limit:
                continue
            if best is None or t < best[0]:
                best = (t, occluder.label, occluder.target)
    return best


# ---------------------------------------------------------------------------
# local patch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalPatch:
    """One observed surface fragment, entirely in the robot frame plus a world normal.

    ``points_robot`` are the range returns; ``segment_robot`` is the least-squares segment
    through them; ``normal_world`` points from the robot towards the surface (the direction
    the object lies in, i.e. "into the object" along the approach direction). There is no
    face identity here: association across scans is done by observed distance and normal
    continuity only.
    """

    points_robot: Tuple[Vec, ...]
    segment_robot: Tuple[Vec, Vec]
    distance: float
    normal_world: Vec
    tangent_world: Vec
    confidence: float
    timestamp: float
    is_target: bool
    arc_rays: int
    occluder_labels: Tuple[str, ...]

    def centroid_robot(self) -> Vec:
        count = len(self.points_robot)
        return (sum(p[0] for p in self.points_robot) / count,
                sum(p[1] for p in self.points_robot) / count)


def _fit_patch(points_robot: Sequence[Vec], robot_angle: float, timestamp: float,
               is_target: bool, arc_rays: int, labels: Sequence[str]) -> LocalPatch:
    count = len(points_robot)
    cx = sum(p[0] for p in points_robot) / count
    cy = sum(p[1] for p in points_robot) / count
    sxx = sum((p[0] - cx) ** 2 for p in points_robot) / count
    syy = sum((p[1] - cy) ** 2 for p in points_robot) / count
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in points_robot) / count
    # principal direction of the point set
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    direction = (math.cos(theta), math.sin(theta))
    projections = [(p[0] - cx) * direction[0] + (p[1] - cy) * direction[1] for p in points_robot]
    low, high = min(projections), max(projections)
    segment = (
        (cx + direction[0] * low, cy + direction[1] * low),
        (cx + direction[0] * high, cy + direction[1] * high),
    )
    distance = math.hypot(cx, cy)
    normal_robot = (-direction[1], direction[0])
    if normal_robot[0] * cx + normal_robot[1] * cy < 0.0:
        normal_robot = (-normal_robot[0], -normal_robot[1])
    # Rotate the *fitted* robot-frame normal into the shared world frame. The robot's own yaw
    # is known; the object's orientation is not, and is never used here.
    cos, sin = math.cos(robot_angle), math.sin(robot_angle)
    normal_world = (
        normal_robot[0] * cos - normal_robot[1] * sin,
        normal_robot[0] * sin + normal_robot[1] * cos,
    )
    length = math.hypot(normal_world[0], normal_world[1])
    normal_world = (normal_world[0] / length, normal_world[1] / length)
    tangent_world = (-normal_world[1], normal_world[0])
    spread = math.sqrt(max(0.0, high - low))
    confidence = min(1.0, count / 6.0) * min(1.0, spread / 0.05 + 0.3)
    return LocalPatch(
        points_robot=tuple(points_robot),
        segment_robot=segment,
        distance=distance,
        normal_world=normal_world,
        tangent_world=tangent_world,
        confidence=float(confidence),
        timestamp=timestamp,
        is_target=is_target,
        arc_rays=arc_rays,
        occluder_labels=tuple(labels),
    )


def scan_surface(
    scene,
    object_pose: Tuple[float, float, float],
    object_size: Tuple[float, float],
    robots: Sequence[Tuple[float, float, float, float, float, float]],
    robot_id: int,
    *,
    scan_range: float,
    scan_rays: int,
    timestamp: float,
    min_points: int = 3,
    gap_tolerance: float = 0.03,
    noise_sigma: float = 0.0,
    noise_rng=None,
) -> List[LocalPatch]:
    """Return the local surface patches visible from ``robot_id``.

    Grouping uses only angular adjacency and range continuity, so a patch is "a run of
    neighbouring rays that stopped at about the same distance", not a labelled face.
    """

    robot = robots[robot_id]
    ox, oy, oangle = robot[0], robot[1], robot[2]
    occluders = _occluders(scene, object_pose, object_size, robots, robot_id)
    hits: List[Optional[Tuple[float, str, bool]]] = []
    for index in range(scan_rays):
        angle = oangle + 2.0 * math.pi * index / scan_rays
        dx, dy = math.cos(angle), math.sin(angle)
        hit = cast_ray(ox, oy, dx, dy, scan_range, occluders)
        if hit is not None and noise_sigma > 0.0 and noise_rng is not None:
            hit = (max(0.0, hit[0] + noise_rng.gauss(0.0, noise_sigma)), hit[1], hit[2])
        hits.append(hit)

    patches: List[LocalPatch] = []

    def _linked(first: int, second: int) -> bool:
        """Are rays ``first`` and ``second`` (adjacent round the circle) one run?"""
        a, b = hits[first], hits[second]
        if a is None or b is None:
            return False
        return abs(b[0] - a[0]) <= gap_tolerance

    # The scanner is a full 360 degree ring, so runs must be grouped *circularly*. Grouping
    # them linearly split any patch that straddles the 0/2pi boundary: a robot facing the
    # 0.25 m end face of the object at 0.18 m saw 14 rays, but they arrived as runs of 8 and 6
    # (a 0.088 m "observed span") instead of one 0.2 m patch, and the surface-position arm then
    # reported patch_support_lost against a span that was an artefact of the index origin.
    # Start the sweep at a genuine break so the remaining runs are linear and complete.
    start = 0
    for index in range(scan_rays):
        if not _linked((index - 1) % scan_rays, index):
            start = index
            break
    visited = 0
    while visited < scan_rays:
        index = (start + visited) % scan_rays
        if hits[index] is None:
            visited += 1
            continue
        length = 1
        while length < scan_rays - visited and _linked((start + visited + length - 1) % scan_rays,
                                                       (start + visited + length) % scan_rays):
            length += 1
        run = [hits[(start + visited + offset) % scan_rays] for offset in range(length)]
        if len(run) >= min_points:
            points_robot: List[Vec] = []
            for offset in range(length):
                angle = oangle + 2.0 * math.pi * ((start + visited + offset) % scan_rays) / scan_rays
                dx, dy = math.cos(angle), math.sin(angle)
                distance = run[offset][0]  # type: ignore[index]
                world_x = ox + dx * distance
                world_y = oy + dy * distance
                cos, sin = math.cos(-oangle), math.sin(-oangle)
                rx, ry = world_x - ox, world_y - oy
                points_robot.append((rx * cos - ry * sin, rx * sin + ry * cos))
            patches.append(_fit_patch(
                points_robot, oangle, timestamp, bool(run[0][2]), len(run), [run[0][1]],
            ))
        visited += length
    return patches


def best_target_patch(patches: Sequence[LocalPatch]) -> Optional[LocalPatch]:
    candidates = [patch for patch in patches if patch.is_target]
    if not candidates:
        return None
    return min(candidates, key=lambda patch: patch.distance)


# ---------------------------------------------------------------------------
# the observation whitelist
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyKinematics:
    """Own or teammate kinematics: global pose and velocity. Never the object's."""

    x: float
    y: float
    angle: float
    vx: float
    vy: float
    omega: float


@dataclass(frozen=True)
class TeammateMessage:
    """What the teammate shares: its own kinematics, stage and contact flag only."""

    robot_id: int
    kinematics: BodyKinematics
    stage: str
    ready: bool
    contact: bool
    normal_world: Optional[Vec] = None
    reference_velocity_world: Optional[Vec] = None


@dataclass(frozen=True)
class LocalContactObservation:
    """The complete, closed input set of a local-contact controller.

    Deliberately absent: object position, object angle, object velocity, object corners,
    object face id, true object-local contact coordinates. ``contact_point_robot`` is the
    sensor's own reading in the robot frame, not a point expressed in object coordinates.
    """

    robot_id: int
    tick: int
    time: float
    #: Simulation time at which this observation was **sampled** (not when it was delivered).
    sample_time: float
    own: BodyKinematics
    teammate: Optional[TeammateMessage]
    patch: Optional[LocalPatch]
    contacting: bool
    contact_point_robot: Optional[Vec]
    contact_force_world: Vec
    contact_impulse: float
    arena: Tuple[float, float]
    robot_radius: float
    force_limit: float
    skill: Dict[str, float] = field(default_factory=dict)
    stage: str = "approach"
    surface_speed_estimate: Optional[float] = None
    surface_speed_signed: Optional[float] = None
    surface_speed_matches: int = 0
    surface_unavailable_reason: Optional[str] = None
    flags: Tuple[str, ...] = ()

    # -- convenience -------------------------------------------------------
    def unit_normal_world(self) -> Optional[Vec]:
        if self.patch is None:
            return None
        return self.patch.normal_world

    def unit_tangent_world(self) -> Optional[Vec]:
        if self.patch is None:
            return None
        return self.patch.tangent_world

    def forbidden_fields(self) -> Tuple[str, ...]:
        """Names a controller must never find on its observation."""

        banned = (
            "object", "object_state", "object_pose", "object_position", "object_velocity",
            "object_angle", "object_omega", "object_corners", "object_size", "face_id",
            "object_local_contact", "true_state", "env", "environment", "backend", "space",
        )
        return tuple(name for name in banned if hasattr(self, name))
