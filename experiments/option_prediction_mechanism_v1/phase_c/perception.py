"""Declared sensing model for the phase-C mechanism experiment.

The observation below is the *complete* allowed shareable input of a phase-C decision.  It
is deliberately built from raw measurements, not from a compressed feature vector:

* both robots' own global poses and velocities (these are known and shareable, and are the
  reason a *pose* difference between two worlds could never be a valid ambiguity);
* the raw local range scan of each robot: one range and one hit label per ray, with an
  explicit validity flag per ray and a whole-scan valid flag;
* the fitted local surface patch of each robot (points, least-squares segment, distance,
  world normal, tangent, confidence, arc length, occluder label);
* contact readings: whether the robot touches the object, the contact point in the robot
  frame, the contact normal and the applied contact force;
* the known static map (arena size, obstacle boxes, goal box, object size) -- identical in
  the two worlds by construction;
* the joint command that produced the current state (for the "history" column).

Sensing geometry is the *existing* frozen configuration (``SCAN_RANGE = 0.6`` m,
``SCAN_RAYS = 72``, ideal sensing, noise off) with one explicitly declared narrowing: a
forward sensing cone of half-angle 60 degrees in the robot frame, which removes the arena
behind each robot.  This is declared here, recorded in the result JSON, and is **not** used
to crop object endpoints: the mechanism object is long enough that both short faces are far
outside the 0.6 m range in both worlds, and that is verified per candidate pair.

No controller reads this module at execution time.  It is an offline measurement of what the
robots are allowed to share, which is exactly what CONTRACT section 3.1 asks to compare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from coimagine.continuous2d.backend import BodyState
from coimagine.continuous2d.scenes import SceneSpec
from coimagine.local_contact_option.sensing import (
    LocalPatch,
    Occluder,
    _occluders,
    best_target_patch,
    cast_ray,
)

#: Frozen local-scan constants, imported by value so the record states what was used.
SCAN_RANGE = 0.6
SCAN_RAYS = 72
RANGE_NOISE_SIGMA = 0.002  # m, only used by the explicit noise variant
#: Declared forward sensing cone: +/- this half-angle around the robot's own yaw.
FOV_HALF_ANGLE = math.radians(60.0)
#: A patch needs this many returns, matching ``MIN_POINTS_FOR_NORMAL``.
MIN_POINTS_FOR_NORMAL = 3
PATCH_GAP_TOLERANCE = 0.03

Vec = Tuple[float, float]


@dataclass(frozen=True)
class RawScan:
    """One robot's raw scan: per-ray range/label/validity plus the fitted patch."""

    robot_id: int
    origin: Vec
    yaw: float
    #: Ray angles in the *robot* frame (fixed by the declared cone, not by the world).
    ray_angles: Tuple[float, ...]
    #: Range of the nearest hit, ``None`` when the ray left the range or the cone.
    ranges: Tuple[Optional[float], ...]
    hits: Tuple[Optional[str], ...]
    is_target: Tuple[bool, ...]
    valid: Tuple[bool, ...]
    patch: Optional[LocalPatch]
    patch_points_robot: Tuple[Vec, ...] = ()
    patch_segment_robot: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0))
    patch_distance: float = float("nan")
    patch_normal_world: Vec = (float("nan"), float("nan"))
    patch_tangent_world: Vec = (float("nan"), float("nan"))
    patch_confidence: float = float("nan")
    patch_arc_rays: int = 0
    patch_span: float = float("nan")
    patch_occluder: str = ""
    #: Short faces of the target object inside the declared range: "none" means the observed
    #: edge is seen as an unbounded straight segment at this stance.
    short_face_visible: bool = False


@dataclass(frozen=True)
class RobotObservation:
    """Everything one robot may share at the decision time (cold start, no history)."""

    robot_id: int
    pose: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    scan: RawScan
    contacting: bool
    contact_point_robot: Optional[Vec]
    contact_normal_world: Optional[Vec]
    contact_force_world: Vec
    contact_impulse: float


@dataclass(frozen=True)
class SharedObservation:
    """The complete joint input available to both robots at a decision time."""

    tick: int
    time: float
    robots: Tuple[RobotObservation, RobotObservation]
    arena: Tuple[float, float]
    goal: Tuple[float, float, float, float]
    obstacles: Tuple[Tuple[float, float, float, float], ...]
    object_size: Tuple[float, float]
    history: Tuple[str, ...] = ()
    flags: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# ray casting with the declared cone
# ---------------------------------------------------------------------------


def _cone_angles(rays: int = SCAN_RAYS, half_angle: float = FOV_HALF_ANGLE) -> Tuple[float, ...]:
    """Uniform angles inside ``[-half_angle, +half_angle]`` in the robot frame."""

    if rays < 2:
        raise ValueError("need at least two rays to define a cone")
    return tuple(-half_angle + 2.0 * half_angle * index / (rays - 1) for index in range(rays))


def _cast_cone(
    scene: SceneSpec,
    object_pose: Tuple[float, float, float],
    object_size: Tuple[float, float],
    robots: Sequence[Tuple[float, float, float, float, float, float]],
    robot_id: int,
    *,
    scan_range: float = SCAN_RANGE,
    rays: int = SCAN_RAYS,
    noise_sigma: float = 0.0,
    noise_rng=None,
) -> Tuple[Tuple[Optional[float], ...], Tuple[Optional[str], ...], Tuple[bool, ...], Tuple[float, ...]]:
    robot = robots[robot_id]
    ox, oy, oangle = robot[0], robot[1], robot[2]
    occluders = _occluders(scene, object_pose, object_size, robots, robot_id)
    angles = _cone_angles(rays)
    ranges: List[Optional[float]] = []
    labels: List[Optional[str]] = []
    targets: List[bool] = []
    for relative in angles:
        angle = oangle + relative
        dx, dy = math.cos(angle), math.sin(angle)
        hit = cast_ray(ox, oy, dx, dy, scan_range, occluders)
        if hit is not None and noise_sigma > 0.0 and noise_rng is not None:
            hit = (max(0.0, hit[0] + noise_rng.gauss(0.0, noise_sigma)), hit[1], hit[2])
        if hit is None:
            ranges.append(None)
            labels.append(None)
            targets.append(False)
        else:
            ranges.append(float(hit[0]))
            labels.append(str(hit[1]))
            targets.append(bool(hit[2]))
    return tuple(ranges), tuple(labels), tuple(targets), angles


def _fit_patch_from_scan(
    ranges: Sequence[Optional[float]],
    targets: Sequence[bool],
    labels: Sequence[Optional[str]],
    angles: Sequence[float],
    yaw: float,
    timestamp: float,
    raw_robot_xy: Vec,
) -> Optional[LocalPatch]:
    """Fit a patch exactly like ``local_contact_option.sensing.scan_surface`` does.

    Reimplemented here over the cone (the frozen helper groups a full 360 degree ring).
    """

    runs: List[List[int]] = []
    current: List[int] = []
    for index, value in enumerate(ranges):
        if value is None:
            current = []
            continue
        if current and abs(value - ranges[current[-1]]) > PATCH_GAP_TOLERANCE:
            runs.append(current)
            current = []
        current.append(index)
    if current:
        runs.append(current)
    # Prefer the target run closest to the robot, matching ``best_target_patch``.
    fitted: List[LocalPatch] = []
    for run in runs:
        if len(run) < MIN_POINTS_FOR_NORMAL:
            continue
        if not targets[run[0]]:
            continue
        points_robot: List[Vec] = []
        for index in run:
            distance = ranges[index]
            angle = yaw + angles[index]
            world_x = raw_robot_xy[0] + math.cos(angle) * distance  # type: ignore[arg-type]
            world_y = raw_robot_xy[1] + math.sin(angle) * distance  # type: ignore[arg-type]
            rx, ry = world_x - raw_robot_xy[0], world_y - raw_robot_xy[1]
            cos, sin = math.cos(-yaw), math.sin(-yaw)
            points_robot.append((rx * cos - ry * sin, rx * sin + ry * cos))
        count = len(points_robot)
        cx = sum(p[0] for p in points_robot) / count
        cy = sum(p[1] for p in points_robot) / count
        sxx = sum((p[0] - cx) ** 2 for p in points_robot) / count
        syy = sum((p[1] - cy) ** 2 for p in points_robot) / count
        sxy = sum((p[0] - cx) * (p[1] - cy) for p in points_robot) / count
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
        cos, sin = math.cos(yaw), math.sin(yaw)
        normal_world = (
            normal_robot[0] * cos - normal_robot[1] * sin,
            normal_robot[0] * sin + normal_robot[1] * cos,
        )
        length = math.hypot(normal_world[0], normal_world[1])
        normal_world = (normal_world[0] / length, normal_world[1] / length)
        tangent_world = (-normal_world[1], normal_world[0])
        spread = math.sqrt(max(0.0, high - low))
        confidence = min(1.0, count / 6.0) * min(1.0, spread / 0.05 + 0.3)
        fitted.append(LocalPatch(
            points_robot=tuple(points_robot),
            segment_robot=segment,
            distance=distance,
            normal_world=normal_world,
            tangent_world=tangent_world,
            confidence=float(confidence),
            timestamp=timestamp,
            is_target=True,
            arc_rays=len(run),
            occluder_labels=(str(labels[run[0]]),),
        ))
    if not fitted:
        return None
    return min(fitted, key=lambda patch: patch.distance)


def _short_face_visible(
    scene: SceneSpec,
    object_pose: Tuple[float, float, float],
    object_size: Tuple[float, float],
    robots: Sequence[Tuple[float, float, float, float, float, float]],
    robot_id: int,
    ranges: Sequence[Optional[float]],
    labels: Sequence[Optional[str]],
    angles: Sequence[float],
) -> bool:
    """True when any cone ray hits a *short* (x-normal) face of the target object."""

    half_w, half_h = object_size[0] / 2.0, object_size[1] / 2.0
    cos, sin = math.cos(object_pose[2]), math.sin(object_pose[2])
    short_faces = []
    for sx in (-1.0, 1.0):
        corners = (
            (sx * half_w, -half_h),
            (sx * half_w, half_h),
        )
        short_faces.append(tuple(
            (object_pose[0] + dx * cos - dy * sin, object_pose[1] + dx * sin + dy * cos)
            for dx, dy in corners
        ))
    robot = robots[robot_id]
    for index, value in enumerate(ranges):
        if value is None or labels[index] != "target_object":
            continue
        angle = robot[2] + angles[index]
        px = robot[0] + math.cos(angle) * value
        py = robot[1] + math.sin(angle) * value
        for face in short_faces:
            (ax, ay), (bx, by) = face
            ex, ey = bx - ax, by - ay
            length = math.hypot(ex, ey)
            if length < 1e-12:
                continue
            # distance from the hit point to this short-face segment
            t = max(0.0, min(1.0, ((px - ax) * ex + (py - ay) * ey) / (length * length)))
            qx, qy = ax + t * ex, ay + t * ey
            if math.hypot(px - qx, py - qy) < 1e-6:
                return True
    return False


# ---------------------------------------------------------------------------
# the joint observation
# ---------------------------------------------------------------------------


def observe(
    scene: SceneSpec,
    object_pose: Tuple[float, float, float],
    robots: Sequence[BodyState],
    *,
    tick: int = 0,
    time: float = 0.0,
    contact_state: Optional[Sequence[Dict[str, object]]] = None,
    history: Sequence[str] = (),
    scan_range: float = SCAN_RANGE,
    rays: int = SCAN_RAYS,
    noise_sigma: float = 0.0,
    noise_rng=None,
) -> SharedObservation:
    """Compute the full allowed joint observation of both robots."""

    physics_object_size = (scene.object_width, scene.object_height)
    robot_tuples = tuple((state.x, state.y, state.angle, state.vx, state.vy, state.omega)
                         for state in robots)
    observations: List[RobotObservation] = []
    for robot_id in (0, 1):
        ranges, labels, targets, angles = _cast_cone(
            scene, object_pose, physics_object_size, robot_tuples, robot_id,
            scan_range=scan_range, rays=rays, noise_sigma=noise_sigma, noise_rng=noise_rng,
        )
        patch = _fit_patch_from_scan(
            ranges, targets, labels, angles, robots[robot_id].angle, time,
            (robots[robot_id].x, robots[robot_id].y),
        )
        visible_short = _short_face_visible(
            scene, object_pose, physics_object_size, robot_tuples, robot_id, ranges, labels, angles,
        )
        patch_points = patch.points_robot if patch is not None else ()
        patch_segment = patch.segment_robot if patch is not None else ((0.0, 0.0), (0.0, 0.0))
        scan = RawScan(
            robot_id=robot_id,
            origin=(robots[robot_id].x, robots[robot_id].y),
            yaw=robots[robot_id].angle,
            ray_angles=angles,
            ranges=ranges,
            hits=labels,
            is_target=targets,
            valid=tuple(value is not None for value in ranges),
            patch=patch,
            patch_points_robot=tuple(patch_points),
            patch_segment_robot=patch_segment,
            patch_distance=patch.distance if patch is not None else float("nan"),
            patch_normal_world=patch.normal_world if patch is not None else (float("nan"), float("nan")),
            patch_tangent_world=patch.tangent_world if patch is not None else (float("nan"), float("nan")),
            patch_confidence=patch.confidence if patch is not None else float("nan"),
            patch_arc_rays=patch.arc_rays if patch is not None else 0,
            patch_span=float("nan") if patch is None else _segment_length(patch.segment_robot),
            patch_occluder=patch.occluder_labels[0] if patch is not None else "",
            short_face_visible=visible_short,
        )
        contact = _contact_reading(contact_state or (), robot_id, patch, robots[robot_id])
        observations.append(RobotObservation(
            robot_id=robot_id,
            pose=(robots[robot_id].x, robots[robot_id].y, robots[robot_id].angle),
            velocity=(robots[robot_id].vx, robots[robot_id].vy, robots[robot_id].omega),
            scan=scan,
            contacting=contact["contacting"],
            contact_point_robot=contact["point"],
            contact_normal_world=contact["normal"],
            contact_force_world=contact["force"],
            contact_impulse=contact["impulse"],
        ))
    goal = scene.goal
    return SharedObservation(
        tick=tick,
        time=time,
        robots=(observations[0], observations[1]),
        arena=scene.arena,
        goal=(goal.cx, goal.cy, goal.half_w, goal.half_h),
        obstacles=tuple((box.cx, box.cy, box.half_w, box.half_h) for box in scene.obstacles),
        object_size=physics_object_size,
        history=tuple(history),
    )


def _segment_length(segment) -> float:
    (ax, ay), (bx, by) = segment
    return math.hypot(bx - ax, by - ay)


def _contact_reading(
    contacts: Sequence[Dict[str, object]],
    robot_id: int,
    patch: Optional[LocalPatch],
    robot: BodyState,
) -> Dict[str, object]:
    points = [
        entry for entry in contacts
        if entry.get("pair") == "robot_object" and int(entry.get("robot_id", -1)) == robot_id
    ]
    if not points:
        return {"contacting": False, "point": None, "normal": None, "force": (0.0, 0.0),
                "impulse": 0.0}
    # Single contact point in the robot frame, plus the sensed surface normal.
    best = min(points, key=lambda entry: float(entry.get("distance", 0.0)))
    px, py = float(best["point"][0]), float(best["point"][1])  # type: ignore[index]
    sin, cos = math.sin(-robot.angle), math.cos(-robot.angle)
    rx, ry = px - robot.x, py - robot.y
    return {
        "contacting": True,
        "point": (rx * cos - ry * sin, rx * sin + ry * cos),
        "normal": None if patch is None else patch.normal_world,
        "force": (0.0, 0.0),  # filled by the caller if it recorded the command
        "impulse": 0.0,
    }


# ---------------------------------------------------------------------------
# observation-identity comparison
# ---------------------------------------------------------------------------


def observation_differences(obs_a: SharedObservation, obs_b: SharedObservation) -> Dict[str, float]:
    """Maximum absolute difference per compared quantity (NaN-free: misses compare equal)."""

    diff: Dict[str, float] = {}

    def track(name: str, value: float) -> None:
        if math.isnan(value):
            return
        diff[name] = max(diff.get(name, 0.0), value)

    for robot_id in (0, 1):
        a, b = obs_a.robots[robot_id], obs_b.robots[robot_id]
        for index in range(3):
            track(f"robot{robot_id}.pose[{index}]", abs(a.pose[index] - b.pose[index]))
            track(f"robot{robot_id}.velocity[{index}]", abs(a.velocity[index] - b.velocity[index]))
        sa, sb = a.scan, b.scan
        track(f"robot{robot_id}.scan.origin", max(abs(sa.origin[0] - sb.origin[0]),
                                                  abs(sa.origin[1] - sb.origin[1])))
        track(f"robot{robot_id}.scan.yaw", abs(sa.yaw - sb.yaw))
        ranges_diff, ranges_mismatch = _optional_compare(sa.ranges, sb.ranges)
        diff[f"robot{robot_id}.scan.range"] = ranges_diff
        diff[f"robot{robot_id}.scan.range_mismatch"] = float(ranges_mismatch)
        diff[f"robot{robot_id}.scan.valid_mismatch"] = float(
            sum(1 for x, y in zip(sa.valid, sb.valid) if x != y)
        )
        diff[f"robot{robot_id}.scan.hit_label_mismatch"] = float(
            sum(1 for x, y in zip(sa.hits, sb.hits) if x != y)
        )
        diff[f"robot{robot_id}.scan.target_flag_mismatch"] = float(
            sum(1 for x, y in zip(sa.is_target, sb.is_target) if x != y)
        )
        diff[f"robot{robot_id}.scan.short_face_mismatch"] = float(
            int(sa.short_face_visible != sb.short_face_visible)
        )
        # patch
        diff[f"robot{robot_id}.patch.present_mismatch"] = float(
            int((sa.patch is None) != (sb.patch is None))
        )
        track(f"robot{robot_id}.patch.distance", abs(sa.patch_distance - sb.patch_distance))
        track(f"robot{robot_id}.patch.span", abs(sa.patch_span - sb.patch_span))
        track(f"robot{robot_id}.patch.arc_rays", abs(sa.patch_arc_rays - sb.patch_arc_rays))
        track(f"robot{robot_id}.patch.confidence", abs(sa.patch_confidence - sb.patch_confidence))
        for index in range(2):
            track(f"robot{robot_id}.patch.normal[{index}]",
                  abs(sa.patch_normal_world[index] - sb.patch_normal_world[index]))
            track(f"robot{robot_id}.patch.tangent[{index}]",
                  abs(sa.patch_tangent_world[index] - sb.patch_tangent_world[index]))
        points_diff, points_mismatch = _points_compare(sa.patch_points_robot, sb.patch_points_robot)
        diff[f"robot{robot_id}.patch.points_mismatch"] = float(points_mismatch)
        diff[f"robot{robot_id}.patch.points"] = points_diff
        seg_diff = max(
            abs(sa.patch_segment_robot[0][i] - sb.patch_segment_robot[0][i]) for i in range(2)
        )
        seg_diff = max(seg_diff, max(
            abs(sa.patch_segment_robot[1][i] - sb.patch_segment_robot[1][i]) for i in range(2)
        ))
        diff[f"robot{robot_id}.patch.segment"] = seg_diff
        diff[f"robot{robot_id}.patch.occluder_mismatch"] = float(
            int(sa.patch_occluder != sb.patch_occluder)
        )
        # contact
        diff[f"robot{robot_id}.contact.flag_mismatch"] = float(int(a.contacting != b.contacting))
        if a.contact_point_robot is not None and b.contact_point_robot is not None:
            track(f"robot{robot_id}.contact.point", max(
                abs(a.contact_point_robot[i] - b.contact_point_robot[i]) for i in range(2)
            ))
        if a.contact_normal_world is not None and b.contact_normal_world is not None:
            track(f"robot{robot_id}.contact.normal", max(
                abs(a.contact_normal_world[i] - b.contact_normal_world[i]) for i in range(2)
            ))
        track(f"robot{robot_id}.contact.impulse", abs(a.contact_impulse - b.contact_impulse))
    # map (must be bit-identical)
    diff["map.goal"] = max(abs(x - y) for x, y in zip(obs_a.goal, obs_b.goal))
    diff["map.arena"] = max(abs(x - y) for x, y in zip(obs_a.arena, obs_b.arena))
    diff["map.object_size"] = max(abs(x - y) for x, y in zip(obs_a.object_size, obs_b.object_size))
    obs_diff = 0.0
    for box_a, box_b in zip(obs_a.obstacles, obs_b.obstacles):
        obs_diff = max(obs_diff, max(abs(x - y) for x, y in zip(box_a, box_b)))
    diff["map.obstacles"] = obs_diff
    diff["map.obstacles_count_mismatch"] = float(
        abs(len(obs_a.obstacles) - len(obs_b.obstacles))
    )
    diff["history_mismatch"] = float(int(len(obs_a.history) != len(obs_b.history)))
    assert all(math.isfinite(value) for value in diff.values()), "non-finite difference reported"
    return diff


def _optional_compare(a: Sequence[Optional[float]], b: Sequence[Optional[float]]) -> Tuple[float, int]:
    """Max abs difference over entries present in **both**, plus a mismatch count.

    A quantity present in one world and absent in the other is a *mismatch*, not an infinite
    numeric difference; it is returned separately so no caller can see ``inf``.
    """

    if len(a) != len(b):
        return 0.0, abs(len(a) - len(b))
    worst = 0.0
    mismatch = 0
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        if x is None or y is None:
            mismatch += 1
            continue
        worst = max(worst, abs(x - y))
    return worst, mismatch


def _points_compare(a: Sequence[Vec], b: Sequence[Vec]) -> Tuple[float, int]:
    """Max abs difference over the paired prefix, plus a length-mismatch flag."""

    worst = 0.0
    for pa, pb in zip(a, b):
        worst = max(worst, abs(pa[0] - pb[0]), abs(pa[1] - pb[1]))
    return worst, abs(len(a) - len(b))


def observation_is_identical(diff: Dict[str, float], *, tolerance: float = 1e-12) -> bool:
    """Identity from finite quantities only: every mismatch count must be zero."""

    for name, value in diff.items():
        if not math.isfinite(value):
            return False
        if name.endswith("_mismatch"):
            if value != 0.0:
                return False
        elif value > tolerance:
            return False
    return True
