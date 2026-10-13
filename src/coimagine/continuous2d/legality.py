"""Exact geometric legality of a scene's initial configuration.

The engine's ``contacts`` list is not a legality check: a body can start overlapping and the
solver may not report every pair, and the arena boundary is a capsule of radius 0.05, not
the line ``x = 0``. This module computes the signed clearance of every relevant pair from
the **scene geometry**:

* robot / object against each of the four boundary capsule surfaces;
* robot / object against each obstacle box;
* robot against the object polygon and against each other.

A non-contact pair must keep a uniform margin (default 0.005 m). Contacting pairs (e.g. a
robot resting on the object in an intended-contact stance) are not required to, but the
scene library's initial states are all non-contact.

:func:`minimal_robot_fix` moves the violating robots along the violated constraints'
inward normals until the margin is met, choosing the smallest displacement it finds by
iterating the constraints deterministically. It never consults success rates.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import _distance_to_box, _distance_to_polygon
from .scenes import Box, Pose, SceneSpec

Vec = Tuple[float, float]

#: Library version written into every scene and every result table.
SCENE_VERSION = "closeout_v2"
#: Uniform non-contact margin required of every initial state.
DEFAULT_MARGIN = 0.005
#: The arena boundary is a capsule of this radius centred on the arena edge.
BOUNDARY_RADIUS = 0.05


@dataclass(frozen=True)
class Clearance:
    pair: str
    a: str
    b: str
    clearance: float
    required: float
    satisfied: bool

    def as_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def _object_corners(scene: SceneSpec) -> Tuple[Vec, ...]:
    half_w = scene.object_width / 2.0
    half_h = scene.object_height / 2.0
    cos, sin = math.cos(scene.object_pose.angle), math.sin(scene.object_pose.angle)
    return tuple(
        (
            scene.object_pose.x + dx * cos - dy * sin,
            scene.object_pose.y + dx * sin + dy * cos,
        )
        for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
    )


def boundary_clearance(point: Vec, arena: Tuple[float, float]) -> float:
    """Distance from a point to the inner surface of the boundary capsule."""

    width, height = arena
    x, y = point
    return min(
        x - BOUNDARY_RADIUS,
        width - BOUNDARY_RADIUS - x,
        y - BOUNDARY_RADIUS,
        height - BOUNDARY_RADIUS - y,
    )


def scene_clearances(scene: SceneSpec, margin: float = DEFAULT_MARGIN) -> List[Clearance]:
    """Signed clearances of every pair, with the uniform non-contact margin applied."""

    radius = scene.robot_radius
    out: List[Clearance] = []
    for robot_id, pose in enumerate(scene.robot_poses):
        point = (pose.x, pose.y)
        out.append(Clearance(
            "robot_boundary", f"robot{robot_id}", "boundary",
            boundary_clearance(point, scene.arena) - radius, margin,
            boundary_clearance(point, scene.arena) - radius >= margin,
        ))
        for index, box in enumerate(scene.obstacles):
            clearance = _distance_to_box(point[0], point[1], box) - radius
            out.append(Clearance(
                "robot_obstacle", f"robot{robot_id}", f"obstacle{index}", clearance, margin,
                clearance >= margin,
            ))
        object_clearance = _distance_to_polygon(point[0], point[1], _object_corners(scene)) - radius
        out.append(Clearance(
            "robot_object", f"robot{robot_id}", "object", object_clearance, margin,
            object_clearance >= margin,
        ))
    # robot-robot
    first, second = scene.robot_poses
    distance = math.hypot(first.x - second.x, first.y - second.y) - 2 * radius
    out.append(Clearance("robot_robot", "robot0", "robot1", distance, margin, distance >= margin))
    # object against boundary and obstacles
    object_boundary = min(boundary_clearance(corner, scene.arena) for corner in _object_corners(scene))
    out.append(Clearance("object_boundary", "object", "boundary", object_boundary, margin, object_boundary >= margin))
    for index, box in enumerate(scene.obstacles):
        clearance = min(
            _distance_to_box(corner[0], corner[1], box) for corner in _object_corners(scene)
        )
        out.append(Clearance(
            "object_obstacle", "object", f"obstacle{index}", clearance, margin, clearance >= margin,
        ))
    return out


def legality_report(scene: SceneSpec, margin: float = DEFAULT_MARGIN) -> Dict[str, object]:
    clearances = [entry.as_dict() for entry in scene_clearances(scene, margin)]
    violations = [entry for entry in clearances if not entry["satisfied"]]
    return {
        "scene": scene.name,
        "scene_version": scene.scene_version,
        "content_hash": scene_content_hash(scene),
        "margin": margin,
        "legal": not violations,
        "min_clearance": min(entry["clearance"] for entry in clearances),
        "violations": violations,
        "clearances": clearances,
    }


def scene_content_hash(scene: SceneSpec) -> str:
    """Stable content hash over the geometry that decides the physics."""

    payload = {
        "name": scene.name,
        "category": scene.category,
        "variant": scene.variant,
        "arena": list(scene.arena),
        "obstacles": [dataclasses.asdict(box) for box in scene.obstacles],
        "goal": dataclasses.asdict(scene.goal),
        "object": dataclasses.asdict(scene.object_pose),
        "robots": [dataclasses.asdict(pose) for pose in scene.robot_poses],
        "object_size": [scene.object_width, scene.object_height],
        "robot_radius": scene.robot_radius,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _violation_normal(
    point: Vec, scene: SceneSpec, radius: float, margin: float
) -> Optional[Tuple[Vec, float]]:
    """Largest single violation of a robot at ``point``: (inward normal, push distance)."""

    best: Optional[Tuple[Vec, float]] = None
    x, y = point
    width, height = scene.arena
    boundary = [
        ((1.0, 0.0), (x - BOUNDARY_RADIUS) - radius),      # left
        ((-1.0, 0.0), (width - BOUNDARY_RADIUS - x) - radius),  # right
        ((0.0, 1.0), (y - BOUNDARY_RADIUS) - radius),      # bottom
        ((0.0, -1.0), (height - BOUNDARY_RADIUS - y) - radius),  # top
    ]
    for normal, clearance in boundary:
        if clearance < margin:
            push = margin - clearance
            if best is None or push > best[1]:
                best = (normal, push)

    def _consider(nearest: Vec, required: float) -> None:
        nonlocal best
        dx, dy = x - nearest[0], y - nearest[1]
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            normal = (1.0, 0.0)
        else:
            normal = (dx / distance, dy / distance)
        push = required - distance
        if push > 0 and (best is None or push > best[1]):
            best = (normal, push)

    for box in scene.obstacles:
        if _distance_to_box(x, y, box) < radius + margin:
            # nearest point on the box
            cos, sin = math.cos(-box.angle), math.sin(-box.angle)
            local_x = (x - box.cx) * cos - (y - box.cy) * sin
            local_y = (x - box.cx) * sin + (y - box.cy) * cos
            clamped_x = max(-box.half_w, min(box.half_w, local_x))
            clamped_y = max(-box.half_h, min(box.half_h, local_y))
            world_x = box.cx + clamped_x * math.cos(box.angle) - clamped_y * math.sin(box.angle)
            world_y = box.cy + clamped_x * math.sin(box.angle) + clamped_y * math.cos(box.angle)
            _consider((world_x, world_y), radius + margin)
    corners = _object_corners(scene)
    if _distance_to_polygon(x, y, corners) < radius + margin:
        # nearest point on the object polygon
        nearest = None
        best_distance = float("inf")
        for index, corner in enumerate(corners):
            nxt = corners[(index + 1) % len(corners)]
            from .geometry import _point_segment_distance

            distance = _point_segment_distance(x, y, corner[0], corner[1], nxt[0], nxt[1])
            if distance < best_distance:
                best_distance = distance
                t = 0.0 if distance < 1e-12 else max(
                    0.0,
                    min(
                        1.0,
                        ((x - corner[0]) * (nxt[0] - corner[0]) + (y - corner[1]) * (nxt[1] - corner[1]))
                        / max(1e-12, (nxt[0] - corner[0]) ** 2 + (nxt[1] - corner[1]) ** 2),
                    ),
                )
                nearest = (corner[0] + t * (nxt[0] - corner[0]), corner[1] + t * (nxt[1] - corner[1]))
        if nearest is not None:
            _consider(nearest, radius + margin)
    return best


def minimal_robot_fix(
    scene: SceneSpec, margin: float = DEFAULT_MARGIN, *, max_iterations: int = 5000
) -> Tuple[SceneSpec, List[Dict[str, object]]]:
    """Move each illegal robot the smallest amount we find by deterministic projection.

    Robots are resolved one at a time in id order; each is iteratively pushed along the
    inward normal of its current worst violation until every non-contact pair meets the
    margin. The object, goal, obstacles and physics are untouched.
    """

    poses = list(scene.robot_poses)
    records: List[Dict[str, object]] = []
    # Aim slightly outside each constraint so that the alternating projection converges to
    # a point that is strictly feasible for the true margin even when two constraints are
    # nearly tangent.
    working_margin = margin + 1e-6
    for robot_id in range(len(poses)):
        original = poses[robot_id]
        current = (original.x, original.y)
        moves: List[Vec] = []
        for _ in range(max_iterations):
            # robot-robot is handled explicitly against the other robot's current pose
            violation = _violation_normal(current, scene, scene.robot_radius, working_margin)
            other_id = 1 - robot_id
            other = (poses[other_id].x, poses[other_id].y)
            distance = math.hypot(current[0] - other[0], current[1] - other[1])
            required = 2 * scene.robot_radius + working_margin
            if distance < required:
                if distance < 1e-9:
                    normal = (1.0, 0.0)
                else:
                    normal = ((current[0] - other[0]) / distance, (current[1] - other[1]) / distance)
                push = required - distance
                if violation is None or push > violation[1]:
                    violation = (normal, push)
            if violation is None:
                break
            normal, push = violation
            current = (current[0] + normal[0] * push, current[1] + normal[1] * push)
            moves.append((normal[0] * push, normal[1] * push))
        poses[robot_id] = Pose(current[0], current[1], original.angle)
        records.append({
            "robot_id": robot_id,
            "before": [original.x, original.y],
            "after": [current[0], current[1]],
            "total_displacement": math.hypot(current[0] - original.x, current[1] - original.y),
            "iterations": len(moves),
        })
    fixed = dataclasses.replace(
        scene, scene_version=SCENE_VERSION, robot_poses=tuple(poses)
    )
    # verify
    report = legality_report(fixed, margin)
    if not report["legal"]:
        raise RuntimeError(f"minimal fix did not reach a legal state for {scene.name}: {report['violations']}")
    return fixed, records
