"""Random scene sampling: object, two robots and goal, with rejection statistics.

Contract (task book section 2.2). The object centre is drawn uniformly in the legal region,
its heading uniformly in [-pi, pi). Each robot centre is drawn uniformly in the bounding box
of the object expanded by the largest admissible gap; the candidate is accepted when the
disc-to-object edge gap is in [0.12, 0.50] m and the disc clears the obstacle, the boundary
and the other robot by at least 0.05 m. Both robot headings are uniform, and *no* robot is
labelled active: any robot may attempt any skill. The goal centre is drawn uniformly in
direction and in distance [0.80, 1.20] m from the object's initial centre.

Privileged truth is used here and only here: collision screening. A sampled instance is
called "geometrically legal", never "reachable".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..continuous2d.scenes import Box, Pose, SceneSpec
from . import geometry as G
from .spec import (
    ARENA,
    BOUNDARY_RADIUS,
    GOAL_DISTANCE_MAX,
    GOAL_DISTANCE_MIN,
    GOAL_RADIUS,
    GOAL_TERMINAL_CLEARANCE,
    L1_CENTRE,
    L1_FULL_HEIGHT,
    L1_FULL_WIDTH,
    MAX_SAMPLE_ATTEMPTS,
    OBJECT_EDGE_MARGIN,
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    ROBOT_CLEARANCE,
    ROBOT_DRAWS_PER_STAGE,
    ROBOT_OBJECT_GAP_MAX,
    ROBOT_OBJECT_GAP_MIN,
    ROBOT_RADIUS,
    SAMPLE_GOAL_DRAWS,
    SPLIT_SEEDS,
    SPLIT_SIZES,
    ENV_VERSION,
)


@dataclass(frozen=True)
class MapSpec:
    name: str
    arena: Tuple[float, float]
    obstacles: Tuple[Box, ...]

    def obstacle_polygons(self) -> Tuple[G.Poly, ...]:
        return tuple(G.box_polygon(box.cx, box.cy, box.angle, box.half_w, box.half_h)
                     for box in self.obstacles)


def map_spec(name: str) -> MapSpec:
    if name == "L0":
        return MapSpec(name="L0", arena=ARENA, obstacles=())
    if name == "L1":
        return MapSpec(name="L1", arena=ARENA, obstacles=(
            Box(L1_CENTRE[0], L1_CENTRE[1], L1_FULL_WIDTH / 2.0, L1_FULL_HEIGHT / 2.0, 0.0),
        ))
    raise KeyError(name)


@dataclass(frozen=True)
class InstanceSpec:
    """One independent task instance; every exploration branch belongs to its split."""

    name: str
    map: str
    split: str
    index: int
    seed: int
    arena: Tuple[float, float]
    obstacles: Tuple[Box, ...]
    object_pose: Pose
    robot_poses: Tuple[Pose, Pose]
    goal_centre: Tuple[float, float]
    goal_radius: float = GOAL_RADIUS
    env_version: str = ENV_VERSION

    def to_scene(self) -> SceneSpec:
        """The engine-facing scene. ``goal`` is a square of the goal radius, used for
        rendering and for the scene record only; success is the centre/tolerance rule of
        section 5.1, evaluated by the environment, not ``Continuous2DEnv.task_success``."""

        return SceneSpec(
            name=self.name, category=self.map, variant=self.index,
            arena=self.arena, obstacles=self.obstacles,
            goal=Box(self.goal_centre[0], self.goal_centre[1], self.goal_radius,
                     self.goal_radius, 0.0),
            object_pose=self.object_pose, robot_poses=self.robot_poses,
            description=f"random-start {self.map} instance {self.index}",
            scene_version=ENV_VERSION,
            object_width=OBJECT_WIDTH, object_height=OBJECT_HEIGHT,
            robot_radius=ROBOT_RADIUS,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name, "map": self.map, "split": self.split, "index": self.index,
            "seed": self.seed, "arena": list(self.arena),
            "obstacles": [asdict(box) for box in self.obstacles],
            "object": asdict(self.object_pose),
            "robots": [asdict(pose) for pose in self.robot_poses],
            "goal_centre": list(self.goal_centre), "goal_radius": self.goal_radius,
            "env_version": self.env_version,
        }


@dataclass
class RejectionLog:
    """Counts of candidate draws and the reason each was rejected."""

    draws: int = 0
    object_draws: int = 0
    robot_draws: int = 0
    goal_draws: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def as_dict(self) -> Dict[str, object]:
        return {"draws": self.draws, "object_draws": self.object_draws,
                "robot_draws": self.robot_draws, "goal_draws": self.goal_draws,
                "total_rejections": int(sum(self.reasons.values())),
                "reasons": dict(sorted(self.reasons.items()))}


def object_legal(map_name: str, poly: G.Poly, arena: Tuple[float, float],
                 margin: float) -> bool:
    if not G.corners_inside(poly, arena, margin):
        return False
    for obstacle in map_spec(map_name).obstacle_polygons():
        if G.polygon_distance(poly, obstacle) < margin:
            return False
    return True


def robot_legal(map_name: str, centre: G.Vec, arena: Tuple[float, float],
                object_poly: G.Poly, other: Optional[G.Vec]) -> Optional[str]:
    """Return ``None`` when legal, else the rejection reason."""

    gap = G.circle_polygon_gap(centre, ROBOT_RADIUS, object_poly)
    if gap < ROBOT_OBJECT_GAP_MIN:
        return "robot_too_close_to_object" if gap >= 0.0 else "robot_overlaps_object"
    if gap > ROBOT_OBJECT_GAP_MAX:
        return "robot_too_far_from_object"
    wall_gap = G.wall_distance(centre, arena) - ROBOT_RADIUS
    if wall_gap < ROBOT_CLEARANCE:
        return "robot_boundary_clearance" if wall_gap >= 0.0 else "robot_outside_boundary"
    for obstacle in map_spec(map_name).obstacle_polygons():
        obstacle_gap = G.circle_polygon_gap(centre, ROBOT_RADIUS, obstacle)
        if obstacle_gap < ROBOT_CLEARANCE:
            return ("robot_obstacle_clearance" if obstacle_gap >= 0.0
                    else "robot_overlaps_obstacle")
    if other is not None:
        pair_gap = G.circle_circle_gap(centre, ROBOT_RADIUS, other, ROBOT_RADIUS)
        if pair_gap < ROBOT_CLEARANCE:
            return "robot_robot_clearance" if pair_gap >= 0.0 else "robots_overlap"
    return None


def goal_legal(map_name: str, centre: G.Vec, object_angle: float,
               arena: Tuple[float, float]) -> Optional[str]:
    width, height = arena
    if (centre[0] - GOAL_RADIUS < BOUNDARY_RADIUS or centre[0] + GOAL_RADIUS > width - BOUNDARY_RADIUS
            or centre[1] - GOAL_RADIUS < BOUNDARY_RADIUS
            or centre[1] + GOAL_RADIUS > height - BOUNDARY_RADIUS):
        return "goal_outside_arena"
    for obstacle in map_spec(map_name).obstacle_polygons():
        if G.circle_polygon_gap(centre, GOAL_RADIUS, obstacle) < 0.0:
            return "goal_overlaps_obstacle"
    placed = G.box_polygon(centre[0], centre[1], object_angle, OBJECT_WIDTH / 2.0,
                           OBJECT_HEIGHT / 2.0)
    if not G.corners_inside(placed, arena, GOAL_TERMINAL_CLEARANCE):
        return "goal_placement_leaves_arena"
    for obstacle in map_spec(map_name).obstacle_polygons():
        if G.polygon_distance(placed, obstacle) < GOAL_TERMINAL_CLEARANCE:
            return "goal_placement_hits_obstacle"
    return None


def _expanded_box(poly: G.Poly, margin: float) -> Tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = G.aabb(poly)
    return min_x - margin, min_y - margin, max_x + margin, max_y + margin


def sample_instance(map_name: str, split: str, index: int, seed: int,
                    log: Optional[RejectionLog] = None) -> Optional[InstanceSpec]:
    """Sample one instance, or ``None`` when the 2000-draw budget is exhausted."""

    import numpy as np

    rng = np.random.default_rng(seed)
    spec = map_spec(map_name)
    arena = spec.arena
    log = log if log is not None else RejectionLog()
    name = f"{map_name}_{split}_{index:04d}"

    while log.draws < MAX_SAMPLE_ATTEMPTS:
        # ---- object ----------------------------------------------------
        log.draws += 1
        log.object_draws += 1
        object_angle = float(rng.uniform(-math.pi, math.pi))
        object_centre = (float(rng.uniform(BOUNDARY_RADIUS, arena[0] - BOUNDARY_RADIUS)),
                         float(rng.uniform(BOUNDARY_RADIUS, arena[1] - BOUNDARY_RADIUS)))
        object_poly = G.box_polygon(object_centre[0], object_centre[1], object_angle,
                                    OBJECT_WIDTH / 2.0, OBJECT_HEIGHT / 2.0)
        if not object_legal(map_name, object_poly, arena, OBJECT_EDGE_MARGIN):
            log.reject("object_illegal")
            continue

        # ---- robots ----------------------------------------------------
        min_x, min_y, max_x, max_y = _expanded_box(
            object_poly, ROBOT_OBJECT_GAP_MAX + 2.0 * ROBOT_RADIUS)
        centres: List[G.Vec] = []
        robot_reason: Optional[str] = None
        for robot_id in (0, 1):
            other = centres[0] if robot_id == 1 else None
            accepted: Optional[G.Vec] = None
            for _ in range(ROBOT_DRAWS_PER_STAGE):
                log.draws += 1
                log.robot_draws += 1
                if log.draws >= MAX_SAMPLE_ATTEMPTS:
                    break
                candidate = (float(rng.uniform(min_x, max_x)), float(rng.uniform(min_y, max_y)))
                reason = robot_legal(map_name, candidate, arena, object_poly, other)
                if reason is None:
                    accepted = candidate
                    break
                robot_reason = reason
                log.reject(reason)
            if accepted is None:
                break
            centres.append(accepted)
        if len(centres) != 2:
            if robot_reason:
                log.reject(f"outer_retry_after_{robot_reason}")
            continue
        robot_angles = [float(rng.uniform(-math.pi, math.pi)) for _ in range(2)]

        # ---- goal ------------------------------------------------------
        goal_reason: Optional[str] = None
        goal_centre: Optional[G.Vec] = None
        for _ in range(SAMPLE_GOAL_DRAWS):
            log.draws += 1
            log.goal_draws += 1
            if log.draws >= MAX_SAMPLE_ATTEMPTS:
                break
            bearing = float(rng.uniform(-math.pi, math.pi))
            distance = float(rng.uniform(GOAL_DISTANCE_MIN, GOAL_DISTANCE_MAX))
            candidate = (object_centre[0] + distance * math.cos(bearing),
                         object_centre[1] + distance * math.sin(bearing))
            reason = goal_legal(map_name, candidate, object_angle, arena)
            if reason is None:
                goal_centre = candidate
                break
            goal_reason = reason
            log.reject(reason)
        if goal_centre is None:
            if goal_reason:
                log.reject(f"outer_retry_after_{goal_reason}")
            continue

        return InstanceSpec(
            name=name, map=map_name, split=split, index=index, seed=seed, arena=arena,
            obstacles=spec.obstacles,
            object_pose=Pose(object_centre[0], object_centre[1], object_angle),
            robot_poses=(Pose(centres[0][0], centres[0][1], robot_angles[0]),
                         Pose(centres[1][0], centres[1][1], robot_angles[1])),
            goal_centre=goal_centre,
        )
    return None


SPLIT_VERSION = "independent-v2"
LEGACY_SPLIT_VERSION = "overlapping-v1"


def content_hash(instance: InstanceSpec) -> str:
    """Hash physical task content, never names, seeds, indices or split labels."""
    import hashlib
    import json

    data = instance.as_dict()
    physical = {key: data[key] for key in
                ("map", "arena", "obstacles", "object", "robots", "goal_centre", "goal_radius")}
    return hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()


def audit_split_content(panels: Dict[str, Sequence[InstanceSpec]]) -> Dict[str, object]:
    """Detect both within-panel and cross-panel exact physical duplicates."""
    seen = {}
    duplicates = []
    for panel, instances in panels.items():
        for instance in instances:
            digest = content_hash(instance)
            current = {"panel": panel, "instance": instance.name}
            if digest in seen:
                duplicates.append({"first": seen[digest], "second": current, "sha256": digest})
            else:
                seen[digest] = current
    return {"ok": not duplicates, "unique": len(seen), "duplicates": duplicates}


def build_split(map_name: str, split: str, count: Optional[int] = None,
                base_seed: Optional[int] = None, *,
                split_version: str = SPLIT_VERSION) -> Tuple[List[InstanceSpec], Dict[str, object]]:
    """Versioned panels; retain all original train seeds and geometries.

    v2 evaluation seeds live above 2**64 in separate 2**32-wide namespaces, even
    with an explicit base_seed. Legacy evaluation panels require an explicit version
    for historical replay; they must never be presented as independent holdouts.
    """

    if split_version not in (SPLIT_VERSION, LEGACY_SPLIT_VERSION):
        raise ValueError(f"Unknown split version: {split_version}")
    if split not in SPLIT_SIZES:
        raise ValueError(f"Unknown split: {split}")

    count = SPLIT_SIZES[split] if count is None else count
    base_seed = SPLIT_SEEDS[map_name] if base_seed is None else base_seed
    if not 0 <= count < (2**32 - 1) // 10 or not 0 <= base_seed < 2**63:
        raise ValueError("count/base_seed outside the versioned seed namespace")
    instances: List[InstanceSpec] = []
    total = RejectionLog()
    failures: List[Dict[str, object]] = []
    index = 0
    while len(instances) < count:
        if index > count * 10:
            raise RuntimeError(f"{map_name}/{split}: too many failed instance indices")
        seed = base_seed + index
        if split_version == SPLIT_VERSION and split != "train":
            seed = 2**64 + (base_seed << 33) + ((split == "test") << 32) + index
        log = RejectionLog()
        instance = sample_instance(map_name, split, index, seed, log)
        total.draws += log.draws
        total.object_draws += log.object_draws
        total.robot_draws += log.robot_draws
        total.goal_draws += log.goal_draws
        for reason, value in log.reasons.items():
            total.reasons[reason] = total.reasons.get(reason, 0) + value
        if instance is None:
            failures.append({"index": index, "seed": seed, "attempts": log.draws})
        else:
            if split_version == SPLIT_VERSION and split != "train":
                instance = replace(instance, name=f"{map_name}_{split}_v2_{index:04d}")
            instances.append(instance)
        index += 1
    return instances, {"map": map_name, "split": split, "split_version": split_version,
                       "base_seed": base_seed, "count": len(instances),
                       "failed_indices": failures, "rejections": total.as_dict()}


def instance_statistics(instances: Sequence[InstanceSpec]) -> Dict[str, object]:
    """Post-hoc distribution of the realised geometry (task book section 2.3)."""

    import numpy as np

    rows = []
    for instance in instances:
        object_poly = G.box_polygon(instance.object_pose.x, instance.object_pose.y,
                                    instance.object_pose.angle, OBJECT_WIDTH / 2.0,
                                    OBJECT_HEIGHT / 2.0)
        object_centre = (instance.object_pose.x, instance.object_pose.y)
        for robot_id in (0, 1):
            pose = instance.robot_poses[robot_id]
            gap = G.circle_polygon_gap((pose.x, pose.y), ROBOT_RADIUS, object_poly)
            bearing = math.atan2(pose.y - object_centre[1], pose.x - object_centre[0])
            goal_distance = math.hypot(instance.goal_centre[0] - object_centre[0],
                                       instance.goal_centre[1] - object_centre[1])
            goal_bearing = math.atan2(instance.goal_centre[1] - object_centre[1],
                                      instance.goal_centre[0] - object_centre[0])
            rows.append({"gap": gap, "bearing": bearing, "goal_distance": goal_distance,
                         "goal_bearing": goal_bearing,
                         "goal_minus_bearing": math.atan2(
                             math.sin(goal_bearing - bearing), math.cos(goal_bearing - bearing)),
                         "object_angle": instance.object_pose.angle,
                         "axis_relative_bearing": math.atan2(
                             math.sin(bearing - instance.object_pose.angle),
                             math.cos(bearing - instance.object_pose.angle))})
    gaps = np.asarray([row["gap"] for row in rows])
    bearings = np.asarray([row["bearing"] for row in rows])
    robot_distance = np.asarray([
        math.hypot(inst.robot_poses[0].x - inst.robot_poses[1].x,
                   inst.robot_poses[0].y - inst.robot_poses[1].y) for inst in instances])
    goal_distances = np.asarray([row["goal_distance"] for row in rows[0::2]])
    goal_relative = np.asarray([row["goal_minus_bearing"] for row in rows[0::2]])
    axis_relative = np.asarray([row["axis_relative_bearing"] for row in rows[0::2]])

    def summary(values: np.ndarray) -> Dict[str, float]:
        return {"min": float(values.min()), "p10": float(np.quantile(values, 0.10)),
                "median": float(np.median(values)), "p90": float(np.quantile(values, 0.90)),
                "max": float(values.max()), "mean": float(values.mean())}

    return {
        "instances": len(instances),
        "robot_object_gap_m": summary(gaps),
        "robot_robot_distance_m": summary(robot_distance),
        "robot_bearing_rad": summary(bearings),
        "goal_distance_m": summary(goal_distances),
        "goal_relative_bearing_rad": summary(goal_relative),
        "robot_bearing_in_object_frame_rad": summary(axis_relative),
        # coverage of the object's neighbourhood: 12 sectors over 360 degrees
        "robot_bearing_histogram_30deg": [int(np.sum((bearings >= -math.pi + k * math.pi / 6)
                                                      & (bearings < -math.pi + (k + 1) * math.pi / 6)))
                                          for k in range(12)],
        "goal_bearing_histogram_30deg": [int(np.sum((goal_relative >= -math.pi + k * math.pi / 6)
                                                     & (goal_relative < -math.pi + (k + 1) * math.pi / 6)))
                                         for k in range(12)],
        "min_pair_gap_m": float(min(row["gap"] for row in rows)),
        "objects_near_boundary": int(sum(
            1 for inst in instances
            if not G.corners_inside(
                G.box_polygon(inst.object_pose.x, inst.object_pose.y, inst.object_pose.angle,
                              OBJECT_WIDTH / 2.0, OBJECT_HEIGHT / 2.0),
                inst.arena, OBJECT_EDGE_MARGIN - 1e-9))),
    }


def scene_legality_report(instances: Sequence[InstanceSpec]) -> Dict[str, object]:
    """Independent re-check of every sampled instance against the section 2.2 contract."""

    problems: List[Dict[str, object]] = []
    for instance in instances:
        object_poly = G.box_polygon(instance.object_pose.x, instance.object_pose.y,
                                    instance.object_pose.angle, OBJECT_WIDTH / 2.0,
                                    OBJECT_HEIGHT / 2.0)
        if not object_legal(instance.map, object_poly, instance.arena, OBJECT_EDGE_MARGIN - 1e-9):
            problems.append({"instance": instance.name, "problem": "object_illegal"})
        reason = robot_legal(instance.map, (instance.robot_poses[0].x, instance.robot_poses[0].y),
                             instance.arena, object_poly,
                             (instance.robot_poses[1].x, instance.robot_poses[1].y))
        if reason is not None:
            problems.append({"instance": instance.name, "problem": f"robot0_{reason}"})
        reason = robot_legal(instance.map, (instance.robot_poses[1].x, instance.robot_poses[1].y),
                             instance.arena, object_poly,
                             (instance.robot_poses[0].x, instance.robot_poses[0].y))
        if reason is not None:
            problems.append({"instance": instance.name, "problem": f"robot1_{reason}"})
        reason = goal_legal(instance.map, instance.goal_centre, instance.object_pose.angle,
                            instance.arena)
        if reason is not None:
            problems.append({"instance": instance.name, "problem": reason})
        distance = math.hypot(instance.goal_centre[0] - instance.object_pose.x,
                              instance.goal_centre[1] - instance.object_pose.y)
        if not (GOAL_DISTANCE_MIN - 1e-9 <= distance <= GOAL_DISTANCE_MAX + 1e-9):
            problems.append({"instance": instance.name, "problem": "goal_distance_out_of_range"})
    return {"checked": len(instances), "problems": problems, "ok": not problems}


def split_manifest(instances: Sequence[InstanceSpec]) -> Dict[str, object]:
    import hashlib
    import json

    payload = json.dumps([instance.as_dict() for instance in instances], sort_keys=True).encode()
    return {"count": len(instances), "sha256": hashlib.sha256(payload).hexdigest()[:32],
            "instances": [instance.name for instance in instances]}
