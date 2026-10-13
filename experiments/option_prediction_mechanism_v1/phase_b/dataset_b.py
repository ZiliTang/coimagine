"""Phase B: turn the collector's JSONL records into the normalised arrays the trainer consumes.

Field choices are recorded in ``FIELDS.md``. Two rules drive the mapping:

* every **input** column is deployable: both robots' shareable start local observations, their own
  kinematics, the static scene, and the joint skill command. Nothing in ``x`` is a function of the
  object's true global pose;
* every **label** that needs ground truth lives in ``targets`` (and in ``x_priv`` for the
  privileged reference arm), never in ``x``.

Masks are per column. A missing terminal patch keeps its sample with mask 0 rather than being
dropped, and a zero is never substituted into a geometry loss.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .model import GroupSpec, SupervisedData

EXIT_CLASSES = (
    "settled_observed", "settle_timeout", "settle_unobserved", "contact_lost", "patch_lost",
    "patch_support_lost", "surface_unobserved", "approach_timeout", "not_ready",
    "push_wall_timeout", "wall_clock_end", "environment_termination",
)


def _f(value, default=0.0) -> float:
    return default if value is None else float(value)


def _vec(values, size: int, default=0.0) -> List[float]:
    if values is None:
        return [default] * size
    out = [float(v) if v is not None else default for v in values]
    return (out + [default] * size)[:size]


def _rot_inverse(theta: float, vector: Sequence[float]) -> List[float]:
    cos, sin = math.cos(-theta), math.sin(-theta)
    x, y = float(vector[0]), float(vector[1])
    return [x * cos - y * sin, x * sin + y * cos]


def _wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _robot_start_features(block: Dict[str, object], *, own_frame: bool) -> List[float]:
    """One robot's shareable start observation, expressed in its own start frame."""

    twist = block.get("start_twist_world") or [0.0, 0.0, 0.0]
    patch_normal = block.get("patch_normal_sin_cos") or [0.0, 0.0]
    centroid = _vec(block.get("patch_centroid_own_frame"), 2)
    raw_segment = block.get("patch_segment_own_frame")
    if raw_segment:
        segment = [float(point[0]) for point in raw_segment] + \
                  [float(point[1]) for point in raw_segment]
    else:
        segment = [0.0, 0.0, 0.0, 0.0]
    contact = _vec(block.get("contact_point_own_frame"), 2)
    return [
        float(twist[0]), float(twist[1]), float(twist[2]),
        _f(block.get("patch_distance")), float(block.get("patch_points") or 0),
        _f(block.get("patch_confidence")),
        float(patch_normal[0]), float(patch_normal[1]),
        centroid[0], centroid[1], segment[0], segment[1], segment[2], segment[3],
        float(block.get("contacting") or 0), contact[0], contact[1],
        float(block.get("mask_own_kinematics") or 0),
        float(block.get("mask_patch_valid") or 0), float(block.get("mask_contact") or 0),
    ]


def _teammate_relation(starts: Sequence[Dict[str, object]]) -> List[float]:
    """Robot 1's start pose relative to robot 0's start frame (shareable)."""

    pose0 = starts[0].get("start_pose_world")
    pose1 = starts[1].get("start_pose_world")
    if not pose0 or not pose1:
        return [0.0] * 4
    dx, dy = float(pose1[0]) - float(pose0[0]), float(pose1[1]) - float(pose0[1])
    local = _rot_inverse(float(pose0[2]), (dx, dy))
    delta = _wrap(float(pose1[2]) - float(pose0[2]))
    return [local[0], local[1], math.cos(delta), math.sin(delta)]


def _scene_features(record: Dict[str, object], frame_pose: Sequence[float]) -> List[float]:
    scene = record["input"]["scene"]
    obstacles = scene.get("obstacles") or []
    features: List[float] = [float(len(obstacles))]
    for index in range(3):
        if index < len(obstacles):
            cx, cy, half_w, half_h, angle = obstacles[index]
            local = _rot_inverse(float(frame_pose[2]), (float(cx) - frame_pose[0],
                                                        float(cy) - frame_pose[1]))
            features.extend([local[0], local[1], float(half_w), float(half_h), float(angle), 1.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    features.extend([float(v) for v in scene["object_size"]])
    features.extend([float(scene["robot_radius"]), float(scene["scan_range"]),
                     float(scene["scan_rays"]), float(scene["control_dt"])])
    return features


def _command_features(record: Dict[str, object]) -> List[float]:
    command = record["input"]["command"]
    alpha = command["alpha_rad"]
    speed = command["speed"]
    active = [1.0 if index in command["active_robots"] else 0.0 for index in (0, 1)]
    features: List[float] = []
    for index in (0, 1):
        a = alpha[index]
        s = speed[index]
        features.extend([math.cos(a) if a is not None else 0.0,
                         math.sin(a) if a is not None else 0.0,
                         float(s) if s is not None else 0.0,
                         active[index]])
    features.extend([float(command["active_duration"]), float(command["force_limit"])])
    return features


def build_features(record: Dict[str, object]) -> tuple:
    """Return ``(deployable, privileged)`` flat feature vectors for one record."""

    starts = [record["input"]["robot0"], record["input"]["robot1"]]
    frame_pose = starts[0].get("start_pose_world") or [0.0, 0.0, 0.0]
    x: List[float] = []
    x.extend(_command_features(record))
    x.extend(_scene_features(record, frame_pose))
    x.extend(_robot_start_features(starts[0], own_frame=True))
    x.extend(_robot_start_features(starts[1], own_frame=True))
    # the teammate relation is public and shareable, so it is a second view of the same fact
    x.extend(_teammate_relation(starts))

    truth = record["truth"]
    object_world = truth["object_start_world"]
    relative = [object_world[0] - frame_pose[0], object_world[1] - frame_pose[1]]
    local = _rot_inverse(float(frame_pose[2]), relative)
    delta = _wrap(float(object_world[2]) - float(frame_pose[2]))
    velocity = _rot_inverse(float(frame_pose[2]), (float(object_world[3]), float(object_world[4])))
    x_priv = [local[0], local[1], math.cos(delta), math.sin(delta),
              velocity[0], velocity[1], float(object_world[5])]
    return x, x_priv


def build_targets(records: Sequence[Dict[str, object]]):
    """Flat label vectors plus per-column masks, exactly as documented in FIELDS.md."""

    geometry, geometry_mask = [], []
    pose_delta, residual_velocity = [], []
    duration, validity, validity_mask, exit_class = [], [], [], []
    for record in records:
        label = record["label"]
        for robot_id in (0, 1):
            robot = label[f"robot{robot_id}"]
            mask = float(robot["mask_terminal_geometry"])
            normal = _vec(robot["terminal_normal_sin_cos"], 2)
            geometry.extend([_f(robot["terminal_signed_distance"]),
                             _f(robot["terminal_tangential_offset"]),
                             normal[0], normal[1]])
            geometry_mask.extend([mask] * 4)
            pose_delta.extend(_vec(robot["terminal_pose_change_own_start"], 3))
            residual_velocity.extend(_vec(robot["terminal_residual_velocity_own_start"], 3))
            validity.extend([float(robot["mask_terminal_patch_valid"]),
                             float(robot["mask_terminal_contacting"])])
            validity_mask.extend([1.0, 1.0])
        duration.append(float(label["duration_seconds"]))
        exit_class.append(EXIT_CLASSES.index(label["exit_class"])
                          if label["exit_class"] in EXIT_CLASSES else 0)
    n = len(records)
    targets = {
        "terminal_geometry": {"y": np.asarray(geometry, dtype=np.float32).reshape(n, 8),
                              "mask": np.asarray(geometry_mask, dtype=np.float32).reshape(n, 8)},
        "robot_pose_delta": {"y": np.asarray(pose_delta, dtype=np.float32).reshape(n, 6),
                             "mask": np.ones((n, 6), dtype=np.float32)},
        "robot_residual_velocity": {
            "y": np.asarray(residual_velocity, dtype=np.float32).reshape(n, 6),
            "mask": np.ones((n, 6), dtype=np.float32)},
        "duration": {"y": np.asarray(duration, dtype=np.float32).reshape(n, 1),
                     "mask": np.ones((n, 1), dtype=np.float32)},
        "observation_validity": {"y": np.asarray(validity, dtype=np.float32).reshape(n, 4),
                                 "mask": np.asarray(validity_mask, dtype=np.float32).reshape(n, 4)},
        "exit_class": {"y": np.asarray(exit_class, dtype=np.int64).reshape(n),
                       "mask": np.ones(n, dtype=np.float32)},
    }
    return targets


def build_persistence(records: Sequence[Dict[str, object]]):
    """Start-state values for the continuous targets; the contract's persistence baseline."""

    geometry, pose_delta, residual_velocity, duration = [], [], [], []
    for record in records:
        for robot_id in (0, 1):
            robot = record["label"][f"robot{robot_id}"]
            normal = _vec(robot.get("terminal_normal_sin_cos"), 2)
            del normal  # the start normal comes from the input block, not the label
            block = record["input"][f"robot{robot_id}"]
            start_normal = _vec(block.get("patch_normal_sin_cos"), 2)
            geometry.extend([_f(robot["start_signed_distance"]),
                             _f(robot["start_tangential_offset"]),
                             start_normal[0], start_normal[1]])
            pose_delta.extend([0.0, 0.0, 0.0])
            residual_velocity.extend(_vec(block.get("start_twist_world"), 3))
        duration.append(0.0)  # filled with the train median by persistence_predictions
    n = len(records)
    return {
        "terminal_geometry": np.asarray(geometry, dtype=np.float32).reshape(n, 8),
        "robot_pose_delta": np.asarray(pose_delta, dtype=np.float32).reshape(n, 6),
        "robot_residual_velocity": np.asarray(residual_velocity, dtype=np.float32).reshape(n, 6),
        "duration": np.asarray(duration, dtype=np.float32).reshape(n, 1),
    }


def load_records(path: Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.open()]


def build_supervised(records: Sequence[Dict[str, object]]) -> SupervisedData:
    x, x_priv, roots, split, commands = [], [], [], [], []
    for record in records:
        features, privileged = build_features(record)
        x.append(features)
        x_priv.append(privileged)
        roots.append(record["root_id"])
        split.append(record["split"])
        commands.append(f"{record['command_index']}")

    specs = [
        GroupSpec("terminal_geometry", "continuous", 8, 1.0),
        GroupSpec("robot_pose_delta", "continuous", 6, 1.0),
        GroupSpec("robot_residual_velocity", "continuous", 6, 1.0),
        GroupSpec("duration", "continuous", 1, 1.0),
        GroupSpec("observation_validity", "binary", 4, 1.0),
        GroupSpec("exit_class", "multiclass", 0, 1.0, num_classes=len(EXIT_CLASSES)),
    ]
    targets = build_targets(records)
    persistence = build_persistence(records)
    train = np.asarray(split) == "train"
    durations = targets["duration"]["y"][train, 0]
    classes = targets["exit_class"]["y"][train]
    validity_train = targets["observation_validity"]["y"][train]
    validity_majority = (validity_train.mean(axis=0) >= 0.5).astype(np.float32)
    data = SupervisedData(
        x=np.asarray(x, dtype=np.float32),
        x_priv=np.asarray(x_priv, dtype=np.float32),
        roots=roots, split=np.asarray(split), commands=commands,
        targets=targets, specs=specs, persistence=persistence,
        train_duration_median=float(np.median(durations)) if durations.size else 0.0,
        train_exit_majority=int(np.bincount(classes).argmax()) if classes.size else 0,
        train_validity_majority=validity_majority,
    )
    return data
