"""Round-4 dataset: deployable start inputs plus the *observable* terminal labels at T.

The labels come from the exit-instant observation captured by the exact replay (``audit_labels.py``),
not from the true object pose. Everything geometric is expressed in each robot's own start frame
R_i0, defined by that robot's pose at the start of the option:

    q_i0 = R(theta_i0)^T [ R(theta_iT) q_iT + p_iT - p_i0 ]
    n_i0 = R(theta_i0)^T R(theta_iT) n_iT

The six target groups, their dims and their masks are exactly the task book's section 4.2 table.
Inputs are the **same deployable feature vector as the previous round** (so the network keeps its
86-dim input and 97,573 parameters); the privileged slots stay pinned to zero.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.option_prediction_mechanism_v1.phase_b.dataset_b import (  # noqa: E402
    EXIT_CLASSES,
    build_features,
)
from experiments.option_prediction_mechanism_v1.phase_b.model import (  # noqa: E402
    GroupSpec,
    SupervisedData,
)

#: the six output groups of task book section 4.2. Dims match the previous round exactly.
GROUPS: Tuple[Tuple[str, str, int], ...] = (
    ("terminal_observation_geometry", "continuous", 8),
    ("robot_pose_delta", "continuous", 6),
    ("robot_residual_velocity", "continuous", 6),
    ("duration", "continuous", 1),
    ("observation_validity", "binary", 4),
    ("exit_class", "multiclass", 0),
)
GEOMETRY_COLUMNS = ["r0_mid_x_m", "r0_mid_y_m", "r0_normal_sin", "r0_normal_cos",
                    "r1_mid_x_m", "r1_mid_y_m", "r1_normal_sin", "r1_normal_cos"]
POSE_COLUMNS = ["r0_dx_m", "r0_dy_m", "r0_dtheta_rad", "r1_dx_m", "r1_dy_m", "r1_dtheta_rad"]
VELOCITY_COLUMNS = ["r0_vx_mps", "r0_vy_mps", "r0_omega_radps",
                    "r1_vx_mps", "r1_vy_mps", "r1_omega_radps"]
VALIDITY_COLUMNS = ["r0_patch_valid", "r0_contacting", "r1_patch_valid", "r1_contacting"]


def _wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _rotate(theta: float, vector: Sequence[float]) -> Tuple[float, float]:
    cos, sin = math.cos(theta), math.sin(theta)
    return (vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos)


def to_frame(point: Sequence[float], pose: Sequence[float]) -> Tuple[float, float]:
    """Express a world point in the frame ``(pose_x, pose_y, pose_theta)``."""

    local = _rotate(-pose[2], (point[0] - pose[0], point[1] - pose[1]))
    return local


def terminal_geometry(observation: dict, start_pose: Sequence[float],
                      terminal_pose: Sequence[float]) -> Optional[Tuple[float, float, float, float]]:
    """Midpoint and surface normal of the visible fitted segment, in R_i0.

    ``observation`` is the ``observation_summary`` payload: the segment is given in the robot frame
    at T, so it is mapped to the world with the terminal pose and then into R_i0. The normal is
    rotated only, never translated. Returns ``None`` when no valid segment is visible at T.
    """

    if not observation.get("patch_valid"):
        return None
    segment = observation.get("patch_segment_robot")
    normal = observation.get("patch_normal_robot")
    if not segment or normal is None:
        return None
    midpoint_robot = (0.5 * (segment[0][0] + segment[1][0]),
                      0.5 * (segment[0][1] + segment[1][1]))
    world = _rotate(terminal_pose[2], midpoint_robot)
    world = (world[0] + terminal_pose[0], world[1] + terminal_pose[1])
    midpoint = to_frame(world, start_pose)
    normal_world = _rotate(terminal_pose[2], normal)
    normal_start = _rotate(-start_pose[2], normal_world)
    length = math.hypot(normal_start[0], normal_start[1])
    if length < 1e-9:
        return None
    normal_start = (normal_start[0] / length, normal_start[1] / length)
    angle = math.atan2(normal_start[1], normal_start[0])
    return (midpoint[0], midpoint[1], math.sin(angle), math.cos(angle))


def robot_motion(observation: dict, start_pose: Sequence[float],
                 terminal_pose: Sequence[float]) -> Tuple[Tuple[float, float, float],
                                                           Tuple[float, float, float]]:
    """Pose change and terminal velocity, both expressed in R_i0."""

    delta = to_frame((terminal_pose[0], terminal_pose[1]), start_pose)
    pose_delta = (delta[0], delta[1], _wrap(terminal_pose[2] - start_pose[2]))
    velocity = _rotate(-start_pose[2], (observation["own"][3], observation["own"][4]))
    return pose_delta, (velocity[0], velocity[1], observation["own"][5])


def start_pose_of(record: dict, robot_id: int) -> Sequence[float]:
    """R_i0's definition: the robot's own pose at the start of the option (t = 0)."""

    return record["input"][f"robot{robot_id}"]["start_pose_world"]


def build_targets(records: Sequence[dict], observations: Sequence[dict]):
    """Six label groups plus per-column masks, from the exit-instant observations."""

    geometry, geometry_mask = [], []
    pose_delta, residual_velocity, duration = [], [], []
    validity, validity_mask, exit_class = [], [], []
    missing_midpoint: List[int] = []
    for record, observation in zip(records, observations):
        for robot_id in (0, 1):
            start_pose = start_pose_of(record, robot_id)
            summary = observation["terminal_observation"][str(robot_id)]
            terminal_pose = observation["terminal_robots"][robot_id]
            geometry_values = terminal_geometry(summary, start_pose, terminal_pose)
            if geometry_values is None:
                geometry.extend([0.0, 0.0, 0.0, 0.0])
                geometry_mask.extend([0.0, 0.0, 0.0, 0.0])
                missing_midpoint.append(1)
            else:
                geometry.extend(geometry_values)
                geometry_mask.extend([1.0, 1.0, 1.0, 1.0])
                missing_midpoint.append(0)
            delta, velocity = robot_motion(summary, start_pose, terminal_pose)
            pose_delta.extend(delta)
            residual_velocity.extend(velocity)
            validity.extend([1.0 if summary.get("patch_valid") else 0.0,
                             1.0 if summary.get("contacting") else 0.0])
            validity_mask.extend([1.0, 1.0])
        duration.append(float(observation["duration_seconds"]))
        exit_class.append(EXIT_CLASSES.index(observation["exit_class"])
                          if observation["exit_class"] in EXIT_CLASSES else 0)
    n = len(records)
    return {
        "terminal_observation_geometry": {
            "y": np.asarray(geometry, dtype=np.float32).reshape(n, 8),
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
    }, missing_midpoint


def build_persistence(records: Sequence[dict], targets: Dict[str, Dict[str, np.ndarray]]):
    """P0: start observation copied, zero motion, start flags, train statistics for duration/class."""

    n = len(records)
    geometry = np.zeros((n, 8), dtype=np.float32)
    geometry_mask = np.zeros((n, 8), dtype=np.float32)
    pose_delta = np.zeros((n, 6), dtype=np.float32)
    velocity = np.zeros((n, 6), dtype=np.float32)
    validity = np.zeros((n, 4), dtype=np.float32)
    fallback_flags = np.zeros((n, 2), dtype=np.float32)
    for index, record in enumerate(records):
        for robot_id in (0, 1):
            block = record["input"][f"robot{robot_id}"]
            start_pose = start_pose_of(record, robot_id)
            segment = block.get("patch_segment_own_frame")
            normal = block.get("patch_normal_sin_cos")
            valid = bool(block.get("mask_patch_valid")) and segment and normal
            base = 4 * robot_id
            if valid:
                midpoint = (0.5 * (segment[0][0] + segment[1][0]),
                            0.5 * (segment[0][1] + segment[1][1]))
                geometry[index, base + 0] = midpoint[0]
                geometry[index, base + 1] = midpoint[1]
                geometry[index, base + 2] = normal[0]
                geometry[index, base + 3] = normal[1]
                geometry_mask[index, base:base + 4] = 1.0
            else:
                # the start patch is not visible: persist nothing, flag the fallback, and let the
                # caller substitute the per-column train median; measured on the common subset
                fallback_flags[index, robot_id] = 1.0
            start_velocity = block.get("start_twist_world") or [0.0, 0.0, 0.0]
            local = _rotate(-start_pose[2], (start_velocity[0], start_velocity[1]))
            velocity[index, 3 * robot_id + 0] = local[0]
            velocity[index, 3 * robot_id + 1] = local[1]
            velocity[index, 3 * robot_id + 2] = start_velocity[2]
            validity[index, 2 * robot_id + 0] = 1.0 if block.get("mask_patch_valid") else 0.0
            validity[index, 2 * robot_id + 1] = 1.0 if block.get("contacting") else 0.0
    return {
        "terminal_observation_geometry": geometry,
        "robot_pose_delta": pose_delta,
        "robot_residual_velocity": velocity,
        "observation_validity": validity,
        "geometry_mask": geometry_mask,
        "geometry_fallback": fallback_flags,
        "duration": float(np.median(targets["duration"]["y"])),
        "exit_class": int(np.bincount(targets["exit_class"]["y"]).argmax()),
    }


def command_signature(record: dict) -> str:
    """The full public command, including the active mask (task book B2 / P1)."""

    command = record["input"]["command"]
    alpha = command["alpha_rad"]
    speed = command["speed"]
    active = command["active_robots"]
    parts = [f"active={sorted(active)}"]
    for robot_id in (0, 1):
        parts.append(f"a{robot_id}={None if alpha[robot_id] is None else round(alpha[robot_id], 6)}")
        parts.append(f"v{robot_id}={None if speed[robot_id] is None else round(speed[robot_id], 6)}")
    return "|".join(parts)


def build_dataset(records: Sequence[dict], observations: Sequence[dict],
                  split_of: Dict[str, str]) -> Tuple[SupervisedData, Dict[str, object]]:
    x, x_priv, roots, split, commands, signatures = [], [], [], [], [], []
    for record in records:
        features, privileged = build_features(record)
        x.append(features)
        x_priv.append(privileged)
        roots.append(record["root_id"])
        split.append(split_of[record["record_id"]])
        commands.append(str(record["command_index"]))
        signatures.append(command_signature(record))
    targets, missing = build_targets(records, observations)
    specs = [
        GroupSpec("terminal_observation_geometry", "continuous", 8, 1.0),
        GroupSpec("robot_pose_delta", "continuous", 6, 1.0),
        GroupSpec("robot_residual_velocity", "continuous", 6, 1.0),
        GroupSpec("duration", "continuous", 1, 1.0),
        GroupSpec("observation_validity", "binary", 4, 1.0),
        GroupSpec("exit_class", "multiclass", 0, 1.0, num_classes=len(EXIT_CLASSES)),
    ]
    train = np.asarray(split) == "train"
    p0 = build_persistence(records, targets)
    data = SupervisedData(
        x=np.asarray(x, dtype=np.float32), x_priv=np.asarray(x_priv, dtype=np.float32),
        roots=roots, split=np.asarray(split), commands=commands, targets=targets, specs=specs,
        persistence=p0,
        train_duration_median=float(np.median(targets["duration"]["y"][train])),
        train_exit_majority=int(np.bincount(targets["exit_class"]["y"][train]).argmax()),
        train_validity_majority=(targets["observation_validity"]["y"][train].mean(axis=0) >= 0.5)
        .astype(np.float32))
    data.command_signature = signatures  # type: ignore[attr-defined]
    data.observation_count = len(records)  # type: ignore[attr-defined]
    meta = {"missing_terminal_midpoint_slots": int(sum(missing)),
            "records": len(records)}
    return data, meta
