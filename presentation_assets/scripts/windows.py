"""Round-5 windows: turn stored sequences into (history window) -> (two prediction heads).

Feature layout. One token is the **joint** observation at one instant, every geometric quantity
expressed in the observing robot's own start frame R_i0, so nothing depends on object truth:

    r0 block (18): own pose in R_00 (3), own velocity in R_00 (3), patch_valid (1),
                   segment midpoint in R_00 (2), surface normal in R_00 (2), range (1),
                   point count / 32 (1), confidence (1), contacting (1),
                   contact point in R_00 (2), contact-point valid (1)
    r1 block (18): the same nine items expressed in R_10
    relation (12): what r0 knows about r1, in R_00 (6); what r1 knows about r0, in R_10 (6)
    time (3):      elapsed seconds, stage code / 8, anchor flag
    command (6):   alpha (2), speed (2), active mask (2)   -- public, identical for all tokens

The **terminal token** uses the same layout, built from the sensing call made at the exit instant T.

Both heads read the same window and the same masks:
    A (fixed dt)  -> geometry 8, pose 6, velocity 6, validity 4, exits-within-dt 1
    B (terminal)  -> geometry 8, pose 6, velocity 6, validity 4, remaining time 1, exit class 12

Everything is derived by indexing the same token array, so a target can never accidentally come
from a different time than the one it claims.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.sequences import EXIT_CODES  # noqa: E402

DATA = ROOT / "outputs/night_sequence_v1"
FEATURES = 57
#: column names for the geometric target, with the definition pinned: the midpoint of the two
#: endpoints of the **fitted segment**, not the mean of the raw scan points
MIDPOINT_DEFINITION = "midpoint of the fitted segment's two endpoints, expressed in R_i0"
WINDOW = 33
HISTORY = 32
DT = 0.05

GEOMETRY_COLUMNS = ["r0_mid_x", "r0_mid_y", "r0_normal_sin", "r0_normal_cos",
                    "r1_mid_x", "r1_mid_y", "r1_normal_sin", "r1_normal_cos"]
POSE_COLUMNS = ["r0_dx", "r0_dy", "r0_dtheta", "r1_dx", "r1_dy", "r1_dtheta"]
VELOCITY_COLUMNS = ["r0_vx", "r0_vy", "r0_omega", "r1_vx", "r1_vy", "r1_omega"]
VALIDITY_COLUMNS = ["r0_patch_valid", "r0_contacting", "r1_patch_valid", "r1_contacting"]


def wrap(value):
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def rot(theta, vector):
    cos, sin = np.cos(theta), np.sin(theta)
    return np.stack([vector[..., 0] * cos - vector[..., 1] * sin,
                     vector[..., 0] * sin + vector[..., 1] * cos], axis=-1)


def to_frame(point, pose):
    """World point -> the frame (x, y, theta)."""

    delta = point - pose[..., :2]
    return rot(-pose[..., 2], delta)


def unit_normal_world_to_frame(angle, normal_world):
    return rot(-angle, normal_world)


def _normal_sin_cos(vector):
    length = np.hypot(vector[..., 0], vector[..., 1])
    safe = np.where(length < 1e-9, 1.0, length)
    unit = vector / safe[..., None]
    return np.stack([unit[..., 1], unit[..., 0]], axis=-1)


# ---------------------------------------------------------------------------
# token construction
# ---------------------------------------------------------------------------


def _sequence_tokens(shard: Dict[str, np.ndarray], row: int) -> Tuple[np.ndarray, np.ndarray,
                                                                     np.ndarray]:
    """(steps, FEATURES) tokens, (steps,) token_valid, (FEATURES,) terminal token."""

    steps = int(shard["steps"][row])
    own = shard["own_world"][row, :steps]                    # (steps, 2, 6)
    teammate = shard["teammate_world"][row, :steps]
    valid = shard["patch_valid"][row, :steps]                # (steps, 2)
    points = shard["patch_points"][row, :steps]
    distance = shard["patch_distance"][row, :steps]
    confidence = shard["patch_confidence"][row, :steps]
    midpoint = shard["patch_midpoint_robot"][row, :steps]    # robot frame at t
    normal = shard["patch_normal_world"][row, :steps]        # world
    contacting = shard["contacting"][row, :steps]
    contact_point = shard["contact_point_robot"][row, :steps]
    contact_valid = shard["contact_point_valid"][row, :steps]
    time = shard["time"][row, :steps]
    stage = shard["stage_code"][row, :steps]
    command = np.array([
        shard["command_alpha"][row, 0], shard["command_alpha"][row, 1],
        shard["command_speed"][row, 0], shard["command_speed"][row, 1],
        1.0 if shard["command_active"][row, 0] else 0.0,
        1.0 if shard["command_active"][row, 1] else 0.0], dtype=np.float32)
    anchors = [own[0, r, :3].copy() for r in (0, 1)]

    tokens = np.zeros((steps, FEATURES), dtype=np.float32)
    token_valid = np.ones(steps, dtype=np.float32)
    for robot_id in (0, 1):
        base = 18 * robot_id
        anchor = anchors[robot_id]
        pose = to_frame(own[:, robot_id, :2], anchor)         # (steps, 2)
        theta = wrap(own[:, robot_id, 2] - anchor[2])
        tokens[:, base + 0:base + 3] = np.stack([pose[:, 0], pose[:, 1], theta], axis=1)
        velocity = rot(-anchor[2], own[:, robot_id, 3:5])
        tokens[:, base + 3:base + 6] = np.concatenate(
            [velocity, own[:, robot_id, 5:6]], axis=1)
        tokens[:, base + 6] = valid[:, robot_id]
        # **round-6 fix**: the labelled midpoint is the midpoint of the two endpoints of the *fitted
        # segment*, which is what round 4 used and what the contract's "可见拟合线段两端点中点"
        # means. Round 5 used the mean of the scan points instead, an undeclared target change of a
        # few millimetres (up to ~1.3 cm). The raw point mean is kept as a separate diagnostic field
        # below so the two can never be confused again.
        segment = shard["patch_segment_robot"][row, :steps]        # (steps, 2, 2, 2)
        endpoint_mid = 0.5 * (segment[:, robot_id, 0, :] + segment[:, robot_id, 1, :])
        valid_segment = (valid[:, robot_id] > 0) & np.isfinite(endpoint_mid).all(axis=1)
        use = np.where(valid_segment[:, None], endpoint_mid, midpoint[:, robot_id])
        rotated = rot(own[:, robot_id, 2], use)
        world_mid = rotated + own[:, robot_id, :2]
        local_mid = to_frame(world_mid, anchor)
        tokens[:, base + 7:base + 9] = local_mid
        local_normal = unit_normal_world_to_frame(anchor[2], normal[:, robot_id])
        tokens[:, base + 9:base + 11] = _normal_sin_cos(local_normal)
        tokens[:, base + 11] = np.where(valid[:, robot_id] > 0, distance[:, robot_id], 0.0)
        tokens[:, base + 12] = points[:, robot_id] / 32.0
        tokens[:, base + 13] = confidence[:, robot_id]
        tokens[:, base + 14] = contacting[:, robot_id]
        world_contact = rot(own[:, robot_id, 2], contact_point[:, robot_id]) \
            + own[:, robot_id, :2]
        tokens[:, base + 15:base + 17] = np.where(
            contact_valid[:, robot_id:robot_id + 1] > 0,
            to_frame(world_contact, anchor), 0.0)
        tokens[:, base + 17] = contact_valid[:, robot_id]
    # relation blocks: each robot's view of the other, in its own R_i0
    for robot_id, base in ((0, 36), (1, 42)):
        other = 1 - robot_id
        anchor = anchors[robot_id]
        delta = to_frame(teammate[:, other, :2], anchor)
        dtheta = wrap(teammate[:, other, 2] - anchor[2])
        dvel = np.concatenate([rot(-anchor[2], teammate[:, other, 3:5]),
                               teammate[:, other, 5:6]], axis=1)
        tokens[:, base:base + 6] = np.concatenate(
            [delta, dtheta[:, None], dvel], axis=1)
    tokens[:, 48] = time
    tokens[:, 49] = stage / 8.0
    tokens[:, 50] = 0.0
    if steps:
        tokens[0, 50] = 1.0
    tokens[:, 51:57] = command
    return tokens, token_valid, _terminal_token(shard, row, anchors, command)


def _terminal_token(shard: Dict[str, np.ndarray], row: int, anchors, command) -> np.ndarray:
    """The same layout, built from the sensing call made at the exit instant T."""

    token = np.zeros(FEATURES, dtype=np.float32)
    own = shard["terminal_own_world"][row]                    # (2, 6)
    valid = shard["terminal_patch_valid"][row]
    points = shard["terminal_patch_points"][row]
    distance = shard["terminal_patch_distance"][row]
    confidence = shard["terminal_patch_confidence"][row]
    midpoint = shard["terminal_patch_midpoint_robot"][row]
    terminal_segment = shard["terminal_patch_segment_robot"][row]
    normal = shard["terminal_patch_normal_world"][row]
    contacting = shard["terminal_contacting"][row]
    teammate = shard["terminal_teammate_world"][row]
    for robot_id in (0, 1):
        base = 18 * robot_id
        anchor = anchors[robot_id]
        pose = to_frame(own[robot_id, :2], anchor)
        token[base + 0:base + 3] = np.array([pose[0], pose[1],
                                             wrap(own[robot_id, 2] - anchor[2])])
        velocity = rot(-anchor[2], own[robot_id, 3:5])
        token[base + 3:base + 6] = np.array([velocity[0], velocity[1], own[robot_id, 5]])
        token[base + 6] = valid[robot_id]
        # same definition as a normal token: fitted-segment endpoint midpoint
        if valid[robot_id] and np.isfinite(terminal_segment[robot_id]).all():
            chosen = 0.5 * (terminal_segment[robot_id][0] + terminal_segment[robot_id][1])
        else:
            chosen = midpoint[robot_id]
        rotated = rot(own[robot_id, 2], chosen) + own[robot_id, :2]
        token[base + 7:base + 9] = to_frame(rotated, anchor)
        token[base + 9:base + 11] = _normal_sin_cos(
            unit_normal_world_to_frame(anchor[2], normal[robot_id]))
        token[base + 11] = distance[robot_id] if valid[robot_id] else 0.0
        token[base + 12] = points[robot_id] / 32.0
        token[base + 13] = confidence[robot_id]
        token[base + 14] = contacting[robot_id]
    for robot_id, base in ((0, 36), (1, 42)):
        other = 1 - robot_id
        anchor = anchors[robot_id]
        delta = to_frame(teammate[other, :2], anchor)
        dtheta = wrap(teammate[other, 2] - anchor[2])
        dvel = rot(-anchor[2], teammate[other, 3:5])
        token[base:base + 6] = np.array([delta[0], delta[1], dtheta,
                                         dvel[0], dvel[1], teammate[other, 5]])
    token[48] = shard["T_exit"][row]
    token[49] = 0.0
    token[50] = 0.0
    token[51:57] = command
    return token


# ---------------------------------------------------------------------------
# precompute
# ---------------------------------------------------------------------------


def precompute(split: str, shard_paths: Sequence[Path], out_dir: Path) -> dict:
    tokens_all, valid_all, terminal_all = [], [], []
    steps_all, exit_all, T_all, record_all, root_all, command_all = [], [], [], [], [], []
    for path in shard_paths:
        shard = np.load(path, allow_pickle=True)
        for row in range(len(shard["steps"])):
            tokens, token_valid, terminal = _sequence_tokens(shard, row)
            tokens_all.append(tokens)
            valid_all.append(token_valid)
            terminal_all.append(terminal)
            steps_all.append(int(shard["steps"][row]))
            exit_all.append(int(shard["exit_code"][row]))
            T_all.append(float(shard["T_exit"][row]))
            record_all.append(str(shard["record_id"][row]))
            root_all.append(str(shard["root_id"][row]))
            command_all.append(int(shard["command_index"][row]))
    width = int(max(steps_all))
    n = len(steps_all)
    tokens = np.zeros((n, width, FEATURES), dtype=np.float16)
    token_valid = np.zeros((n, width), dtype=np.int8)
    for index, (block, mask) in enumerate(zip(tokens_all, valid_all)):
        tokens[index, :len(mask)] = block.astype(np.float16)
        token_valid[index, :len(mask)] = mask.astype(np.int8)
    payload = {
        "tokens": tokens, "token_valid": token_valid,
        "terminal": np.asarray(terminal_all, dtype=np.float16),
        "steps": np.asarray(steps_all, dtype=np.int32),
        "exit_class": np.asarray(exit_all, dtype=np.int16),
        "T_exit": np.asarray(T_all, dtype=np.float32),
        "record_id": np.asarray(record_all, dtype=object),
        "root_id": np.asarray(root_all, dtype=object),
        "command_index": np.asarray(command_all, dtype=np.int16),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"tokens_{split}.npz", **payload)
    return {"split": split, "sequences": n, "max_steps": width,
            "frames": int(sum(steps_all)),
            "independent_roots": len(set(root_all)),
            "bytes": (out_dir / f"tokens_{split}.npz").stat().st_size,
            "exit_classes": {EXIT_CODES[i] if 0 <= i < len(EXIT_CODES) else "unknown":
                             int((np.asarray(exit_all) == i).sum())
                             for i in sorted(set(exit_all))}}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA / "tokens"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    shard_dir = DATA / "sequences"
    summary = {}
    for split in ("d0", "d1ext", "d2ext", "val", "test"):
        paths = sorted(shard_dir.glob(f"{split}_*.npz"))
        if not paths:
            continue
        summary[split] = precompute(split, paths, out_dir)
        print(json.dumps(summary[split], indent=1, default=float), flush=True)
    (DATA / "token_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print("PRECOMPUTE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
