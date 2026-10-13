"""Phase B dataset collection harness (CONTRACT.md section 2.1, task book section 3 / B1).

One JSONL record per **real joint option execution**: the pre-execution information that the
two robots can share plus the joint skill command, and the terminal information / labels at
exit. The duration is whatever the frozen V protocol produced; it is never a fixed window.

What this harness deliberately does *not* do: train a model, score a model, or evaluate
anything. It only samples roots, generates the eight joint commands per root, runs the frozen
V execution through the existing ``runner.run_execution`` path (full snapshot +
``Continuous2DEnv.restore(exact=True)``), and writes the data plus a ledger.

Usage::

    # 32-root pipeline check (CONTRACT.md section 2.1)
    .venv/bin/python experiments/option_prediction_mechanism_v1/phase_b/collect_b.py \
        --out-dir outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/phase_b/check32 \
        --train-roots 32 --val-roots 0 --test-roots 0 --tag check32

    # full tier-1 collection
    .venv/bin/python experiments/option_prediction_mechanism_v1/phase_b/collect_b.py

    # spot check: eight distinct commands, identical snapshot, identical replay
    .venv/bin/python experiments/option_prediction_mechanism_v1/phase_b/collect_b.py --spot-check
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
# HERE = <repo>/experiments/option_prediction_mechanism_v1/phase_b
ROOT = HERE.parents[2]
for _path in (str(ROOT / "src"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import roots_b as RB  # noqa: E402  (same directory)

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.local_contact_option.contract import OptionConfig  # noqa: E402
from coimagine.local_contact_option.roots import TEMPLATES, VariantSpec  # noqa: E402
from coimagine.local_contact_option.runner import (  # noqa: E402
    build_snapshot,
    root_from_metadata,
    run_execution,
)

#: Frozen V controller configuration (task book section 1.2 / CONTRACT.md section 2.1).
CONTROLLER = dict(
    kv=25.0, a_ref_max=3.0, approach_max_speed=0.30, approach_contact_speed=0.06,
    approach_gain=4.0, ready_precontact_speed=0.02,
)
#: Frozen round-3 settle protocol.
SETTLE = dict(settle_window=0.20, settle_window_min_baseline=0.10, settle_max_sample_gap=0.15)
#: Main dataset sensing: no added noise, no delay. The real scan range / occlusion stays.
SENSING = dict(range_noise_sigma=0.0, observation_delay=0.0)

ARM = "V"
VARIANT_NAME = "baseline"
VARIANT = VariantSpec("baseline")
VERSION = "option-prediction-mechanism-v1/phase_b-v1"
OBJECT_HALF = (0.5, 0.125)  # PhysicsConfig.object_width / object_height
ROBOT_RADIUS = default_config().physics.robot_radius
SCAN_RANGE = OptionConfig().scan_range
CONTROL_DT = default_config().physics.control_dt


def frozen_option_config() -> OptionConfig:
    return replace(OptionConfig(**CONTROLLER), **SETTLE)


# ---------------------------------------------------------------------------
# small geometry helpers (labels only; never part of the model input)
# ---------------------------------------------------------------------------


def _rot(theta: float, vector: Tuple[float, float]) -> Tuple[float, float]:
    cos, sin = math.cos(theta), math.sin(theta)
    return (vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos)


def _to_frame(point: Tuple[float, float], origin: Sequence[float]) -> Tuple[float, float]:
    """World point in the frame of ``origin`` = (x, y, angle)."""

    dx, dy = point[0] - origin[0], point[1] - origin[1]
    return _rot(-origin[2], (dx, dy))


def _point_box_distance(point: Sequence[float], object_pose: Sequence[float]) -> float:
    local = _rot(-object_pose[2], (point[0] - object_pose[0], point[1] - object_pose[1]))
    dx = max(abs(local[0]) - OBJECT_HALF[0], 0.0)
    dy = max(abs(local[1]) - OBJECT_HALF[1], 0.0)
    return math.hypot(dx, dy)


#: face id -> (centre in the object frame, outward normal, tangent = Rot90(outward)).
#: The ids follow the existing generator (``roots._face_point``): 0 = local -y, 1 = +x,
#: 2 = +y, 3 = -x.
_FACES: Dict[int, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
    0: ((0.0, -OBJECT_HALF[1]), (0.0, -1.0), (1.0, 0.0)),
    1: ((OBJECT_HALF[0], 0.0), (1.0, 0.0), (0.0, 1.0)),
    2: ((0.0, OBJECT_HALF[1]), (0.0, 1.0), (-1.0, 0.0)),
    3: ((-OBJECT_HALF[0], 0.0), (-1.0, 0.0), (0.0, -1.0)),
}


def reference_face(object_pose: Sequence[float], robot_point: Sequence[float]) -> int:
    """The box face the robot starts against: the face line it is furthest outside of."""

    local = _rot(-object_pose[2],
                 (robot_point[0] - object_pose[0], robot_point[1] - object_pose[1]))
    outside = {
        0: -local[1] - OBJECT_HALF[1],
        1: local[0] - OBJECT_HALF[0],
        2: local[1] - OBJECT_HALF[1],
        3: -local[0] - OBJECT_HALF[0],
    }
    return max(outside, key=lambda face: outside[face])


def face_frame(object_pose: Sequence[float], face: int) -> Tuple[Tuple[float, float],
                                                                 Tuple[float, float],
                                                                 Tuple[float, float]]:
    """World (face centre, outward normal, tangent) of ``face`` for this object pose."""

    local_centre, local_normal, local_tangent = _FACES[face]
    centre = _rot(object_pose[2], local_centre)
    centre = (centre[0] + object_pose[0], centre[1] + object_pose[1])
    return centre, _rot(object_pose[2], local_normal), _rot(object_pose[2], local_tangent)


def segment_geometry(object_pose: Sequence[float], face: int,
                     robot_point: Sequence[float]) -> Tuple[float, float]:
    """Signed distance (positive = separated) and tangential coordinate on the face line."""

    centre, normal, tangent = face_frame(object_pose, face)
    dx, dy = robot_point[0] - centre[0], robot_point[1] - centre[1]
    return dx * normal[0] + dy * normal[1], dx * tangent[0] + dy * tangent[1]


def snapshot_sha256(snapshot) -> str:
    blob = json.dumps(dataclasses.asdict(snapshot), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# observation extraction
# ---------------------------------------------------------------------------


def _obs_entry(row: Dict[str, object], robot_id: int) -> Optional[Dict[str, object]]:
    observed = row.get("observed") or {}
    entry = observed.get(str(robot_id))
    return entry if isinstance(entry, dict) else None


def _last_observation_row(rows: Sequence[Dict[str, object]], robot_id: int
                          ) -> Optional[Dict[str, object]]:
    for row in reversed(rows):
        if _obs_entry(row, robot_id) is not None:
            return row
    return None


def _start_block(rows: Sequence[Dict[str, object]], robot_id: int) -> Dict[str, object]:
    """The robot's own shareable start observation, in its own start frame."""

    row = _start_row(rows, robot_id)
    entry = _obs_entry(row, robot_id) if row is not None else None
    if entry is None:
        return {
            "start_pose_world": None, "start_twist_world": None, "mask_own_kinematics": 0,
            "patch_distance": None, "patch_points": 0, "patch_confidence": None,
            "patch_normal_own_frame": None, "patch_normal_sin_cos": None,
            "patch_segment_own_frame": None, "patch_centroid_own_frame": None,
            "contacting": 0, "contact_point_own_frame": None,
            "mask_patch_valid": 0, "mask_contact": 0,
        }
    own = list(entry.get("own") or [0.0] * 6)
    # the first observation is taken at t = 0, so the robot's current frame *is* its start frame
    normal_world = entry.get("patch_normal")
    normal_own = None
    normal_sin_cos = None
    if normal_world is not None:
        normal_own = list(_rot(-own[2], (float(normal_world[0]), float(normal_world[1]))))
        normal_sin_cos = [normal_own[1], normal_own[0]]
    segment_own = None
    centroid_own = None
    segment_world = entry.get("segment_world")
    if segment_world:
        segment_own = [list(_to_frame((float(p[0]), float(p[1])), own)) for p in segment_world]
        centroid_own = [
            sum(point[0] for point in segment_own) / len(segment_own),
            sum(point[1] for point in segment_own) / len(segment_own),
        ]
    points = int(entry.get("patch_points") or 0)
    return {
        "start_pose_world": own[:3],
        "start_twist_world": own[3:6],
        "mask_own_kinematics": 1,
        "patch_distance": entry.get("patch_distance"),
        "patch_points": points,
        "patch_confidence": entry.get("patch_confidence"),
        "patch_normal_own_frame": normal_own,
        "patch_normal_sin_cos": normal_sin_cos,
        "patch_segment_own_frame": segment_own,
        "patch_centroid_own_frame": centroid_own,
        "contacting": int(bool(entry.get("contact"))),
        "contact_point_own_frame": (list(entry["contact_point_robot"])
                                    if entry.get("contact_point_robot") else None),
        "mask_patch_valid": int(points >= 3 and normal_own is not None),
        "mask_contact": int(bool(entry.get("contact"))),
    }


def _start_row(rows: Sequence[Dict[str, object]], robot_id: int) -> Optional[Dict[str, object]]:
    for row in rows:
        if _obs_entry(row, robot_id) is not None:
            return row
    return None


def _exit_block(rows: Sequence[Dict[str, object]], robot_id: int,
                start_pose: Sequence[float]) -> Dict[str, object]:
    """The last available on-board observation, re-expressed in the robot's start frame."""

    row = _last_observation_row(rows, robot_id)
    entry = _obs_entry(row, robot_id) if row is not None else None
    if entry is None:
        return {
            "obs_terminal_patch_points": 0, "obs_terminal_patch_distance": None,
            "obs_terminal_patch_normal_own_start_frame": None,
            "obs_terminal_patch_centroid_own_start_frame": None,
            "obs_terminal_contacting": 0, "obs_terminal_contact_point_own_start_frame": None,
            "mask_terminal_patch_valid": 0, "mask_terminal_contacting": 0,
        }
    own = list(entry.get("own") or [0.0] * 6)
    normal_world = entry.get("patch_normal")
    normal_start = None
    if normal_world is not None:
        normal_start = list(_rot(-start_pose[2],
                                 (float(normal_world[0]), float(normal_world[1]))))
    segment_world = entry.get("segment_world")
    centroid_start = None
    if segment_world:
        points = [_to_frame((float(p[0]), float(p[1])), start_pose) for p in segment_world]
        centroid_start = [sum(p[0] for p in points) / len(points),
                          sum(p[1] for p in points) / len(points)]
    contact_start = None
    if entry.get("contact_point_robot") is not None:
        local = entry["contact_point_robot"]
        world = _rot(own[2], (float(local[0]), float(local[1])))
        world = (world[0] + own[0], world[1] + own[1])
        contact_start = list(_to_frame(world, start_pose))
    points = int(entry.get("patch_points") or 0)
    return {
        "obs_terminal_patch_points": points,
        "obs_terminal_patch_distance": entry.get("patch_distance"),
        "obs_terminal_patch_normal_own_start_frame": normal_start,
        "obs_terminal_patch_centroid_own_start_frame": centroid_start,
        "obs_terminal_contacting": int(bool(entry.get("contact"))),
        "obs_terminal_contact_point_own_start_frame": contact_start,
        "mask_terminal_patch_valid": int(points >= 3 and normal_start is not None),
        "mask_terminal_contacting": int(bool(entry.get("contact"))),
    }


def _validity_block(rows: Sequence[Dict[str, object]], robot_id: int) -> Dict[str, object]:
    logged = 0
    valid = 0
    unavailable = 0
    for row in rows:
        entry = _obs_entry(row, robot_id)
        if entry is not None:
            logged += 1
            if int(entry.get("patch_points") or 0) >= 3 and entry.get("patch_normal") is not None:
                valid += 1
        if str(robot_id) in (row.get("observation_unavailable_for") or []):
            unavailable += 1
    return {
        "logged_periods": logged,
        "patch_valid_fraction": (valid / logged) if logged else 0.0,
        "unavailable_periods": unavailable,
    }


# ---------------------------------------------------------------------------
# one execution -> one record
# ---------------------------------------------------------------------------


def build_record(
    root,
    command: RB.JointCommand,
    record: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    option_config: OptionConfig,
    snapshot_hash: str,
    initial_object: Sequence[float],
    initial_robots: Sequence[Sequence[float]],
    template_name: str,
    n_commands: int = RB.N_COMMANDS,
) -> Dict[str, object]:
    outcome = record["arm_outcome"]
    metrics = record["metrics"]
    terminal_object = list(record["terminal_object"])
    terminal_robots = [list(row) for row in record["terminal_robots"]]
    active = list(root.active_robots)

    input_robots: Dict[str, object] = {}
    label_robots: Dict[str, object] = {}
    validity: Dict[str, object] = {}
    start_poses = [list(initial_robots[i])[:3] for i in (0, 1)]

    for robot_id in (0, 1):
        start = _start_block(rows, robot_id)
        exit_block = _exit_block(rows, robot_id, start["start_pose_world"])
        validity_block = _validity_block(rows, robot_id)
        validity[f"robot{robot_id}"] = {
            "patch_valid": exit_block["mask_terminal_patch_valid"],
            "contacting": exit_block["mask_terminal_contacting"],
            "patch_points": exit_block["obs_terminal_patch_points"],
            "patch_valid_fraction": validity_block["patch_valid_fraction"],
            "unavailable_periods": validity_block["unavailable_periods"],
        }
        input_robots[f"robot{robot_id}"] = dict(start, active=int(robot_id in active))

        pose_start = start_poses[robot_id]
        face = reference_face(initial_object, pose_start)
        start_distance, start_tangential = segment_geometry(initial_object, face, pose_start)
        terminal_pose = [float(v) for v in terminal_robots[robot_id][:3]]
        terminal_distance, terminal_tangential = segment_geometry(
            terminal_object, face, terminal_pose
        )
        _, normal_terminal, _ = face_frame(terminal_object, face)
        inward_world = (-normal_terminal[0], -normal_terminal[1])
        inward_start = _rot(-pose_start[2], inward_world)
        pose_change = list(_to_frame(terminal_pose, pose_start)) + [
            RB.wrap_angle(terminal_pose[2] - pose_start[2])
        ]
        velocity_start = list(_rot(-pose_start[2], (float(terminal_robots[robot_id][3]),
                                                    float(terminal_robots[robot_id][4]))))
        velocity_start.append(float(terminal_robots[robot_id][5]))
        observable = bool(
            exit_block["mask_terminal_patch_valid"]
            and _point_box_distance(terminal_pose, terminal_object) <= SCAN_RANGE
        )
        label_robots[f"robot{robot_id}"] = {
            "terminal_pose_change_own_start": pose_change,
            "terminal_residual_velocity_own_start": velocity_start,
            "reference_face": face,
            "start_signed_distance": start_distance,
            "terminal_signed_distance": terminal_distance,
            "start_tangential_offset": start_tangential,
            "terminal_tangential_offset": terminal_tangential,
            "segment_tangential_offset_change": terminal_tangential - start_tangential,
            "terminal_normal_own_start_frame": list(inward_start),
            "terminal_normal_sin_cos": [inward_start[1], inward_start[0]],
            "mask_terminal_geometry": int(observable),
        }
        label_robots[f"robot{robot_id}"].update(exit_block)

    template = TEMPLATES[template_name]
    label = {
        "duration_seconds": float(metrics["wall_clock_seconds"]),
        "duration_steps": int(metrics["wall_clock_steps"]),
        "stage_seconds": dict(outcome["stage_seconds"]),
        "exit_class": outcome["exit_reason"],
        "primary_exit_reason": outcome["primary_exit_reason"],
        "aborted_by": outcome["aborted_by"],
        "effective_push_completed": int(bool(outcome["push_completed"])),
        "effective_push_seconds": float(outcome["effective_push_seconds"]),
        "locally_settled": int(bool(outcome["locally_settled"])),
        "settled_observed": int(bool(outcome["settled_observed"])),
        "settled_after_abort": int(bool(outcome["settled_after_abort"])),
        "exit_truth_low_speed": int(bool(metrics["exit_truth_low_speed"])),
        "window_truth_low_speed": int(bool(metrics["window_truth_low_speed"])),
        "observation_validity": validity,
    }
    label.update({f"robot{robot_id}": label_robots[f"robot{robot_id}"] for robot_id in (0, 1)})

    truth = {
        "object_start_world": list(initial_object),
        "object_terminal_world": terminal_object,
        "object_relative_terminal_initial_frame": list(metrics["relative_object"]),
        "object_displacement": float(metrics["displacement"]),
        "object_rotation": float(metrics["rotation"]),
        "object_final_speed": float(metrics["final_speed"]),
        "object_final_omega": float(metrics["final_omega"]),
        "robots_start_world": [list(row) for row in initial_robots],
        "robots_terminal_world": terminal_robots,
        "robot_travel": list(metrics["robot_travel"]),
        "robot_terminal_speed": list(metrics["terminal_robot_speeds"]),
        "robot_terminal_omega": list(metrics["terminal_robot_omegas"]),
        "micro_motion": int(bool(metrics["micro_motion"])),
        "reached_effective_push": int(bool(metrics["reached_effective_push"])),
        "contact_period_ratio": float(metrics["contact_period_ratio"]),
        "push_contact_ratio": float(metrics["push_contact_ratio"]),
        "settle_evidence_span": metrics["settle_evidence_span"],
        "settle_evidence_lag": metrics["settle_evidence_lag"],
    }

    return {
        "record_id": f"{root.root_id}#cmd{command.index}",
        "version": VERSION,
        "root_id": root.root_id,
        "split": root.split,
        "interaction_class": root.interaction_class,
        "template": root.template,
        "command_index": command.index,
        "n_commands": int(n_commands),
        "arm": ARM,
        "variant": VARIANT_NAME,
        "snapshot_sha256": snapshot_hash,
        "initial_velocities": {
            key: list(value)
            for key, value in record["variant_report"]["initial_velocities"].items()
        },
        "input": {
            "command": command.as_dict(),
            "scene": {
                "template": template_name,
                "arena": list(template.arena),
                "obstacles": [[box.cx, box.cy, box.half_w, box.half_h, box.angle]
                              for box in template.obstacles],
                "object_size": [2.0 * OBJECT_HALF[0], 2.0 * OBJECT_HALF[1]],
                "robot_radius": ROBOT_RADIUS,
                "scan_range": SCAN_RANGE,
                "scan_rays": int(option_config.scan_rays),
                "control_dt": CONTROL_DT,
            },
            **input_robots,
        },
        "label": label,
        "truth": truth,
    }


def process_root_b(job: Dict[str, object]) -> Dict[str, object]:
    """Worker entry point: one root, its eight commands, one shared snapshot."""

    root = root_from_metadata(job["root"])
    base_config = default_config()
    option_config = OptionConfig(**job["option_config"])
    errors: List[Dict[str, object]] = []
    records: List[Dict[str, object]] = []
    try:
        snapshot, _config, initial_object, initial_robots, _report, _velocities = build_snapshot(
            root, VARIANT, base_config
        )
        snapshot_hash = snapshot_sha256(snapshot)
    except Exception:
        return {"root_id": root.root_id, "records": [], "errors": [{
            "root_id": root.root_id, "stage": "snapshot",
            "traceback": traceback.format_exc(),
        }]}

    for command_payload in job["commands"]:
        command = RB.JointCommand(
            index=int(command_payload["command_index"]),
            alpha=(command_payload["alpha_rad"][0], command_payload["alpha_rad"][1]),
            speed=(command_payload["speed"][0], command_payload["speed"][1]),
            active_robots=tuple(int(v) for v in command_payload["active_robots"]),
        )
        root_c = RB.apply_command(root, command)
        try:
            record, rows, _variant_report = run_execution(
                root_c, VARIANT, ARM, option_config, base_config, log_every=1,
            )
            records.append(build_record(
                root, command, record, rows, option_config, snapshot_hash,
                initial_object, initial_robots, root.template,
                n_commands=int(job.get("n_commands", len(job["commands"]))),
            ))
        except Exception:
            errors.append({
                "root_id": root.root_id, "command_index": command.index,
                "stage": "execution", "traceback": traceback.format_exc(),
            })
    return {"root_id": root.root_id, "records": records, "errors": errors}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))


def collect(
    out_dir: Path,
    split_roots: Dict[str, int],
    *,
    workers: int = 12,
    max_commands: int = RB.N_COMMANDS,
    tag: str = "phase_b",
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    roots, reports = RB.build_phase_b_roots(split_roots)
    leakage = RB.leakage_summary(roots)
    # prove the leakage guard is live, not dead code: a deliberately large threshold must
    # reject candidates instead of accepting them (the same seed stream reproduces the same
    # roots, so at least the first 16 candidates are rejected by construction)
    _probe_base, _ = RB.sample_split_roots("train", 16)
    _probe_roots, probe_report = RB.sample_split_roots(
        "train", 16, forbidden=tuple(_probe_base), min_distance=0.30,
    )
    guard_probe = {
        "min_distance": 0.30, "sampled": probe_report.sampled,
        "leakage_rejected": probe_report.leakage_rejected,
        "guard_is_live": bool(probe_report.leakage_rejected > 0),
    }

    roots_artifact = {
        "version": VERSION,
        "salt": RB.SALT_B,
        "split_roots_requested": dict(split_roots),
        "commands_per_root": max_commands,
        "min_root_distance": RB.MIN_ROOT_DISTANCE,
        "angle_scale": RB.ANG_SCALE,
        "sampling_reports": {
            name: dataclasses.asdict(report) for name, report in reports.items()
        },
        "leakage": leakage,
        "leakage_guard_probe": guard_probe,
        "command_choice_space": RB.command_choice_space(),
        "roots": {name: RB.root_records(values) for name, values in roots.items()},
    }
    roots_path = out_dir / "roots.json"
    _write_json(roots_path, roots_artifact)

    commands_artifact = {
        "version": VERSION,
        "commands_per_root": max_commands,
        "selection_rule": RB.command_choice_space()["selection_rule"],
        "roots": [
            command_record
            for name in ("train", "val", "test")
            for command_record in RB.commands_records(roots.get(name, []), max_commands)
        ],
    }
    commands_path = out_dir / "commands.json"
    _write_json(commands_path, commands_artifact)

    executions_paths: Dict[str, Path] = {}
    counts: Dict[str, Dict[str, object]] = {}
    all_errors: List[Dict[str, object]] = []
    for split in ("train", "val", "test"):
        split_root_list = roots.get(split, [])
        jobs = []
        for root in split_root_list:
            commands = [command.as_dict() for command in RB.commands_for_root(root, max_commands)]
            jobs.append({
                "root": RB.root_meta(root),
                "option_config": frozen_option_config().as_dict(),
                "commands": commands,
                "n_commands": max_commands,
            })
        records: List[Dict[str, object]] = []
        split_started = time.monotonic()
        if jobs:
            if workers <= 1:
                payloads = [process_root_b(job) for job in jobs]
                for payload in payloads:
                    records.extend(payload["records"])
                    all_errors.extend(payload["errors"])
            else:
                context = mp.get_context("fork")
                with context.Pool(processes=workers) as pool:
                    for index, payload in enumerate(
                        pool.imap_unordered(process_root_b, jobs, chunksize=1)
                    ):
                        records.extend(payload["records"])
                        all_errors.extend(payload["errors"])
                        if (index + 1) % 10 == 0:
                            print(f"  {split}: {index + 1}/{len(jobs)} roots done", flush=True)
        records.sort(key=lambda item: (item["root_id"], item["command_index"]))
        path = out_dir / f"executions_{split}.jsonl"
        with path.open("w") as handle:
            for item in records:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
        executions_paths[split] = path
        planned = len(split_root_list) * max_commands
        counts[split] = {
            "roots_planned": int(split_roots.get(split, 0)),
            "roots_sampled": len(split_root_list),
            "roots_failed_to_sample": int(split_roots.get(split, 0)) - len(split_root_list),
            "commands_per_root": max_commands,
            "executions_planned": planned,
            "executions_executed": len(records),
            "executions_dropped": planned - len(records),
            "wall_clock_seconds": time.monotonic() - split_started,
        }
        print(f"{split}: {len(records)} executions from {len(split_root_list)} roots", flush=True)

    total_wall = time.monotonic() - started
    sha256 = {
        "roots.json": file_sha256(roots_path),
        "commands.json": file_sha256(commands_path),
    }
    for split, path in executions_paths.items():
        sha256[path.name] = file_sha256(path)

    drop_reasons: Dict[str, int] = {}
    for error in all_errors:
        key = f"{error.get('stage')}:{error.get('traceback', '').strip().splitlines()[-1][:120]}"
        drop_reasons[key] = drop_reasons.get(key, 0) + 1

    ledger = {
        "version": VERSION,
        "phase": "B",
        "contract": "docs/option_prediction_mechanism_v1/CONTRACT.md",
        "task_book": "docs/tasks/20261013_1952_CoImagine_判稳修复与预测机制验证任务书.md",
        "git_commit": _git_commit(),
        "tag": tag,
        "arm": ARM,
        "variant": VARIANT_NAME,
        "controller": CONTROLLER,
        "settle_protocol": SETTLE,
        "sensing": dict(SENSING, scan_range=SCAN_RANGE, scan_rays=int(OptionConfig().scan_rays)),
        "option_config": frozen_option_config().as_dict(),
        "split_roots_requested": dict(split_roots),
        "splits": counts,
        "totals": {
            "roots_planned": sum(int(v) for v in split_roots.values()),
            "roots_sampled": sum(entry["roots_sampled"] for entry in counts.values()),
            "roots_failed_to_sample": sum(entry["roots_failed_to_sample"]
                                          for entry in counts.values()),
            "executions_planned": sum(entry["executions_planned"] for entry in counts.values()),
            "executions_executed": sum(entry["executions_executed"] for entry in counts.values()),
            "executions_dropped": sum(entry["executions_dropped"] for entry in counts.values()),
            "total_wall_clock_seconds": total_wall,
        },
        "executions_dropped_reasons": drop_reasons,
        "errors": all_errors,
        "sampling_failures": {
            name: {
                "sample_none": report.sample_none,
                "leakage_rejected": report.leakage_rejected,
                "unfilled": report.unfilled,
            }
            for name, report in reports.items()
        },
        "leakage_check": leakage,
        "leakage_guard_probe": guard_probe,
        "sha256": sha256,
        "artifacts": {
            "roots": str(roots_path), "commands": str(commands_path),
            "executions": {name: str(path) for name, path in executions_paths.items()},
        },
    }
    ledger_path = out_dir / "ledger.json"
    _write_json(ledger_path, ledger)
    ledger["ledger_path"] = str(ledger_path)
    return ledger


# ---------------------------------------------------------------------------
# spot check: distinct commands, one snapshot, exact replay
# ---------------------------------------------------------------------------


def spot_check(out_dir: Path, *, root_index: int = 0) -> Dict[str, object]:
    roots, _reports = RB.build_phase_b_roots({"train": max(root_index + 1, 1),
                                              "val": 0, "test": 0})
    root = roots["train"][root_index]
    commands = RB.commands_for_root(root)
    base_config = default_config()
    option_config = frozen_option_config()

    snapshot_a, _c, _io, _ir, _vr, _v = build_snapshot(root, VARIANT, base_config)
    snapshot_b, _c, _io, _ir, _vr, _v = build_snapshot(root, VARIANT, base_config)
    hash_a = snapshot_sha256(snapshot_a)
    hash_b = snapshot_sha256(snapshot_b)

    payload_commands = [command.as_dict() for command in commands]
    signatures = {command.signature() for command in commands}

    # exact replay of command 0: two independent executions must be identical
    root_c0 = RB.apply_command(root, commands[0])
    run_a, _rows_a, _vr = run_execution(root_c0, VARIANT, ARM, option_config, base_config)
    run_b, _rows_b, _vr = run_execution(root_c0, VARIANT, ARM, option_config, base_config)
    replay_identical = (
        list(run_a["terminal_object"]) == list(run_b["terminal_object"])
        and [list(row) for row in run_a["terminal_robots"]]
        == [list(row) for row in run_b["terminal_robots"]]
        and run_a["arm_outcome"]["exit_reason"] == run_b["arm_outcome"]["exit_reason"]
        and run_a["metrics"]["wall_clock_steps"] == run_b["metrics"]["wall_clock_steps"]
    )

    # the eight commands share the initial state but not the terminal outcome
    hashes = set()
    terminal_objects = []
    displacements = []
    for command in commands:
        root_c = RB.apply_command(root, command)
        snap, _c, _io, _ir, _vr, _v = build_snapshot(root_c, VARIANT, base_config)
        hashes.add(snapshot_sha256(snap))
        record, _rows, _vr = run_execution(root_c, VARIANT, ARM, option_config, base_config)
        terminal_objects.append(list(record["terminal_object"]))
        displacements.append(float(record["metrics"]["displacement"]))

    distinct_terminals = len({tuple(round(value, 9) for value in row)
                              for row in terminal_objects})
    payload = {
        "version": VERSION,
        "root_id": root.root_id,
        "interaction_class": root.interaction_class,
        "active_robots": list(root.active_robots),
        "commands": payload_commands,
        "distinct_command_signatures": len(signatures),
        "expected_commands": RB.N_COMMANDS,
        "snapshot_sha256_first": hash_a,
        "snapshot_sha256_repeat": hash_b,
        "snapshot_identical_across_commands": len(hashes) == 1,
        "snapshot_hashes": sorted(hashes),
        "replay_terminal_object_first": list(run_a["terminal_object"]),
        "replay_terminal_object_repeat": list(run_b["terminal_object"]),
        "replay_identical": bool(replay_identical),
        "command_object_displacement": displacements,
        "distinct_terminal_objects": distinct_terminals,
        "terminal_objects": terminal_objects,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "spot_check.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs/option_prediction_mechanism_v1/"
                    "option-prediction-mechanism-v1/phase_b"),
    )
    parser.add_argument("--train-roots", type=int, default=RB.SPLIT_ROOTS["train"])
    parser.add_argument("--val-roots", type=int, default=RB.SPLIT_ROOTS["val"])
    parser.add_argument("--test-roots", type=int, default=RB.SPLIT_ROOTS["test"])
    parser.add_argument("--max-commands", type=int, default=RB.N_COMMANDS)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--tag", default="phase_b")
    parser.add_argument("--spot-check", action="store_true",
                        help="write spot_check.json (eight commands, snapshot and replay) and exit")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if args.spot_check:
        payload = spot_check(out_dir)
        print(json.dumps(payload, indent=1, sort_keys=True))
        print("SPOT_CHECK_OK" if (
            payload["replay_identical"]
            and payload["snapshot_identical_across_commands"]
            and payload["distinct_command_signatures"] == RB.N_COMMANDS
        ) else "SPOT_CHECK_FAILED")
        return 0

    split_roots = {
        "train": args.train_roots, "val": args.val_roots, "test": args.test_roots,
    }
    ledger = collect(
        out_dir, split_roots, workers=args.workers,
        max_commands=args.max_commands, tag=args.tag,
    )
    print(json.dumps({
        "totals": ledger["totals"],
        "splits": ledger["splits"],
        "sha256": ledger["sha256"],
    }, indent=1, sort_keys=True))
    print("COLLECT_B_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
