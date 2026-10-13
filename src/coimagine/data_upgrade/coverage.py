"""Phase C6: coverage report for Old / RandomAdd / BranchAdd."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..continuous2d.config import default_config
from ..continuous2d.scenes import scene_by_name
from ..continuous2d_learning.contract import wrap_angle

TRANSLATION_THRESHOLD = 0.01
ROTATION_THRESHOLD = 0.02
SUCCESS_WINDOW_MIN = 20
SUCCESS_ANCHOR_MIN = 4


def _object_corners(pose, size):
    x, y, angle = float(pose[0]), float(pose[1]), float(pose[2])
    half_w, half_h = size[0] / 2.0, size[1] / 2.0
    cos, sin = math.cos(angle), math.sin(angle)
    return [
        (x + dx * cos - dy * sin, y + dx * sin + dy * cos)
        for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
    ]


def near_success(record: Dict[str, object], config=None) -> bool:
    """All object corners inside the goal region but the real hold criterion not met."""

    if int(record["success"]) == 1:
        return False
    config = config or default_config()
    scene = scene_by_name(str(record["scene_id"]))
    corners = _object_corners(record["next_state"]["object"], (config.physics.object_width, config.physics.object_height))
    goal = scene.goal
    margin = config.physics.goal_margin
    for x, y in corners:
        if abs(x - goal.cx) > goal.half_w - margin or abs(y - goal.cy) > goal.half_h - margin:
            return False
    return True


def _combination(record: Dict[str, object]) -> str:
    kinds = [str(spec["kind"]) for spec in record["joint_skill_specs"]]  # type: ignore[index]
    pushes = kinds.count("push_contact")
    moves = kinds.count("move_to")
    holds = kinds.count("hold")
    if pushes == 2:
        return "double_push"
    if pushes == 1 and holds == 1:
        return "single_push"
    if pushes == 1 and moves == 1:
        return "push_move"
    if pushes == 1:
        return "single_push"
    if moves == 2:
        return "move_move"
    if moves == 1 and holds == 1:
        return "move_hold"
    if holds == 2:
        return "hold_hold"
    return "mixed"


def dataset_coverage(records: Sequence[Dict[str, object]], label: str, config=None) -> Dict[str, object]:
    config = config or default_config()
    scenes = set()
    episodes = set()
    anchors = set()
    states = set()
    state_actions = set()
    combos: Dict[str, int] = defaultdict(int)
    contact_types: Dict[str, int] = defaultdict(int)
    terminations: Dict[str, int] = defaultdict(int)
    translated = rotated = stationary = 0
    success_windows = near_success_windows = failure_windows = 0
    success_anchors = set()
    train_scene_success_anchors = set()
    for record in records:
        scenes.add(str(record["scene_id"]))
        episodes.add(str(record["episode_id"]))
        source = record.get("source", {}) or {}
        anchor = source.get("anchor_id")
        if anchor:
            anchors.add(str(anchor))
        states.add(str(record["state"]))
        state_actions.add(str(record["state"]) + "|" + str(record["joint_skill_specs"]))
        combos[_combination(record)] += 1
        pair = record["contact"]["penetration_by_pair"]
        if record["contact"]["had_robot_object_contact"]:
            contact_types["robot_object"] += 1
        for name, value in pair.items():
            if float(value) > 0.0:
                contact_types[str(name)] += 1
        terminations[str(record["termination_class"])] += 1
        before, after = record["state"], record["next_state"]
        dx = float(after["object"][0]) - float(before["object"][0])
        dy = float(after["object"][1]) - float(before["object"][1])
        rotation = abs(wrap_angle(float(after["object"][2]) - float(before["object"][2])))
        moved = math.hypot(dx, dy) > TRANSLATION_THRESHOLD
        turned = rotation > ROTATION_THRESHOLD
        translated += int(moved)
        rotated += int(turned)
        stationary += int(not moved and not turned)
        if int(record["success"]) == 1:
            success_windows += 1
            if anchor:
                success_anchors.add(str(anchor))
                if str(record["scene_id"]) in (
                    "S0_dev", "S0_v1", "S0_v2", "S1_dev", "S1_v1", "S1_v2",
                ):
                    train_scene_success_anchors.add(str(anchor))
        elif near_success(record, config):
            near_success_windows += 1
        if str(record["termination_class"]) == "physical_failure":
            failure_windows += 1
    limited = bool(success_windows < SUCCESS_WINDOW_MIN or len(train_scene_success_anchors) < SUCCESS_ANCHOR_MIN)
    return {
        "label": label,
        "scenes": sorted(scenes),
        "n_scenes": len(scenes),
        "episodes": len(episodes),
        "anchors": len(anchors),
        "distinct_states": len(states),
        "distinct_state_actions": len(state_actions),
        "windows": len(records),
        "object_translation": translated,
        "object_rotation": rotated,
        "stationary": stationary,
        "combinations": dict(combos),
        "contact_types": dict(contact_types),
        "terminations": dict(terminations),
        "success_windows": success_windows,
        "near_success_windows": near_success_windows,
        "failure_windows": failure_windows,
        "success_anchors": len(success_anchors),
        "train_scene_success_anchors": len(train_scene_success_anchors),
        "thresholds": {"translation_m": TRANSLATION_THRESHOLD, "rotation_rad": ROTATION_THRESHOLD},
        "success_coverage": "SUCCESS_COVERAGE_LIMITED" if limited else "SUCCESS_COVERAGE_OK",
    }


def anchor_progress_differences(
    records: Sequence[Dict[str, object]], config=None
) -> Dict[str, object]:
    """Same anchor, different first action: spread of real object progress."""

    config = config or default_config()
    by_anchor: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        source = record.get("source", {}) or {}
        if source.get("branch_window") == 0 and source.get("anchor_id"):
            by_anchor[str(source["anchor_id"])].append(record)
    rows = []
    for anchor, group in sorted(by_anchor.items()):
        progresses = []
        for record in group:
            before, after = record["state"], record["next_state"]
            translation = math.hypot(
                float(after["object"][0]) - float(before["object"][0]),
                float(after["object"][1]) - float(before["object"][1]),
            )
            rotation = abs(wrap_angle(float(after["object"][2]) - float(before["object"][2])))
            # passage distance change: negative means the object got closer to the goal
            scene = scene_by_name(str(record["scene_id"]))
            from ..continuous2d_learning.rollout import distance_of_state
            from ..continuous2d_learning.features import BoundaryState
            from ..continuous2d.backend import BodyState

            def _distance(payload):
                return distance_of_state(
                    BoundaryState(
                        object=BodyState(*[float(v) for v in payload["object"]]),
                        robots=tuple(BodyState(*[float(v) for v in row]) for row in payload["robots"]),
                        success_counter=0.0,
                    ),
                    scene,
                    config,
                )

            progress = _distance(before) - _distance(after)
            progresses.append(
                {
                    "branch_id": source.get("branch_id"),
                    "translation": translation,
                    "rotation": rotation,
                    "passage_progress": progress,
                    "success": int(record["success"]),
                    "termination": record["termination_class"],
                }
            )
        values = [row["passage_progress"] for row in progresses]
        rows.append(
            {
                "anchor_id": anchor,
                "n_actions": len(progresses),
                "progress_min": float(np.min(values)) if values else None,
                "progress_max": float(np.max(values)) if values else None,
                "progress_spread": float(np.max(values) - np.min(values)) if values else None,
                "actions": progresses,
            }
        )
    spreads = [row["progress_spread"] for row in rows if row["progress_spread"] is not None]
    return {
        "anchors": len(rows),
        "mean_progress_spread": float(np.mean(spreads)) if spreads else None,
        "max_progress_spread": float(np.max(spreads)) if spreads else None,
        "rows": rows,
    }
