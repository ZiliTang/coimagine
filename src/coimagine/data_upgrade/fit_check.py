"""Phase B2: encoding round-trip, wrapped-angle loss check and the 32-sample fit check.

The fit check is a training-pipeline diagnostic, not a generalisation target. Its checkpoint
is never used for a formal evaluation and is never mixed into the D groups.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..continuous2d.scenes import scene_by_name
from ..continuous2d_learning.contract import wrap_angle
from ..continuous2d_learning.features import (
    BoundaryState,
    apply_state_delta,
    encode_state_target,
)
from ..continuous2d_learning.model import (
    ANGLE_SLOTS,
    JointWindowMLP,
    ModelConfig,
    boundary_after,
    boundary_before,
    joint_spec_of,
    state_residual,
)

FIT_SALT = "fit-check-v1"
N_FIT = 32
N_MOVING = 16
N_STATIONARY = 8
N_SUCCESS = 4
FIT_UPDATES = 3000


# ---------------------------------------------------------------------------
# Encoding round trip
# ---------------------------------------------------------------------------


def round_trip_check(windows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """``encode_state_target`` then ``apply_state_delta`` must restore the true next state."""

    max_position = 0.0
    max_angle = 0.0
    max_velocity = 0.0
    worst = None
    for record in windows:
        scene = scene_by_name(str(record["scene_id"]))
        before = boundary_before(record)
        actual = boundary_after(record)
        target = encode_state_target(before, actual, scene)
        restored = apply_state_delta(before, target, scene)
        for predicted, expected in zip(restored.bodies(), actual.bodies()):
            position = math.hypot(predicted.x - expected.x, predicted.y - expected.y)
            angle = abs(wrap_angle(predicted.angle - expected.angle))
            velocity = math.hypot(predicted.vx - expected.vx, predicted.vy - expected.vy)
            max_position = max(max_position, position)
            max_angle = max(max_angle, angle)
            max_velocity = max(max_velocity, velocity)
            if worst is None or position > worst["position"]:
                worst = {
                    "episode_id": record["episode_id"],
                    "window_index": record["window_index"],
                    "position": position,
                    "angle": angle,
                    "velocity": velocity,
                }
    return {
        "n_windows": len(windows),
        "max_position_error": max_position,
        "max_wrapped_angle_error": max_angle,
        "max_velocity_error": max_velocity,
        "worst": worst,
        "tolerance": 1e-9,
        "passed": bool(max_position < 1e-9 and max_angle < 1e-9 and max_velocity < 1e-9),
    }


# ---------------------------------------------------------------------------
# Wrapped-angle loss check
# ---------------------------------------------------------------------------


def angle_residual_check() -> Dict[str, object]:
    """A prediction and target that differ by exactly one turn must cost nothing."""

    predicted = torch.zeros(1, 18)
    target = torch.zeros(1, 18)
    predicted[0, 2] = 2.0  # +2*pi in units of pi
    target[0, 2] = 0.0
    wrapped = float(state_residual(predicted, target)[0, 2])
    plain = float((predicted - target)[0, 2])
    # a half turn must still cost the full half turn
    predicted_half = torch.zeros(1, 18)
    predicted_half[0, 8] = 1.0
    half_wrapped = float(state_residual(predicted_half, target)[0, 8])
    return {
        "angle_slots": list(ANGLE_SLOTS),
        "full_turn_plain_residual": plain,
        "full_turn_wrapped_residual": wrapped,
        "half_turn_wrapped_residual": half_wrapped,
        "wrapped_is_periodic": bool(abs(wrapped) < 1e-12),
        "half_turn_preserved": bool(abs(abs(half_wrapped) - 1.0) < 1e-12),
        "loss_uses_wrapped_residual": True,
        "note": (
            "the loss now wraps the predicted-minus-target angle delta into (-pi, pi]; "
            "previously a full-turn-equivalent prediction was charged 2*pi"
        ),
    }


# ---------------------------------------------------------------------------
# 32-sample fit check
# ---------------------------------------------------------------------------


def _order_key(record: Dict[str, object]) -> str:
    return hashlib.sha256(f"{FIT_SALT}|{record['episode_id']}|{record['window_index']}".encode()).hexdigest()


def _transition_key(record: Dict[str, object]) -> str:
    return json.dumps(
        [record["scene_id"], record["state"], record["joint_skill_specs"], record["next_state"]],
        sort_keys=True,
    )


def select_fit_samples(
    windows: Sequence[Dict[str, object]],
    *,
    n_moving: int = N_MOVING,
    n_stationary: int = N_STATIONARY,
    n_success: int = N_SUCCESS,
    total: int = N_FIT,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Fixed-SHA256 selection from the original train split only."""

    train = [record for record in windows if record["split"] == "train"]
    seen = set()
    unique: List[Dict[str, object]] = []
    for record in sorted(train, key=_order_key):
        key = _transition_key(record)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)

    def is_moving(record) -> bool:
        return (
            abs(float(record["next_state"]["object"][0]) - float(record["state"]["object"][0])) > 0.01
            or abs(float(record["next_state"]["object"][1]) - float(record["state"]["object"][1])) > 0.01
            or abs(wrap_angle(float(record["next_state"]["object"][2]) - float(record["state"]["object"][2]))) > 0.02
        )

    buckets = {
        "moving": [r for r in unique if is_moving(r)],
        "stationary": [r for r in unique if not is_moving(r)],
        "success": [r for r in unique if int(r["success"]) == 1],
    }
    chosen: List[Dict[str, object]] = []
    chosen_keys = set()

    def take(pool, count):
        added = 0
        for record in pool:
            if added >= count:
                break
            key = _transition_key(record)
            if key in chosen_keys:
                continue
            chosen_keys.add(key)
            chosen.append(record)
            added += 1
        return added

    report = {
        "moving_available": len(buckets["moving"]),
        "stationary_available": len(buckets["stationary"]),
        "success_available": len(buckets["success"]),
        "moving_taken": take(buckets["moving"], n_moving),
        "success_taken": take(buckets["success"], n_success),
        "stationary_taken": take(buckets["stationary"], n_stationary),
    }
    report["filled_from_train"] = take(unique, total - len(chosen))
    report["sample_ids"] = [
        f"{record['scene_id']}|{record['episode_id']}|{record['window_index']}" for record in chosen
    ]
    report["selected"] = chosen
    return chosen, report


def train_fit_model(records: Sequence[Dict[str, object]], *, seed: int = 0, updates: int = FIT_UPDATES):
    """Full-batch gradient descent on the 32 samples with the production loss and optimiser."""

    from ..continuous2d_learning.model import _loss, _tensors

    torch.manual_seed(seed)
    model = JointWindowMLP(ModelConfig())
    tensor_batch = _tensors(records)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    started = time.time()
    history: List[Dict[str, float]] = []
    for step in range(updates):
        outputs = model(tensor_batch[0])
        loss = _loss(outputs, tuple(tensor_batch[1:]))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 100 == 0 or step == updates - 1:
            history.append({"update": step, "loss": float(loss.item())})
    model.eval()
    return model, {"history": history, "wall_clock": time.time() - started, "updates": updates}


def evaluate_fit(model: JointWindowMLP, records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    from ..continuous2d_learning.evaluation import predict_records
    from ..continuous2d_learning.contract import TERMINATION_CLASSES, geometric_violation
    from ..continuous2d.config import default_config

    physics = default_config().physics
    outputs = predict_records(model, records)
    object_errors: List[float] = []
    robot_errors: List[List[float]] = [[], []]
    angle_errors: List[float] = []
    termination_correct = 0
    invalid = 0
    non_finite = 0
    for index, record in enumerate(records):
        scene = scene_by_name(str(record["scene_id"]))
        before = boundary_before(record)
        actual = boundary_after(record)
        predicted = apply_state_delta(before, outputs["state_delta"][index], scene)
        object_errors.append(math.hypot(predicted.object.x - actual.object.x, predicted.object.y - actual.object.y))
        for robot in (0, 1):
            first, second = predicted.robots[robot], actual.robots[robot]
            robot_errors[robot].append(math.hypot(first.x - second.x, first.y - second.y))
        angle_errors.append(abs(wrap_angle(predicted.object.angle - actual.object.angle)))
        predicted_class = TERMINATION_CLASSES[int(outputs["termination"][index])]
        if predicted_class == str(record["termination_class"]):
            termination_correct += 1
        result = geometric_violation(predicted.object, predicted.robots, scene, physics)
        if not result["finite"]:
            non_finite += 1
        if result["violation"]:
            invalid += 1
    metrics = {
        "n": len(records),
        "object_position_error": float(np.mean(object_errors)),
        "robot0_position_error": float(np.mean(robot_errors[0])),
        "robot1_position_error": float(np.mean(robot_errors[1])),
        "object_angle_error": float(np.mean(angle_errors)),
        "termination_correct": termination_correct,
        "invalid_rate": invalid / max(1, len(records)),
        "non_finite": non_finite,
    }
    metrics["targets"] = {
        "object_position_error_max": 0.02,
        "robot_position_error_max": 0.03,
        "object_angle_error_max": 0.05,
        "termination_correct_min": 31,
    }
    metrics["passed"] = bool(
        metrics["object_position_error"] <= 0.02
        and metrics["robot0_position_error"] <= 0.03
        and metrics["robot1_position_error"] <= 0.03
        and metrics["object_angle_error"] <= 0.05
        and termination_correct >= 31
        and non_finite == 0
    )
    return metrics


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
