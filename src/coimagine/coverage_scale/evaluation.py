"""Per-record metrics and the frozen root / cell weighting (task book C2, C3).

``fit_capacity.metrics.evaluate`` reports unweighted means over windows, which lets a long
trajectory's root dominate a cell. The task book freezes a different aggregation for every
decision of this round:

1. average within a root over its windows;
2. average within a ``family x mode`` cell over its roots;
3. combine cells with the frozen weights (each family 0.5; modes 0.5 / 0.25 / 0.25).

This module computes exactly that, from per-record arrays, and also exposes the unweighted
means so the two can be compared instead of confused.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..continuous2d.config import default_config
from ..continuous2d.scenes import scene_by_name
from ..continuous2d_learning.contract import TERMINATION_CLASSES, geometric_violation, wrap_angle
from ..continuous2d_learning.features import apply_state_delta
from ..continuous2d_learning.model import (
    JointWindowMLP,
    _tensors,
    boundary_after,
    boundary_before,
)
from ..fit_capacity.metrics import loss_components

FAMILY_WEIGHTS: Dict[str, float] = {"S0": 0.5, "S1": 0.5}
MODE_WEIGHTS: Dict[str, float] = {"broad": 0.5, "wall": 0.25, "goal": 0.25}
MAX_OBSTACLE_SLOTS = 2


def weights_dict() -> Dict[str, float]:
    return {
        f"{family}/{mode}": FAMILY_WEIGHTS[family] * MODE_WEIGHTS[mode]
        for family in FAMILY_WEIGHTS
        for mode in MODE_WEIGHTS
    }


def cell_of(record: Dict[str, object]) -> str:
    family = str(record.get("family") or str(record["scene_id"]).split("_")[0])
    mode = str(record.get("mode", "unknown"))
    return f"{family}/{mode}"


def per_record_metrics(
    model: JointWindowMLP,
    records: Sequence[Dict[str, object]],
    *,
    batch_size: int = 512,
    tensors: Optional[Tuple[torch.Tensor, ...]] = None,
) -> Dict[str, np.ndarray]:
    """One batched pass; every metric kept per record so the weighting can be applied later."""

    config = default_config()
    physics = config.physics
    if tensors is None:
        tensors = _tensors(records)
    n = len(records)
    out = {
        "state_loss": np.zeros(n),
        "hold_loss": np.zeros(n),
        "duration_loss": np.zeros(n),
        "termination_loss": np.zeros(n),
        "object_position": np.zeros(n),
        "robot0_position": np.zeros(n),
        "robot1_position": np.zeros(n),
        "object_angle": np.zeros(n),
        "invalid": np.zeros(n, dtype=bool),
        "non_finite": np.zeros(n, dtype=bool),
        "predicted_success": np.zeros(n, dtype=bool),
        "true_success": np.zeros(n, dtype=bool),
        "persistence_object": np.zeros(n),
        "persistence_robot0": np.zeros(n),
        "persistence_robot1": np.zeros(n),
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            index = slice(start, min(start + batch_size, n))
            outputs = model(tensors[0][index])
            components = loss_components(outputs, tuple(t[index] for t in tensors[1:]))
            out["state_loss"][index] = components[0].numpy()
            out["hold_loss"][index] = components[1].numpy()
            out["duration_loss"][index] = components[2].numpy()
            out["termination_loss"][index] = components[3].numpy()
            delta = outputs["state"].numpy()
            termination = outputs["termination"].argmax(dim=1).numpy()
            for row, record_index in enumerate(range(start, min(start + batch_size, n))):
                record = records[record_index]
                scene = scene_by_name(str(record["scene_id"]))
                before = boundary_before(record)
                actual = boundary_after(record)
                predicted = apply_state_delta(before, delta[row], scene)
                out["object_position"][record_index] = math.hypot(
                    predicted.object.x - actual.object.x, predicted.object.y - actual.object.y
                )
                for robot in (0, 1):
                    first, second = predicted.robots[robot], actual.robots[robot]
                    value = math.hypot(first.x - second.x, first.y - second.y)
                    out[f"robot{robot}_position"][record_index] = value
                out["object_angle"][record_index] = abs(
                    wrap_angle(predicted.object.angle - actual.object.angle)
                )
                out["persistence_object"][record_index] = math.hypot(
                    before.object.x - actual.object.x, before.object.y - actual.object.y
                )
                for robot in (0, 1):
                    first, second = before.robots[robot], actual.robots[robot]
                    out[f"persistence_robot{robot}"][record_index] = math.hypot(
                        first.x - second.x, first.y - second.y
                    )
                result = geometric_violation(predicted.object, predicted.robots, scene, physics)
                out["invalid"][record_index] = bool(result["violation"])
                out["non_finite"][record_index] = not bool(result["finite"])
                out["predicted_success"][record_index] = (
                    int(termination[row]) == TERMINATION_CLASSES.index("success")
                )
                out["true_success"][record_index] = str(record["termination_class"]) == "success"
    out["total_loss"] = (
        out["state_loss"] + out["hold_loss"] + out["duration_loss"] + out["termination_loss"]
    )
    out["position_mean"] = (
        out["object_position"] + out["robot0_position"] + out["robot1_position"]
    ) / 3.0
    return out


def _cell_means(metrics: Dict[str, np.ndarray], keys: Sequence[str], records) -> Dict[str, Dict[str, float]]:
    by_root: Dict[str, List[int]] = defaultdict(list)
    root_cell: Dict[str, str] = {}
    for index, record in enumerate(records):
        root_id = str(record["root_id"])
        by_root[root_id].append(index)
        root_cell[root_id] = cell_of(record)
    cell_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for root_id, indices in by_root.items():
        cell = root_cell[root_id]
        for key in keys:
            cell_values[cell][key].append(float(np.mean(metrics[key][indices])))
    return {
        cell: {key: float(np.mean(values[key])) for key in keys}
        for cell, values in cell_values.items()
    }


def weighted_summary(
    model: JointWindowMLP,
    records: Sequence[Dict[str, object]],
    *,
    tensors: Optional[Tuple[torch.Tensor, ...]] = None,
) -> Dict[str, object]:
    """The frozen root -> cell -> weight aggregation, plus the unweighted comparison."""

    metrics = per_record_metrics(model, records, tensors=tensors)
    keys = ("total_loss", "position_mean", "object_position", "robot0_position", "robot1_position",
            "object_angle", "invalid", "state_loss", "termination_loss", "persistence_object")
    cells = _cell_means(metrics, keys, records)
    weights = weights_dict()
    weighted: Dict[str, Optional[float]] = {}
    for key in keys:
        numerator = 0.0
        denominator = 0.0
        for cell, values in cells.items():
            weight = weights.get(cell)
            if weight is None or np.isnan(values[key]):
                continue
            numerator += weight * values[key]
            denominator += weight
        weighted[key] = (numerator / denominator) if denominator > 0 else None
    unweighted = {key: float(np.mean(metrics[key])) for key in keys}
    # The narrow-coverage control's *training* records carry mode ``fixed``, which has no
    # frozen family x mode weight. There the weighted figure falls back to the plain window
    # mean and says so, rather than silently reporting null.
    fallback = all(value is None for value in weighted.values()) or not any(
        cell in weights for cell in cells
    )
    if fallback:
        weighted = dict(unweighted)
    true_success = metrics["true_success"]
    n_success = int(true_success.sum())
    predicted = metrics["predicted_success"]
    true_positive = int(np.sum(true_success & predicted))
    return {
        "n": len(records),
        "independent_roots": len({str(record["root_id"]) for record in records}),
        "cell_weights": weights,
        "cells": cells,
        "weighted": weighted,
        "unweighted": unweighted,
        "weight_fallback_to_unweighted": fallback,
        "legality": {
            "invalid_rate_weighted": weighted["invalid"],
            "invalid_rate_unweighted": unweighted["invalid"],
            "non_finite_count": int(metrics["non_finite"].sum()),
        },
        "success": {
            "true_success_windows": n_success,
            "predicted_success_windows": int(predicted.sum()),
            "recall": (true_positive / n_success) if n_success else None,
            "recall_denominator": n_success,
        },
        "persistence_weighted_object_position": weighted["persistence_object"],
    }


def plain_summary(
    model: JointWindowMLP,
    records: Sequence[Dict[str, object]],
    *,
    tensors: Optional[Tuple[torch.Tensor, ...]] = None,
) -> Dict[str, object]:
    """Unweighted window means, for record sets that carry no root/cell metadata (old data)."""

    metrics = per_record_metrics(model, records, tensors=tensors)
    keys = ("total_loss", "position_mean", "object_position", "robot0_position", "robot1_position",
            "object_angle", "invalid", "state_loss", "termination_loss", "persistence_object")
    true_success = metrics["true_success"]
    predicted = metrics["predicted_success"]
    n_success = int(true_success.sum())
    true_positive = int(np.sum(true_success & predicted))
    return {
        "n": len(records),
        "weighted": {key: float(np.mean(metrics[key])) for key in keys},
        "unweighted": {key: float(np.mean(metrics[key])) for key in keys},
        "legality": {
            "invalid_rate_weighted": float(np.mean(metrics["invalid"])),
            "invalid_rate_unweighted": float(np.mean(metrics["invalid"])),
            "non_finite_count": int(metrics["non_finite"].sum()),
        },
        "success": {
            "true_success_windows": n_success,
            "predicted_success_windows": int(predicted.sum()),
            "recall": (true_positive / n_success) if n_success else None,
            "recall_denominator": n_success,
        },
    }


def group_by_cell(
    records: Sequence[Dict[str, object]]
) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        out[cell_of(record)].append(record)
    return dict(out)


def family_metrics(summary: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """Per-family P and invalid rate, weighting the modes by the frozen mode weights."""

    cells = summary["cells"]  # type: ignore[index]
    out: Dict[str, Dict[str, float]] = {}
    for family in ("S0", "S1"):
        numerator_position = 0.0
        numerator_invalid = 0.0
        denominator = 0.0
        for mode, weight in MODE_WEIGHTS.items():
            cell = cells.get(f"{family}/{mode}")
            if cell is None:
                continue
            numerator_position += weight * float(cell["position_mean"])
            numerator_invalid += weight * float(cell["invalid"])
            denominator += weight
        if denominator > 0:
            out[family] = {
                "position_mean": numerator_position / denominator,
                "invalid_rate": numerator_invalid / denominator,
            }
    return out


# ---------------------------------------------------------------------------
# Grouped metrics: object translation / rotation / stationary, contact / no contact
# ---------------------------------------------------------------------------

#: Frozen grouping thresholds, copied from ``contract`` (never re-tuned here).
GROUP_TRANSLATION_THRESHOLD = 0.01  # m
GROUP_ROTATION_THRESHOLD = 0.02     # rad
MOTION_GROUPS = ("translation", "rotation", "stationary")
CONTACT_GROUPS = ("contact", "no_contact")
GROUP_KEYS = (
    "object_position", "robot0_position", "robot1_position", "position_mean",
    "persistence_object", "persistence_robot0", "persistence_robot1",
    "invalid", "true_success", "object_angle",
)


def record_group(record: Dict[str, object]) -> Dict[str, bool]:
    """The two group partitions of one window.

    The motion partition and the contact partition are independent, so every window belongs to
    exactly one motion group and exactly one contact group, but the two partitions overlap.
    Inside the motion partition ``translation`` and ``rotation`` can both be true at once
    (they are not a partition into disjoint members -- only ``stationary`` is their
    complement); that overlap is measured and reported rather than hidden.
    """

    before, after = record["state"], record["next_state"]
    dx = float(after["object"][0]) - float(before["object"][0])
    dy = float(after["object"][1]) - float(before["object"][1])
    translation = math.hypot(dx, dy) > GROUP_TRANSLATION_THRESHOLD
    rotation = abs(wrap_angle(float(after["object"][2]) - float(before["object"][2]))) > GROUP_ROTATION_THRESHOLD
    contact = bool(record["contact"]["had_robot_object_contact"])
    return {
        "translation": translation,
        "rotation": rotation,
        "stationary": not translation and not rotation,
        "contact": contact,
        "no_contact": not contact,
    }


def grouped_summary(
    model: JointWindowMLP,
    records: Sequence[Dict[str, object]],
    *,
    tensors: Optional[Tuple[torch.Tensor, ...]] = None,
) -> Dict[str, object]:
    """Per-group body-level errors, persistence baselines and invalid rates.

    Every group reports the plain window mean *and* the frozen root -> cell -> weight
    aggregation, so a group dominated by a long trajectory's root cannot masquerade as the
    group's average.
    """

    metrics = per_record_metrics(model, records, tensors=tensors)
    groups = [record_group(record) for record in records]
    membership = {name: np.asarray([flag[name] for flag in groups], dtype=bool)
                  for name in MOTION_GROUPS + CONTACT_GROUPS}
    membership["all"] = np.ones(len(records), dtype=bool)
    weights = weights_dict()
    by_root_cell: Dict[str, str] = {}
    root_indices: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        root_id = str(record["root_id"])
        root_indices[root_id].append(index)
        by_root_cell[root_id] = cell_of(record)

    out: Dict[str, object] = {
        "n_windows": len(records),
        "n_roots": len(root_indices),
        "thresholds": {
            "translation_m": GROUP_TRANSLATION_THRESHOLD,
            "rotation_rad": GROUP_ROTATION_THRESHOLD,
        },
        "partitions": {
            "motion": list(MOTION_GROUPS),
            "contact": list(CONTACT_GROUPS),
            "note": "the two partitions are independent and overlap; translation and rotation "
                    "may both hold in one window, stationary is the complement of their union",
        },
        "overlap": {
            "translation_and_rotation": int(np.sum(membership["translation"] & membership["rotation"])),
            "translation_only": int(np.sum(membership["translation"] & ~membership["rotation"])),
            "rotation_only": int(np.sum(membership["rotation"] & ~membership["translation"])),
            "stationary": int(np.sum(membership["stationary"])),
            "contact_and_translation": int(np.sum(membership["contact"] & membership["translation"])),
            "no_contact_and_stationary": int(np.sum(membership["no_contact"] & membership["stationary"])),
        },
        "groups": {},
    }
    for name, mask in membership.items():
        indices = np.flatnonzero(mask)
        entry: Dict[str, object] = {"n_windows": int(indices.size)}
        if indices.size == 0:
            out["groups"][name] = entry
            continue
        for key in GROUP_KEYS:
            values = metrics[key][indices]
            entry[key] = float(np.mean(values)) if key != "invalid" and key != "true_success" else float(np.mean(values))
        entry["n_roots"] = len({str(records[index]["root_id"]) for index in indices})
        entry["success_windows"] = int(np.sum(metrics["true_success"][indices]))
        # root -> cell -> frozen weight aggregation, restricted to the group
        weighted: Dict[str, Optional[float]] = {}
        for key in GROUP_KEYS:
            cell_values: Dict[str, List[float]] = defaultdict(list)
            for root_id, root_index in root_indices.items():
                selected = [index for index in root_index if mask[index]]
                if not selected:
                    continue
                cell_values[by_root_cell[root_id]].append(float(np.mean(metrics[key][selected])))
            numerator = 0.0
            denominator = 0.0
            for cell, values in cell_values.items():
                weight = weights.get(cell)
                if weight is None:
                    continue
                numerator += weight * float(np.mean(values))
                denominator += weight
            weighted[key] = (numerator / denominator) if denominator > 0 else None
        if all(value is None for value in weighted.values()):
            weighted = {key: entry[key] for key in GROUP_KEYS}
            entry["weight_fallback_to_unweighted"] = True
        entry["weighted"] = weighted
        out["groups"][name] = entry
    return out
