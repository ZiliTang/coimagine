"""Round-4 evaluation: the observable-terminal targets, in their own physical units.

Everything here works in **original units** for continuous groups (the same convention
``train_b`` documents), and every geometry error is restricted to the **true** visible mask -- never
to the model's own predicted visibility.

Reported per task book C2: the segment-midpoint 2-D Euclidean error plus its x and y components, the
normal angle error, the robot pose/velocity/duration errors, the visibility/contact/exit confusion
with support and recall, the same-root action-difference error and rank correlation with a label
permutation control, and the same numbers aggregated by execution and by root.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence

import numpy as np

from experiments.observation_terminal_v1.dataset_terminal import (
    EXIT_CLASSES,
    GEOMETRY_COLUMNS,
    POSE_COLUMNS,
    VALIDITY_COLUMNS,
    VELOCITY_COLUMNS,
)
from experiments.option_prediction_mechanism_v1.phase_b.action_sensitivity import spearman


def percentiles(values: Sequence[float]) -> dict:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if array.size == 0:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    return {"n": int(array.size), "p50": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)), "max": float(array.max())}


def _geometry_errors(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    out: Dict[str, dict] = {}
    euclidean: List[float] = []
    for robot_id in (0, 1):
        base = 4 * robot_id
        both = (mask[:, base] > 0) & (mask[:, base + 1] > 0)
        dx = np.abs(prediction[both, base] - truth[both, base])
        dy = np.abs(prediction[both, base + 1] - truth[both, base + 1])
        euclidean.extend(np.hypot(dx, dy).tolist())
        out[f"r{robot_id}_mid_x_m"] = percentiles(dx.tolist())
        out[f"r{robot_id}_mid_y_m"] = percentiles(dy.tolist())
        out[f"r{robot_id}_mid_2d_m"] = percentiles(np.hypot(dx, dy).tolist())
        valid = (mask[:, base + 2] > 0) & (mask[:, base + 3] > 0)
        predicted_norm = np.hypot(prediction[valid, base + 2], prediction[valid, base + 3])
        true_norm = np.hypot(truth[valid, base + 2], truth[valid, base + 3])
        angle_pred = np.arctan2(prediction[valid, base + 3], prediction[valid, base + 2])
        angle_true = np.arctan2(truth[valid, base + 3], truth[valid, base + 2])
        difference = np.abs((angle_pred - angle_true + np.pi) % (2 * np.pi) - np.pi)
        out[f"r{robot_id}_normal_angle_deg"] = percentiles(np.degrees(difference).tolist())
        out[f"r{robot_id}_normal_degenerate_rate"] = {
            "threshold": 0.1, "n": int(valid.sum()),
            "rate": float((predicted_norm < 0.1).mean()) if valid.any() else None,
            "true_norm_below_threshold": float((true_norm < 0.1).mean()) if valid.any() else None}
    out["both_robots_mid_2d_m"] = percentiles(euclidean)
    return out


def _continuous_errors(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                       columns: Sequence[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for index, name in enumerate(columns):
        valid = mask[:, index] > 0
        out[name] = percentiles(np.abs(prediction[valid, index] - truth[valid, index]).tolist())
    return out


def _classification(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                    num_classes: int, names: Sequence[str]) -> dict:
    """Class indices (1-D) or per-class scores (2-D) are both accepted; scores go through argmax.

    ``argmax`` is the same for logits and for probabilities, so no conversion is needed -- but the
    two must never be mixed with the binary path, where a sigmoid would be applied.
    """

    valid = mask > 0
    accepted = CLASS_INDEX if prediction.ndim == 1 else CLASS_SCORES
    if prediction.ndim == 1:
        predicted = prediction[valid].astype(int)
    else:
        predicted = np.argmax(prediction[valid], axis=1)
    actual = truth[valid].astype(int)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for a, p in zip(actual, predicted):
        confusion[a, p] += 1
    support = {names[i]: int((actual == i).sum()) for i in range(num_classes)}
    recall = {names[i]: (float(confusion[i, i] / support[names[i]]) if support[names[i]] else None)
              for i in range(num_classes)}
    return {"prediction_kind": accepted,
            "confusion": confusion.tolist(), "class_names": list(names), "support": support,
            "recall": recall, "accuracy": float((predicted == actual).mean()) if valid.any()
            else None, "n": int(valid.sum())}


#: the three prediction types the shared interface must keep apart
CONTINUOUS_UNITS = "original physical units"
BINARY_PROBABILITY = "probability in [0, 1]"
BINARY_LOGIT = "logit (any real number)"
CLASS_INDEX = "class index"
CLASS_SCORES = "per-class scores (logits or probabilities)"


def coerce_binary(prediction: np.ndarray, kind: str) -> np.ndarray:
    """Turn a declared binary prediction into probabilities, refusing to guess the type.

    A baseline emits 0/1 **probabilities**; the network emits **logits**. Applying a sigmoid to an
    already-binary baseline maps both 0 and 1 above 0.5, so every sample is called positive and the
    baseline comparison silently becomes meaningless. The declared kind is therefore required, and a
    mismatch between the declared kind and the values is an error rather than a silent conversion.
    """

    values = np.asarray(prediction, dtype=float)
    if kind == BINARY_PROBABILITY:
        if values.size and (values.min() < -1e-9 or values.max() > 1.0 + 1e-9):
            raise ValueError(
                f"declared probabilities outside [0, 1]: min={values.min()} max={values.max()}; "
                f"pass kind={BINARY_LOGIT!r} if these are logits")
        return values
    if kind == BINARY_LOGIT:
        return 1.0 / (1.0 + np.exp(-values))
    raise ValueError(f"unknown binary prediction kind {kind!r}")


def _binary(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray,
            columns: Sequence[str], kind: str) -> Dict[str, dict]:
    probabilities = coerce_binary(prediction, kind)
    out: Dict[str, dict] = {}
    for index, name in enumerate(columns):
        valid = mask[:, index] > 0
        predicted = probabilities[valid, index] >= 0.5
        actual = truth[valid, index] >= 0.5
        positives = int(actual.sum())
        true_positive = int((predicted & actual).sum())
        false_positive = int((predicted & ~actual).sum())
        false_negative = int((~predicted & actual).sum())
        true_negative = int((~predicted & ~actual).sum())
        out[name] = {
            "prediction_kind": kind,
            "n": int(valid.sum()), "positives": positives,
            "predicted_positives": int(predicted.sum()),
            "recall": float(true_positive / positives) if positives else None,
            "precision": float(true_positive / max(1, true_positive + false_positive)),
            "accuracy": float((predicted == actual).mean()) if valid.any() else None,
            "confusion": [[true_negative, false_positive], [false_negative, true_positive]],
            "evidence_insufficient": positives < 10}
    return out


def _by_root(values: np.ndarray, roots: Sequence[str]) -> dict:
    roots = np.asarray(roots)
    per_root: List[float] = []
    for root in sorted(set(roots.tolist())):
        selected = values[roots == root]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            per_root.append(float(np.median(selected)))
    out = percentiles(per_root)
    out["roots"] = len(per_root)
    return out


def action_sensitivity(roots: Sequence[str], truth: np.ndarray, predicted: np.ndarray,
                       mask: np.ndarray, columns: Sequence[str],
                       permutation: np.random.Generator | None = None) -> dict:
    """Same-root action-difference error and rank correlation, with an optional permutation."""

    difference: Dict[str, List[float]] = {name: [] for name in columns}
    correlation: Dict[str, List[float]] = {name: [] for name in columns}
    attempted: Dict[str, int] = {name: 0 for name in columns}
    roots = np.asarray(roots)
    for root in sorted(set(roots.tolist())):
        index = np.where(roots == root)[0]
        if len(index) < 3:
            continue
        block_predicted = predicted[index].copy()
        if permutation is not None:
            block_predicted = block_predicted[permutation.permutation(len(index))]
        for column, name in enumerate(columns):
            keep = mask[index, column] > 0
            if keep.sum() < 3:
                continue
            true_values = truth[index][keep, column]
            predicted_values = block_predicted[keep, column]
            differences = [abs((predicted_values[i] - predicted_values[j])
                               - (true_values[i] - true_values[j]))
                           for i, j in itertools.combinations(range(len(true_values)), 2)]
            if differences:
                difference[name].append(float(np.median(differences)))
            attempted[name] += 1
            value = spearman(true_values, predicted_values)
            if math.isfinite(value):
                correlation[name].append(value)
    return {
        "action_difference_error": {name: percentiles(values)
                                    for name, values in difference.items()},
        "rank_correlation": {
            name: {"mean": float(np.mean(values)) if values else None,
                   "median": float(np.median(values)) if values else None,
                   "roots_defined": len(values), "roots_attempted": attempted[name],
                   "roots_undefined": attempted[name] - len(values)}
            for name, values in correlation.items()},
    }


def evaluate(data, observations: Sequence[dict], predictions: Dict[str, np.ndarray], split: str,
             label: str, *, permutation_seed: int | None = None,
             start_patch_valid: Sequence[bool] | None = None,
             binary_kind: str = BINARY_LOGIT) -> dict:
    """Score one model or baseline.

    ``predictions`` are in **original units** for continuous groups, so every comparison here uses
    ``y_raw`` (the pre-standardisation copy). Comparing a raw prediction against the standardised
    ``y`` would silently rescale every error -- the same class of mistake the previous round made
    with the persistence baseline, and the reason the B1 unit check now asserts ``y != y_raw``.
    """


    index = np.where(data.split == split)[0]
    geometry_truth = data.targets["terminal_observation_geometry"]["y_raw"]
    geometry_mask = data.targets["terminal_observation_geometry"]["mask"]
    geometry_pred = predictions["terminal_observation_geometry"]
    out: Dict[str, object] = {
        "label": label, "split": split, "n": int(len(index)),
        "n_roots": len({data.roots[i] for i in index}),
        "terminal_observation_geometry": _geometry_errors(
            geometry_pred[index], geometry_truth[index], geometry_mask[index]),
        "robot_pose_delta": _continuous_errors(
            predictions["robot_pose_delta"][index], data.targets["robot_pose_delta"]["y_raw"][index],
            data.targets["robot_pose_delta"]["mask"][index], POSE_COLUMNS),
        "robot_residual_velocity": _continuous_errors(
            predictions["robot_residual_velocity"][index],
            data.targets["robot_residual_velocity"]["y_raw"][index],
            data.targets["robot_residual_velocity"]["mask"][index], VELOCITY_COLUMNS),
        "duration": _continuous_errors(
            predictions["duration"][index], data.targets["duration"]["y_raw"][index],
            data.targets["duration"]["mask"][index], ["duration_s"]),
        "observation_validity": _binary(
            predictions["observation_validity"][index],
            data.targets["observation_validity"]["y"][index],
            data.targets["observation_validity"]["mask"][index], VALIDITY_COLUMNS, binary_kind),
        "exit_class": _classification(
            predictions["exit_class"][index], data.targets["exit_class"]["y"][index],
            data.targets["exit_class"]["mask"][index], len(EXIT_CLASSES), EXIT_CLASSES),
    }
    # root aggregation of the headline geometry errors
    base = 0
    both = (geometry_mask[index][:, base] > 0) & (geometry_mask[index][:, base + 1] > 0)
    midpoint_error = np.hypot(geometry_pred[index][:, base] - geometry_truth[index][:, base],
                              geometry_pred[index][:, base + 1] - geometry_truth[index][:, base + 1])
    out["per_root"] = {
        "r0_mid_2d_m": _by_root(midpoint_error[both],
                                np.asarray([data.roots[i] for i in index])[both]),
    }
    # action-difference and rank metrics on the midpoint x/y columns
    generator = np.random.default_rng(permutation_seed) if permutation_seed is not None else None
    columns = ["r0_mid_x", "r0_mid_y", "r1_mid_x", "r1_mid_y"]
    column_index = [0, 1, 4, 5]
    truth_columns = np.stack([geometry_truth[index][:, c] for c in column_index], axis=1)
    predicted_columns = np.stack([geometry_pred[index][:, c] for c in column_index], axis=1)
    mask_columns = np.stack([geometry_mask[index][:, c] for c in column_index], axis=1)
    roots_index = [data.roots[i] for i in index]
    out["action_sensitivity"] = action_sensitivity(roots_index, truth_columns, predicted_columns,
                                                   mask_columns, columns)
    if generator is not None:
        out["action_sensitivity_permuted"] = action_sensitivity(
            roots_index, truth_columns, predicted_columns, mask_columns, columns,
            permutation=generator)
    # groupings: end type, visibility transition, configuration class
    end = np.asarray([
        "normal" if observations[i]["exit_class"] == "settled_observed" else
        "after_abort" if observations[i]["settled_after_abort"] else "unsettled"
        for i in index])
    visibility_by_robot = {}
    for robot_id in (0, 1):
        base = 4 * robot_id
        robot_error = np.hypot(
            geometry_pred[index][:, base] - geometry_truth[index][:, base],
            geometry_pred[index][:, base + 1] - geometry_truth[index][:, base + 1])
        robot_both = ((geometry_mask[index][:, base] > 0)
                      & (geometry_mask[index][:, base + 1] > 0))
        if start_patch_valid is None:
            transition = np.asarray(["unknown->unknown"] * len(index))
        else:
            start = np.asarray(start_patch_valid)
            if start.ndim == 1:
                start = np.stack([start, start], axis=1)
            transition = np.asarray([
                f"{'visible' if start[i, robot_id] else 'invisible'}->"
                f"{'visible' if geometry_mask[i, base] > 0 else 'invisible'}" for i in index])
        visibility_by_robot[f"r{robot_id}"] = _grouped(robot_error, robot_both, transition,
                                                       np.asarray(data.roots)[index])
    configuration = np.asarray([observations[i]["interaction_class"] for i in index])
    index_roots = np.asarray(data.roots)[index]
    out["grouped_mid_2d"] = {
        "by_end": _grouped(midpoint_error, both, end, index_roots),
        "by_visibility_transition_r0": visibility_by_robot["r0"],
        "by_visibility_transition_r1": visibility_by_robot["r1"],
        "by_configuration": _grouped(midpoint_error, both, configuration, index_roots),
    }
    out["geometry_by_end"] = _grouped(midpoint_error, both, end, index_roots)
    # per-robot visibility denominators, so the report cannot mix sample and slot percentages
    out["visibility_denominators"] = {}
    for robot_id in (0, 1):
        base = 4 * robot_id
        robot_both = (geometry_mask[index][:, base] > 0) & (geometry_mask[index][:, base + 1] > 0)
        start_valid = (np.asarray(start_patch_valid)[index][:, robot_id] > 0
                       if start_patch_valid is not None else None)
        out["visibility_denominators"][f"r{robot_id}"] = {
            "terminal_visible_samples": int(robot_both.sum()),
            "terminal_invisible_samples": int((~robot_both).sum()),
            "start_invisible_terminal_visible_samples":
                int((~start_valid & robot_both).sum()) if start_valid is not None else None,
            "terminal_invisible_root_count":
                int(len(set(index_roots[~robot_both].tolist()))),
            "all_samples": int(len(index)),
        }
    return out


def _grouped(error: np.ndarray, valid: np.ndarray, groups: np.ndarray,
             roots: Sequence[str] | None = None) -> Dict[str, dict]:
    """Per-group error summary with the **sample count and the independent-root count**.

    Both are reported because a group with few samples can also have very few distinct roots, and a
    percentage quoted over samples must not be read as a percentage over independent starts.
    """

    out: Dict[str, dict] = {}
    root_array = None if roots is None else np.asarray(roots)
    for value in sorted(set(groups[valid].tolist())):
        selected = error[valid][groups[valid] == value]
        entry = percentiles(selected.tolist())
        entry["samples"] = int(selected.size)
        entry["roots"] = (int(len(set(root_array[valid][groups[valid] == value].tolist())))
                          if root_array is not None else None)
        out[str(value)] = entry
    return out
