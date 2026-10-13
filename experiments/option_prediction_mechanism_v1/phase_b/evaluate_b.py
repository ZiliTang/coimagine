"""Phase B evaluation: physical-unit errors, confusions, grouping, and same-root action spread.

Contract section 2.4: report the median and p90 of each continuous target's absolute error in its
own physical unit (metres, metres/second, seconds, or degrees for angles), the classification
confusion/counts/recall, the same grouping breakdowns, and the per-root aggregation. Three seeds
are always listed separately; a root's eight command branches are never treated as eight
independent experiments.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .model import GroupSpec


def _percentiles(values: np.ndarray, fractions: Sequence[float] = (0.5, 0.9)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"p{int(f * 100)}": None for f in fractions} | {"n": 0}
    out = {f"p{int(f * 100)}": float(np.quantile(values, f)) for f in fractions}
    out["n"] = int(values.size)
    return out


def continuous_errors(prediction: np.ndarray, truth_raw: np.ndarray, mask: np.ndarray,
                      spec: GroupSpec, columns: Sequence[str]) -> Dict[str, dict]:
    """Per-column absolute error in the target's own unit (angles stay in degrees)."""

    result: Dict[str, dict] = {}
    for index, name in enumerate(columns):
        valid = mask[:, index] > 0
        error = np.abs(prediction[valid, index] - truth_raw[valid, index])
        result[name] = _percentiles(error)
    return result


def angle_from_sincos(prediction: np.ndarray, truth_raw: np.ndarray, mask: np.ndarray,
                      sin_index: int, cos_index: int) -> dict:
    """Angular error in degrees from a (sin, cos) pair, wrapped to [0, 180]."""

    valid = (mask[:, sin_index] > 0) & (mask[:, cos_index] > 0)
    predicted = np.arctan2(prediction[valid, sin_index], prediction[valid, cos_index])
    actual = np.arctan2(truth_raw[valid, sin_index], truth_raw[valid, cos_index])
    difference = np.abs((predicted - actual + np.pi) % (2 * np.pi) - np.pi)
    return _percentiles(np.degrees(difference))


def classification_report(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                          num_classes: int, names: Sequence[str]) -> dict:
    valid = mask > 0
    if prediction.ndim == 1:
        # a baseline that emits a class index rather than logits
        predicted = prediction[valid].astype(int)
    else:
        predicted = np.argmax(prediction[valid], axis=1)
    actual = truth[valid].astype(int)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for a, p in zip(actual, predicted):
        confusion[a, p] += 1
    counts = {names[i]: int((actual == i).sum()) for i in range(num_classes)}
    recall = {}
    for i in range(num_classes):
        total = int((actual == i).sum())
        recall[names[i]] = float(confusion[i, i] / total) if total else None
    accuracy = float((predicted == actual).mean()) if valid.any() else None
    return {"confusion": confusion.tolist(), "class_names": list(names),
            "support": counts, "recall": recall, "accuracy": accuracy,
            "n": int(valid.sum())}


def binary_report(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                  columns: Sequence[str]) -> dict:
    probabilities = 1.0 / (1.0 + np.exp(-prediction))
    out: Dict[str, dict] = {}
    for index, name in enumerate(columns):
        valid = mask[:, index] > 0
        predicted = probabilities[valid, index] >= 0.5
        actual = truth[valid, index] >= 0.5
        true_positive = int((predicted & actual).sum())
        false_positive = int((predicted & ~actual).sum())
        false_negative = int((~predicted & actual).sum())
        true_negative = int((~predicted & ~actual).sum())
        positives = int(actual.sum())
        out[name] = {
            "n": int(valid.sum()), "positives": positives,
            "recall": float(true_positive / positives) if positives else None,
            "precision": float(true_positive / max(1, true_positive + false_positive)),
            "accuracy": float((predicted == actual).mean()) if valid.any() else None,
            "confusion": [[true_negative, false_positive], [false_negative, true_positive]],
        }
    return out


def per_root_error(error: np.ndarray, roots: Sequence[str]) -> dict:
    """Aggregate by root first, then across roots (CONTRACT section 2.4).

    A root's eight command branches are one experiment, so the root's own median error is the
    unit and the reported percentiles are taken over roots.
    """

    roots = np.asarray(roots)
    per_root = []
    for root in sorted(set(roots.tolist())):
        selected = error[roots == root]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            per_root.append(float(np.median(selected)))
    per_root = np.asarray(per_root)
    out = _percentiles(per_root)
    out["roots"] = int(per_root.size)
    return out


def grouped_errors(name: str, error: np.ndarray, groups: np.ndarray) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for value in sorted(set(groups.tolist())):
        selected = error[groups == value]
        out[str(value)] = _percentiles(selected)
    return out


def same_root_spread(roots: Sequence[str], commands: Sequence[str],
                     truth_raw: np.ndarray, mask: np.ndarray,
                     prediction: np.ndarray) -> dict:
    """Does the action actually change the terminal state, and does the model capture it?

    For every root with more than one distinct command this reports the spread of the *true*
    terminal values across the commands, and the spread of the model's predictions, so a model
    that ignores the action can be told apart from a task where the action does not matter.
    """

    roots = np.asarray(roots)
    commands = np.asarray(commands)
    truth_spreads: List[float] = []
    prediction_spreads: List[float] = []
    per_root: Dict[str, dict] = {}
    for root in sorted(set(roots.tolist())):
        selected = roots == root
        if len(set(commands[selected].tolist())) < 2:
            continue
        values = truth_raw[selected]
        valid = mask[selected] > 0
        if values.size == 0 or not valid.any():
            continue
        covered = valid.all(axis=0)
        if not covered.any():
            continue
        spread = np.nanmax(values[:, covered], axis=0) - np.nanmin(values[:, covered], axis=0)
        predicted = prediction[selected][:, covered]
        predicted_spread = (np.nanmax(predicted, axis=0) - np.nanmin(predicted, axis=0))
        truth_spreads.extend(spread.tolist())
        prediction_spreads.extend(predicted_spread.tolist())
        per_root[root] = {"n_commands": int(len(set(commands[selected].tolist()))),
                          "max_true_spread": float(np.max(spread)),
                          "max_predicted_spread": float(np.max(predicted_spread))}
    truth_spreads = np.asarray(truth_spreads, dtype=float)
    prediction_spreads = np.asarray(prediction_spreads, dtype=float)
    return {
        "roots_with_multiple_commands": len(per_root),
        "true_spread": _percentiles(truth_spreads),
        "predicted_spread": _percentiles(prediction_spreads),
        "fraction_of_targets_with_true_spread_above_zero": (
            float((truth_spreads > 1e-6).mean()) if truth_spreads.size else None),
        "mean_ratio_predicted_over_true_spread": (
            float(prediction_spreads.mean() / truth_spreads.mean())
            if truth_spreads.size and truth_spreads.mean() > 1e-12 else None),
        "example_roots": dict(list(per_root.items())[:3]),
    }
