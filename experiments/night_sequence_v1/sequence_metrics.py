"""Metric helpers for round 5: percentiles, tie-correct rank correlation, action sensitivity.

Kept in one small module so the evaluation, the assets and the tests all use the same definitions --
in particular the same **average-rank** Spearman with an explicit ``nan`` when the ordering is
undefined, which was a corrected defect in the previous round.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence

import numpy as np


def percentiles(values: Sequence[float]) -> Dict[str, object]:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if array.size == 0:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    return {"n": int(array.size), "p50": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)), "max": float(array.max())}


def average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(len(array), dtype=float)
    index = 0
    while index < len(array):
        end = index
        while end + 1 < len(array) and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        ranks[order[index:end + 1]] = 0.5 * (index + end) + 1.0
        index = end + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ranks_a, ranks_b = average_ranks(a), average_ranks(b)
    if np.std(ranks_a) < 1e-12 or np.std(ranks_b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def action_sensitivity(roots: Sequence[str], truth: np.ndarray, predicted: np.ndarray,
                       mask: np.ndarray, columns: Sequence[str],
                       permutation: np.random.Generator | None = None) -> Dict[str, object]:
    difference: Dict[str, List[float]] = {name: [] for name in columns}
    correlation: Dict[str, List[float]] = {name: [] for name in columns}
    attempted: Dict[str, int] = {name: 0 for name in columns}
    root_array = np.asarray(roots)
    for root in sorted(set(root_array.tolist())):
        index = np.where(root_array == root)[0]
        if len(index) < 3:
            continue
        block = predicted[index].copy()
        if permutation is not None:
            block = block[permutation.permutation(len(index))]
        for column, name in enumerate(columns):
            keep = mask[index, column] > 0
            if keep.sum() < 3:
                continue
            true_values = truth[index][keep, column]
            predicted_values = block[keep, column]
            gaps = [abs((predicted_values[i] - predicted_values[j])
                        - (true_values[i] - true_values[j]))
                    for i, j in itertools.combinations(range(len(true_values)), 2)]
            if gaps:
                difference[name].append(float(np.median(gaps)))
            attempted[name] += 1
            value = spearman(true_values, predicted_values)
            if math.isfinite(value):
                correlation[name].append(value)
    return {"action_difference_error": {n: percentiles(v) for n, v in difference.items()},
            "rank_correlation": {n: {"mean": float(np.mean(v)) if v else None,
                                     "roots_defined": len(v), "roots_attempted": attempted[n]}
                                 for n, v in correlation.items()}}
