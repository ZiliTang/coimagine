"""A4: paired candidate comparison on the common feasible subset.

The old comparison subtracted two aggregate means that had been computed on *different* row
sets. Here every pair is formed on ``(start_id, candidate_seed)`` and only kept when both
models have a feasible candidate on that row; the dropped rows are counted, never hidden, and
the uniform / candidate_best references are recomputed on the same subset.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PAIR_KEY = ("start_id", "candidate_seed")

UNFILTERED_KEYS = (
    "chosen_true_return", "uniform_candidate_true_return", "candidate_best_true_return",
    "selected_bias", "selection_excess", "unfiltered_regret", "candidate_mean_bias",
)
EXECUTION_KEYS = (
    "rejection_rate", "feasible_chosen_true_return", "misrejection_loss", "ranking_loss",
    "execution_regret",
)


def _key(row: Dict[str, object]) -> Tuple:
    return tuple(row[field] for field in PAIR_KEY)


def select_common(
    rows_a: Sequence[Dict[str, object]], rows_b: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """Keep only rows present in both and feasible in both."""

    index_a = {_key(row): row for row in rows_a}
    index_b = {_key(row): row for row in rows_b}
    both = sorted(set(index_a) & set(index_b))
    common = [
        key for key in both
        if not index_a[key]["no_feasible_candidate"] and not index_b[key]["no_feasible_candidate"]
    ]
    only_a = sorted(set(index_a) - set(index_b))
    only_b = sorted(set(index_b) - set(index_a))
    return {
        "rows_a": len(rows_a),
        "rows_b": len(rows_b),
        "keys_both": len(both),
        "common_feasible": len(common),
        "dropped_not_common": len(only_a) + len(only_b),
        "dropped_infeasible_a": sum(1 for key in both if index_a[key]["no_feasible_candidate"]),
        "dropped_infeasible_b": sum(1 for key in both if index_b[key]["no_feasible_candidate"]),
        "a_subset": [index_a[key] for key in common],
        "b_subset": [index_b[key] for key in common],
        "keys": common,
    }


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values]
    return float(np.mean(values)) if values else None


def summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Mean within one start over candidate seeds, then mean over starts, with denominators."""

    per_start: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        per_start[str(row["start_id"])].append(row)
    keys = UNFILTERED_KEYS + EXECUTION_KEYS
    out: Dict[str, object] = {
        "rows": len(rows),
        "starts": len(per_start),
        "starts_without_feasible": sum(
            1 for group in per_start.values() if all(bool(r["no_feasible_candidate"]) for r in group)
        ),
        "rows_without_feasible": sum(1 for row in rows if bool(row["no_feasible_candidate"])),
    }
    for key in keys:
        start_means = []
        for group in per_start.values():
            values = [r[key] for r in group if r[key] is not None]
            if values:
                start_means.append(_mean_or_none(values))
        out[key] = _mean_or_none([v for v in start_means if v is not None])
        out[key + "_n_starts"] = len([v for v in start_means if v is not None])
    return out


def compare(
    rows_a: Sequence[Dict[str, object]],
    rows_b: Sequence[Dict[str, object]],
    *,
    label_a: str,
    label_b: str,
) -> Dict[str, object]:
    """Full-sample rates plus common-subset paired deltas (b - a)."""

    common = select_common(rows_a, rows_b)
    summary_a_full = summarize_rows(rows_a)
    summary_b_full = summarize_rows(rows_b)
    summary_a = summarize_rows(common["a_subset"])
    summary_b = summarize_rows(common["b_subset"])
    deltas: Dict[str, object] = {}
    for key in UNFILTERED_KEYS + EXECUTION_KEYS:
        first, second = summary_a.get(key), summary_b.get(key)
        deltas[key] = None if first is None or second is None else float(second - first)
        deltas[key + "_n_starts"] = min(
            int(summary_a.get(key + "_n_starts", 0) or 0), int(summary_b.get(key + "_n_starts", 0) or 0)
        )
    # a->no-feasible / no-feasible->a transitions on the keys present in both
    index_a = {_key(row): row for row in rows_a}
    index_b = {_key(row): row for row in rows_b}
    both = sorted(set(index_a) & set(index_b))
    lost = sum(1 for key in both if not index_a[key]["no_feasible_candidate"] and index_b[key]["no_feasible_candidate"])
    gained = sum(1 for key in both if index_a[key]["no_feasible_candidate"] and not index_b[key]["no_feasible_candidate"])
    return {
        "label_a": label_a,
        "label_b": label_b,
        "full": {"a": summary_a_full, "b": summary_b_full},
        "common": {**{k: v for k, v in common.items() if not k.endswith("_subset")}},
        "summary_a_common": summary_a,
        "summary_b_common": summary_b,
        "delta_common": deltas,
        "feasible_lost": lost,
        "feasible_gained": gained,
    }
