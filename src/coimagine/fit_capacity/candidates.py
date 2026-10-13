"""Phase D2: fixed-candidate review for the 12 ``best_mixed_val`` models.

Two protocols are reported separately and never merged:

* **unfiltered** — rank every candidate by ``J_hat`` (the original diagnostic);
* **execution** — keep only candidates the model calls feasible, then pick the best of those.

The feasibility split is decomposed exactly:
``misrejection_loss = best_true(all) - best_true(kept)`` and
``ranking_loss = best_true(kept) - chosen_true(kept)``; their sum is the regret relative to
choosing the best feasible candidate's true value, i.e. the total gap to the pool optimum.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..continuous2d.scenes import scene_by_name
from ..continuous2d.config import Continuous2DConfig
from ..continuous2d_learning.features import BoundaryState
from ..continuous2d_learning.model import JointWindowMLP
from ..continuous2d_learning.pools import candidate_pool
from ..continuous2d_learning.rollout import evaluate_sequence_learned
from ..continuous2d_learning.starts import EvalStart

DEPTH = 3
SIZE = 128
CANDIDATE_SEEDS = (0, 1, 2)


def feasibility_decomposition(
    j_ref: np.ndarray, best_all: int, best_feasible: int, chosen_feasible: int
) -> Dict[str, float]:
    """Exact split: misrejection + within-set ranking == regret versus the pool optimum."""

    misrejection = float(j_ref[best_all] - j_ref[best_feasible])
    ranking = float(j_ref[best_feasible] - j_ref[chosen_feasible])
    return {
        "misrejection_loss": misrejection,
        "ranking_loss": ranking,
        "execution_regret": float(j_ref[best_all] - j_ref[chosen_feasible]),
        "identity_residual": float(misrejection + ranking - (j_ref[best_all] - j_ref[chosen_feasible])),
    }


def pair_episodes(rows):
    """Closed-loop pairing on (scene_id, episode_seed, H, model_seed)."""

    def key(row):
        return (row["scene_id"], row["episode_seed"], row["H"], row["model_seed"])

    return [key(row) for row in rows]


def evaluate_model(
    model: JointWindowMLP,
    starts: Sequence[EvalStart],
    reference,
    config: Continuous2DConfig,
    *,
    label: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for start in starts:
        scene = scene_by_name(start.scene_id)
        start_boundary = BoundaryState(
            object=start.snapshot.object, robots=start.snapshot.robots,
            success_counter=start.snapshot.success_counter,
        )
        for candidate_seed in CANDIDATE_SEEDS:
            key = (start.start_id, DEPTH, candidate_seed)
            if key not in reference:
                continue
            stored = reference[key]
            pool = candidate_pool(start.observation, config, DEPTH, candidate_seed, cap=SIZE)
            k = min(SIZE, len(pool))
            j_ref = np.asarray([stored[index][1] for index in range(k)], dtype=np.float64)
            true_success = np.asarray([stored[index][3] for index in range(k)], dtype=np.int64)
            evaluations = [
                evaluate_sequence_learned(model, scene, start_boundary, sequence, config, candidate_id=index)
                for index, sequence in enumerate(pool[:k])
            ]
            j_hat = np.asarray([evaluation.j for evaluation in evaluations], dtype=np.float64)
            feasible = [index for index, evaluation in enumerate(evaluations) if evaluation.feasible]
            best_all = int(np.argmax(j_ref))
            chosen_unfiltered = max(range(k), key=lambda index: (j_hat[index], -index))
            selected_bias = float(j_hat[chosen_unfiltered] - j_ref[chosen_unfiltered])
            candidate_mean_bias = float(np.mean(j_hat - j_ref))
            entry: Dict[str, object] = {
                "label": label,
                "start_id": start.start_id,
                "scene_id": start.scene_id,
                "candidate_seed": candidate_seed,
                "H": DEPTH,
                "K": SIZE,
                "actual_K": k,
                # unfiltered diagnostic
                "chosen_true_return": float(j_ref[chosen_unfiltered]),
                "chosen_true_success": int(true_success[chosen_unfiltered]),
                "candidate_best_true_return": float(j_ref[best_all]),
                "uniform_candidate_true_return": float(np.mean(j_ref)),
                "selected_bias": selected_bias,
                "candidate_mean_bias": candidate_mean_bias,
                "selection_excess": selected_bias - candidate_mean_bias,
                "unfiltered_regret": float(j_ref[best_all] - j_ref[chosen_unfiltered]),
                # execution protocol
                "feasible_count": len(feasible),
                "rejection_rate": (k - len(feasible)) / k,
                "no_feasible_candidate": len(feasible) == 0,
                "feasible_best_true_return": None,
                "feasible_chosen_true_return": None,
                "feasible_chosen_true_success": None,
                "misrejection_loss": None,
                "ranking_loss": None,
                "execution_regret": None,
            }
            if feasible:
                best_feasible = max(feasible, key=lambda index: (j_ref[index], -index))
                chosen = max(feasible, key=lambda index: (j_hat[index], -index))
                split = feasibility_decomposition(j_ref, best_all, best_feasible, chosen)
                entry.update(
                    {
                        "feasible_best_true_return": float(j_ref[best_feasible]),
                        "feasible_chosen_true_return": float(j_ref[chosen]),
                        "feasible_chosen_true_success": int(true_success[chosen]),
                        "misrejection_loss": split["misrejection_loss"],
                        "ranking_loss": split["ranking_loss"],
                        "execution_regret": split["execution_regret"],
                    }
                )
            rows.append(entry)
    return rows


METRIC_KEYS = (
    "chosen_true_return", "candidate_best_true_return", "uniform_candidate_true_return",
    "selected_bias", "candidate_mean_bias", "selection_excess", "unfiltered_regret",
    "feasible_count", "rejection_rate", "feasible_best_true_return", "feasible_chosen_true_return",
    "misrejection_loss", "ranking_loss", "execution_regret",
)


def aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Mean within one start over candidate seeds, then mean over starts, with denominators."""

    by_label: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_label[str(row["label"])][str(row["start_id"])].append(row)
    out = []
    for label, per_start in sorted(by_label.items()):
        entry: Dict[str, object] = {"label": label, "n_starts": len(per_start)}
        for key in METRIC_KEYS:
            start_means = []
            for start_id, group in per_start.items():
                values = [float(r[key]) for r in group if r[key] is not None]
                if values:
                    start_means.append(float(np.mean(values)))
            entry[key] = float(np.mean(start_means)) if start_means else None
            entry[key + "_n_starts"] = len(start_means)
        entry["starts_without_feasible"] = sum(
            1 for group in per_start.values() if all(bool(r["no_feasible_candidate"]) for r in group)
        )
        out.append(entry)
    return out
