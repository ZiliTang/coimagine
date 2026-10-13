"""D3: fixed-candidate decision validation on new test roots.

The candidate set, horizon, seeds and root selection are frozen before any model is scored:

* 12 test roots, 6 per family, chosen by ``sha256(root_id)`` (never by difficulty);
* the start of every root is that root's legal initial state, so the exact reference is one
  cached computation and the snapshot needs no prefix replay;
* ``H in {1, 3}``, ``K = 64``, candidate seeds ``{0, 1}``.

Both protocols of the corrected diagnostic are reported separately: the unfiltered ranking and
the execution-filtered ranking, with the exact decomposition
``misrejection + ranking == regret``. A start with no feasible candidate keeps ``null`` values
and is counted, never filled with 0 and never executed illegally.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..continuous2d.config import Continuous2DConfig, default_config
from ..continuous2d.environment import Continuous2DEnv
from ..continuous2d.scenes import scene_by_name
from ..continuous2d_learning.features import BoundaryState
from ..continuous2d_learning.model import JointWindowMLP, ModelConfig
from ..continuous2d_learning.pools import candidate_pool
from ..continuous2d_learning.rollout import evaluate_sequence_exact, evaluate_sequence_learned
from ..fit_capacity.candidates import feasibility_decomposition
from .plan import RootPlan

HORIZONS = (1, 3)
POOL_SIZE = 64
CANDIDATE_SEEDS = (0, 1)
ROOTS_PER_FAMILY = 6
SELECTION_SALT = "coverage-scale-decision-v1"


def select_test_starts(run_dir: Path, *, roots_per_family: int = ROOTS_PER_FAMILY):
    """The 12 fixed test roots, ordered by a stable hash of ``root_id`` within each family."""

    plan = RootPlan(run_dir / "roots")
    roots = plan.peek("test")
    out = []
    for family in ("S0", "S1"):
        group = [root for root in roots if root.family == family]
        group.sort(key=lambda root: hashlib.sha256(f"{SELECTION_SALT}|{root.root_id}".encode()).hexdigest())
        out.extend(group[:roots_per_family])
    return out


def make_start(root, config: Continuous2DConfig):
    scene = scene_by_name(root.scene_id)
    env = Continuous2DEnv(config)
    env.reset(scene)
    snapshot = env.export_state()
    observation = env.observation()
    return {
        "start_id": f"{root.root_id}/initial",
        "root_id": root.root_id,
        "scene_id": root.scene_id,
        "family": root.family,
        "mode": root.mode,
        "snapshot": snapshot,
        "observation": observation,
        "boundary": BoundaryState(
            object=snapshot.object, robots=snapshot.robots, success_counter=snapshot.success_counter
        ),
        "scene": scene,
    }


def build_reference(starts, config: Continuous2DConfig) -> Dict[str, object]:
    """Compute the exact true return of every frozen candidate exactly once."""

    planner = Continuous2DEnv(config)
    reference: Dict[str, object] = {}
    for start in starts:
        for horizon in HORIZONS:
            for seed in CANDIDATE_SEEDS:
                pool = candidate_pool(start["observation"], config, horizon, seed, cap=POOL_SIZE)
                rows = []
                for index, sequence in enumerate(pool[:POOL_SIZE]):
                    evaluation = evaluate_sequence_exact(
                        planner, start["snapshot"], start["scene"], sequence, config,
                        candidate_id=index,
                    )
                    rows.append({
                        "candidate_id": index,
                        "j": float(evaluation.j),
                        "success": int(evaluation.success),
                        "exit_class": evaluation.exit_class,
                        "feasible": bool(evaluation.feasible),
                    })
                reference[f"{start['start_id']}|{horizon}|{seed}"] = {
                    "start_id": start["start_id"],
                    "root_id": start["root_id"],
                    "scene_id": start["scene_id"],
                    "family": start["family"],
                    "mode": start["mode"],
                    "H": horizon,
                    "candidate_seed": seed,
                    "K": len(rows),
                    "candidates": rows,
                }
    return reference


def evaluate_model_on_starts(
    model: JointWindowMLP, starts, reference, config: Continuous2DConfig, *, label: str
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for start in starts:
        for horizon in HORIZONS:
            for seed in CANDIDATE_SEEDS:
                stored = reference[f"{start['start_id']}|{horizon}|{seed}"]
                pool = candidate_pool(start["observation"], config, horizon, seed, cap=POOL_SIZE)
                k = min(POOL_SIZE, len(pool))
                candidates = stored["candidates"][:k]
                j_ref = np.asarray([row["j"] for row in candidates], dtype=np.float64)
                success_ref = np.asarray([row["success"] for row in candidates], dtype=np.int64)
                evaluations = [
                    evaluate_sequence_learned(
                        model, start["scene"], start["boundary"], sequence, config, candidate_id=index
                    )
                    for index, sequence in enumerate(pool[:k])
                ]
                j_hat = np.asarray([evaluation.j for evaluation in evaluations], dtype=np.float64)
                feasible = [index for index, evaluation in enumerate(evaluations) if evaluation.feasible]
                best_all = int(np.argmax(j_ref))
                chosen_unfiltered = max(range(k), key=lambda index: (j_hat[index], -index))
                entry: Dict[str, object] = {
                    "label": label,
                    "start_id": start["start_id"],
                    "root_id": start["root_id"],
                    "scene_id": start["scene_id"],
                    "family": start["family"],
                    "mode": start["mode"],
                    "H": horizon,
                    "candidate_seed": seed,
                    "K": k,
                    "chosen_true_return": float(j_ref[chosen_unfiltered]),
                    "chosen_true_success": int(success_ref[chosen_unfiltered]),
                    "candidate_best_true_return": float(j_ref[best_all]),
                    "candidate_best_true_success": int(success_ref[best_all]),
                    "candidate_worst_true_return": float(np.min(j_ref)),
                    "uniform_candidate_true_return": float(np.mean(j_ref)),
                    "progress_spread": float(np.max(j_ref) - np.min(j_ref)),
                    "any_true_success": int(np.any(success_ref > 0)),
                    "true_success_count": int(np.sum(success_ref > 0)),
                    "selected_bias": float(j_hat[chosen_unfiltered] - j_ref[chosen_unfiltered]),
                    "candidate_mean_bias": float(np.mean(j_hat - j_ref)),
                    "selection_excess": float(
                        (j_hat[chosen_unfiltered] - j_ref[chosen_unfiltered]) - np.mean(j_hat - j_ref)
                    ),
                    "unfiltered_regret": float(j_ref[best_all] - j_ref[chosen_unfiltered]),
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
                    entry.update({
                        "feasible_best_true_return": float(j_ref[best_feasible]),
                        "feasible_chosen_true_return": float(j_ref[chosen]),
                        "feasible_chosen_true_success": int(success_ref[chosen]),
                        "misrejection_loss": split["misrejection_loss"],
                        "ranking_loss": split["ranking_loss"],
                        "execution_regret": split["execution_regret"],
                        "identity_residual": split["identity_residual"],
                    })
                rows.append(entry)
    return rows


METRIC_KEYS = (
    "chosen_true_return", "candidate_best_true_return", "uniform_candidate_true_return",
    "candidate_worst_true_return", "progress_spread", "selected_bias", "candidate_mean_bias",
    "selection_excess", "unfiltered_regret", "rejection_rate", "feasible_best_true_return",
    "feasible_chosen_true_return", "misrejection_loss", "ranking_loss", "execution_regret",
    "any_true_success", "true_success_count",
)


def aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Mean over candidate seeds within a start, then over starts; denominators recorded."""

    from collections import defaultdict

    by_label: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_label[str(row["label"])][str(row["start_id"])].append(row)
    out = []
    for label, per_start in sorted(by_label.items()):
        entry: Dict[str, object] = {"label": label, "n_starts": len(per_start)}
        for key in METRIC_KEYS:
            start_means = []
            for start_id, group in per_start.items():
                values = [float(row[key]) for row in group if row[key] is not None]
                if values:
                    start_means.append(float(np.mean(values)))
            entry[key] = float(np.mean(start_means)) if start_means else None
            entry[key + "_denominator"] = len(start_means)
        entry["starts_without_feasible"] = sum(
            1 for group in per_start.values() if all(bool(row["no_feasible_candidate"]) for row in group)
        )
        entry["starts_with_any_true_success"] = sum(
            1 for group in per_start.values() if any(int(row["any_true_success"]) for row in group)
        )
        all_rows = [row for group in per_start.values() for row in group]
        entry["rows"] = len(all_rows)
        entry["no_feasible_ratio"] = (
            sum(1 for row in all_rows if row["no_feasible_candidate"]) / len(all_rows)
            if all_rows else None
        )
        entry["no_true_success_ratio"] = (
            sum(1 for row in all_rows if not int(row["any_true_success"])) / len(all_rows)
            if all_rows else None
        )
        entry["mean_K"] = float(np.mean([row["K"] for row in all_rows])) if all_rows else None
        entry["mean_true_success_count"] = (
            float(np.mean([row["true_success_count"] for row in all_rows])) if all_rows else None
        )
        out.append(entry)
    return out


def aggregate_by_horizon(rows: Sequence[Dict[str, object]], horizon: int) -> List[Dict[str, object]]:
    """The same aggregation restricted to one planning horizon (H=1 and H=3 are reported apart)."""

    return aggregate([row for row in rows if int(row["H"]) == int(horizon)])


def load_checkpoint(path: Path) -> JointWindowMLP:
    import torch

    blob = torch.load(path, map_location="cpu")
    model = JointWindowMLP(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model
