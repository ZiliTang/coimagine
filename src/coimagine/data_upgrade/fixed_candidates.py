"""Phase D3: fixed-candidate re-evaluation of the trained groups on the frozen pools.

The truth (``J_ref``) is reused from the previous round's records after the rebuilt pools are
verified to match bit-for-bit. Both the original "rank all candidates by J" view and the A3
"feasible candidates only" view are reported, never merged.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..continuous2d.config import Continuous2DConfig, default_config
from ..continuous2d.skills import joint_spec_from_dicts
from ..continuous2d_learning.diagnostic import FEASIBILITY_KEYS, selection_metrics
from ..continuous2d_learning.features import BoundaryState
from ..continuous2d_learning.model import JointWindowMLP
from ..continuous2d_learning.pools import candidate_pool
from ..continuous2d_learning.rollout import evaluate_sequence_learned
from ..continuous2d_learning.starts import EvalStart, select_eval_starts

DEPTHS = (1, 3)
SIZES = (16, 128)
CANDIDATE_SEEDS = (0, 1, 2)


def load_reference(source_run: Path, config: Continuous2DConfig):
    records = [json.loads(line) for line in (source_run / "diagnostic" / "candidate_records.jsonl").open()]
    reference: Dict[Tuple[str, int, int], Dict[int, Tuple[List, float, str, int]]] = defaultdict(dict)
    for record in records:
        if int(record["H"]) not in DEPTHS:
            continue
        key = (str(record["start_id"]), int(record["H"]), int(record["candidate_seed"]))
        reference[key][int(record["candidate_id"])] = (
            record["sequence"], float(record["J_ref"]), str(record["J_ref_exit_class"]),
            int(record["J_ref_components"]["success"]),
        )
    return reference


def verify_and_rebuild(
    source_run: Path, config: Continuous2DConfig, reference
) -> Tuple[List[EvalStart], Dict[str, object]]:
    starts = select_eval_starts(
        source_run / "dataset" / "windows.jsonl", config=config
    )
    stored = json.loads((source_run / "diagnostic" / "eval_starts.json").read_text())
    stored_ids = [entry["start_id"] for entry in stored]
    verification = {
        "starts": len(starts),
        "start_ids_match": stored_ids == [start.start_id for start in starts],
        "pool_mismatches": 0,
        "pools_checked": 0,
        "candidate_count_mismatches": 0,
    }
    for start in starts:
        for depth in DEPTHS:
            for seed in CANDIDATE_SEEDS:
                key = (start.start_id, depth, seed)
                if key not in reference:
                    continue
                pool = candidate_pool(start.observation, config, depth, seed, cap=128)
                stored_sequences = reference[key]
                verification["pools_checked"] += 1
                if len(pool) != len(stored_sequences):
                    verification["candidate_count_mismatches"] += 1
                    continue
                for index, sequence in enumerate(pool):
                    if _serialise(sequence) != stored_sequences[index][0]:
                        verification["pool_mismatches"] += 1
                        break
    verification["reused"] = bool(
        verification["start_ids_match"]
        and verification["pool_mismatches"] == 0
        and verification["candidate_count_mismatches"] == 0
    )
    return starts, verification


def _serialise(sequence) -> List[List[Dict[str, object]]]:
    """Structured, full-precision sequence, in the stored candidate-record format."""

    from ..continuous2d.skills import joint_spec_to_dicts

    return [joint_spec_to_dicts(spec) for spec in sequence]


def _pool_sequences(start: EvalStart, config: Continuous2DConfig, depth: int, seed: int):
    return candidate_pool(start.observation, config, depth, seed, cap=128)


def evaluate_group(
    model: JointWindowMLP,
    starts: Sequence[EvalStart],
    reference,
    config: Continuous2DConfig,
    *,
    group: str,
    seed: int,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    from ..continuous2d.scenes import scene_by_name

    for start in starts:
        scene = scene_by_name(start.scene_id)
        start_boundary = BoundaryState(
            object=start.snapshot.object, robots=start.snapshot.robots,
            success_counter=start.snapshot.success_counter,
        )
        for depth in DEPTHS:
            for candidate_seed in CANDIDATE_SEEDS:
                pool = _pool_sequences(start, config, depth, candidate_seed)
                stored = reference[(start.start_id, depth, candidate_seed)]
                j_ref = np.asarray([stored[index][1] for index in range(len(pool))], dtype=np.float64)
                evaluations = [
                    evaluate_sequence_learned(model, scene, start_boundary, sequence, config, candidate_id=index)
                    for index, sequence in enumerate(pool)
                ]
                j_hat = np.asarray([evaluation.j for evaluation in evaluations], dtype=np.float64)
                for size in SIZES:
                    k = min(size, len(pool))
                    ids = list(range(k))
                    metrics = selection_metrics(ids, j_hat[:k], j_ref[:k])
                    chosen = int(metrics["chosen_index"])
                    feasible = [index for index, evaluation in enumerate(evaluations[:k]) if evaluation.feasible]
                    rejected = k - len(feasible)
                    entry = {
                        "group": group, "model_seed": seed, "start_id": start.start_id,
                        "H": depth, "K": size, "actual_K": k, "candidate_seed": candidate_seed,
                        **metrics,
                        "chosen_true_success": int(stored[chosen][3]),
                        "chosen_exit_class": evaluations[chosen].exit_class,
                        "candidate_true_failure_rate": float(
                            np.mean([float(stored[index][2] == "physical_failure") for index in range(k)])
                        ),
                        "candidate_invalid_rate": float(
                            np.mean([float(evaluation.exit_class in ("invalid_prediction", "non_finite_prediction")) for evaluation in evaluations[:k]])
                        ),
                        "feasible_count": len(feasible),
                        "rejected_count": rejected,
                        "rejection_rate": rejected / max(1, k),
                        "feasible_chosen_candidate_id": None,
                        "feasible_chosen_true_return": None,
                        "conditional_regret": None,
                    }
                    if feasible:
                        best = max(feasible, key=lambda index: (j_hat[index], -index))
                        entry.update(
                            {
                                "feasible_chosen_candidate_id": int(best),
                                "feasible_chosen_true_return": float(j_ref[best]),
                                "conditional_regret": float(np.max(j_ref[feasible]) - j_ref[best]),
                            }
                        )
                    rows.append(entry)
    return {"rows": rows}


def aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    metric_keys = (
        "selected_bias", "candidate_mean_bias", "selection_excess", "candidate_regret",
        "chosen_true_return", "uniform_candidate_true_return", "candidate_best_true_return",
        "chosen_true_success", "candidate_invalid_rate", "candidate_true_failure_rate",
        "rejection_rate", "feasible_count", "rejected_count",
        "feasible_chosen_true_return", "conditional_regret",
    )
    grouped: Dict[Tuple[str, int, int, int, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["group"]), int(row["H"]), int(row["K"]), int(row["model_seed"]), int(row["candidate_seed"]))].append(row)
    # mean within one start over candidate seeds, then mean over starts
    per_start: Dict[Tuple[str, int, int, int], Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        per_start[(str(row["group"]), int(row["H"]), int(row["K"]), int(row["model_seed"]))][str(row["start_id"])].append(row)
    out: List[Dict[str, object]] = []
    for (group, depth, size, seed), by_start in sorted(per_start.items()):
        entry: Dict[str, object] = {"group": group, "H": depth, "K": size, "model_seed": seed, "n_starts": len(by_start)}
        for key in metric_keys:
            values = []
            for start_id, group_rows in by_start.items():
                inner = [float(r[key]) for r in group_rows if r[key] is not None]
                if inner:
                    values.append(float(np.mean(inner)))
            entry[key] = float(np.mean(values)) if values else None
            entry[key + "_n_starts"] = len(values)
        out.append(entry)
    # mean over model seeds per group
    means: List[Dict[str, object]] = []
    for (group, depth, size) in sorted({(r["group"], r["H"], r["K"]) for r in out}):
        subset = [r for r in out if r["group"] == group and r["H"] == depth and r["K"] == size]
        entry = {"group": group, "H": depth, "K": size, "model_seed": "mean", "n_starts": subset[0]["n_starts"] if subset else 0}
        for key in metric_keys:
            values = [float(r[key]) for r in subset if r[key] is not None]
            entry[key] = float(np.mean(values)) if values else None
            entry[key + "_min"] = float(np.min(values)) if values else None
            entry[key + "_max"] = float(np.max(values)) if values else None
            entry[key + "_n_starts"] = len(values)
        means.append(entry)
    return out + means
