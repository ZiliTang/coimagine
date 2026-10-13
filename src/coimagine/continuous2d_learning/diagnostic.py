"""Phase C: fixed-candidate ranking diagnostic for the continuous platform.

Contract: ``docs/forecast_baseline_v1/CONTRACT.md`` section 3. Truth for a
``(start, depth, candidate seed)`` pool is computed once with the true simulator and shared
by all three model seeds. Candidate coverage is reported before any model metric, and the
pre-registered coverage verdict can stop phase D.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..continuous2d.config import Continuous2DConfig
from ..continuous2d.environment import Continuous2DEnv
from ..continuous2d.scenes import SceneSpec, scene_by_name
from ..continuous2d.skills import joint_spec_to_dicts
from .contract import (
    CHAIN_ANGLE_TOLERANCE,
    CHAIN_POSITION_TOLERANCE,
    geometric_violation,
    wrap_angle,
)
from .features import BoundaryState
from .model import JointWindowMLP, predict_window
from .pools import candidate_pool
from .rollout import (
    EXIT_INVALID,
    EXIT_NON_FINITE,
    EXIT_PHYSICAL_FAILURE,
    SequenceEvaluation,
    boundary_from_dict,
    evaluate_sequence_exact,
    evaluate_sequence_learned,
)
from .starts import EvalStart

EXACT_SEED = -1

#: A3 feasibility-filtered metrics. Kept in a separate table from the all-candidate metrics:
#: ``None`` means "no feasible candidate", which is reported as NA and never replaced by 0.
FEASIBILITY_KEYS = (
    "feasible_count", "rejected_count", "rejection_rate",
    "feasible_chosen_true_return", "feasible_chosen_model_return",
    "feasible_chosen_true_success", "conditional_regret",
    "feasible_candidate_best_true_return",
)


@dataclass(frozen=True)
class DiagnosticConfigV1:
    depths: Tuple[int, ...] = (1, 2, 3)
    sizes: Tuple[int, ...] = (16, 64, 128)
    candidate_seeds: Tuple[int, ...] = (0, 1, 2)
    model_seeds: Tuple[int, ...] = (0, 1, 2)
    pool_cap: int = 128
    recursive_depth: int = 3
    recursive_candidate_seed: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "depths": list(self.depths),
            "sizes": list(self.sizes),
            "candidate_seeds": list(self.candidate_seeds),
            "model_seeds": list(self.model_seeds),
            "pool_cap": self.pool_cap,
            "recursive_depth": self.recursive_depth,
            "recursive_candidate_seed": self.recursive_candidate_seed,
        }


def selection_metrics(
    candidate_ids: Sequence[int], j_hat: np.ndarray, j_ref: np.ndarray
) -> Dict[str, float]:
    j_hat = np.asarray(j_hat, dtype=np.float64)
    j_ref = np.asarray(j_ref, dtype=np.float64)
    chosen = max(range(len(j_hat)), key=lambda index: (j_hat[index], -candidate_ids[index]))
    selected_bias = float(j_hat[chosen] - j_ref[chosen])
    candidate_mean_bias = float(np.mean(j_hat - j_ref))
    return {
        "chosen_index": int(chosen),
        "chosen_candidate_id": int(candidate_ids[chosen]),
        "selected_bias": selected_bias,
        "candidate_mean_bias": candidate_mean_bias,
        "selection_excess": selected_bias - candidate_mean_bias,
        "candidate_regret": float(np.max(j_ref) - j_ref[chosen]),
        "chosen_true_return": float(j_ref[chosen]),
        "uniform_candidate_true_return": float(np.mean(j_ref)),
        "candidate_best_true_return": float(np.max(j_ref)),
        "chosen_model_return": float(j_hat[chosen]),
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _feasibility_metrics(
    ids: Sequence[int],
    j_hat: np.ndarray,
    j_ref: np.ndarray,
    evaluations: Sequence[SequenceEvaluation],
    truth: Sequence[SequenceEvaluation],
) -> Dict[str, object]:
    """A3 feasibility filter, reported next to (never merged with) the all-candidate table.

    Only ``window_complete`` / ``success`` are feasible. ``conditional_regret`` is ``None``
    when no feasible candidate exists: NA is never replaced by 0 and the start is not dropped.
    """

    feasible = [index for index, evaluation in enumerate(evaluations) if evaluation.feasible]
    rejected = len(evaluations) - len(feasible)
    out: Dict[str, object] = {
        "feasible_count": len(feasible),
        "rejected_count": rejected,
        "rejection_rate": (rejected / len(evaluations)) if evaluations else float("nan"),
        "feasible_chosen_candidate_id": None,
        "feasible_chosen_true_return": None,
        "feasible_chosen_true_success": None,
        "feasible_chosen_model_return": None,
        "feasible_chosen_exit_class": None,
        "conditional_regret": None,
        "feasible_candidate_best_true_return": None,
    }
    if not feasible:
        return out
    best = max(feasible, key=lambda index: (j_hat[index], -ids[index]))
    out.update(
        {
            "feasible_chosen_candidate_id": int(ids[best]),
            "feasible_chosen_true_return": float(j_ref[best]),
            "feasible_chosen_true_success": int(truth[best].success),
            "feasible_chosen_model_return": float(j_hat[best]),
            "feasible_chosen_exit_class": evaluations[best].exit_class,
            "conditional_regret": float(np.max(j_ref[feasible]) - j_ref[best]),
            "feasible_candidate_best_true_return": float(np.max(j_ref[feasible])),
        }
    )
    return out


def _start_boundary(start: EvalStart) -> BoundaryState:
    snapshot = start.snapshot
    return BoundaryState(
        object=snapshot.object, robots=snapshot.robots, success_counter=snapshot.success_counter
    )


def _distance_of(state: BoundaryState, scene: SceneSpec, config: Continuous2DConfig) -> float:
    from .rollout import distance_of_state

    return distance_of_state(state, scene, config)


def _components_dict(evaluation: SequenceEvaluation) -> Dict[str, float]:
    return evaluation.components.as_dict()


def run_diagnostic(
    starts: Sequence[EvalStart],
    models: Dict[int, JointWindowMLP],
    *,
    config: Optional[Continuous2DConfig] = None,
    diagnostic: Optional[DiagnosticConfigV1] = None,
    checkpoint_hashes: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    config = config or Continuous2DConfig()
    diagnostic = diagnostic or DiagnosticConfigV1()
    planner = Continuous2DEnv(config)
    started = time.time()

    per_run: List[Dict[str, object]] = []
    coverage_rows: List[Dict[str, object]] = []
    candidate_records: List[Dict[str, object]] = []
    truth_cache: Dict[Tuple[str, int, int], Tuple[List, List]] = {}
    start_rows: List[Dict[str, object]] = []

    for start in starts:
        scene = scene_by_name(start.scene_id)
        start_boundary = _start_boundary(start)
        d_start = _distance_of(start_boundary, scene, config)
        for depth in diagnostic.depths:
            for candidate_seed in diagnostic.candidate_seeds:
                pool = candidate_pool(
                    start.observation, config, depth, candidate_seed, cap=diagnostic.pool_cap
                )
                truth = [
                    evaluate_sequence_exact(
                        planner, start.snapshot, scene, sequence, config, candidate_id=index
                    )
                    for index, sequence in enumerate(pool)
                ]
                learned = {
                    model_seed: [
                        evaluate_sequence_learned(
                            models[model_seed], scene, start_boundary, sequence, config,
                            candidate_id=index,
                        )
                        for index, sequence in enumerate(pool)
                    ]
                    for model_seed in diagnostic.model_seeds
                }
                truth_cache[(start.start_id, depth, candidate_seed)] = (pool, truth)

                feasible = [
                    evaluation
                    for evaluation in truth
                    if evaluation.exit_class != EXIT_PHYSICAL_FAILURE
                ]
                feasible_distances = [
                    0.0 if evaluation.success else evaluation.components.distance
                    for evaluation in feasible
                ]
                coverage_rows.append(
                    {
                        "start_id": start.start_id,
                        "scene_id": start.scene_id,
                        "H": depth,
                        "candidate_seed": candidate_seed,
                        "n_candidates": len(pool),
                        "start_passage_distance": d_start,
                        "any_success": bool(any(evaluation.success for evaluation in truth)),
                        "candidate_best_true_return": float(max(e.j for e in truth)),
                        "uniform_candidate_true_return": _mean([e.j for e in truth]),
                        "candidate_true_failure_rate": _mean(
                            [float(e.exit_class == EXIT_PHYSICAL_FAILURE) for e in truth]
                        ),
                        "min_feasible_distance": min(feasible_distances) if feasible_distances else float("nan"),
                        "has_d_lowering_feasible": bool(
                            feasible_distances and min(feasible_distances) < d_start
                        ),
                        "d_lowering_fraction": (
                            _mean([float(value < d_start) for value in feasible_distances])
                            if feasible_distances
                            else float("nan")
                        ),
                        "candidate_distinct": len(
                            {repr([joint_spec_to_dicts(spec) for spec in sequence]) for sequence in pool}
                        ),
                    }
                )

                for size in diagnostic.sizes:
                    k = min(size, len(pool))
                    if k <= 0:
                        continue
                    ids = list(range(k))
                    j_ref = np.asarray([evaluation.j for evaluation in truth[:k]])
                    for model_seed in (EXACT_SEED, *diagnostic.model_seeds):
                        if model_seed == EXACT_SEED:
                            evaluations = truth[:k]
                        else:
                            evaluations = learned[model_seed][:k]
                        j_hat = np.asarray([evaluation.j for evaluation in evaluations])
                        metrics = selection_metrics(ids, j_hat, j_ref)
                        chosen = int(metrics["chosen_index"])
                        chosen_eval = evaluations[chosen]
                        row = {
                            "start_id": start.start_id,
                            "scene_id": start.scene_id,
                            "H": depth,
                            "K": size,
                            "actual_K": k,
                            "candidate_seed": candidate_seed,
                            "model_seed": model_seed,
                            "checkpoint_sha256": (checkpoint_hashes or {}).get(
                                model_seed, "exact" if model_seed == EXACT_SEED else ""
                            ),
                            **metrics,
                            "chosen_true_success": int(truth[chosen].success),
                            "chosen_model_success": int(chosen_eval.success),
                            "chosen_false_success": int(chosen_eval.success and not truth[chosen].success),
                            "chosen_exit_class": chosen_eval.exit_class,
                            "chosen_true_exit_class": truth[chosen].exit_class,
                            "candidate_success_rate": _mean([float(e.success) for e in truth[:k]]),
                            "candidate_true_failure_rate": _mean(
                                [float(e.exit_class == EXIT_PHYSICAL_FAILURE) for e in truth[:k]]
                            ),
                            "candidate_invalid_rate": _mean(
                                [
                                    float(e.exit_class in (EXIT_INVALID, EXIT_NON_FINITE))
                                    for e in evaluations
                                ]
                            ),
                            "candidate_predicted_failure_rate": _mean(
                                [
                                    float(e.exit_class in (EXIT_PHYSICAL_FAILURE, EXIT_INVALID, EXIT_NON_FINITE))
                                    for e in evaluations
                                ]
                            ),
                            "chosen_components": _components_dict(chosen_eval),
                            "candidate_mean_components": {
                                key: _mean([evaluation.components.as_dict()[key] for evaluation in evaluations])
                                for key in ("success_reward", "time_penalty", "distance_penalty", "failure_penalty", "J")
                            },
                            "chosen_true_components": _components_dict(truth[chosen]),
                            "candidate_mean_true_components": {
                                key: _mean([evaluation.components.as_dict()[key] for evaluation in truth[:k]])
                                for key in ("success_reward", "time_penalty", "distance_penalty", "failure_penalty", "J")
                            },
                            **_feasibility_metrics(ids, j_hat, j_ref, evaluations, truth[:k]),
                        }
                        per_run.append(row)

                for index, sequence in enumerate(pool):
                    candidate_records.append(
                        {
                            "start_id": start.start_id,
                            "scene_id": start.scene_id,
                            "H": depth,
                            "candidate_seed": candidate_seed,
                            "candidate_id": index,
                            "sequence": [joint_spec_to_dicts(spec) for spec in sequence],
                            "J_ref": truth[index].j,
                            "J_ref_components": _components_dict(truth[index]),
                            "J_ref_exit_class": truth[index].exit_class,
                            "J_hat": {
                                str(model_seed): learned[model_seed][index].j
                                for model_seed in diagnostic.model_seeds
                            },
                            "J_hat_exit_class": {
                                str(model_seed): learned[model_seed][index].exit_class
                                for model_seed in diagnostic.model_seeds
                            },
                        }
                    )

    # ---- aggregation: mean within one start, then mean/range over starts ----------------
    grouped: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
    for row in per_run:
        grouped.setdefault((int(row["H"]), int(row["K"]), int(row["model_seed"])), []).append(row)
    metric_keys = (
        "selected_bias", "candidate_mean_bias", "selection_excess", "candidate_regret",
        "chosen_true_return", "uniform_candidate_true_return", "candidate_best_true_return",
        "chosen_true_success", "chosen_false_success", "candidate_invalid_rate",
        "candidate_true_failure_rate", "candidate_success_rate",
    )
    per_start_rows: List[Dict[str, object]] = []
    for (depth, size, model_seed), rows in sorted(grouped.items()):
        by_start: Dict[str, List[Dict[str, object]]] = {}
        for row in rows:
            by_start.setdefault(str(row["start_id"]), []).append(row)
        for start_id, start_group in sorted(by_start.items()):
            entry: Dict[str, object] = {
                "start_id": start_id, "H": depth, "K": size, "model_seed": model_seed,
                "candidate_seeds": len(start_group),
            }
            for key in metric_keys:
                entry[key] = _mean([float(row[key]) for row in start_group])
            per_start_rows.append(entry)
    across_rows: List[Dict[str, object]] = []
    for (depth, size, model_seed) in sorted({(r["H"], r["K"], r["model_seed"]) for r in per_start_rows}):
        subset = [
            row for row in per_start_rows
            if row["H"] == depth and row["K"] == size and row["model_seed"] == model_seed
        ]
        entry = {
            "H": depth, "K": size, "model_seed": model_seed, "n_starts": len(subset),
            "candidate_seed_runs_per_start": max(int(row["candidate_seeds"]) for row in subset) if subset else 0,
        }
        for key in metric_keys:
            values = [float(row[key]) for row in subset]
            entry[key] = _mean(values)
            entry[key + "_min"] = float(np.min(values)) if values else float("nan")
            entry[key + "_max"] = float(np.max(values)) if values else float("nan")
        across_rows.append(entry)

    # ---- A3 feasibility-filtered aggregation (a separate table, never merged) -----------
    feasible_per_start: List[Dict[str, object]] = []
    grouped_feasible: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
    for row in per_run:
        grouped_feasible.setdefault(
            (int(row["H"]), int(row["K"]), int(row["model_seed"])), []
        ).append(row)
    for (depth, size, model_seed), rows in sorted(grouped_feasible.items()):
        by_start_f: Dict[str, List[Dict[str, object]]] = {}
        for row in rows:
            by_start_f.setdefault(str(row["start_id"]), []).append(row)
        for start_id, start_group in sorted(by_start_f.items()):
            entry_f: Dict[str, object] = {
                "start_id": start_id, "H": depth, "K": size, "model_seed": model_seed,
                "candidate_seeds": len(start_group),
            }
            for key in FEASIBILITY_KEYS:
                values = [float(row[key]) for row in start_group if row[key] is not None]
                entry_f[key] = float(np.mean(values)) if values else None
                entry_f[key + "_n"] = len(values)
            feasible_per_start.append(entry_f)
    across_feasible: List[Dict[str, object]] = []
    for (depth, size, model_seed) in sorted(
        {(r["H"], r["K"], r["model_seed"]) for r in feasible_per_start}
    ):
        subset = [
            row for row in feasible_per_start
            if row["H"] == depth and row["K"] == size and row["model_seed"] == model_seed
        ]
        entry = {
            "H": depth, "K": size, "model_seed": model_seed, "n_starts": len(subset),
            "candidate_seed_runs_per_start": max(int(r["candidate_seeds"]) for r in subset) if subset else 0,
        }
        for key in FEASIBILITY_KEYS:
            values = [float(row[key]) for row in subset if row[key] is not None]
            entry[key] = float(np.mean(values)) if values else None
            entry[key + "_n_starts"] = len(values)
        across_feasible.append(entry)

    # ---- coverage verdict ---------------------------------------------------------------
    start_has_lowering = {
        start.start_id: any(
            row["has_d_lowering_feasible"] for row in coverage_rows if row["start_id"] == start.start_id
        )
        for start in starts
    }
    coverage_summary = {
        "n_starts": len(starts),
        "starts_with_success_candidate": sum(
            1 for start in starts
            if any(row["any_success"] for row in coverage_rows if row["start_id"] == start.start_id)
        ),
        "starts_with_d_lowering_feasible": sum(1 for value in start_has_lowering.values() if value),
        "per_start": start_has_lowering,
        "verdict": "COVERAGE_LIMITED"
        if sum(1 for value in start_has_lowering.values() if value) < 3
        else "COVERAGE_OK",
    }

    recursive = recursive_contrast(
        starts, models, config, diagnostic, truth_cache
    )
    traced = traced_cases(
        per_start_rows, per_run, starts, truth_cache, diagnostic, models, config
    )

    return {
        "config": diagnostic.as_dict(),
        "starts": [
            {**start.recoverable(), "kind": start.kind} for start in starts
        ],
        "coverage_rows": coverage_rows,
        "coverage_summary": coverage_summary,
        "per_run": per_run,
        "per_start": per_start_rows,
        "aggregate": across_rows,
        "per_start_feasible": feasible_per_start,
        "aggregate_feasible": across_feasible,
        "candidate_records": candidate_records,
        "recursive": recursive,
        "traced_cases": traced,
        "wall_clock": time.time() - started,
    }


# ---------------------------------------------------------------------------
# C4: recursive vs teacher-forced
# ---------------------------------------------------------------------------


def _state_error(predicted: BoundaryState, actual: BoundaryState) -> Tuple[float, float]:
    positions = []
    for first, second in zip(predicted.bodies(), actual.bodies()):
        positions.append(math.hypot(first.x - second.x, first.y - second.y))
    angle = abs(wrap_angle(predicted.object.angle - actual.object.angle))
    return float(np.mean(positions)), float(angle)


def recursive_contrast(
    starts: Sequence[EvalStart],
    models: Dict[int, JointWindowMLP],
    config: Continuous2DConfig,
    diagnostic: DiagnosticConfigV1,
    truth_cache: Dict[Tuple[str, int, int], Tuple[List, List]],
) -> Dict[str, object]:
    depth = diagnostic.recursive_depth
    seed = diagnostic.recursive_candidate_seed
    physics = config.physics
    out: Dict[str, object] = {}
    for model_seed, model in models.items():
        teacher_error = [0.0] * depth
        teacher_angle = [0.0] * depth
        teacher_n = [0] * depth
        teacher_invalid = [0] * depth
        rec_error = [0.0] * depth
        rec_angle = [0.0] * depth
        rec_n = [0] * depth
        rec_invalid = [0] * depth
        rec_success_exit = [0] * depth
        rec_failure_exit = [0] * depth
        rec_alive = [0] * depth
        all_predicted = [0] * depth
        all_invalid = [0] * depth
        position_error_at = [0] * depth
        angle_error_at = [0] * depth
        # Per-chain outcomes, so the cumulative rates below are real cumulative rates and
        # not per-window counts masquerading as cumulative ones.
        terminated_at: List[int] = []
        invalid_at: List[int] = []
        mismatch_at: List[int] = []
        chains = 0

        for start in starts:
            key = (start.start_id, depth, seed)
            if key not in truth_cache:
                continue
            pool, truth = truth_cache[key]
            scene = scene_by_name(start.scene_id)
            start_boundary = _start_boundary(start)
            for index, sequence in enumerate(pool):
                chains += 1
                true_trajectory = truth[index].trajectory
                chain_terminated: Optional[int] = None
                chain_invalid: Optional[int] = None
                chain_mismatch: Optional[int] = None
                # teacher-forced: always predict from the true current state
                for window in range(len(true_trajectory)):
                    entry = true_trajectory[window]
                    before = boundary_from_dict(entry["before"])
                    actual = boundary_from_dict(entry["after"])
                    prediction = predict_window(model, scene, before, sequence[window], physics)
                    violation = geometric_violation(
                        prediction.next_state.object, prediction.next_state.robots, scene, physics
                    )
                    all_predicted[window] += 1
                    invalid = bool(violation["violation"] or not violation["finite"])
                    if invalid:
                        all_invalid[window] += 1
                    position, angle = _state_error(prediction.next_state, actual)
                    teacher_error[window] += position
                    teacher_angle[window] += angle
                    teacher_n[window] += 1
                    teacher_invalid[window] += int(invalid)

                # recursive: feed the model's own prediction back in. A4: the paired
                # comparison stops where the true chain stopped, so a correct success
                # absorption is never scored as a recursive failure.
                state = start_boundary
                for window in range(min(len(sequence), len(true_trajectory))):
                    rec_alive[window] += 1
                    prediction = predict_window(model, scene, state, sequence[window], physics)
                    violation = geometric_violation(
                        prediction.next_state.object, prediction.next_state.robots, scene, physics
                    )
                    invalid = bool(violation["violation"] or not violation["finite"])
                    true_after = boundary_from_dict(true_trajectory[window]["after"])
                    position, angle = _state_error(prediction.next_state, true_after)
                    rec_error[window] += position
                    rec_angle[window] += angle
                    rec_n[window] += 1
                    over_position = position > CHAIN_POSITION_TOLERANCE
                    over_angle = angle > CHAIN_ANGLE_TOLERANCE
                    position_error_at[window] += int(over_position)
                    angle_error_at[window] += int(over_angle)
                    if (over_position or over_angle) and chain_mismatch is None:
                        chain_mismatch = window
                    if invalid:
                        rec_invalid[window] += 1
                        chain_invalid = window
                        chain_terminated = window
                        break
                    if prediction.termination == "success":
                        rec_success_exit[window] += 1
                        chain_terminated = window
                        break
                    if prediction.termination == "physical_failure":
                        rec_failure_exit[window] += 1
                        chain_terminated = window
                        break
                    state = prediction.next_state

                terminated_at.append(chain_terminated)
                invalid_at.append(chain_invalid)
                mismatch_at.append(chain_mismatch)

        def _cumulative(values: List[Optional[int]], window: int) -> int:
            return sum(1 for value in values if value is not None and value <= window)

        cumulative_terminated = [
            _cumulative(terminated_at, window) for window in range(depth)
        ]
        cumulative_invalid = [_cumulative(invalid_at, window) for window in range(depth)]
        cumulative_bad = []
        for window in range(depth):
            bad = 0
            for index in range(chains):
                terminated = terminated_at[index]
                mismatch = mismatch_at[index]
                if (terminated is not None and terminated <= window) or (
                    mismatch is not None and mismatch <= window
                ):
                    bad += 1
            cumulative_bad.append(bad)

        out[str(model_seed)] = {
            "depth": depth,
            "candidate_seed": seed,
            "n_chains": chains,
            "teacher_forced": {
                "n_predictions": teacher_n,
                "invalid_count": teacher_invalid,
                "invalid_rate": [value / max(1, n) for value, n in zip(teacher_invalid, teacher_n)],
                "position_error": [value / max(1, n) for value, n in zip(teacher_error, teacher_n)],
                "object_angle_error": [value / max(1, n) for value, n in zip(teacher_angle, teacher_n)],
            },
            "recursive": {
                "alive_at_window_start": rec_alive,
                "position_error_rate_at_window": [
                    value / max(1, n) for value, n in zip(position_error_at, rec_n)
                ],
                "angle_error_rate_at_window": [
                    value / max(1, n) for value, n in zip(angle_error_at, rec_n)
                ],
                "invalid_at_window": rec_invalid,
                "success_exit_at_window": rec_success_exit,
                "failure_exit_at_window": rec_failure_exit,
                "n_surviving_predictions": rec_n,
                "position_error_survivors": [value / max(1, n) for value, n in zip(rec_error, rec_n)],
                "object_angle_error_survivors": [value / max(1, n) for value, n in zip(rec_angle, rec_n)],
            },
            "all_chains": {
                "n_predicted_teacher": all_predicted,
                "invalid_count_teacher": all_invalid,
                "terminated_cumulative": cumulative_terminated,
                "invalid_cumulative": cumulative_invalid,
                "cumulative_termination_rate": [value / max(1, chains) for value in cumulative_terminated],
                "cumulative_invalid_rate": [value / max(1, chains) for value in cumulative_invalid],
                "cumulative_failure_or_error_rate": [value / max(1, chains) for value in cumulative_bad],
                "failure_or_error_definition": (
                    "a chain counts once its recursive prediction terminated (invalid or "
                    "predicted physical failure) or its first tolerance breach (position "
                    f">{CHAIN_POSITION_TOLERANCE} m or angle >{CHAIN_ANGLE_TOLERANCE} rad) "
                    "occurred at or before this window; the paired comparison stops where the "
                    "true chain stopped, so a correct success absorption is not an error"
                ),
            },
        }
    return out


# ---------------------------------------------------------------------------
# Traced cases
# ---------------------------------------------------------------------------


def traced_cases(
    per_start_rows: Sequence[Dict[str, object]],
    per_run: Sequence[Dict[str, object]],
    starts: Sequence[EvalStart],
    truth_cache: Dict[Tuple[str, int, int], Tuple[List, List]],
    diagnostic: DiagnosticConfigV1,
    models: Dict[int, JointWindowMLP],
    config: Continuous2DConfig,
) -> List[Dict[str, object]]:
    """The 3 largest mean-regret starts and 1 low-regret control, at the largest H and K."""

    depth = max(diagnostic.depths)
    size = max(diagnostic.sizes)
    subset = [
        row for row in per_start_rows
        if row["H"] == depth and row["K"] == size and row["model_seed"] != EXACT_SEED
    ]
    by_start: Dict[str, List[float]] = {}
    for row in subset:
        by_start.setdefault(str(row["start_id"]), []).append(float(row["candidate_regret"]))
    ranking = sorted(by_start.items(), key=lambda item: (-_mean(item[1]), item[0]))
    if len(ranking) <= 3:
        chosen = [start_id for start_id, _ in ranking]
    else:
        chosen = [start_id for start_id, _ in ranking[:3]] + [ranking[-1][0]]

    start_lookup = {start.start_id: start for start in starts}
    cases: List[Dict[str, object]] = []
    for start_id in chosen:
        candidates = [
            row for row in per_run
            if row["start_id"] == start_id and row["H"] == depth and row["K"] == size
            and row["model_seed"] != EXACT_SEED
        ]
        if not candidates:
            continue
        worst = max(
            candidates,
            key=lambda row: (
                float(row["candidate_regret"]),
                -int(row["candidate_seed"]),
                -int(row["model_seed"]),
            ),
        )
        key = (start_id, depth, int(worst["candidate_seed"]))
        pool, truth = truth_cache[key]
        chosen_id = int(worst["chosen_candidate_id"])
        best_id = int(np.argmax([evaluation.j for evaluation in truth[: int(worst["actual_K"])]]))
        model_seed = int(worst["model_seed"])
        start = start_lookup[start_id]
        scene = scene_by_name(start.scene_id)

        model_evaluation = evaluate_sequence_learned(
            models[model_seed], scene, _start_boundary(start), pool[chosen_id], config,
            candidate_id=chosen_id,
        )

        cases.append(
            {
                "start_id": start_id,
                "kind": "max_regret" if len(cases) < 3 else "control",
                "scene_id": start.scene_id,
                "H": depth,
                "K": size,
                "candidate_seed": int(worst["candidate_seed"]),
                "model_seed": model_seed,
                "mean_start_regret": _mean(by_start[start_id]),
                "run_candidate_regret": float(worst["candidate_regret"]),
                "selected_bias": float(worst["selected_bias"]),
                "chosen_candidate_id": chosen_id,
                "exact_best_candidate_id": best_id,
                "chosen_sequence": [joint_spec_to_dicts(spec) for spec in pool[chosen_id]],
                "best_sequence": [joint_spec_to_dicts(spec) for spec in pool[best_id]],
                "chosen_true_trajectory": truth[chosen_id].trajectory,
                "chosen_true_components": truth[chosen_id].components.as_dict(),
                "chosen_true_exit_class": truth[chosen_id].exit_class,
                "chosen_model_trajectory": model_evaluation.trajectory,
                "chosen_model_components": model_evaluation.components.as_dict(),
                "chosen_model_exit_class": model_evaluation.exit_class,
                "chosen_model_penalty_windows": model_evaluation.predicted_windows,
                "best_true_trajectory": truth[best_id].trajectory,
                "best_true_components": truth[best_id].components.as_dict(),
                "best_true_exit_class": truth[best_id].exit_class,
            }
        )
    return cases
