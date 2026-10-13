"""Phase A: finite-Grid audit for ``forecast_baseline_v1`` (task book 20261012_2007).

This module is purely additive. It never regenerates the saved dataset, never retrains and
never rewrites an old table. It re-reads the frozen ``samples.jsonl`` / ``splits.jsonl`` and
the three saved checkpoints and recomputes:

* A1 - corrected ``object_moved`` / ``object_stationary`` / ``full_state_unchanged`` group
  statistics, per map, per model seed, plus the corrected single-step headline;
* A2 - a coordinate-unbiased 20-state validation set ordered by
  ``SHA256("grid-audit-v2|" + state_key)``, and the fixed-candidate diagnostic on it;
* A3 - the four-term decomposition of ``J_hat - J_ref`` and the traced error cases;
* A4 - a protocol-identical recursive vs teacher-forced contrast (illegal prediction
  terminates the chain, success is absorbing, no state is fabricated).

The candidate generator, the scoring entry and the checkpoints are the frozen ones; the
only new logic is the loader, the statistics, the ordering rule, the decomposition and the
chain bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .candidates import CandidateConfig, candidate_sequences
from .contract import (
    ANCHOR_INDEX,
    ANCHORS,
    CELL_INDEX,
    CELLS,
    GridMap,
    GridState,
    is_goal,
    state_is_valid,
    terminal_distance,
)
from .data import DataConfig, Dataset, TransitionSample
from .diagnostic import EXACT_SEED, select_eval_states, selection_metrics
from .model import PointModel, decode, encode_input, load_model
from .scoring import ScoreComponents, rollout, rollout_exact, score_trajectory

#: Selection tag for the coordinate-unbiased validation set (CONTRACT section 1, A2).
AUDIT_TAG = "grid-audit-v2"
#: Grouping is fixed by the true transition, never by the model output.
GROUPS = ("object_moved", "object_stationary", "full_state_unchanged")
#: The four decomposition terms of ``J_hat - J_ref`` (CONTRACT section 1, A3).
DECOMP_TERMS = ("success_reward", "executed_steps", "terminal_distance", "invalid_penalty")
RUN_METRIC_KEYS = (
    "selected_bias",
    "candidate_mean_bias",
    "selection_excess",
    "candidate_regret",
    "chosen_true_return",
    "uniform_candidate_return",
    "candidate_best_true_return",
    "chosen_true_success",
    "candidate_invalid_rate",
    "candidate_success_rate",
    "chosen_model_return",
)


# ---------------------------------------------------------------------------
# Hashing and loading (read-only)
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hashes(run_dir: Path) -> Dict[str, str]:
    """SHA256 of the two saved data files that fix the whole phase-A sample space."""

    run_dir = Path(run_dir)
    return {
        str(Path("dataset") / name): sha256_file(run_dir / "dataset" / name)
        for name in ("samples.jsonl", "splits.jsonl")
    }


def checkpoint_hashes(run_dir: Path, seeds: Sequence[int]) -> Dict[str, str]:
    run_dir = Path(run_dir)
    return {
        str(seed): sha256_file(run_dir / "models" / f"seed{seed}.pt") for seed in seeds
    }


def _cell(value) -> Tuple[int, int]:
    return (int(value[0]), int(value[1]))


def _state_from_payload(payload) -> GridState:
    """Accept both the saved dict form and the ``[r0, r1, object]`` list form."""

    if isinstance(payload, dict):
        return GridState(_cell(payload["robot0"]), _cell(payload["robot1"]), _cell(payload["object"]))
    return GridState(_cell(payload[0]), _cell(payload[1]), _cell(payload[2]))


def state_payload(state: GridState) -> Dict[str, List[int]]:
    return {
        "robot0": [int(state.robot0[0]), int(state.robot0[1])],
        "robot1": [int(state.robot1[0]), int(state.robot1[1])],
        "object": [int(state.object_anchor[0]), int(state.object_anchor[1])],
    }


def parse_state_key(key: str) -> Tuple[str, GridState]:
    map_name, rest = key.split("|", 1)
    fields = dict(part.split("=") for part in rest.split("|"))
    return map_name, GridState(
        _cell(fields["r0"].split(",")),
        _cell(fields["r1"].split(",")),
        _cell(fields["o"].split(",")),
    )


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_saved_dataset(run_dir: Path) -> Dataset:
    """Read the saved dataset. The generator is never called to rebuild it."""

    directory = Path(run_dir) / "dataset"
    config = DataConfig(**json.loads((directory / "config.json").read_text()))
    samples: List[TransitionSample] = []
    for row in read_jsonl(directory / "samples.jsonl"):
        samples.append(
            TransitionSample(
                map_name=str(row["map"]),
                state=_state_from_payload(row["state"]),
                joint_action=int(row["joint"]),
                next_state=_state_from_payload(row["next"]),
                object_moved=bool(row["object_moved"]),
                robot_moved=bool(row["robot_moved"]),
                rollback=bool(row["rollback"]),
                invalid_actions=int(row["invalid_actions"]),
            )
        )
    state_split = {
        str(row["state_key"]): str(row["split"])
        for row in read_jsonl(directory / "splits.jsonl")
    }
    return Dataset(samples=samples, state_split=state_split, config=config)


def load_saved_models(run_dir: Path, seeds: Sequence[int]) -> Dict[int, PointModel]:
    return {int(seed): load_model(Path(run_dir) / "models" / f"seed{seed}.pt") for seed in seeds}


# ---------------------------------------------------------------------------
# A1 - corrected group statistics
# ---------------------------------------------------------------------------


def _rate(numerator: float, denominator: int) -> Optional[float]:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _predict_arrays(model: PointModel, samples: Sequence[TransitionSample], maps: Dict[str, GridMap]):
    features = np.asarray(
        [encode_input(maps[s.map_name], s.state, s.joint_action) for s in samples], dtype=np.float32
    )
    model.eval()
    with torch.no_grad():
        object_logits, robot0_logits, robot1_logits = model(torch.from_numpy(features))
    return (
        object_logits.argmax(dim=1).numpy(),
        robot0_logits.argmax(dim=1).numpy(),
        robot1_logits.argmax(dim=1).numpy(),
    )


def _split_samples(dataset: Dataset, split: str) -> List[TransitionSample]:
    return [s for s in dataset.samples if dataset.state_split[s.state.key(s.map_name)] == split]


def group_statistics(
    dataset: Dataset, maps: Dict[str, GridMap], models: Dict[int, PointModel], split: str = "test"
) -> Dict[str, object]:
    """A1: per (map, group, model seed) and pooled rows, from the saved dataset.

    ``full_state_unchanged`` is a separate group and is never merged with
    ``object_stationary``; the grouping uses the true transition only.
    """

    samples = _split_samples(dataset, split)
    object_target = np.asarray([ANCHOR_INDEX[s.next_state.object_anchor] for s in samples])
    robot0_target = np.asarray([CELL_INDEX[s.next_state.robot0] for s in samples])
    robot1_target = np.asarray([CELL_INDEX[s.next_state.robot1] for s in samples])
    object_moved = np.asarray([bool(s.object_moved) for s in samples])
    full_state_unchanged = np.asarray([s.next_state == s.state for s in samples])
    group_masks = {
        "object_moved": object_moved,
        "object_stationary": ~object_moved,
        "full_state_unchanged": full_state_unchanged,
    }
    scopes = sorted(maps)
    rows: List[Dict[str, object]] = []
    overall: List[Dict[str, object]] = []
    relations: Dict[str, object] = {
        "split": split,
        "n_samples": len(samples),
        "full_state_unchanged_subset_of_object_stationary": bool(
            np.all(~full_state_unchanged | ~object_moved)
        ),
        "object_moved_plus_object_stationary_equals_all": bool(
            int(object_moved.sum()) + int((~object_moved).sum()) == len(samples)
        ),
        "full_state_unchanged_implies_object_stationary": bool(
            np.all(~full_state_unchanged | ~object_moved)
        ),
    }

    for seed, model in sorted(models.items()):
        object_pred, r0_pred, r1_pred = _predict_arrays(model, samples, maps)
        object_ok = object_pred == object_target
        robot0_ok = r0_pred == robot0_target
        robot1_ok = r1_pred == robot1_target
        full_ok = object_ok & robot0_ok & robot1_ok
        invalid = np.asarray(
            [
                not state_is_valid(
                    GridState(CELLS[int(r0_pred[i])], CELLS[int(r1_pred[i])], ANCHORS[int(object_pred[i])]),
                    maps[samples[i].map_name],
                )
                for i in range(len(samples))
            ]
        )
        map_arrays = {name: np.asarray([s.map_name == name for s in samples]) for name in scopes}

        def make_row(scope: str, group: str, mask: np.ndarray) -> Dict[str, object]:
            n = int(mask.sum())
            return {
                "map": scope,
                "group": group,
                "model_seed": int(seed),
                "n": n,
                "object_accuracy": _rate(object_ok[mask].sum(), n),
                "robot0_accuracy": _rate(robot0_ok[mask].sum(), n),
                "robot1_accuracy": _rate(robot1_ok[mask].sum(), n),
                "full_state_accuracy": _rate(full_ok[mask].sum(), n),
                "invalid_state_rate": _rate(invalid[mask].sum(), n),
            }

        for scope in scopes + ["both"]:
            scope_mask = (
                np.ones(len(samples), dtype=bool)
                if scope == "both"
                else map_arrays[scope]
            )
            for group, mask in group_masks.items():
                rows.append(make_row(scope, group, scope_mask & mask))
            rows.append(make_row(scope, "all", scope_mask))
            overall.append(make_row(scope, "all", scope_mask))

    # pooled row: mean over the three model seeds
    keys = sorted({(r["map"], r["group"]) for r in rows})
    pooled: List[Dict[str, object]] = []
    for scope, group in keys:
        subset = [r for r in rows if r["map"] == scope and r["group"] == group]
        row: Dict[str, object] = {
            "map": scope, "group": group, "model_seed": "mean",
            "n": int(np.mean([r["n"] for r in subset])),
        }
        for metric in ("object_accuracy", "robot0_accuracy", "robot1_accuracy", "full_state_accuracy", "invalid_state_rate"):
            values = [r[metric] for r in subset if r[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
        pooled.append(row)
    rows.extend(pooled)

    # corrected overall single-step numbers on every split
    splits: List[Dict[str, object]] = []
    for other in ("train", "val", "test"):
        other_samples = _split_samples(dataset, other)
        if not other_samples:
            continue
        object_t = np.asarray([ANCHOR_INDEX[s.next_state.object_anchor] for s in other_samples])
        robot0_t = np.asarray([CELL_INDEX[s.next_state.robot0] for s in other_samples])
        robot1_t = np.asarray([CELL_INDEX[s.next_state.robot1] for s in other_samples])
        for seed, model in sorted(models.items()):
            object_pred, r0_pred, r1_pred = _predict_arrays(model, other_samples, maps)
            object_ok = object_pred == object_t
            robot0_ok = r0_pred == robot0_t
            robot1_ok = r1_pred == robot1_t
            full_ok = object_ok & robot0_ok & robot1_ok
            invalid = np.asarray(
                [
                    not state_is_valid(
                        GridState(CELLS[int(r0_pred[i])], CELLS[int(r1_pred[i])], ANCHORS[int(object_pred[i])]),
                        maps[other_samples[i].map_name],
                    )
                    for i in range(len(other_samples))
                ]
            )
            splits.append(
                {
                    "split": other, "model_seed": int(seed), "n": len(other_samples),
                    "object_accuracy": float(object_ok.mean()),
                    "robot0_accuracy": float(robot0_ok.mean()),
                    "robot1_accuracy": float(robot1_ok.mean()),
                    "full_state_accuracy": float(full_ok.mean()),
                    "invalid_state_rate": float(invalid.mean()),
                    "object_stationary_full_state_accuracy": _rate(
                        (full_ok & ~np.asarray([s.object_moved for s in other_samples])).sum(),
                        int((~np.asarray([s.object_moved for s in other_samples])).sum()),
                    ),
                    "full_state_unchanged_full_state_accuracy": _rate(
                        (full_ok & np.asarray([s.next_state == s.state for s in other_samples])).sum(),
                        int(np.asarray([s.next_state == s.state for s in other_samples]).sum()),
                    ),
                }
            )
    return {"rows": rows, "overall": overall, "by_split": splits, "relations": relations}


# ---------------------------------------------------------------------------
# A2 - coordinate-unbiased validation set and the fixed-candidate diagnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalState:
    map_name: str
    state: GridState
    state_key: str
    digest: str
    rank: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "map": self.map_name,
            "state_key": self.state_key,
            "state": state_payload(self.state),
            "split": "test",
            "sha256": self.digest,
            "rank_within_map": self.rank,
            "selection_rule": f'SHA256("{AUDIT_TAG}|" + state_key)',
        }


def select_eval_states_v2(
    dataset: Dataset, maps: Dict[str, GridMap], *, states_per_map: int = 10, tag: str = AUDIT_TAG
) -> List[EvalState]:
    """Distinct legal non-terminal test states, ordered by SHA256(tag + state_key).

    Coordinates, model error, return and success are never read by this function.
    """

    selected: List[EvalState] = []
    for map_name in sorted(maps):
        seen: set = set()
        entries: List[Tuple[str, GridState, str]] = []
        for key, split in dataset.state_split.items():
            if split != "test":
                continue
            key_map, state = parse_state_key(key)
            if key_map != map_name or key in seen:
                continue
            seen.add(key)
            if is_goal(state, maps[map_name]):
                continue
            digest = hashlib.sha256(f"{tag}|{key}".encode()).hexdigest()
            entries.append((digest, state, key))
        entries.sort(key=lambda item: (item[0], item[2]))
        for rank, (digest, state, key) in enumerate(entries[:states_per_map]):
            selected.append(EvalState(map_name, state, key, digest, rank))
    return selected


def eval_state_coverage(entries: Sequence[EvalState]) -> Dict[str, object]:
    by_map: Dict[str, Dict[str, object]] = {}
    for map_name in sorted({e.map_name for e in entries}):
        subset = [e for e in entries if e.map_name == map_name]
        by_map[map_name] = {
            "n_states": len(subset),
            "distinct_robot0_cells": sorted({e.state.robot0 for e in subset}),
            "distinct_robot1_cells": sorted({e.state.robot1 for e in subset}),
            "n_distinct_robot0_cells": len({e.state.robot0 for e in subset}),
            "n_distinct_robot1_cells": len({e.state.robot1 for e in subset}),
            "n_distinct_object_anchors": len({e.state.object_anchor for e in subset}),
        }
    return {
        "per_map": by_map,
        "overall": {
            "n_states": len(entries),
            "n_distinct_robot0_cells": len({e.state.robot0 for e in entries}),
            "n_distinct_robot1_cells": len({e.state.robot1 for e in entries}),
            "n_distinct_object_anchors": len({e.state.object_anchor for e in entries}),
            "n_distinct_full_states": len({e.state_key for e in entries}),
        },
    }


def point_step_fn(model: PointModel, grid_map: GridMap):
    """A ``StepFn`` that keeps the illegal decoded state visible instead of repairing it."""

    def _step(state: GridState, joint_action: int) -> Tuple[Optional[GridState], bool]:
        predicted = decode(model, grid_map, state, joint_action)
        return predicted, state_is_valid(predicted, grid_map)

    return _step


@dataclass(frozen=True)
class AuditConfig:
    depths: Tuple[int, ...] = (1, 2, 4)
    sizes: Tuple[int, ...] = (16, 64, 256)
    candidate_seeds: Tuple[int, ...] = (0, 1, 2)
    model_seeds: Tuple[int, ...] = (0, 1, 2)
    max_candidates: int = 256
    trace_depth: int = 4
    trace_size: int = 256
    teacher_horizon: int = 4
    teacher_candidates: int = 16

    def as_dict(self) -> Dict[str, object]:
        return {
            "depths": list(self.depths),
            "sizes": list(self.sizes),
            "candidate_seeds": list(self.candidate_seeds),
            "model_seeds": list(self.model_seeds),
            "max_candidates": self.max_candidates,
            "trace_depth": self.trace_depth,
            "trace_size": self.trace_size,
            "teacher_horizon": self.teacher_horizon,
            "teacher_candidates": self.teacher_candidates,
        }


def _sequences_for(grid_map: GridMap, state: GridState, depth: int, seed: int, cap: int):
    return candidate_sequences(
        state, grid_map, depth, cap, seed, CandidateConfig(seed=seed, max_sequences=cap)
    )


def run_fixed_candidate_audit(
    maps: Dict[str, GridMap],
    models: Dict[int, PointModel],
    eval_states: Sequence[EvalState],
    config: AuditConfig,
    *,
    eval_set: str = "v2",
    want_candidate_records: bool = False,
    want_traces: bool = False,
    checkpoint_hashes_by_seed: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    """The frozen fixed-candidate diagnostic, re-run on the given state set.

    Truth is computed once per candidate and reused by all model seeds; the candidate set
    of a smaller ``K`` is a prefix of the largest one. The four-term decomposition of the
    chosen candidate and of the candidate mean is attached to every run.
    """

    per_run: List[Dict[str, object]] = []
    candidate_records: List[Dict[str, object]] = []
    traces: List[Dict[str, object]] = []
    candidate_counts: List[Dict[str, object]] = []

    for entry in eval_states:
        grid_map = maps[entry.map_name]
        state = entry.state
        for depth in config.depths:
            for candidate_seed in config.candidate_seeds:
                sequences = _sequences_for(grid_map, state, depth, candidate_seed, config.max_candidates)
                candidate_counts.append(
                    {
                        "eval_set": eval_set, "map": entry.map_name, "state_key": entry.state_key,
                        "H": depth, "candidate_seed": candidate_seed, "actual_K": len(sequences),
                        "distinct_sequences": len(set(sequences)),
                    }
                )
                exact_results = [rollout_exact(state, sequence, grid_map) for sequence in sequences]
                truth = [score_trajectory(result) for result in exact_results]
                model_scores: Dict[int, List[ScoreComponents]] = {}
                for model_seed in config.model_seeds:
                    step_fn = point_step_fn(models[model_seed], grid_map)
                    model_scores[model_seed] = [
                        score_trajectory(rollout(state, sequence, grid_map, step_fn))
                        for sequence in sequences
                    ]

                for size in config.sizes:
                    k = min(size, len(sequences))
                    if k <= 0:
                        continue
                    ids = list(range(k))
                    j_ref = np.asarray([truth[i].J for i in range(k)], dtype=np.float64)
                    for model_seed in (EXACT_SEED, *config.model_seeds):
                        scores = truth if model_seed == EXACT_SEED else model_scores[model_seed]
                        j_hat = np.asarray([scores[i].J for i in range(k)], dtype=np.float64)
                        metrics = selection_metrics(ids, j_hat, j_ref)
                        chosen = int(metrics["chosen_index"])
                        invalid_rate = float(np.mean([scores[i].invalid_prediction for i in range(k)]))
                        chosen_terms = decompose_components(scores[chosen].as_dict(), truth[chosen].as_dict())
                        mean_terms = {
                            term: float(np.mean([
                                decompose_components(scores[i].as_dict(), truth[i].as_dict())[term]
                                for i in range(k)
                            ]))
                            for term in DECOMP_TERMS
                        }
                        run = {
                            "eval_set": eval_set,
                            "map": entry.map_name,
                            "state_key": entry.state_key,
                            "state": state_payload(state),
                            "H": depth,
                            "K": size,
                            "actual_K": k,
                            "candidate_seed": candidate_seed,
                            "model_seed": model_seed,
                            "checkpoint_sha256": (
                                "exact"
                                if model_seed == EXACT_SEED
                                else (checkpoint_hashes_by_seed or {}).get(model_seed, "")
                            ),
                            "chosen_index": chosen,
                            "chosen_candidate_id": int(metrics["chosen_candidate_id"]),
                            "selected_bias": float(metrics["selected_bias"]),
                            "candidate_mean_bias": float(metrics["candidate_mean_bias"]),
                            "selection_excess": float(metrics["selection_excess"]),
                            "candidate_regret": float(metrics["candidate_regret"]),
                            "chosen_true_return": float(metrics["chosen_true_return"]),
                            "uniform_candidate_return": float(metrics["uniform_candidate_return"]),
                            "candidate_best_true_return": float(metrics["candidate_best_true_return"]),
                            "chosen_model_return": float(metrics["chosen_model_return"]),
                            "chosen_true_success": int(truth[chosen].success),
                            "chosen_true_steps": int(truth[chosen].executed_steps),
                            "chosen_true_invalid": int(truth[chosen].invalid_prediction),
                            "chosen_model_success": int(scores[chosen].success),
                            "chosen_model_invalid": int(scores[chosen].invalid_prediction),
                            "candidate_invalid_rate": invalid_rate,
                            "candidate_success_rate": float(np.mean([truth[i].success for i in range(k)])),
                            "decomposition_chosen": chosen_terms,
                            "decomposition_candidate_mean": mean_terms,
                        }
                        per_run.append(run)

                        if want_traces and depth == config.trace_depth and size == config.trace_size and model_seed != EXACT_SEED:
                            best_true = max(range(k), key=lambda i: (j_ref[i], -i))
                            traces.append(
                                {
                                    "eval_set": eval_set,
                                    "map": entry.map_name,
                                    "state_key": entry.state_key,
                                    "H": depth,
                                    "K": size,
                                    "candidate_seed": candidate_seed,
                                    "model_seed": model_seed,
                                    "chosen_candidate_id": int(metrics["chosen_candidate_id"]),
                                    "best_true_candidate_id": int(best_true),
                                    "chosen_sequence": [int(a) for a in sequences[chosen]],
                                    "best_true_sequence": [int(a) for a in sequences[best_true]],
                                    "chosen_true_states": [state_payload(s) for s in exact_results[chosen].states],
                                    "chosen_true_J": float(truth[chosen].J),
                                    "chosen_model_J": float(scores[chosen].J),
                                }
                            )

                if want_candidate_records:
                    for candidate_id, sequence in enumerate(sequences):
                        candidate_records.append(
                            {
                                "eval_set": eval_set,
                                "map": entry.map_name,
                                "state_key": entry.state_key,
                                "H": depth,
                                "candidate_seed": candidate_seed,
                                "candidate_id": candidate_id,
                                "sequence": [int(a) for a in sequence],
                                "J_ref": float(truth[candidate_id].J),
                                "J_ref_components": truth[candidate_id].as_dict(),
                                "J_hat": {
                                    str(seed): float(model_scores[seed][candidate_id].J)
                                    for seed in config.model_seeds
                                },
                                "J_hat_components": {
                                    str(seed): model_scores[seed][candidate_id].as_dict()
                                    for seed in config.model_seeds
                                },
                            }
                        )

    return {
        "per_run": per_run,
        "candidate_records": candidate_records,
        "traces": traces,
        "candidate_counts": candidate_counts,
    }


def candidate_count_summary(candidate_counts: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Actual distinct candidate count per (eval_set, H) - H=1 caps at 81."""

    grouped: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for row in candidate_counts:
        grouped[(str(row["eval_set"]), int(row["H"]))].append(int(row["actual_K"]))
    rows = []
    for (eval_set, depth), values in sorted(grouped.items()):
        rows.append(
            {
                "eval_set": eval_set,
                "H": depth,
                "n_state_candidate_seed_runs": len(values),
                "actual_K_min": int(np.min(values)),
                "actual_K_max": int(np.max(values)),
                "actual_K_mean": float(np.mean(values)),
                "capped_at_81": bool(depth == 1),
            }
        )
    return rows


def aggregate_by_state(
    per_run: Sequence[Dict[str, object]],
    *,
    eval_set: Optional[str] = None,
    metric_keys: Sequence[str] = RUN_METRIC_KEYS,
) -> List[Dict[str, object]]:
    """Mean within one true initial state first (over candidate seeds), then over states.

    The model seeds and candidate seeds are repeated measurements, not independent scenes.
    """

    runs = [r for r in per_run if eval_set is None or r["eval_set"] == eval_set]
    per_state: Dict[Tuple, List[Dict[str, object]]] = defaultdict(list)
    for run in runs:
        per_state[(int(run["H"]), int(run["K"]), int(run["model_seed"]), run["map"], run["state_key"])].append(run)

    state_rows: List[Dict[str, object]] = []
    for key, group in per_state.items():
        row: Dict[str, object] = {
            "H": key[0], "K": key[1], "model_seed": key[2], "map": key[3], "state_key": key[4],
            "n_runs": len(group),
        }
        for metric in metric_keys:
            values = [float(g[metric]) for g in group]
            row[metric] = float(np.mean(values))
        state_rows.append(row)

    rows: List[Dict[str, object]] = []
    scopes = sorted({str(r["map"]) for r in state_rows}) + ["all"]
    for scope in scopes:
        subset = [r for r in state_rows if scope == "all" or r["map"] == scope]
        for depth, size in sorted({(int(r["H"]), int(r["K"])) for r in subset}):
            for model_seed in sorted({r["model_seed"] for r in subset}):
                group = [r for r in subset if r["H"] == depth and r["K"] == size and r["model_seed"] == model_seed]
                if not group:
                    continue
                row: Dict[str, object] = {
                    "eval_set": eval_set, "H": depth, "K": size, "map": scope,
                    "model_seed": model_seed, "n_states": len(group),
                }
                for metric in metric_keys:
                    values = [float(r[metric]) for r in group]
                    row[metric] = float(np.mean(values))
                    row[metric + "_min"] = float(np.min(values))
                    row[metric + "_max"] = float(np.max(values))
                rows.append(row)

    # The ``mean`` row averages the learned seeds only. The exact model
    # (``model_seed = EXACT_SEED``) keeps its own row in ``rows`` and never enters a mean:
    # its bias is identically zero, so mixing it in would silently shrink every headline by
    # a factor len(learned_seeds) / (len(learned_seeds) + 1).
    learned_seeds = sorted({int(r["model_seed"]) for r in rows if int(r["model_seed"]) != EXACT_SEED})
    seed_mean_rows: List[Dict[str, object]] = []
    for scope in scopes:
        for depth, size in sorted({(int(r["H"]), int(r["K"])) for r in rows if r["map"] == scope}):
            group = [
                r for r in rows
                if r["map"] == scope and r["H"] == depth and r["K"] == size
                and int(r["model_seed"]) != EXACT_SEED
            ]
            if not group:
                continue
            row = {
                "eval_set": eval_set, "H": depth, "K": size, "map": scope,
                "model_seed": "mean", "n_states": int(np.mean([r["n_states"] for r in group])),
                "aggregation": "learned_seeds_only",
                "seed_set": list(learned_seeds),
                "exact_row_excluded": True,
            }
            for metric in metric_keys:
                values = [float(r[metric]) for r in group]
                row[metric] = float(np.mean(values))
                row[metric + "_min"] = float(np.min([float(r[metric + "_min"]) for r in group]))
                row[metric + "_max"] = float(np.max([float(r[metric + "_max"]) for r in group]))
            seed_mean_rows.append(row)
    return rows + seed_mean_rows


# ---------------------------------------------------------------------------
# A3 - decomposition, tracing and first-error counterfactuals
# ---------------------------------------------------------------------------


def decompose_components(hat: Dict[str, object], ref: Dict[str, object]) -> Dict[str, float]:
    """The four terms of ``J_hat - J_ref``; they sum to the total deviation exactly."""

    return {
        "success_reward": 10.0 * (float(hat["success"]) - float(ref["success"])),
        "executed_steps": -(float(hat["executed_steps"]) - float(ref["executed_steps"])),
        "terminal_distance": -0.5 * (float(hat["terminal_distance"]) - float(ref["terminal_distance"])),
        "invalid_penalty": -10.0 * (float(hat["invalid_prediction"]) - float(ref["invalid_prediction"])),
    }


def decomposition_total(terms: Dict[str, float]) -> float:
    return float(sum(float(terms[term]) for term in DECOMP_TERMS))


def summarize_decomposition(
    candidate_records: Sequence[Dict[str, object]],
    per_run: Sequence[Dict[str, object]],
    *,
    eval_set: str = "v2",
) -> Dict[str, object]:
    """Aggregate the four terms for the candidate mean and the chosen candidate."""

    max_residual = 0.0
    checks = 0
    for record in candidate_records:
        ref = record["J_ref_components"]
        for hat in record["J_hat_components"].values():
            terms = decompose_components(hat, ref)
            residual = abs(decomposition_total(terms) - (float(hat["J"]) - float(ref["J"])))
            max_residual = max(max_residual, residual)
            checks += 1

    # candidate mean: average the four terms over all candidates, per (H, candidate seed,
    # model seed), then over candidate seeds within a state (the records cover all states).
    model_seeds = sorted({key for record in candidate_records for key in record["J_hat_components"]})
    mean_bucket: Dict[Tuple[int, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for record in candidate_records:
        if record["eval_set"] != eval_set:
            continue
        mean_bucket[(int(record["H"]), str(record["state_key"]), str(record["candidate_seed"]), "all")].append(record)
    per_state_mean: List[Dict[str, object]] = []
    for (depth, state_key, candidate_seed, _), records in mean_bucket.items():
        row: Dict[str, object] = {"H": depth, "state_key": state_key, "candidate_seed": int(candidate_seed)}
        for seed_key in model_seeds:
            totals = {term: [] for term in DECOMP_TERMS}
            for record in records:
                terms = decompose_components(record["J_hat_components"][seed_key], record["J_ref_components"])
                for term in DECOMP_TERMS:
                    totals[term].append(terms[term])
            row[seed_key] = {term: float(np.mean(totals[term])) for term in DECOMP_TERMS}
            row[seed_key]["total"] = float(sum(row[seed_key][term] for term in DECOMP_TERMS))
        per_state_mean.append(row)
    mean_rows: List[Dict[str, object]] = []
    for depth in sorted({int(r["H"]) for r in per_state_mean}):
        for seed_key in model_seeds:
            group = [r for r in per_state_mean if r["H"] == depth]
            if not group:
                continue
            row = {
                "H": depth, "model_seed": int(seed_key), "n_states": len(group),
            }
            for term in DECOMP_TERMS:
                row[term] = float(np.mean([r[seed_key][term] for r in group]))
            row["total"] = float(sum(row[term] for term in DECOMP_TERMS))
            mean_rows.append(row)

    # chosen candidate: mean over candidate seeds within one state first, then over states
    chosen_rows: List[Dict[str, object]] = []
    bucket: Dict[Tuple[int, int, int, str, str], List[Dict[str, object]]] = defaultdict(list)
    for run in per_run:
        if run["eval_set"] != eval_set or int(run["model_seed"]) == EXACT_SEED:
            continue
        bucket[(int(run["H"]), int(run["K"]), int(run["model_seed"]), run["map"], run["state_key"])].append(run)
    per_state: List[Dict[str, object]] = []
    for key, group in bucket.items():
        row: Dict[str, object] = {
            "H": key[0], "K": key[1], "model_seed": key[2], "map": key[3], "state_key": key[4],
        }
        for term in DECOMP_TERMS:
            row[term] = float(np.mean([float(g["decomposition_chosen"][term]) for g in group]))
        row["total"] = float(sum(row[term] for term in DECOMP_TERMS))
        per_state.append(row)
    for depth, size in sorted({(r["H"], r["K"]) for r in per_state}):
        for model_seed in sorted({r["model_seed"] for r in per_state}):
            group = [r for r in per_state if r["H"] == depth and r["K"] == size and r["model_seed"] == model_seed]
            if not group:
                continue
            chosen_rows.append(
                {
                    "H": depth, "K": size, "model_seed": model_seed, "n_states": len(group),
                    **{term: float(np.mean([r[term] for r in group])) for term in DECOMP_TERMS},
                    "total": float(np.mean([r["total"] for r in group])),
                }
            )

    return {
        "terms": list(DECOMP_TERMS),
        "algebraic_check": {
            "n_checked": checks,
            "max_abs_residual": max_residual,
            "passed": bool(max_residual < 1e-9),
        },
        "candidate_mean": mean_rows,
        "chosen": chosen_rows,
    }


def _run_id(run: Dict[str, object]) -> str:
    return (
        f"{run['map']}|{run['state_key']}|H={run['H']}|K={run['K']}"
        f"|cs={run['candidate_seed']}|ms={run['model_seed']}"
    )


def select_traced_states(
    per_run: Sequence[Dict[str, object]],
    *,
    eval_set: str = "v2",
    depth: int = 4,
    size: int = 256,
    n_top: int = 3,
) -> Dict[str, object]:
    """States with the largest mean regret (over model and candidate seeds), plus a control."""

    runs = [
        r for r in per_run
        if r["eval_set"] == eval_set and int(r["H"]) == depth and int(r["K"]) == size
    ]
    by_state: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for run in runs:
        by_state[(str(run["map"]), str(run["state_key"]))].append(float(run["candidate_regret"]))
    means = {key: float(np.mean(values)) for key, values in by_state.items()}
    ordered = sorted(means.items(), key=lambda item: (-item[1], item[0]))
    top = [
        {"map": key[0], "state_key": key[1], "mean_regret": means[key]}
        for key, _ in ordered[:n_top]
    ]
    control_key = sorted(means.items(), key=lambda item: (item[1], item[0]))[0][0]
    return {
        "depth": depth,
        "size": size,
        "n_states": len(means),
        "selection": (
            "mean candidate_regret per state over model seeds and candidate seeds; top 3 and "
            "the minimum as control (each state has the same number of candidate seeds per "
            "model seed, so this equals the mean over model seeds of the within-state mean)"
        ),
        "top": top,
        "control": {
            "map": control_key[0], "state_key": control_key[1], "mean_regret": means[control_key]
        },
        "mean_regret_by_state": [
            {"map": key[0], "state_key": key[1], "mean_regret": means[key]} for key, _ in ordered
        ],
    }


def pick_traced_run(
    per_run: Sequence[Dict[str, object]],
    map_name: str,
    state_key: str,
    *,
    eval_set: str = "v2",
    depth: int = 4,
    size: int = 256,
) -> Dict[str, object]:
    runs = [
        r for r in per_run
        if r["eval_set"] == eval_set
        and r["map"] == map_name
        and r["state_key"] == state_key
        and int(r["H"]) == depth
        and int(r["K"]) == size
        and int(r["model_seed"]) != EXACT_SEED
    ]
    # largest regret; a stable run id breaks ties
    return max(runs, key=lambda run: (float(run["candidate_regret"]), _run_id(run)))


def _position_error(predicted: GridState, actual: GridState) -> float:
    return float(
        abs(predicted.robot0[0] - actual.robot0[0])
        + abs(predicted.robot0[1] - actual.robot0[1])
        + abs(predicted.robot1[0] - actual.robot1[0])
        + abs(predicted.robot1[1] - actual.robot1[1])
        + abs(predicted.object_anchor[0] - actual.object_anchor[0])
        + abs(predicted.object_anchor[1] - actual.object_anchor[1])
    )


def model_rollout_detail(
    grid_map: GridMap,
    initial: GridState,
    sequence: Sequence[int],
    true_states: Sequence[GridState],
    step_fn,
    repair: Optional[str] = None,
) -> Tuple[ScoreComponents, List[Dict[str, object]]]:
    """A protocol-identical model rollout that keeps every attempted step.

    Semantics match ``scoring.rollout``: an illegal prediction terminates the chain at the
    failing step (the previous legal state is never reused to fabricate a next state), and
    a predicted goal is absorbing. With ``repair`` set to ``full_state``,
    ``object_anchor`` or ``false_success`` the *first* occurrence of that error is replaced
    by the true next state once, and the chain then continues from the repair. This is a
    counterfactual ranking test only; the reported diagnostic itself never repairs.

    ``object_anchor`` replaces the object anchor only. If the resulting combined state is
    invalid, the chain terminates under the original protocol; the robot positions are never
    additionally repaired. ``false_success`` substitutes the whole true next state.
    """

    state = initial
    success = 0
    invalid = 0
    executed = 0
    repaired = False
    records: List[Dict[str, object]] = []
    for index, joint_action in enumerate(sequence):
        true_next = true_states[index + 1] if index + 1 < len(true_states) else None
        predicted, legal = step_fn(state, joint_action)
        if repair and not repaired and predicted is not None and true_next is not None:
            if repair == "full_state" and predicted != true_next:
                predicted, legal, repaired = true_next, True, True
            elif repair == "object_anchor" and predicted.object_anchor != true_next.object_anchor:
                predicted = GridState(predicted.robot0, predicted.robot1, true_next.object_anchor)
                legal = state_is_valid(predicted, grid_map)
                repaired = True
            elif repair == "false_success" and is_goal(predicted, grid_map) and not is_goal(true_next, grid_map):
                predicted, legal, repaired = true_next, True, True
        executed += 1
        is_invalid = predicted is None or not legal
        predicted_success = (not is_invalid) and is_goal(predicted, grid_map)  # type: ignore[arg-type]
        record = {
            "step": index,
            "action": int(joint_action),
            "input_state": state,
            "predicted_state": predicted,
            "true_next_state": true_next,
            "legal": not is_invalid,
            "predicted_success": bool(predicted_success),
            "true_next_is_goal": bool(true_next is not None and is_goal(true_next, grid_map)),
            "full_state_match": None if (predicted is None or true_next is None) else bool(predicted == true_next),
            "object_anchor_match": (
                None if (predicted is None or true_next is None)
                else bool(predicted.object_anchor == true_next.object_anchor)
            ),
            "position_error": (
                None if (predicted is None or true_next is None) else _position_error(predicted, true_next)
            ),
            "false_success": bool(
                predicted_success and true_next is not None and not is_goal(true_next, grid_map)
            ),
            "invalid": bool(is_invalid),
            "termination_reason": None,
        }
        if is_invalid:
            invalid = 1
            record["termination_reason"] = "invalid_prediction"
            records.append(record)
            break
        state = predicted  # type: ignore[assignment]
        records.append(record)
        if predicted_success:
            success = 1
            record["termination_reason"] = "success"
            break
    distance = 0.0 if success else terminal_distance(state, grid_map)
    score = ScoreComponents(
        success=success,
        executed_steps=executed,
        terminal_distance=distance,
        invalid_prediction=invalid,
        J=float(10.0 * success - executed - 0.5 * distance - 10.0 * invalid),
    )
    return score, records


def _step_payload(record: Dict[str, object]) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "step": int(record["step"]) + 1,
        "action": int(record["action"]),
        "legal": bool(record["legal"]),
        "predicted_success": bool(record["predicted_success"]),
        "false_success": bool(record["false_success"]),
        "invalid": bool(record["invalid"]),
        "termination_reason": record["termination_reason"],
        "full_state_match": record["full_state_match"],
        "object_anchor_match": record["object_anchor_match"],
        "position_error": record["position_error"],
        "input_state": state_payload(record["input_state"]),  # type: ignore[arg-type]
        "predicted_state": (
            None if record["predicted_state"] is None else state_payload(record["predicted_state"])  # type: ignore[arg-type]
        ),
        "true_next_state": (
            None if record["true_next_state"] is None else state_payload(record["true_next_state"])  # type: ignore[arg-type]
        ),
    }
    return payload


def build_traced_case(
    maps: Dict[str, GridMap],
    models: Dict[int, PointModel],
    map_name: str,
    state: GridState,
    run: Dict[str, object],
    config: AuditConfig,
) -> Dict[str, object]:
    """Steps, errors and the single-error counterfactual ranking test for one run."""

    grid_map = maps[map_name]
    depth = int(run["H"])
    size = int(run["K"])
    candidate_seed = int(run["candidate_seed"])
    model_seed = int(run["model_seed"])
    sequences = _sequences_for(grid_map, state, depth, candidate_seed, config.max_candidates)
    k = min(size, len(sequences))
    exact = [rollout_exact(state, sequence, grid_map) for sequence in sequences[:k]]
    truth = [score_trajectory(result) for result in exact]
    step_fn = point_step_fn(models[model_seed], grid_map)
    j_ref = np.asarray([score.J for score in truth], dtype=np.float64)
    j_hat = np.asarray(
        [score_trajectory(rollout(state, sequence, grid_map, step_fn)).J for sequence in sequences[:k]],
        dtype=np.float64,
    )
    metrics = selection_metrics(list(range(k)), j_hat, j_ref)
    chosen = int(metrics["chosen_index"])
    best_true = max(range(k), key=lambda index: (j_ref[index], -index))

    chosen_score, chosen_records = model_rollout_detail(
        grid_map, state, sequences[chosen], exact[chosen].states, step_fn
    )

    def first(records: Sequence[Dict[str, object]], key: str):
        for record in records:
            if key == "false_success":
                if record["false_success"]:
                    return int(record["step"])
            elif record[key] is False:
                return int(record["step"])
        return None

    first_full = first(chosen_records, "full_state_match")
    first_object = first(chosen_records, "object_anchor_match")
    first_false = first(chosen_records, "false_success")

    # A2 primary analysis: cache the original ``J_hat`` of every candidate, repair the *first*
    # occurrence of the specified error in the originally chosen candidate ``chosen`` only,
    # continue that candidate's own model chain, replace exactly ``J_hat[chosen]`` and re-rank.
    # Every other entry must stay bit-identical. "The ranking changed" is reported separately
    # from "the newly selected candidate has a better true return"; a repair can help even
    # when the selection does not move (and can move the selection backwards).
    def single_candidate_counterfactual(
        repair: str, error_step: Optional[int], label: str
    ) -> Dict[str, object]:
        repaired_score, _ = model_rollout_detail(
            grid_map, state, sequences[chosen], exact[chosen].states, step_fn, repair=repair
        )
        modified = j_hat.copy()
        modified[chosen] = float(repaired_score.J)
        new_chosen = int(max(range(k), key=lambda index: (modified[index], -index)))
        other_identical = bool(
            all(float(modified[index]) == float(j_hat[index]) for index in range(k) if index != chosen)
        )
        changed = int(new_chosen) != chosen
        true_improved = bool(float(j_ref[new_chosen]) > float(j_ref[chosen]))
        if error_step is None:
            note = "该错误在选中候选的模型链中未出现，J_hat 不变，排序不变"
        elif changed and true_improved:
            note = "单候选干预改变了选中候选，且新选择的真实收益更高"
        elif changed:
            note = "单候选干预改变了选中候选，但新选择的真实收益未改善（选择改变≠修复有收益）"
        else:
            note = "单候选干预未改变选中候选；该错误与本次排序无直接联系"
        return {
            "intervention": label,
            "repair": repair,
            "scope": "chosen_candidate_only",
            "first_step": None if error_step is None else error_step + 1,
            "error_absent": error_step is None,
            "repaired_candidate_id": int(chosen),
            "baseline_chosen_candidate_id": int(metrics["chosen_candidate_id"]),
            "counterfactual_chosen_candidate_id": int(new_chosen),
            "changed_ranking": bool(changed),
            "baseline_chosen_model_return": float(j_hat[chosen]),
            "repaired_candidate_model_return": float(repaired_score.J),
            "counterfactual_chosen_model_return": float(modified[new_chosen]),
            "baseline_new_chosen_model_return": float(j_hat[new_chosen]),
            "baseline_chosen_true_return": float(j_ref[chosen]),
            "new_chosen_true_return": float(j_ref[new_chosen]),
            "true_return_delta": float(j_ref[new_chosen] - j_ref[chosen]),
            "true_return_improved": true_improved,
            "only_chosen_entry_replaced": other_identical,
            "n_changed_entries": int(np.count_nonzero(modified != j_hat)),
            "changed_entry_indices": [int(index) for index in np.nonzero(modified != j_hat)[0]],
            "baseline_j_hat": [float(value) for value in j_hat],
            "counterfactual_j_hat": [float(value) for value in modified],
            "note": note,
        }

    counterfactuals: Dict[str, object] = {
        "full_state": single_candidate_counterfactual(
            "full_state", first_full, "首次完整状态不一致处的完整状态干预"
        ),
        "object_anchor": single_candidate_counterfactual(
            "object_anchor", first_object, "首次物体锚点不一致处的物体锚点干预"
        ),
        "false_success": {
            **single_candidate_counterfactual(
                "false_success", first_false, "首次虚假成功处的完整状态干预"
            ),
            # This repair substitutes the whole true next state; it is *not* a fix of the
            # success classifier alone, and must never be described as one.
            "replaces_whole_true_state": True,
            "success_classification_only": False,
        },
    }

    # Historical attachment: the old implementation repaired every candidate and then
    # reported it as a single-candidate intervention. It is kept, clearly labelled, with
    # corrected causal wording, and is not the primary analysis.
    historical: Dict[str, object] = {}
    for repair, error_step, label in (
        ("full_state", first_full, "首次完整状态不一致处的完整状态干预"),
        ("object_anchor", first_object, "首次物体锚点不一致处的物体锚点干预"),
        ("false_success", first_false, "首次虚假成功处的完整状态干预"),
    ):
        repaired_j: List[float] = []
        for index in range(k):
            score, _ = model_rollout_detail(
                grid_map, state, sequences[index], exact[index].states, step_fn, repair=repair
            )
            repaired_j.append(float(score.J))
        repaired_hat = np.asarray(repaired_j, dtype=np.float64)
        new_chosen = int(max(range(k), key=lambda index: (repaired_hat[index], -index)))
        changed = int(new_chosen) != chosen
        true_improved = bool(float(j_ref[new_chosen]) > float(j_ref[chosen]))
        historical[repair] = {
            "intervention": label,
            "repair": repair,
            "scope": "all_candidates_historical",
            "first_step": None if error_step is None else error_step + 1,
            "error_absent": error_step is None,
            "baseline_chosen_candidate_id": int(metrics["chosen_candidate_id"]),
            "counterfactual_chosen_candidate_id": int(new_chosen),
            "changed_ranking": bool(changed),
            "baseline_chosen_true_return": float(j_ref[chosen]),
            "counterfactual_chosen_true_return": float(j_ref[new_chosen]),
            "true_return_delta": float(j_ref[new_chosen] - j_ref[chosen]),
            "true_return_improved": true_improved,
            "n_changed_entries": int(np.count_nonzero(repaired_hat != j_hat)),
            "note": (
                "历史附件（已更正因果表述）：对**全部候选**各自修复其首次同类错误后重排，"
                "而不是只修原选中候选；因此不能读作单候选干预的因果效果"
            ),
        }

    return {
        "run_id": _run_id(run),
        "map": map_name,
        "state_key": run["state_key"],
        "state": state_payload(state),
        "H": depth,
        "K": size,
        "actual_K": k,
        "candidate_seed": candidate_seed,
        "model_seed": model_seed,
        "chosen_candidate_id": int(metrics["chosen_candidate_id"]),
        "best_true_candidate_id": int(best_true),
        "chosen_candidate_regret": float(metrics["candidate_regret"]),
        "selected_bias": float(metrics["selected_bias"]),
        "candidate_mean_bias": float(metrics["candidate_mean_bias"]),
        "chosen_true_J": float(truth[chosen].J),
        "chosen_model_J": float(chosen_score.J),
        "chosen_true_components": truth[chosen].as_dict(),
        "chosen_model_components": chosen_score.as_dict(),
        "decomposition_chosen": decompose_components(chosen_score.as_dict(), truth[chosen].as_dict()),
        "chosen_actions": [int(a) for a in sequences[chosen]],
        "chosen_steps": [_step_payload(record) for record in chosen_records],
        "best_true_actions": [int(a) for a in sequences[best_true]],
        "best_true_true_components": truth[best_true].as_dict(),
        "best_true_model_components": score_trajectory(
            rollout(state, sequences[best_true], grid_map, step_fn)
        ).as_dict(),
        "first_full_state_mismatch_step": None if first_full is None else first_full + 1,
        "first_object_anchor_mismatch_step": None if first_object is None else first_object + 1,
        "first_false_success_step": None if first_false is None else first_false + 1,
        "counterfactuals": counterfactuals,
        "counterfactuals_all_candidates_historical": historical,
        "counterfactual_scope_note": (
            "counterfactuals 是单候选主分析：只替换原选中候选的 J_hat 并重排；"
            "counterfactuals_all_candidates_historical 是旧全候选干预的历史附件，已更正因果表述。"
        ),
    }


def traced_cases(
    maps: Dict[str, GridMap],
    models: Dict[int, PointModel],
    per_run: Sequence[Dict[str, object]],
    config: AuditConfig,
    *,
    eval_set: str = "v2",
) -> Dict[str, object]:
    selection = select_traced_states(per_run, eval_set=eval_set, depth=config.trace_depth, size=config.trace_size)
    cases: List[Dict[str, object]] = []
    entries = list(selection["top"]) + [
        {**selection["control"], "role": "control"}
    ]
    for item in entries:
        map_name = str(item["map"])
        state_key = str(item["state_key"])
        _, state = parse_state_key(state_key)
        run = pick_traced_run(
            per_run, map_name, state_key, eval_set=eval_set,
            depth=config.trace_depth, size=config.trace_size,
        )
        case = build_traced_case(maps, models, map_name, state, run, config)
        case["role"] = item.get("role", "top_regret")
        case["mean_regret_over_seeds"] = item["mean_regret"]
        cases.append(case)
    return {"selection": selection, "cases": cases}


# ---------------------------------------------------------------------------
# A4 - protocol-identical recursive vs teacher-forced contrast
# ---------------------------------------------------------------------------


def a4_chain(
    grid_map: GridMap,
    initial: GridState,
    sequence: Sequence[int],
    true_states: Sequence[GridState],
    step_fn,
    mode: str,
) -> List[Dict[str, object]]:
    """One chain over the fixed true trajectory; ``mode`` is ``teacher`` or ``recursive``.

    Both modes use the main diagnostic rules: an illegal prediction terminates the chain at
    the failing step and no next state is fabricated; a predicted goal is absorbing. The
    chain is evaluated only over the steps of the fixed true trajectory
    (``len(true_states) - 1``).
    """

    records: List[Dict[str, object]] = []
    recursive_state = initial
    limit = min(len(sequence), max(0, len(true_states) - 1))
    for index in range(limit):
        joint_action = sequence[index]
        input_state = true_states[index] if mode == "teacher" else recursive_state
        predicted, legal = step_fn(input_state, joint_action)
        true_next = true_states[index + 1]
        is_invalid = predicted is None or not legal
        predicted_success = (not is_invalid) and is_goal(predicted, grid_map)  # type: ignore[arg-type]
        record = {
            "step": index,
            "action": int(joint_action),
            "input_state": input_state,
            "predicted_state": predicted,
            "true_next_state": true_next,
            "legal": not is_invalid,
            "predicted_success": bool(predicted_success),
            "true_next_is_goal": bool(is_goal(true_next, grid_map)),
            "full_state_match": None if predicted is None else bool(predicted == true_next),
            "position_error": None if predicted is None else _position_error(predicted, true_next),
            "false_success": bool(predicted_success and not is_goal(true_next, grid_map)),
            "invalid": bool(is_invalid),
            "termination_reason": None,
        }
        if is_invalid:
            record["termination_reason"] = "invalid_prediction"
            records.append(record)
            break
        recursive_state = predicted  # type: ignore[assignment]
        records.append(record)
        if predicted_success:
            record["termination_reason"] = "success"
            break
    return records


def _chain_summary(chain: Sequence[Dict[str, object]], true_success: bool) -> Dict[str, object]:
    termination = next((int(r["step"]) for r in chain if r["termination_reason"] is not None), None)
    invalid = next((int(r["step"]) for r in chain if r["invalid"]), None)
    false_success = next((int(r["step"]) for r in chain if r["false_success"]), None)
    mismatch = next((int(r["step"]) for r in chain if r["full_state_match"] is False), None)
    predicted_success = next((int(r["step"]) for r in chain if r["predicted_success"]), None)
    missed_true_success = bool(true_success and predicted_success is None)
    bad_steps = [s for s in (termination, mismatch) if s is not None]
    return {
        "termination_step": termination,
        "invalid_step": invalid,
        "false_success_step": false_success,
        "mismatch_step": mismatch,
        "predicted_success_step": predicted_success,
        "missed_true_success": missed_true_success,
        "any_bad_step": min(bad_steps) if bad_steps else None,
    }


def recursive_vs_teacher(
    maps: Dict[str, GridMap],
    models: Dict[int, PointModel],
    eval_states: Sequence[EvalState],
    config: AuditConfig,
    *,
    eval_set: str = "v2",
    candidate_seed: int = 0,
) -> Dict[str, object]:
    """Per-step termination, survivor error with its denominator, and all-chain rates."""

    depth = config.teacher_horizon
    per_model: Dict[str, object] = {}
    for model_seed, model in sorted(models.items()):
        chains: Dict[str, List[List[Dict[str, object]]]] = {"teacher": [], "recursive": []}
        true_success_flags: List[bool] = []
        for entry in eval_states:
            grid_map = maps[entry.map_name]
            sequences = _sequences_for(
                grid_map, entry.state, depth, candidate_seed, config.teacher_candidates
            )
            step_fn = point_step_fn(model, grid_map)
            for sequence in sequences:
                true_result = rollout_exact(entry.state, sequence, grid_map)
                true_states = true_result.states
                if len(true_states) - 1 <= 0:
                    continue
                true_success_flags.append(bool(true_result.success))
                for mode in ("teacher", "recursive"):
                    chains[mode].append(
                        a4_chain(grid_map, entry.state, sequence, true_states, step_fn, mode)
                    )
        n_chains = len(true_success_flags)
        per_model[str(model_seed)] = {
            mode: _summarize_a4(chains[mode], true_success_flags, depth, n_chains)
            for mode in ("teacher", "recursive")
        }
        per_model[str(model_seed)]["n_chains"] = n_chains

    return {
        "eval_set": eval_set,
        "horizon": depth,
        "candidates_per_state": config.teacher_candidates,
        "candidate_seed": candidate_seed,
        "protocol": (
            "same as the main diagnostic: an illegal prediction terminates the chain at the "
            "failing step (no fabricated next state) and a predicted goal is absorbing; the "
            "chain is evaluated over the fixed true trajectory's steps"
        ),
        "per_model": per_model,
        "pooled": _pool_a4(per_model, depth),
    }


def _summarize_a4(
    chains: Sequence[Sequence[Dict[str, object]]],
    true_success_flags: Sequence[bool],
    depth: int,
    n_chains: int,
) -> Dict[str, object]:
    summaries = [
        _chain_summary(chain, true_success_flags[i]) for i, chain in enumerate(chains)
    ]
    steps: List[Dict[str, object]] = []
    for index in range(depth):
        at = [chain[index] for chain in chains if len(chain) > index]
        legal_at = [record for record in at if record["legal"]]
        invalid_at = sum(1 for record in at if record["invalid"])
        success_at = sum(1 for record in at if record["predicted_success"])
        false_at = sum(1 for record in at if record["false_success"])
        mismatch_at = sum(1 for record in legal_at if record["full_state_match"] is False)
        position = [float(record["position_error"]) for record in legal_at if record["position_error"] is not None]
        terminated_cum = sum(1 for s in summaries if s["termination_step"] is not None and s["termination_step"] <= index)
        invalid_cum = sum(1 for s in summaries if s["invalid_step"] is not None and s["invalid_step"] <= index)
        false_cum = sum(1 for s in summaries if s["false_success_step"] is not None and s["false_success_step"] <= index)
        missed_cum = sum(1 for s in summaries if s["missed_true_success"])
        bad_cum = sum(1 for s in summaries if s["any_bad_step"] is not None and s["any_bad_step"] <= index)
        steps.append(
            {
                "step": index + 1,
                "n_chains": n_chains,
                "predictions_made": len(at),
                "survivor_denominator": len(legal_at),
                "invalid_at_step": invalid_at,
                "success_at_step": success_at,
                "false_success_at_step": false_at,
                "full_state_error_on_survivors": _rate(mismatch_at, len(legal_at)),
                "position_error_on_survivors": float(np.mean(position)) if position else None,
                "termination_rate_cumulative": _rate(terminated_cum, n_chains),
                "invalid_rate_cumulative": _rate(invalid_cum, n_chains),
                "false_success_rate_cumulative": _rate(false_cum, n_chains),
                "missed_true_success_rate_cumulative": _rate(missed_cum, n_chains),
                "failure_or_error_rate_cumulative_all_chains": _rate(bad_cum, n_chains),
                "full_state_error_all_chains": _rate(mismatch_at + sum(1 for s in summaries if s["termination_step"] is not None and s["termination_step"] < index), n_chains),
            }
        )
    return {
        "n_chains": n_chains,
        "chains_terminated": sum(1 for s in summaries if s["termination_step"] is not None),
        "chains_invalid": sum(1 for s in summaries if s["invalid_step"] is not None),
        "chains_false_success": sum(1 for s in summaries if s["false_success_step"] is not None),
        "chains_missed_true_success": sum(1 for s in summaries if s["missed_true_success"]),
        "steps": steps,
    }


def _pool_a4(per_model: Dict[str, object], depth: int) -> Dict[str, object]:
    pooled: Dict[str, object] = {}
    for mode in ("teacher", "recursive"):
        rows = [per_model[key][mode]["steps"] for key in per_model]
        steps = []
        for index in range(depth):
            row: Dict[str, object] = {"step": index + 1}
            for metric in (
                "termination_rate_cumulative",
                "invalid_rate_cumulative",
                "false_success_rate_cumulative",
                "missed_true_success_rate_cumulative",
                "failure_or_error_rate_cumulative_all_chains",
                "full_state_error_on_survivors",
                "position_error_on_survivors",
                "full_state_error_all_chains",
            ):
                values = [float(r[index][metric]) for r in rows if r[index][metric] is not None]
                row[metric] = float(np.mean(values)) if values else None
            steps.append(row)
        pooled[mode] = {"steps": steps}
    return pooled


# ---------------------------------------------------------------------------
# Small helpers used by the report
# ---------------------------------------------------------------------------


def state_from_key(state_key: str) -> GridState:
    return parse_state_key(state_key)[1]
