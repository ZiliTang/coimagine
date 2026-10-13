"""Phase A orchestration for ``forecast_baseline_v1`` (task book 20261012_2007).

Thin layer. It loads the frozen inputs, calls :mod:`coimagine.deterministic_grid.audit`,
and returns one artifact bundle. All heavy logic and every number live in ``audit``; the
experiment CLI only serialises the bundle and renders the tables.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..deterministic_grid.audit import (
    AUDIT_TAG,
    RUN_METRIC_KEYS,
    AuditConfig,
    EvalState,
    aggregate_by_state,
    candidate_count_summary,
    checkpoint_hashes,
    dataset_hashes,
    eval_state_coverage,
    group_statistics,
    load_saved_dataset,
    load_saved_models,
    read_jsonl,
    recursive_vs_teacher,
    run_fixed_candidate_audit,
    select_eval_states_v2,
    state_payload,
    summarize_decomposition,
    traced_cases,
)
from ..deterministic_grid.contract import default_maps
from ..deterministic_grid.diagnostic import DiagnosticConfig, select_eval_states


def _old_diagnostic_entries(dataset, maps) -> List[EvalState]:
    """The old 20 states, reproduced with the untouched old ordering rule (read-only)."""

    entries: List[EvalState] = []
    for rank, (map_name, state) in enumerate(select_eval_states(dataset, maps, DiagnosticConfig())):
        entries.append(
            EvalState(
                map_name=map_name,
                state=state,
                state_key=state.key(map_name),
                digest="",
                rank=rank,
            )
        )
    return entries


def _old_invalid_penalty_term(run_dir: Path) -> Dict[str, object]:
    """Fact 3: the invalid-prediction penalty sitting inside candidate_mean_bias at H=4."""

    rows = read_jsonl(Path(run_dir) / "diagnostic" / "candidate_records.jsonl")
    terms: List[float] = []
    totals: List[float] = []
    per_seed: Dict[str, List[float]] = {"0": [], "1": [], "2": []}
    for row in rows:
        if int(row["H"]) != 4:
            continue
        ref = row["J_ref_components"]
        for seed in ("0", "1", "2"):
            hat = row["J_hat_components"][seed]
            term = -10.0 * (float(hat["invalid_prediction"]) - float(ref["invalid_prediction"]))
            terms.append(term)
            totals.append(float(hat["J"]) - float(ref["J"]))
            per_seed[seed].append(term)
    return {
        "source": "outputs/deterministic_grid_v1/grid-20261012-194632/diagnostic/candidate_records.jsonl (read-only)",
        "H": 4,
        "n_candidate_seed_model_rows": len(terms),
        "mean_invalid_penalty_term": float(np.mean(terms)) if terms else None,
        "mean_total_J_hat_minus_J_ref": float(np.mean(totals)) if totals else None,
        "per_model_seed_mean_invalid_penalty_term": {
            seed: float(np.mean(values)) for seed, values in per_seed.items() if values
        },
    }


def _row_mean_metrics(per_run: Sequence[Dict[str, object]], eval_set: str) -> List[Dict[str, object]]:
    """The legacy aggregation rule: plain mean over all per_run rows.

    The old run's ``metrics.json`` averaged every (state, candidate seed, model seed) row
    together; CONTRACT section 1 A2 requires state-first aggregation instead. Both are
    reported so the old headline stays traceable.
    """

    from collections import defaultdict

    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for run in per_run:
        if run["eval_set"] != eval_set or int(run["model_seed"]) == -1:
            continue
        grouped[(int(run["H"]), int(run["K"]))].append(run)
    rows = []
    for (depth, size), group in sorted(grouped.items()):
        row: Dict[str, object] = {"H": depth, "K": size, "n_rows": len(group)}
        for metric in RUN_METRIC_KEYS:
            row[metric] = float(np.mean([float(r[metric]) for r in group]))
        rows.append(row)
    return rows


def _old_stored_metrics(run_dir: Path) -> List[Dict[str, object]]:
    """The old run's own ``metrics.json`` rows (model-seed mean), read-only."""

    stored = json.loads((Path(run_dir) / "diagnostic" / "metrics.json").read_text())
    keys = ("H", "K", "selected_bias", "candidate_mean_bias", "selection_excess", "candidate_regret",
            "chosen_true_return", "uniform_candidate_return", "candidate_best_true_return",
            "chosen_true_success", "candidate_invalid_rate", "candidate_success_rate")
    return [
        {key: row[key] for key in keys if key in row}
        for row in stored
        if row.get("model_seed") == "mean"
    ]


def _aggregation_reconciliation(
    run_dir: Path, per_run: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    stored = _old_stored_metrics(run_dir)
    recomputed = _row_mean_metrics(per_run, "old")
    index = {(int(r["H"]), int(r["K"])): r for r in recomputed}
    diffs = []
    for row in stored:
        key = (int(row["H"]), int(row["K"]))
        other = index.get(key)
        if other is None:
            continue
        for metric in ("selected_bias", "candidate_mean_bias", "selection_excess"):
            diffs.append(abs(float(row[metric]) - float(other[metric])))
    return {
        "note": (
            "旧表按行平均（每 state/候选种子/模型种子一行）汇总；CONTRACT 要求的聚合是先在状态内"
            "对候选种子取均值再对状态平均。两种口径都列出，旧表数字不被覆盖。"
        ),
        "legacy_row_mean_aggregation": recomputed,
        "old_stored_row_mean_metrics": stored,
        "max_abs_diff_legacy_vs_stored": float(max(diffs)) if diffs else None,
        "legacy_reproduced": bool(diffs and max(diffs) < 1e-9),
    }


def _old_set_reproduction(run_dir: Path, per_run: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Confirm the re-run reproduces the stored old per_run metrics exactly."""

    stored = read_jsonl(Path(run_dir) / "diagnostic" / "per_run.jsonl")
    index = {
        (r["state_key"], int(r["H"]), int(r["K"]), int(r["candidate_seed"]), int(r["model_seed"])): r
        for r in stored
    }
    diffs: List[float] = []
    matched = 0
    missing = 0
    for run in per_run:
        if run["eval_set"] != "old":
            continue
        key = (
            run["state_key"], int(run["H"]), int(run["K"]),
            int(run["candidate_seed"]), int(run["model_seed"]),
        )
        other = index.get(key)
        if other is None:
            missing += 1
            continue
        matched += 1
        for metric in RUN_METRIC_KEYS:
            diffs.append(abs(float(run[metric]) - float(other[metric])))
    return {
        "stored_rows": len(stored),
        "matched_rows": matched,
        "missing_rows": missing,
        "metrics_compared_per_row": len(RUN_METRIC_KEYS),
        "max_abs_diff": float(np.max(diffs)) if diffs else None,
        "reproduced_exactly": bool(diffs and max(diffs) < 1e-12),
    }


def _stored_moved_metric(run_dir: Path) -> Dict[str, object]:
    """Fact 2: the stored 0.53 is ``moved_full_state_accuracy`` (object_moved full state)."""

    summary = json.loads((Path(run_dir) / "training" / "summary.json").read_text())
    per_seed = {
        seed: payload["test"]["moved_full_state_accuracy"]
        for seed, payload in summary["seeds"].items()
    }
    return {
        "source": "outputs/deterministic_grid_v1/grid-20261012-194632/training/summary.json (read-only)",
        "stored_field": "moved_full_state_accuracy",
        "stored_per_model_seed": per_seed,
        "stored_mean": float(np.mean(list(per_seed.values()))),
        "corrected_name": "object_moved full-state accuracy",
        "not_the_same_as": "object push success accuracy",
    }


def run_phase_a(
    run_dir: Path,
    *,
    config: Optional[AuditConfig] = None,
    seeds: Sequence[int] = (0, 1, 2),
    states_per_map: int = 10,
) -> Dict[str, object]:
    config = config or AuditConfig()
    run_dir = Path(run_dir)
    started = time.time()

    start_hashes = {**dataset_hashes(run_dir), **checkpoint_hashes(run_dir, seeds)}
    dataset = load_saved_dataset(run_dir)
    maps = {grid_map.name: grid_map for grid_map in default_maps()}
    models = load_saved_models(run_dir, seeds)
    checkpoint_digests = checkpoint_hashes(run_dir, seeds)

    # ---- A1 ---------------------------------------------------------------
    group_stats = group_statistics(dataset, maps, models, split="test")

    # ---- A2 ---------------------------------------------------------------
    new_entries = select_eval_states_v2(dataset, maps, states_per_map=states_per_map)
    old_entries = _old_diagnostic_entries(dataset, maps)
    new_audit = run_fixed_candidate_audit(
        maps, models, new_entries, config,
        eval_set="v2", want_candidate_records=True, want_traces=True,
        checkpoint_hashes_by_seed=checkpoint_digests,
    )
    old_audit = run_fixed_candidate_audit(
        maps, models, old_entries, config,
        eval_set="old", want_candidate_records=False, want_traces=False,
        checkpoint_hashes_by_seed=checkpoint_digests,
    )
    per_run = new_audit["per_run"] + old_audit["per_run"]
    candidate_counts = new_audit["candidate_counts"] + old_audit["candidate_counts"]
    metrics_v2 = aggregate_by_state(per_run, eval_set="v2")
    metrics_old = aggregate_by_state(per_run, eval_set="old")

    eval_states_v2 = {
        "selection_tag": AUDIT_TAG,
        "selection_rule": (
            'from the saved splits.jsonl take the distinct legal non-terminal test states per map, '
            'order by SHA256("' + AUDIT_TAG + '|" + state_key), take the first '
            f"{states_per_map} per map"
        ),
        "used_coordinates_for_selection": False,
        "used_model_error_for_selection": False,
        "used_return_or_success_for_selection": False,
        "n_new_states": len(new_entries),
        "new_states": [entry.as_dict() for entry in new_entries],
        "coverage": eval_state_coverage(new_entries),
        "old_states_unchanged": [
            {
                "map": entry.map_name,
                "state_key": entry.state_key,
                "state": state_payload(entry.state),
                "split": dataset.state_split.get(entry.state_key, "test"),
                "selection_rule": "legacy: sort by state.key() string, first 10 per map",
            }
            for entry in old_entries
        ],
        "old_coverage": eval_state_coverage(old_entries),
        "candidate_counts": candidate_count_summary(candidate_counts),
        "aggregation_rule": (
            "mean within one true initial state over candidate seeds first, then mean/min/max "
            "over states; model seeds and candidate seeds are repeated measurements, not "
            "independent scenes, so no significance claim is made"
        ),
        "metrics_v2": metrics_v2,
        "metrics_old": metrics_old,
    }

    # ---- A3 ---------------------------------------------------------------
    decomposition = summarize_decomposition(new_audit["candidate_records"], per_run, eval_set="v2")
    traced = traced_cases(maps, models, per_run, config, eval_set="v2")

    # ---- A4 ---------------------------------------------------------------
    a4 = {
        "v2": recursive_vs_teacher(maps, models, new_entries, config, eval_set="v2"),
        "old": recursive_vs_teacher(maps, models, old_entries, config, eval_set="old"),
    }

    # ---- facts and reproduction checks ------------------------------------
    old_robot0 = {entry.state.robot0 for entry in old_entries}
    facts = {
        "1_old_states_robot0_single_cell": {
            "distinct_robot0_cells": sorted(old_robot0),
            "n_distinct_robot0_cells": len(old_robot0),
            "all_robot0_at_single_cell": len(old_robot0) == 1,
            "note": "legacy coordinate-string ordering put every old start at one cell",
        },
        "2_stored_0p53_metric": _stored_moved_metric(run_dir),
        "3_invalid_penalty_inside_candidate_mean_bias": _old_invalid_penalty_term(run_dir),
    }
    reproduction = _old_set_reproduction(run_dir, per_run)
    reconciliation = _aggregation_reconciliation(run_dir, per_run)
    eval_states_v2["aggregation_reconciliation"] = reconciliation

    # ---- hashes -----------------------------------------------------------
    end_hashes = {**dataset_hashes(run_dir), **checkpoint_hashes(run_dir, seeds)}
    changed = {name: {"start": start_hashes[name], "end": end_hashes[name]}
               for name in start_hashes if start_hashes[name] != end_hashes.get(name)}
    hash_check = {
        "start": start_hashes,
        "end": end_hashes,
        "unchanged": not changed,
        "changed": changed,
        "files": sorted(start_hashes),
    }

    return {
        "run_dir": str(run_dir),
        "config": config.as_dict(),
        "seeds": [int(seed) for seed in seeds],
        "start_hashes": start_hashes,
        "end_hashes": end_hashes,
        "hash_check": hash_check,
        "dataset_size": {"samples": len(dataset.samples), "split_states": len(dataset.state_split)},
        "group_stats": group_stats,
        "eval_states_v2": eval_states_v2,
        "per_run": per_run,
        "candidate_records": new_audit["candidate_records"],
        "traces": new_audit["traces"],
        "decomposition": decomposition,
        "traced_cases": traced,
        "recursive_vs_teacher": a4,
        "facts": facts,
        "old_set_reproduction": reproduction,
        "aggregation_reconciliation": reconciliation,
        "wall_clock": time.time() - started,
    }
