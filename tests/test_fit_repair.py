"""A-phase tests for fit_repair_v1: loss reduction, checkpoint snapshots, real resume.

The failures these guard against were all present in ``fit_capacity_v1``: a batch-summed loss,
checkpoints that all held the final parameters, and a resume file with no resume entry point.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d_learning.model import JointWindowMLP, ModelConfig, _tensors
from coimagine.continuous2d_learning.records import CollectionConfig, collect_episode
from coimagine.data_upgrade.training import per_sample_loss
from coimagine.fit_capacity.metrics import loss_components
from coimagine.fit_capacity.trainer import data_content_hash, run


def _records(windows: int = 12):
    config = default_config()
    records, _ = collect_episode(
        "S0_dev", "S0_dev/00", "train", config,
        CollectionConfig(episodes_per_scene=1, max_windows=windows, full_step_episodes=0),
    )
    return records


# ---------------------------------------------------------------------------
# A1: the batch loss is a per-sample mean
# ---------------------------------------------------------------------------


def test_batch_loss_and_gradients_match_the_reference_function():
    records = _records()
    tensors = _tensors(records)
    torch.manual_seed(0)
    model = JointWindowMLP(ModelConfig(hidden=32))

    outputs = model(tensors[0])
    components = loss_components(outputs, tuple(tensors[1:]))
    fixed = torch.stack(components, dim=0).sum(dim=0).mean()
    fixed.backward()
    fixed_grads = [parameter.grad.detach().clone() for parameter in model.parameters()]

    model.zero_grad()
    reference = per_sample_loss(model(tensors[0]), tuple(tensors[1:])).mean()
    reference.backward()
    reference_grads = [parameter.grad.detach().clone() for parameter in model.parameters()]

    assert float(fixed) == pytest.approx(float(reference), rel=1e-6)
    for first, second in zip(fixed_grads, reference_grads):
        assert torch.allclose(first, second, atol=1e-7)


def test_duplicating_the_batch_does_not_change_mean_loss_or_gradients():
    records = _records()
    tensors = _tensors(records)
    torch.manual_seed(0)
    model = JointWindowMLP(ModelConfig(hidden=32))
    outputs = model(tensors[0])
    single = torch.stack(loss_components(outputs, tuple(tensors[1:])), dim=0).sum(dim=0).mean()
    single.backward()
    single_grads = [parameter.grad.detach().clone() for parameter in model.parameters()]

    model.zero_grad()
    features = torch.cat([tensors[0], tensors[0]], dim=0)
    targets = tuple(torch.cat([t, t], dim=0) for t in tensors[1:])
    doubled = torch.stack(loss_components(model(features), targets), dim=0).sum(dim=0).mean()
    doubled.backward()
    doubled_grads = [parameter.grad.detach().clone() for parameter in model.parameters()]

    assert float(doubled) == pytest.approx(float(single), rel=1e-6)
    for first, second in zip(single_grads, doubled_grads):
        assert torch.allclose(first, second, atol=1e-7)


def test_legacy_batch_sum_would_have_been_batch_size_times_too_large():
    records = _records()
    tensors = _tensors(records)
    torch.manual_seed(0)
    model = JointWindowMLP(ModelConfig(hidden=32))
    components = loss_components(model(tensors[0]), tuple(tensors[1:]))
    legacy = float(torch.stack(components).sum())
    fixed = float(torch.stack(components, dim=0).sum(dim=0).mean())
    assert legacy == pytest.approx(fixed * len(records), rel=1e-5)


# ---------------------------------------------------------------------------
# A2: intermediate checkpoints are real snapshots
# ---------------------------------------------------------------------------


def test_intermediate_checkpoint_equals_the_model_at_that_update(tmp_path):
    records = _records()
    val = records[:4]
    full = run("Old+Random", 16, 0, records, val, val, tmp_path / "full",
               total_updates=20, eval_updates=(10, 20), checkpoint_updates=(10, 20), order_log_limit=0)
    early = run("Old+Random", 16, 0, records, val, val, tmp_path / "early",
                total_updates=10, eval_updates=(10,), checkpoint_updates=(10,), order_log_limit=0)

    stored = torch.load(full.checkpoints["fixed_10"]["path"], map_location="cpu")
    reference = torch.load(early.checkpoints["fixed_10"]["path"], map_location="cpu")
    for name, tensor in stored["state_dict"].items():
        assert torch.equal(tensor, reference["state_dict"][name]), (
            "update10 checkpoint must hold the parameters from update 10, not the final ones"
        )
    final = torch.load(full.checkpoints["fixed_20"]["path"], map_location="cpu")
    assert final["parameter_hash"] != stored["parameter_hash"]
    assert final["updates"] == 20


def test_early_checkpoint_metrics_match_the_inline_eval_log(tmp_path):
    from coimagine.fit_capacity.metrics import evaluate

    records = _records()
    val = records[:4]
    result = run("Old+Random", 16, 0, records, val, val, tmp_path / "run",
                 total_updates=20, eval_updates=(10, 20), checkpoint_updates=(10,), order_log_limit=0)
    inline = [json.loads(line) for line in (tmp_path / "run" / "eval_metrics.jsonl").open()]
    entry = next(row for row in inline if row["update"] == 10)
    blob = torch.load(result.checkpoints["fixed_10"]["path"], map_location="cpu")
    model = JointWindowMLP(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    replayed = evaluate(model, records)
    assert replayed["loss"]["total"] == pytest.approx(entry["metrics"]["train"]["loss"]["total"], rel=1e-5)
    assert replayed["position"]["object_mean"] == pytest.approx(
        entry["metrics"]["train"]["position"]["object_mean"], rel=1e-5
    )


# ---------------------------------------------------------------------------
# A3: resume actually continues
# ---------------------------------------------------------------------------


def _fingerprint(result):
    blob = torch.load(result.checkpoints["fixed_20"]["path"], map_location="cpu")
    return {
        "state": blob["state_dict"],
        "updates": result.updates,
        "seen_samples": result.seen_samples,
        "sample_order_hash": result.sample_order_hash,
        "best_updates": {k: v["updates"] for k, v in result.checkpoints.items() if k.startswith("best_")},
    }


def test_resume_reproduces_the_continuous_run(tmp_path):
    records = _records()
    val = records[:4]
    schedule = (5, 10, 15, 20)
    checkpoints = (10, 20)
    continuous = run("Old+Random", 16, 0, records, val, val, tmp_path / "continuous",
                     total_updates=20, eval_updates=schedule, checkpoint_updates=checkpoints, order_log_limit=0)
    part_dir = tmp_path / "part"
    run("Old+Random", 16, 0, records, val, val, part_dir,
        total_updates=10, eval_updates=schedule, checkpoint_updates=checkpoints, order_log_limit=0)
    resumed = run("Old+Random", 16, 0, records, val, val, part_dir,
                  total_updates=20, eval_updates=schedule, checkpoint_updates=checkpoints,
                  resume=True, order_log_limit=0)

    assert resumed.resumed_from == 10
    first = _fingerprint(continuous)
    second = _fingerprint(resumed)
    assert first["updates"] == second["updates"] == 20
    assert first["seen_samples"] == second["seen_samples"]
    assert first["sample_order_hash"] == second["sample_order_hash"]
    assert first["best_updates"] == second["best_updates"]
    for name, tensor in first["state"].items():
        assert torch.equal(tensor, second["state"][name]), "resumed weights must match the continuous run"


def test_resume_refuses_mismatched_data(tmp_path):
    records = _records()
    val = records[:4]
    run("Old+Random", 16, 0, records, val, val, tmp_path / "run",
        total_updates=5, eval_updates=(5,), checkpoint_updates=(5,), order_log_limit=0)
    other = records[:6]
    with pytest.raises(ValueError):
        run("Old+Random", 16, 0, other, val, val, tmp_path / "run",
            total_updates=10, eval_updates=(5, 10), checkpoint_updates=(5,), resume=True, order_log_limit=0)


def test_data_content_hash_changes_with_payload():
    records = _records()
    first = data_content_hash(records)
    second = data_content_hash(records)
    assert first == second
    mutated = [dict(record) for record in records]
    mutated[0]["duration"] = float(mutated[0]["duration"]) + 1e-6
    assert data_content_hash(mutated) != first


# ---------------------------------------------------------------------------
# A4: paired comparison on the common feasible subset
# ---------------------------------------------------------------------------


def _row(start_id, candidate_seed, *, feasible=True, chosen=0.0, best=1.0, uniform=0.5,
         misrejection=0.0, ranking=0.0, execution_regret=0.0, rejection=0.0, chosen_bias=0.0,
         excess=0.0, candidate_best=1.0, regret=0.0, mean_bias=0.0):
    return {
        "start_id": start_id, "candidate_seed": candidate_seed,
        "no_feasible_candidate": not feasible,
        "rejection_rate": rejection if feasible else 1.0,
        "chosen_true_return": chosen, "uniform_candidate_true_return": uniform,
        "candidate_best_true_return": candidate_best, "selected_bias": chosen_bias,
        "selection_excess": excess, "unfiltered_regret": regret, "candidate_mean_bias": mean_bias,
        "feasible_chosen_true_return": chosen if feasible else None,
        "misrejection_loss": misrejection if feasible else None,
        "ranking_loss": ranking if feasible else None,
        "execution_regret": execution_regret if feasible else None,
    }


def test_common_subset_drops_infeasible_rows_and_reports_them():
    from coimagine.fit_capacity.paired import compare

    rows_a = [_row("s0", 0, chosen=1.0), _row("s0", 1, feasible=False), _row("s1", 0, chosen=2.0)]
    rows_b = [_row("s0", 0, chosen=3.0), _row("s0", 1, chosen=4.0), _row("s1", 0, chosen=5.0)]
    result = compare(rows_a, rows_b, label_a="a", label_b="b")
    assert result["common"]["keys_both"] == 3
    assert result["common"]["common_feasible"] == 2
    assert result["common"]["dropped_infeasible_a"] == 1
    assert result["delta_common"]["feasible_chosen_true_return"] == pytest.approx(2.5)
    # the full-sample summaries keep the dropped row and show the a-side no-solution rate
    assert result["full"]["a"]["rows_without_feasible"] == 1
    assert result["full"]["b"]["rows_without_feasible"] == 0


def test_common_subset_recomputes_references_on_the_same_rows():
    from coimagine.fit_capacity.paired import summarize_rows

    rows = [_row("s0", 0, uniform=1.0, candidate_best=9.0), _row("s0", 1, uniform=3.0, candidate_best=5.0)]
    summary = summarize_rows(rows)
    # per start: uniform mean 2.0, candidate best 7.0 (the max of the per-start max)
    assert summary["uniform_candidate_true_return"] == pytest.approx(2.0)
    assert summary["candidate_best_true_return"] == pytest.approx(7.0)


def test_all_rows_infeasible_keeps_null_not_zero():
    from coimagine.fit_capacity.paired import summarize_rows

    summary = summarize_rows([_row("s0", 0, feasible=False), _row("s0", 1, feasible=False)])
    assert summary["feasible_chosen_true_return"] is None
    assert summary["misrejection_loss"] is None
    assert summary["execution_regret"] is None
    assert summary["starts_without_feasible"] == 1
    assert summary["rows_without_feasible"] == 2
