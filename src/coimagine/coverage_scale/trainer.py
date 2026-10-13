"""The coverage-scale trainer: incremental data, two widths, a frozen selection rule.

It is the ``fit_repair_v1`` trainer (batch-mean reduction, true intermediate snapshots, real
resume, a shared sample-index stream) with three changes the task book requires:

* periodic evaluation reads only the fixed ``train_monitor`` subset and the new validation
  set; the full training set is decoded once, at the end, for the two selected points;
* the main selection point is ``best_new_val_total``: the minimum of the *root -> cell ->
  weight* total loss on the new validation set (task book C2). Every evaluation point,
  including update 0, is eligible; nothing is re-selected by test, candidate return or
  illegality;
* the update budget grows with the data -- ``max(6000, 60 * ceil(N/128))``, capped at 60000 --
  so a larger tier is not penalised merely for having seen the data fewer times.
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..continuous2d_learning.model import ModelConfig, JointWindowMLP, _tensors
from ..fit_capacity.metrics import loss_components
from ..fit_capacity.trainer import (
    BATCH_SIZE,
    LEARNING_RATE,
    data_content_hash,
    file_sha256,
    state_dict_hash,
)
from .evaluation import weighted_summary

SELECTION_KIND = "best_new_val_total"
FIXED_BUDGET_KIND = "final"
MONITOR_LIMIT = 4096


def update_budget(windows: int) -> int:
    return min(60000, max(6000, 60 * math.ceil(windows / BATCH_SIZE)))


def eval_schedule(total_updates: int) -> Tuple[int, ...]:
    """Every ``max(250, ceil(Umax/20/250)*250)`` updates, plus update 100 and update 0."""

    step = max(250, math.ceil((total_updates / 20) / 250) * 250)
    points = {0, 100, total_updates}
    points.update(range(step, total_updates + 1, step))
    return tuple(sorted(value for value in points if value <= total_updates))


def _atomic_torch_save(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@dataclass
class CellResult:
    tier: str
    width: int
    seed: int
    windows: int
    updates: int
    max_updates: int
    seen_samples: int
    epochs_seen: float
    wall_clock: float
    initial_parameter_hash: str
    sample_order_hash: str = ""
    schedule: List[Dict[str, object]] = field(default_factory=list)
    checkpoints: Dict[str, Dict[str, object]] = field(default_factory=dict)
    stopped_reason: str = "completed"
    resumed_from: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "tier": self.tier, "width": self.width, "seed": self.seed,
            "windows": self.windows, "updates": self.updates, "max_updates": self.max_updates,
            "seen_samples": self.seen_samples, "epochs_seen": self.epochs_seen,
            "wall_clock": self.wall_clock, "initial_parameter_hash": self.initial_parameter_hash,
            "sample_order_hash": self.sample_order_hash, "schedule": self.schedule,
            "checkpoints": self.checkpoints, "stopped_reason": self.stopped_reason,
            "resumed_from": self.resumed_from,
        }


def run_cell(
    tier: str,
    width: int,
    seed: int,
    train_records: Sequence[Dict[str, object]],
    monitor_records: Sequence[Dict[str, object]],
    val_records: Sequence[Dict[str, object]],
    out_dir: Path,
    *,
    max_updates: Optional[int] = None,
    resume: bool = False,
    device: str = "cpu",
    data_hashes: Optional[Dict[str, str]] = None,
) -> CellResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = len(train_records)
    if count < BATCH_SIZE:
        raise ValueError(f"tier has only {count} windows; the frozen batch is {BATCH_SIZE}")
    total_updates = max_updates or update_budget(count)
    schedule_points = eval_schedule(total_updates)
    data_hash = data_content_hash(train_records)

    monitor_tensors = _tensors(monitor_records)
    val_tensors = _tensors(val_records)

    torch.manual_seed(seed)
    model = JointWindowMLP(ModelConfig(hidden=width)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_tensors = _tensors(train_records)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(count, generator=generator)

    result = CellResult(
        tier=tier, width=width, seed=seed, windows=count, updates=0,
        max_updates=total_updates, seen_samples=0, epochs_seen=0.0, wall_clock=0.0,
        initial_parameter_hash=state_dict_hash(model.state_dict()),
    )
    cursor = 0
    epoch = 0
    best_row: Optional[Dict[str, object]] = None
    best_copy: Optional[Dict[str, torch.Tensor]] = None
    final_copy: Optional[Dict[str, torch.Tensor]] = None
    resume_path = out_dir / "resume_state.pt"

    if resume and resume_path.exists():
        blob = torch.load(resume_path, map_location="cpu")
        if blob["data_hash"] != data_hash:
            raise ValueError("resume refused: training data content hash differs")
        if blob["config"] != ModelConfig(hidden=width).as_dict():
            raise ValueError("resume refused: model config differs")
        model.load_state_dict(blob["model"])
        optimiser.load_state_dict(blob["optimizer"])
        generator.set_state(blob["data_generator"])
        order = blob["order"]
        cursor = int(blob["cursor"])
        epoch = int(blob["epoch"])
        result.updates = int(blob["updates"])
        result.seen_samples = int(blob["seen_samples"])
        result.schedule = list(blob.get("schedule", []))
        best_row = blob.get("best_row")
        best_copy = blob.get("best_copy")
        final_copy = blob.get("final_copy")
        result.resumed_from = result.updates

    started = time.time()
    batch_log = (out_dir / "batch_loss.jsonl").open("a")
    eval_log = (out_dir / "eval_metrics.jsonl").open("a")
    order_file = (out_dir / "sample_order.bin").open("ab")

    def save_resume() -> None:
        _atomic_torch_save(
            resume_path,
            {
                "model": model.state_dict(), "optimizer": optimiser.state_dict(),
                "data_generator": generator.get_state(), "order": order, "cursor": cursor,
                "epoch": epoch, "updates": result.updates, "seen_samples": result.seen_samples,
                "schedule": result.schedule, "best_row": best_row, "best_copy": best_copy,
                "final_copy": final_copy, "config": ModelConfig(hidden=width).as_dict(),
                "data_hash": data_hash, "tier": tier, "width": width, "seed": seed,
                "max_updates": total_updates, "data_hashes": data_hashes or {},
            },
        )

    def evaluate_now(update: int) -> None:
        nonlocal best_row, best_copy
        monitor = weighted_summary(model, monitor_records, tensors=monitor_tensors)
        val = weighted_summary(model, val_records, tensors=val_tensors)
        row = {
            "update": update,
            "new_val_total_loss": val["weighted"]["total_loss"],
            "new_val_total_loss_unweighted": val["unweighted"]["total_loss"],
            "new_val_position_mean": val["weighted"]["position_mean"],
            "new_val_invalid_rate": val["legality"]["invalid_rate_weighted"],
            "train_monitor_total_loss": monitor["weighted"]["total_loss"],
            "train_monitor_position_mean": monitor["weighted"]["position_mean"],
            "train_monitor_invalid_rate": monitor["legality"]["invalid_rate_weighted"],
        }
        result.schedule.append(row)
        eval_log.write(json.dumps({"update": update, "monitor": monitor, "new_val": val}) + "\n")
        eval_log.flush()
        if best_row is None or float(row["new_val_total_loss"]) < float(best_row["new_val_total_loss"]) - 1e-12:
            best_row = dict(row)
            best_copy = copy.deepcopy(model.state_dict())
        save_resume()

    try:
        if 0 in schedule_points:
            evaluate_now(0)
        while result.updates < total_updates:
            if cursor + BATCH_SIZE > count:
                epoch += 1
                cursor = 0
                order = torch.randperm(count, generator=generator)
            index = order[cursor : cursor + BATCH_SIZE]
            order_file.write(index.numpy().astype("<i8").tobytes())
            cursor += BATCH_SIZE
            outputs = model(train_tensors[0][index])
            components = loss_components(outputs, tuple(t[index] for t in train_tensors[1:]))
            loss = torch.stack(components, dim=0).sum(dim=0).mean()
            if not torch.isfinite(loss):
                result.stopped_reason = f"non_finite_loss_at_update_{result.updates}"
                batch_log.write(json.dumps({"update": result.updates, "status": "non_finite"}) + "\n")
                break
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            result.updates += 1
            result.seen_samples += int(index.numel())
            batch_log.write(
                json.dumps(
                    {
                        "update": result.updates, "epoch": epoch, "batch_size": int(index.numel()),
                        "state": float(components[0].mean().detach()),
                        "hold": float(components[1].mean().detach()),
                        "duration": float(components[2].mean().detach()),
                        "termination": float(components[3].mean().detach()),
                        "total": float(loss.detach()),
                    }
                )
                + "\n"
            )
            if result.updates % 100 == 0:
                batch_log.flush()
            if result.updates in schedule_points:
                evaluate_now(result.updates)
            if result.updates == total_updates:
                final_copy = copy.deepcopy(model.state_dict())
    finally:
        batch_log.close()
        eval_log.close()
        order_file.close()

    result.wall_clock = time.time() - started
    result.seen_samples = int(result.updates) * BATCH_SIZE
    result.epochs_seen = result.seen_samples / max(1, count)
    result.sample_order_hash = file_sha256(out_dir / "sample_order.bin")

    def _write(kind: str, state: Dict[str, torch.Tensor], update: int, row: Optional[Dict[str, object]]) -> None:
        payload = {
            "state_dict": state, "config": ModelConfig(hidden=width).as_dict(), "seed": seed,
            "tier": tier, "width": width, "kind": kind, "updates": update,
            "parameter_hash": state_dict_hash(state), "data_hash": data_hash,
            "data_hashes": data_hashes or {},
            "new_val_total_loss": None if row is None else row["new_val_total_loss"],
            "new_val_position_mean": None if row is None else row["new_val_position_mean"],
            "new_val_invalid_rate": None if row is None else row["new_val_invalid_rate"],
            "selection_rule": SELECTION_KIND,
        }
        path = out_dir / f"{kind}.pt"
        _atomic_torch_save(path, payload)
        result.checkpoints[kind] = {
            "path": str(path), "file_sha256": file_sha256(path),
            "parameter_hash": payload["parameter_hash"], "updates": update,
            "new_val_total_loss": payload["new_val_total_loss"],
            "new_val_position_mean": payload["new_val_position_mean"],
            "new_val_invalid_rate": payload["new_val_invalid_rate"],
        }

    final_state = final_copy if final_copy is not None else copy.deepcopy(model.state_dict())
    if best_copy is not None and best_row is not None:
        _write(SELECTION_KIND, best_copy, int(best_row["update"]), best_row)
    _write(FIXED_BUDGET_KIND, final_state, result.updates, None)

    # Full-training-set decode happens exactly once, for the two selected points (task book C2).
    selection_eval: Dict[str, object] = {}
    for kind, state, row in (
        (SELECTION_KIND, best_copy, best_row),
        (FIXED_BUDGET_KIND, final_state, None),
    ):
        if state is None:
            continue
        probe = JointWindowMLP(ModelConfig(hidden=width))
        probe.load_state_dict(state)
        new_val = weighted_summary(probe, val_records, tensors=val_tensors)
        train_full = weighted_summary(probe, train_records, tensors=train_tensors)
        selection_eval[kind] = {
            "kind": kind,
            "update": None if row is None else int(row["update"]),
            "new_val": new_val,
            "train_full": {
                "n": train_full["n"],
                "weighted_total_loss": train_full["weighted"]["total_loss"],
                "weighted_position_mean": train_full["weighted"]["position_mean"],
                "weighted_invalid_rate": train_full["legality"]["invalid_rate_weighted"],
                "success_recall": train_full["success"]["recall"],
                "true_success_windows": train_full["success"]["true_success_windows"],
            },
        }
    (out_dir / "selection_eval.json").write_text(json.dumps(selection_eval, indent=1, sort_keys=True))
    (out_dir / "full_train_eval.json").write_text(
        json.dumps({kind: {"train_full": value["train_full"]} for kind, value in selection_eval.items()},
                   indent=2, sort_keys=True)
    )

    save_resume()
    (out_dir / "run_summary.json").write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return result
