"""Phase A/B trainer for fit_repair_v1.

Fixes over the ``fit_capacity_v1`` trainer:

* **A1** the batch loss is the mean of the per-sample four-term sum, not a sum over the batch
  (the old ``torch.stack(components).sum()`` was batch-size times too large);
* **A2** every fixed-progress checkpoint is a snapshot taken *at* its update; parameter-content
  hashes are recorded separately from file hashes;
* **A3** the whole run state is written atomically at every evaluation point, and ``run`` can
  resume from it, so an interrupted run continues instead of restarting.

One run = one (data group, width, seed) cell. The minibatch order comes from a generator seeded
only by the run seed, so both widths and both data groups with the same seed iterate the identical
sample-index stream; the stream is hashed incrementally over the whole run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from ..continuous2d_learning.model import JointWindowMLP, ModelConfig, _tensors
from .metrics import evaluate, loss_components

#: The frozen evaluation schedule: {0, 100} plus every 250 from 250 to 6000.
EVAL_UPDATES: Tuple[int, ...] = tuple(sorted({0, 100} | set(range(250, 6001, 250))))
FIXED_CHECKPOINTS = (1000, 3000, 6000)
TOTAL_UPDATES = 6000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def parameter_hash(model: nn.Module) -> str:
    return state_dict_hash(model.state_dict())


def state_dict_hash(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:32]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_content_hash(records: Sequence[Dict[str, object]]) -> str:
    """Full SHA256 over the raw record payloads, not just scene/episode/index fields."""

    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@dataclass
class RunResult:
    group: str
    width: int
    seed: int
    updates: int
    seen_samples: int
    wall_clock: float
    initial_parameter_hash: str
    schedule: List[Dict[str, object]] = field(default_factory=list)
    checkpoints: Dict[str, Dict[str, object]] = field(default_factory=dict)
    stopped_reason: str = "completed"
    sample_order: List[List[int]] = field(default_factory=list)
    sample_order_hash: str = ""
    resumed_from: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "group": self.group, "width": self.width, "seed": self.seed,
            "updates": self.updates, "seen_samples": self.seen_samples,
            "wall_clock": self.wall_clock, "initial_parameter_hash": self.initial_parameter_hash,
            "schedule": self.schedule, "checkpoints": self.checkpoints,
            "stopped_reason": self.stopped_reason, "sample_order_hash": self.sample_order_hash,
            "resumed_from": self.resumed_from,
        }


def _evaluate_all(model: JointWindowMLP, datasets: Dict[str, Sequence[Dict[str, object]]]):
    return {name: evaluate(model, records) for name, records in datasets.items()}


def run(
    group: str,
    width: int,
    seed: int,
    train_records: Sequence[Dict[str, object]],
    val_original: Sequence[Dict[str, object]],
    val_branch: Sequence[Dict[str, object]],
    out_dir: Path,
    *,
    total_updates: int = TOTAL_UPDATES,
    device: str = "cpu",
    eval_updates: Optional[Sequence[int]] = None,
    checkpoint_updates: Sequence[int] = FIXED_CHECKPOINTS,
    resume: bool = False,
    order_log_limit: int = 64,
    data_hashes: Optional[Dict[str, str]] = None,
) -> RunResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {"train": train_records, "original_val": val_original, "branch_val": val_branch}
    tensors = _tensors(train_records)
    count = len(train_records)
    data_hash = data_content_hash(train_records)
    schedule_points = tuple(sorted({v for v in (eval_updates if eval_updates is not None else EVAL_UPDATES) if v <= total_updates}))
    checkpoint_points = tuple(sorted({v for v in checkpoint_updates if v <= total_updates}))

    torch.manual_seed(seed)
    model = JointWindowMLP(ModelConfig(hidden=width)).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    data_generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(count, generator=data_generator)

    result = RunResult(group=group, width=width, seed=seed, updates=0, seen_samples=0,
                       wall_clock=0.0, initial_parameter_hash=parameter_hash(model))
    epoch = 0
    cursor = 0
    snapshots: Dict[int, Dict[str, torch.Tensor]] = {}
    best_state: Dict[str, Dict[str, object]] = {}
    best_copy: Dict[str, Dict[str, torch.Tensor]] = {}
    resume_path = out_dir / "resume_state.pt"

    if resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"no resume state at {resume_path}")
        blob = torch.load(resume_path, map_location="cpu")
        if blob["data_hash"] != data_hash:
            raise ValueError("resume refused: training data content hash differs")
        if blob["config"] != ModelConfig(hidden=width).as_dict():
            raise ValueError("resume refused: model config differs")
        model.load_state_dict(blob["model"])
        optimiser.load_state_dict(blob["optimizer"])
        data_generator.set_state(blob["data_generator"])
        order = blob["order"]
        cursor = int(blob["cursor"])
        epoch = int(blob["epoch"])
        result.updates = int(blob["updates"])
        result.seen_samples = int(blob["seen_samples"])
        result.schedule = list(blob.get("schedule", []))
        result.sample_order = list(blob.get("sample_order", []))
        best_state = blob.get("best_state", {})
        best_copy = blob.get("best_copy", {})
        result.resumed_from = result.updates

    started = time.time()
    batch_log = (out_dir / "batch_loss.jsonl").open("a")
    eval_log = (out_dir / "eval_metrics.jsonl").open("a")
    # The full sample-index stream is appended to disk, so the rolling hash can be rebuilt
    # exactly after a resume instead of hashing an un-serialisable digest state.
    order_file = (out_dir / "sample_order.bin").open("ab")

    def _score(row: Dict[str, object], kind: str) -> float:
        return float(row["mixed_val_score"] if kind == "best_mixed_val" else row["original_val_total_loss"])

    def _maybe_best(row: Dict[str, object]) -> None:
        if int(row["update"]) <= 0:
            return
        for kind in ("best_mixed_val", "best_original_val"):
            current = best_state.get(kind)
            if current is None or _score(row, kind) < _score(current, kind) - 1e-12:
                best_state[kind] = row
                best_copy[kind] = copy.deepcopy(model.state_dict())

    def save_resume() -> None:
        _atomic_torch_save(
            resume_path,
            {
                "model": model.state_dict(), "optimizer": optimiser.state_dict(),
                "data_generator": data_generator.get_state(), "order": order, "cursor": cursor,
                "epoch": epoch, "updates": result.updates, "seen_samples": result.seen_samples,
                "schedule": result.schedule, "sample_order": result.sample_order,
                "best_state": best_state, "best_copy": best_copy,
                "group": group, "width": width, "seed": seed,
                "config": ModelConfig(hidden=width).as_dict(), "data_hash": data_hash,
                "data_hashes": data_hashes or {}, "eval_updates": list(schedule_points),
            },
        )

    def evaluate_now(update: int) -> None:
        metrics = _evaluate_all(model, datasets)
        row = {
            "update": update,
            "train_total_loss": metrics["train"]["loss"]["total"],
            "original_val_total_loss": metrics["original_val"]["loss"]["total"],
            "branch_val_total_loss": metrics["branch_val"]["loss"]["total"],
            "mixed_val_score": 0.5 * metrics["original_val"]["loss"]["total"]
            + 0.5 * metrics["branch_val"]["loss"]["total"],
            "metrics": metrics,
        }
        result.schedule.append(
            {
                "update": update,
                "train_total_loss": row["train_total_loss"],
                "original_val_total_loss": row["original_val_total_loss"],
                "branch_val_total_loss": row["branch_val_total_loss"],
                "mixed_val_score": row["mixed_val_score"],
                "train_object_position": metrics["train"]["position"]["object_mean"],
                "train_invalid_rate": metrics["train"]["legality"]["invalid_rate"],
                "original_val_object_position": metrics["original_val"]["position"]["object_mean"],
                "original_val_invalid_rate": metrics["original_val"]["legality"]["invalid_rate"],
                "branch_val_object_position": metrics["branch_val"]["position"]["object_mean"],
                "branch_val_invalid_rate": metrics["branch_val"]["legality"]["invalid_rate"],
            }
        )
        eval_log.write(json.dumps(row) + "\n")
        eval_log.flush()
        _maybe_best(row)
        save_resume()

    try:
        if result.updates == 0 and 0 in schedule_points:
            evaluate_now(0)
        while result.updates < total_updates:
            if cursor >= count:
                epoch += 1
                cursor = 0
                order = torch.randperm(count, generator=data_generator)
            index = order[cursor : cursor + BATCH_SIZE]
            if len(result.sample_order) < order_log_limit:
                result.sample_order.append([int(v) for v in index])
            order_file.write(index.numpy().astype("<i8").tobytes())
            cursor += BATCH_SIZE
            outputs = model(tensors[0][index])
            components = loss_components(outputs, tuple(t[index] for t in tensors[1:]))
            per_sample = torch.stack(components, dim=0).sum(dim=0)  # A1 fix
            loss = per_sample.mean()
            if not torch.isfinite(loss):
                result.stopped_reason = f"non_finite_loss_at_update_{result.updates}"
                batch_log.write(json.dumps({"update": result.updates, "loss": None, "status": "non_finite"}) + "\n")
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
            if result.updates in checkpoint_points:
                snapshots[result.updates] = copy.deepcopy(model.state_dict())
            if result.updates in schedule_points:
                evaluate_now(result.updates)
    finally:
        batch_log.close()
        eval_log.close()
        order_file.close()

    result.wall_clock = time.time() - started
    result.sample_order_hash = file_sha256(out_dir / "sample_order.bin")

    def _write(path: Path, state: Dict[str, torch.Tensor], kind: str, update: int, row: Optional[Dict[str, object]]) -> None:
        payload = {
            "state_dict": state, "config": ModelConfig(hidden=width).as_dict(), "seed": seed,
            "group": group, "updates": update, "kind": kind,
            "parameter_hash": state_dict_hash(state), "data_hash": data_hash,
            "data_hashes": data_hashes or {},
            "mixed_val_score": None if row is None else row["mixed_val_score"],
            "original_val_total_loss": None if row is None else row["original_val_total_loss"],
            "branch_val_total_loss": None if row is None else row["branch_val_total_loss"],
        }
        _atomic_torch_save(path, payload)
        result.checkpoints[kind] = {
            "path": str(path), "file_sha256": file_sha256(path),
            "parameter_hash": payload["parameter_hash"], "updates": update,
            "mixed_val_score": payload["mixed_val_score"],
            "original_val_total_loss": payload["original_val_total_loss"],
        }

    for update in checkpoint_points:
        state = snapshots.get(update, copy.deepcopy(model.state_dict()))
        row = next((entry for entry in result.schedule if entry["update"] == update), None)
        _write(out_dir / f"update{update}.pt", state, f"fixed_{update}", update, row)
    for kind in ("best_mixed_val", "best_original_val"):
        row = best_state.get(kind)
        if row is None:
            continue
        state = best_copy.get(kind, copy.deepcopy(model.state_dict()))
        _write(out_dir / f"{kind}.pt", state, kind, int(row["update"]), row)

    save_resume()
    (out_dir / "run_summary.json").write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return result
