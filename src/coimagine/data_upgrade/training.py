"""Phase D1: the three same-structure data groups (Old / Old+Random / Old+Branch).

Only the training data differs. Network, loss, optimiser, batch size, epoch budget and the
checkpoint rule are identical across groups; the same seed uses the same initialisation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..continuous2d_learning.model import (
    JointWindowMLP,
    ModelConfig,
    _tensors,
    state_residual,
)
from ..continuous2d_learning.contract import TERMINATION_CLASSES

GROUP_OLD = "Old"
GROUP_RANDOM = "Old+Random"
GROUP_BRANCH = "Old+Branch"


def per_sample_loss(outputs: Dict[str, torch.Tensor], targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    state_target, hold_target, duration_target, termination_target = targets
    state = (state_residual(outputs["state"], state_target) ** 2).mean(dim=1)
    hold = (torch.sigmoid(outputs["hold"]) - hold_target) ** 2
    duration = (torch.sigmoid(outputs["duration"]) - duration_target) ** 2
    termination = nn.functional.cross_entropy(
        outputs["termination"], termination_target, reduction="none"
    )
    return state + hold.squeeze(1) + duration.squeeze(1) + termination


def mean_loss(model: JointWindowMLP, records: Sequence[Dict[str, object]], batch_size: int = 512) -> float:
    if not records:
        return float("nan")
    tensors = _tensors(records)
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(tensors[0]), batch_size):
            index = slice(start, start + batch_size)
            values = per_sample_loss(model(tensors[0][index]), tuple(t[index] for t in tensors[1:]))
            total += float(values.sum().item())
            count += int(values.numel())
    return total / max(1, count)


@dataclass
class GroupTraining:
    group: str
    seed: int
    train_samples: int
    val_original_samples: int
    val_branch_samples: int
    epochs: List[Dict[str, float]]
    best_epoch: int
    best_val_score: float
    updates: int
    wall_clock: float
    parameter_count: int
    checkpoint_sha256: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "group": self.group, "seed": self.seed, "train_samples": self.train_samples,
            "val_original_samples": self.val_original_samples,
            "val_branch_samples": self.val_branch_samples,
            "epochs": self.epochs, "best_epoch": self.best_epoch,
            "best_val_score": self.best_val_score, "updates": self.updates,
            "wall_clock": self.wall_clock, "parameter_count": self.parameter_count,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def train_group(
    group: str,
    train_records: Sequence[Dict[str, object]],
    val_original: Sequence[Dict[str, object]],
    val_branch: Sequence[Dict[str, object]],
    *,
    seed: int,
    config: Optional[ModelConfig] = None,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[JointWindowMLP, GroupTraining]:
    """One formal training run. ``val_score`` weights are frozen in the contract."""

    config = config or ModelConfig()
    torch.manual_seed(seed)
    model = JointWindowMLP(config)
    train = _tensors(train_records)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_score = float("inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []
    updates = 0
    started = time.time()

    for epoch in range(config.max_epochs):
        model.train()
        order = torch.randperm(len(train[0]), generator=generator)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            values = per_sample_loss(model(train[0][index]), tuple(t[index] for t in train[1:]))
            loss = values.mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            batches += 1
            updates += 1
        model.eval()
        original_loss = mean_loss(model, val_original)
        branch_loss = mean_loss(model, val_branch) if val_branch else float("nan")
        if val_branch:
            score = 0.5 * original_loss + 0.5 * branch_loss
        else:
            score = original_loss
        history.append(
            {
                "epoch": epoch, "train_loss": epoch_loss / max(1, batches),
                "val_original_loss": original_loss, "val_branch_loss": branch_loss,
                "val_score": score,
            }
        )
        if score < best_score - 1e-9:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    training = GroupTraining(
        group=group, seed=seed, train_samples=len(train_records),
        val_original_samples=len(val_original), val_branch_samples=len(val_branch),
        epochs=history, best_epoch=best_epoch, best_val_score=best_score, updates=updates,
        wall_clock=time.time() - started,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        checkpoint_sha256="",
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config.as_dict(), "seed": seed,
                    "group": group}, checkpoint_path)
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        training.checkpoint_sha256 = digest.hexdigest()
    return model, training
