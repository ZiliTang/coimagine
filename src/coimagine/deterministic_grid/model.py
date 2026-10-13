"""Deterministic point-prediction baseline (CONTRACT B2).

A plain MLP with two hidden ReLU layers of width 256 and three classification heads:
object anchor (25 legal in-bounds anchors), robot A cell (36), robot B cell (36). Softmax
cross-entropy per head, equal weight; argmax gives the predicted next state. Rollouts feed
the decoded **discrete** state back in; walls and the goal are static inputs and are never
predicted. A decoded state that overlaps or hits a wall is an explicit invalid prediction,
not a repaired one.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .contract import (
    ANCHOR_INDEX,
    ANCHORS,
    CELL_INDEX,
    CELLS,
    GRID,
    GridMap,
    GridState,
    N_JOINT_ACTIONS,
    decode_joint,
    state_is_valid,
)
from .data import Dataset, TransitionSample
from .scoring import ScoreComponents, rollout, score_trajectory

INPUT_DIM = 2 * (GRID * GRID) + len(ANCHORS) + (GRID * GRID) + (GRID * GRID) + N_JOINT_ACTIONS
OBJECT_DIM = len(ANCHORS)
ROBOT_DIM = GRID * GRID
HEAD_WEIGHTS = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 256
    layers: int = 2
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 5

    def as_dict(self) -> Dict[str, object]:
        return {
            "hidden": self.hidden, "layers": self.layers, "lr": self.lr,
            "batch_size": self.batch_size, "max_epochs": self.max_epochs, "patience": self.patience,
        }


def encode_input(grid_map: GridMap, state: GridState, joint_action: int) -> List[float]:
    walls = grid_map.wall_vector()
    goal = grid_map.goal_vector()
    features = [0.0] * INPUT_DIM
    offset = 0
    features[offset + CELL_INDEX[state.robot0]] = 1.0
    offset += GRID * GRID
    features[offset + CELL_INDEX[state.robot1]] = 1.0
    offset += GRID * GRID
    features[offset + ANCHOR_INDEX[state.object_anchor]] = 1.0
    offset += OBJECT_DIM
    features[offset : offset + GRID * GRID] = list(walls)
    offset += GRID * GRID
    features[offset : offset + GRID * GRID] = list(goal)
    offset += GRID * GRID
    features[offset + joint_action] = 1.0
    return features


class PointModel(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        layers: List[nn.Module] = []
        width = INPUT_DIM
        for _ in range(self.config.layers):
            layers += [nn.Linear(width, self.config.hidden), nn.ReLU()]
            width = self.config.hidden
        self.trunk = nn.Sequential(*layers)
        self.object_head = nn.Linear(width, OBJECT_DIM)
        self.robot0_head = nn.Linear(width, ROBOT_DIM)
        self.robot1_head = nn.Linear(width, ROBOT_DIM)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.object_head(h), self.robot0_head(h), self.robot1_head(h)


@dataclass
class TrainingLog:
    seed: int
    config: Dict[str, object] = field(default_factory=dict)
    epochs: List[Dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    checkpoint_sha256: str = ""
    wall_clock: float = 0.0
    parameter_count: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed, "config": self.config, "epochs": self.epochs,
            "best_epoch": self.best_epoch, "best_val_loss": self.best_val_loss,
            "checkpoint_sha256": self.checkpoint_sha256, "wall_clock": self.wall_clock,
            "parameter_count": self.parameter_count,
        }


def _split_samples(dataset: Dataset, split: str) -> List[TransitionSample]:
    return [s for s in dataset.samples if dataset.split_of(s.map_name, s.state) == split]


def _tensors(samples: Sequence[TransitionSample], maps: Dict[str, GridMap]):
    features = []
    object_targets = []
    robot0_targets = []
    robot1_targets = []
    for sample in samples:
        grid_map = maps[sample.map_name]
        features.append(encode_input(grid_map, sample.state, sample.joint_action))
        object_targets.append(ANCHOR_INDEX[sample.next_state.object_anchor])
        robot0_targets.append(CELL_INDEX[sample.next_state.robot0])
        robot1_targets.append(CELL_INDEX[sample.next_state.robot1])
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(object_targets, dtype=torch.long),
        torch.tensor(robot0_targets, dtype=torch.long),
        torch.tensor(robot1_targets, dtype=torch.long),
    )


def train_model(
    dataset: Dataset,
    maps: Dict[str, GridMap],
    *,
    seed: int,
    config: Optional[ModelConfig] = None,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[PointModel, TrainingLog]:
    config = config or ModelConfig()
    torch.manual_seed(seed)
    model = PointModel(config)
    log = TrainingLog(seed=seed, config=config.as_dict())
    log.parameter_count = sum(p.numel() for p in model.parameters())

    train_samples = _split_samples(dataset, "train")
    val_samples = _split_samples(dataset, "val")
    train = _tensors(train_samples, maps)
    val = _tensors(val_samples, maps)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()
    rng = torch.Generator().manual_seed(seed)
    started = time.time()
    best_state = None

    for epoch in range(config.max_epochs):
        model.train()
        order = torch.randperm(len(train[0]), generator=rng)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            x = train[0][index]
            targets = (train[1][index], train[2][index], train[3][index])
            outputs = model(x)
            loss = sum(w * loss_fn(output, target) for w, output, target in zip(HEAD_WEIGHTS, outputs, targets))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            batches += 1
        model.eval()
        with torch.no_grad():
            val_outputs = model(val[0])
            val_loss = float(
                sum(w * loss_fn(output, target) for w, output, target in zip(HEAD_WEIGHTS, val_outputs, (val[1], val[2], val[3]))).item()
            )
            train_loss = epoch_loss / max(1, batches)
        log.epochs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < log.best_val_loss - 1e-9:
            log.best_val_loss = val_loss
            log.best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif epoch - log.best_epoch >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    log.wall_clock = time.time() - started
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": config.as_dict(), "seed": seed}, checkpoint_path)
        log.checkpoint_sha256 = _sha256(checkpoint_path)
    return model, log


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path, config: Optional[ModelConfig] = None) -> PointModel:
    blob = torch.load(path, map_location="cpu")
    model = PointModel(ModelConfig(**blob["config"]) if config is None else config)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def decode(model: PointModel, grid_map: GridMap, state: GridState, joint_action: int) -> GridState:
    with torch.no_grad():
        features = torch.tensor([encode_input(grid_map, state, joint_action)], dtype=torch.float32)
        object_logits, robot0_logits, robot1_logits = model(features)
        anchor = ANCHORS[int(object_logits.argmax(dim=1).item())]
        robot0 = CELLS[int(robot0_logits.argmax(dim=1).item())]
        robot1 = CELLS[int(robot1_logits.argmax(dim=1).item())]
    return GridState(robot0, robot1, anchor)


def model_step(model: PointModel, grid_map: GridMap):
    def _step(state: GridState, joint_action: int):
        decoded = decode(model, grid_map, state, joint_action)
        if not state_is_valid(decoded, grid_map):
            # Explicit invalid prediction: stop, never repair by consulting the true successor.
            return None, False
        return decoded, True

    return _step


def rollout_model(
    model: PointModel, grid_map: GridMap, initial: GridState, sequence: Sequence[int]
) -> Tuple[ScoreComponents, object]:
    result = rollout(initial, sequence, grid_map, model_step(model, grid_map))
    return score_trajectory(result), result


@dataclass
class SingleStepMetrics:
    split: str
    n: int
    object_accuracy: float
    robot0_accuracy: float
    robot1_accuracy: float
    full_state_accuracy: float
    invalid_state_rate: float
    moved_n: int
    moved_full_state_accuracy: float
    noop_n: int
    noop_full_state_accuracy: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "split": self.split, "n": self.n,
            "object_accuracy": self.object_accuracy,
            "robot0_accuracy": self.robot0_accuracy,
            "robot1_accuracy": self.robot1_accuracy,
            "full_state_accuracy": self.full_state_accuracy,
            "invalid_state_rate": self.invalid_state_rate,
            "moved_n": self.moved_n, "moved_full_state_accuracy": self.moved_full_state_accuracy,
            "noop_n": self.noop_n, "noop_full_state_accuracy": self.noop_full_state_accuracy,
        }


def single_step_metrics(
    model: PointModel, dataset: Dataset, maps: Dict[str, GridMap], split: str
) -> SingleStepMetrics:
    samples = _split_samples(dataset, split)
    features, object_t, robot0_t, robot1_t = _tensors(samples, maps)
    model.eval()
    with torch.no_grad():
        object_logits, robot0_logits, robot1_logits = model(features)
    object_pred = object_logits.argmax(dim=1)
    robot0_pred = robot0_logits.argmax(dim=1)
    robot1_pred = robot1_logits.argmax(dim=1)
    correct = (object_pred == object_t) & (robot0_pred == robot0_t) & (robot1_pred == robot1_t)
    invalid = 0
    moved_correct = moved_n = noop_correct = noop_n = 0
    for index, sample in enumerate(samples):
        decoded = GridState(
            CELLS[int(robot0_pred[index])], CELLS[int(robot1_pred[index])], ANCHORS[int(object_pred[index])]
        )
        if not state_is_valid(decoded, maps[sample.map_name]):
            invalid += 1
        if sample.object_moved:
            moved_n += 1
            moved_correct += int(bool(correct[index]))
        else:
            noop_n += 1
            noop_correct += int(bool(correct[index]))
    n = max(1, len(samples))
    return SingleStepMetrics(
        split=split,
        n=len(samples),
        object_accuracy=float((object_pred == object_t).float().mean()),
        robot0_accuracy=float((robot0_pred == robot0_t).float().mean()),
        robot1_accuracy=float((robot1_pred == robot1_t).float().mean()),
        full_state_accuracy=float(correct.float().mean()),
        invalid_state_rate=invalid / n,
        moved_n=moved_n,
        moved_full_state_accuracy=moved_correct / max(1, moved_n),
        noop_n=noop_n,
        noop_full_state_accuracy=noop_correct / max(1, noop_n),
    )
