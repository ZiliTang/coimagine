"""The single default joint-window model and its training loop (contract B2).

One plain MLP, one configuration, three seeds. No hyper-parameter search, no loss-weight
search, no checkpoint selection by test or planning success.
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

from ..continuous2d.scenes import SceneSpec, scene_by_name
from .contract import DURATION_SCALE, HOLD_SCALE, TERMINATION_CLASSES
from .features import (
    INPUT_DIM,
    STATE_DIM,
    BoundaryState,
    apply_state_delta,
    encode_input,
    encode_state_target,
)
from .records import SPLIT_SCENES


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 256
    layers: int = 2
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 12

    def as_dict(self) -> Dict[str, object]:
        return {
            "hidden": self.hidden,
            "layers": self.layers,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }


class JointWindowMLP(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        layers: List[nn.Module] = []
        width = INPUT_DIM
        for _ in range(self.config.layers):
            layers += [nn.Linear(width, self.config.hidden), nn.ReLU()]
            width = self.config.hidden
        self.trunk = nn.Sequential(*layers)
        self.state_head = nn.Linear(width, STATE_DIM)
        self.hold_head = nn.Linear(width, 1)
        self.duration_head = nn.Linear(width, 1)
        self.termination_head = nn.Linear(width, len(TERMINATION_CLASSES))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.trunk(x)
        return {
            "state": self.state_head(hidden),
            "hold": self.hold_head(hidden),
            "duration": self.duration_head(hidden),
            "termination": self.termination_head(hidden),
        }


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
    train_samples: int = 0
    val_samples: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "config": self.config,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "checkpoint_sha256": self.checkpoint_sha256,
            "wall_clock": self.wall_clock,
            "parameter_count": self.parameter_count,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
        }


def load_window_records(path: Path, split: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] == split:
                records.append(record)
    return records


def boundary_from_states(object_state, robots, success_counter: float) -> BoundaryState:
    return BoundaryState(object=object_state, robots=tuple(robots), success_counter=success_counter)


def boundary_before(record: Dict[str, object]) -> BoundaryState:
    from ..continuous2d.backend import BodyState

    return BoundaryState(
        object=BodyState(*[float(v) for v in record["state"]["object"]]),  # type: ignore[arg-type]
        robots=tuple(  # type: ignore[arg-type]
            BodyState(*[float(v) for v in row]) for row in record["state"]["robots"]  # type: ignore[arg-type]
        ),
        success_counter=float(record["success_counter_before"]),
    )


def boundary_after(record: Dict[str, object]) -> BoundaryState:
    from ..continuous2d.backend import BodyState

    return BoundaryState(
        object=BodyState(*[float(v) for v in record["next_state"]["object"]]),  # type: ignore[arg-type]
        robots=tuple(  # type: ignore[arg-type]
            BodyState(*[float(v) for v in row]) for row in record["next_state"]["robots"]  # type: ignore[arg-type]
        ),
        success_counter=float(record["success_counter_after"]),
    )


def joint_spec_of(record: Dict[str, object]):
    from ..continuous2d.skills import joint_spec_from_dicts

    return joint_spec_from_dicts(record["joint_skill_specs"])  # type: ignore[arg-type]


def _tensors(records: Sequence[Dict[str, object]]) -> Tuple[torch.Tensor, ...]:
    from ..continuous2d.config import default_config

    physics = default_config().physics
    features: List[List[float]] = []
    states: List[List[float]] = []
    holds: List[float] = []
    durations: List[float] = []
    terminations: List[int] = []
    for record in records:
        scene = scene_by_name(str(record["scene_id"]))
        before = boundary_before(record)
        after = boundary_after(record)
        features.append(encode_input(scene, before, joint_spec_of(record), physics))
        states.append(encode_state_target(before, after, scene))
        holds.append(float(record["success_counter_after"]) / HOLD_SCALE)
        durations.append(float(record["duration"]) / DURATION_SCALE)
        terminations.append(TERMINATION_CLASSES.index(str(record["termination_class"])))
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(holds, dtype=torch.float32).unsqueeze(1),
        torch.tensor(durations, dtype=torch.float32).unsqueeze(1),
        torch.tensor(terminations, dtype=torch.long),
    )


#: Indices of the per-body angle-delta slots inside the 18-value state target. Angles are
#: stored in units of pi, so the wrapped residual is computed in those same units.
ANGLE_SLOTS = (2, 8, 14)


def state_residual(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-value residual with the angle slots compared modulo a full turn.

    B2 fix: the previous loss compared the raw angle *deltas*, so a prediction and a target
    that differ by exactly 2*pi were charged a full-turn error. The delta is now wrapped
    into ``(-pi, pi]`` (i.e. ``(-1, 1]`` in the stored units) before squaring, which is the
    contract's "wrapped difference" applied to the quantity the network actually predicts.
    """

    residual = predicted - target
    if residual.shape[-1] >= STATE_DIM:
        wrapped = torch.remainder(residual[..., list(ANGLE_SLOTS)] + 1.0, 2.0) - 1.0
        residual = residual.clone()
        residual[..., list(ANGLE_SLOTS)] = wrapped
    return residual


def _loss(outputs: Dict[str, torch.Tensor], targets: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    state_target, hold_target, duration_target, termination_target = targets
    state_mse = (state_residual(outputs["state"], state_target) ** 2).mean()
    hold_mse = nn.functional.mse_loss(torch.sigmoid(outputs["hold"]), hold_target)
    duration_mse = nn.functional.mse_loss(torch.sigmoid(outputs["duration"]), duration_target)
    termination_ce = nn.functional.cross_entropy(outputs["termination"], termination_target)
    return state_mse + hold_mse + duration_mse + termination_ce


def train_model(
    train_records: Sequence[Dict[str, object]],
    val_records: Sequence[Dict[str, object]],
    *,
    seed: int,
    config: Optional[ModelConfig] = None,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[JointWindowMLP, TrainingLog]:
    config = config or ModelConfig()
    torch.manual_seed(seed)
    model = JointWindowMLP(config)
    log = TrainingLog(seed=seed, config=config.as_dict())
    log.parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log.train_samples = len(train_records)
    log.val_samples = len(val_records)

    train = _tensors(train_records)
    val = _tensors(val_records)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    generator = torch.Generator().manual_seed(seed)
    started = time.time()
    best_state = None

    for epoch in range(config.max_epochs):
        model.train()
        order = torch.randperm(len(train[0]), generator=generator)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            outputs = model(train[0][index])
            targets = tuple(target[index] for target in train[1:])
            loss = _loss(outputs, targets)  # type: ignore[arg-type]
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            batches += 1
        model.eval()
        with torch.no_grad():
            val_loss = float(_loss(model(val[0]), val[1:]).item())  # type: ignore[arg-type]
        log.epochs.append(
            {"epoch": epoch, "train_loss": epoch_loss / max(1, batches), "val_loss": val_loss}
        )
        if val_loss < log.best_val_loss - 1e-9:
            log.best_val_loss = val_loss
            log.best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif epoch - log.best_epoch >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    log.wall_clock = time.time() - started
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "config": config.as_dict(), "seed": seed},
            checkpoint_path,
        )
        log.checkpoint_sha256 = _sha256(checkpoint_path)
    return model, log


def load_model(path: Path, config: Optional[ModelConfig] = None) -> JointWindowMLP:
    blob = torch.load(path, map_location="cpu")
    model = JointWindowMLP(ModelConfig(**blob["config"]) if config is None else config)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WindowPrediction:
    next_state: BoundaryState
    success_hold: float
    duration: float
    termination: str
    termination_logits: Tuple[float, float, float]
    raw_state_delta: Tuple[float, ...]


def predict_window(
    model: JointWindowMLP,
    scene: SceneSpec,
    before: BoundaryState,
    joint_spec,
    physics,
) -> WindowPrediction:
    """One recursive model step. Only the predicted state and static geometry are used."""

    features = torch.tensor(
        [encode_input(scene, before, joint_spec, physics)], dtype=torch.float32
    )
    with torch.no_grad():
        outputs = model(features)
        delta = outputs["state"][0].tolist()
        hold = float(torch.sigmoid(outputs["hold"])[0, 0].item()) * HOLD_SCALE
        duration = float(torch.sigmoid(outputs["duration"])[0, 0].item()) * DURATION_SCALE
        logits = outputs["termination"][0].tolist()
    index = int(max(range(len(logits)), key=lambda i: (logits[i], -i)))
    next_state = apply_state_delta(before, delta, scene)
    next_state = BoundaryState(
        object=next_state.object, robots=next_state.robots, success_counter=hold
    )
    return WindowPrediction(
        next_state=next_state,
        success_hold=hold,
        duration=duration,
        termination=TERMINATION_CLASSES[index],
        termination_logits=(float(logits[0]), float(logits[1]), float(logits[2])),
        raw_state_delta=tuple(float(value) for value in delta),
    )
