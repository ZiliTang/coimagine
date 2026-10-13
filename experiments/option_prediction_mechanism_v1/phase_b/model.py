"""Phase B model and training core, independent of the dataset's on-disk schema.

The freeze (CONTRACT section 2.3): a 2x256 ReLU MLP with one head per target group, Adam at
1e-3, batch 128, at most 200 epochs, validation patience 20, seeds 0/1/2, and **equal loss weight
1.0 for all six groups**. Continuous targets are standardised with train statistics and scored
with a masked MSE; observation validity is multi-label binary cross entropy; the exit class is
cross entropy. The checkpoint is chosen on the validation composite only -- the test split is
never touched during training or model selection.

Three arms share the *same* architecture and the *same* input slots:

  persistence  terminal geometry / robot state copied from the start; duration and class from the
               train statistics. No network.
  deployable   whitelist inputs; the privileged slots are present but pinned to zero.
  privileged   identical network and budget, privileged slots filled with the object's true
               initial pose and velocity. Evaluated only as an information-gain reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

#: the six target groups, all weighted 1.0 by contract
GROUP_WEIGHTS: Dict[str, float] = {
    "terminal_geometry": 1.0,
    "robot_pose_delta": 1.0,
    "robot_residual_velocity": 1.0,
    "duration": 1.0,
    "observation_validity": 1.0,
    "exit_class": 1.0,
}


@dataclass
class GroupSpec:
    name: str
    kind: str            # "continuous" | "binary" | "multiclass"
    dim: int
    weight: float = 1.0
    num_classes: int = 0


@dataclass
class SupervisedData:
    """Normalised arrays. ``mask`` is 1 where a label exists; masked entries never contribute."""

    x: np.ndarray
    x_priv: np.ndarray
    roots: Sequence[str]
    split: np.ndarray
    commands: Sequence[str]
    targets: Dict[str, Dict[str, np.ndarray]]
    specs: List[GroupSpec]
    #: for the persistence baseline: the start-state values of each continuous target
    persistence: Dict[str, np.ndarray] = field(default_factory=dict)
    #: train statistics for duration (median), exit class and observation validity (majority)
    train_duration_median: float = 0.0
    train_exit_majority: int = 0
    train_validity_majority: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float32))


class MultiHeadMLP(nn.Module):
    def __init__(self, input_dim: int, specs: Sequence[GroupSpec], hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.heads = nn.ModuleDict()
        self.specs = list(specs)
        for spec in self.specs:
            out = spec.num_classes if spec.kind == "multiclass" else spec.dim
            self.heads[spec.name] = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.trunk(x)
        return {spec.name: self.heads[spec.name](features) for spec in self.specs}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(prediction.dtype)
    denominator = mask.sum().clamp(min=1.0)
    return ((prediction - target) ** 2 * mask).sum() / denominator


def masked_bce(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(prediction.dtype)
    loss = nn.functional.binary_cross_entropy_with_logits(
        prediction, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def masked_ce(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(prediction.dtype)
    loss = nn.functional.cross_entropy(prediction, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def batch_loss(outputs: Dict[str, torch.Tensor], batch_targets: Dict[str, Dict[str, torch.Tensor]],
               specs: Sequence[GroupSpec]) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = torch.zeros((), dtype=torch.float32)
    parts: Dict[str, float] = {}
    for spec in specs:
        target = batch_targets[spec.name]
        if spec.kind == "continuous":
            loss = masked_mse(outputs[spec.name], target["y"], target["mask"])
        elif spec.kind == "binary":
            loss = masked_bce(outputs[spec.name], target["y"], target["mask"])
        else:
            loss = masked_ce(outputs[spec.name], target["y"], target["mask"])
        total = total + spec.weight * loss
        parts[spec.name] = float(loss.detach())
    return total, parts


def _standardise(data: SupervisedData) -> None:
    """Fit mean/std on the train split of each continuous group, in place."""

    train = data.split == "train"
    for spec in data.specs:
        if spec.kind != "continuous":
            continue
        block = data.targets[spec.name]
        values = block["y"][train]
        mask = block["mask"][train]
        selected = values[mask > 0]
        mean = selected.mean(axis=0) if selected.size else np.zeros(spec.dim)
        std = selected.std(axis=0) if selected.size else np.ones(spec.dim)
        mean = np.atleast_1d(np.asarray(mean, dtype=np.float64))
        std = np.atleast_1d(np.asarray(std, dtype=np.float64))
        std[std < 1e-9] = 1.0
        block["mean"] = mean.astype(np.float32)
        block["std"] = std.astype(np.float32)
        block["y_raw"] = block["y"].copy()
        block["y"] = (block["y"] - mean) / std


def _torch_batch(indices: np.ndarray, x: np.ndarray, targets: Dict[str, Dict[str, np.ndarray]],
                 specs: Sequence[GroupSpec]):
    xb = torch.from_numpy(x[indices]).float()
    batch: Dict[str, Dict[str, torch.Tensor]] = {}
    for spec in specs:
        block = targets[spec.name]
        entry = {"mask": torch.from_numpy(block["mask"][indices]).float()}
        if spec.kind == "multiclass":
            entry["y"] = torch.from_numpy(block["y"][indices]).long()
        elif spec.kind == "binary":
            entry["y"] = torch.from_numpy(block["y"][indices]).float()
        else:
            entry["y"] = torch.from_numpy(block["y"][indices]).float()
        batch[spec.name] = entry
    return xb, batch


def train_one_seed(data: SupervisedData, *, use_privileged: bool, seed: int,
                   epochs: int = 200, batch_size: int = 128, lr: float = 1e-3,
                   patience: int = 20, verbose: bool = False) -> Dict[str, object]:
    """Train one seed and return the best-validation state and its curves."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    x = np.hstack([data.x, data.x_priv if use_privileged else np.zeros_like(data.x_priv)])
    train_idx = np.where(data.split == "train")[0]
    val_idx = np.where(data.split == "val")[0]
    model = MultiHeadMLP(x.shape[1], data.specs)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    best = {"val": float("inf"), "state": None, "epoch": -1}
    curves: List[Dict[str, float]] = []
    stale = 0
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train_idx)
        train_total = 0.0
        batches = 0
        for start in range(0, len(shuffled), batch_size):
            chunk = shuffled[start:start + batch_size]
            xb, batch = _torch_batch(chunk, x, data.targets, data.specs)
            loss, _parts = batch_loss(model(xb), batch, data.specs)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_total += float(loss.detach())
            batches += 1
        model.eval()
        with torch.no_grad():
            xb, batch = _torch_batch(val_idx, x, data.targets, data.specs)
            val_loss, val_parts = batch_loss(model(xb), batch, data.specs)
        curves.append({"epoch": epoch, "train": train_total / max(1, batches),
                       "val": float(val_loss), **{f"val_{k}": v for k, v in val_parts.items()}})
        if float(val_loss) < best["val"] - 1e-6:
            best = {"val": float(val_loss), "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
        if verbose and epoch % 20 == 0:
            print(f"  seed {seed} epoch {epoch} train {train_total / max(1, batches):.4f} "
                  f"val {float(val_loss):.4f}", flush=True)
    model.load_state_dict(best["state"])
    return {"model": model, "best_val": best["val"], "best_epoch": best["epoch"],
            "curves": curves, "parameters": model.parameter_count(),
            "input_dim": int(x.shape[1]), "x": x}


def predict(model: nn.Module, x: np.ndarray, specs: Sequence[GroupSpec], batch: int = 512):
    model.eval()
    outputs: Dict[str, List[np.ndarray]] = {spec.name: [] for spec in specs}
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = torch.from_numpy(x[start:start + batch]).float()
            result = model(chunk)
            for spec in specs:
                outputs[spec.name].append(result[spec.name].numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in outputs.items()}


def unstandardise_predictions(data: SupervisedData,
                              predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert a model's **standardised** outputs into the targets' original physical units.

    The evaluation interface is defined in original units for every continuous group, so anything
    coming out of the network must pass through here exactly once. Baselines that already produce
    original units (``persistence_predictions``) must **not** be passed through it -- doing so was
    the round-3 unit bug, where the persistence baseline was scaled by ``std`` and offset by
    ``mean`` a second time.
    """

    out: Dict[str, np.ndarray] = {}
    for spec in data.specs:
        if spec.kind == "continuous":
            block = data.targets[spec.name]
            out[spec.name] = predictions[spec.name] * block["std"] + block["mean"]
        else:
            out[spec.name] = predictions[spec.name]
    return out


def persistence_predictions(data: SupervisedData, split: str) -> Dict[str, np.ndarray]:
    """The contract's simple baseline, in the *original* target units.

    Terminal geometry and robot state are copied from the start; duration uses the train median
    and the discrete targets use the train majority, exactly as CONTRACT section 2.3 requires.
    Arrays are returned for **every** row (not only the requested split) so the evaluator can use
    one indexing convention for every model, network or baseline.
    """

    total = len(data.split)
    out: Dict[str, np.ndarray] = {}
    for spec in data.specs:
        block = data.targets[spec.name]
        if spec.name == "duration":
            out[spec.name] = np.full((total, 1), data.train_duration_median, dtype=np.float32)
        elif spec.kind == "continuous":
            out[spec.name] = data.persistence[spec.name].astype(np.float32)
        elif spec.kind == "binary":
            out[spec.name] = np.tile(data.train_validity_majority, (total, 1)).astype(np.float32)
        else:
            out[spec.name] = np.full(total, data.train_exit_majority, dtype=np.int64)
    return out
