"""The 2x2 single-head diagnostic: {direct, residual} x {per-group, per-column} scales.

Four engineering configurations, three seeds each. Everything that could confound the comparison is
held fixed:

* all random sources (Python ``random``, NumPy, PyTorch) are seeded **before** the model is built;
* one initial ``state_dict`` is created per seed and **reused by all four cells**, so the comparison
  is between parameterisations of the same starting weights, not between two random draws;
* the batch order comes from a generator seeded the same way in every cell;
* the loss and every reported error are restricted to the **target visibility mask**; a normalisation
  statistic is fitted on visible training labels only;
* the training panel, the held-out-from-optimisation diagnostic panel and the validation panel are
  fixed and hashed, and the validation records are disjoint from the training records.

This compares *engineering configurations*. It does not isolate the pure structural effect of the
residual form from the effect of the target's amplitude, and the report says so.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import Tokens  # noqa: E402
from experiments.night_sequence_v1.models import parameter_count  # noqa: E402
from experiments.night_sequence_v1.windows import FEATURES, WINDOW  # noqa: E402
from experiments.residual_prediction_v1 import data as D  # noqa: E402
from experiments.residual_prediction_v1.targets import (  # noqa: E402
    GEOMETRY_COLUMNS,
    MIDPOINT_COLUMNS,
    TransformSpec,
    decode_prediction,
    encode_target,
    fit_transform,
    hold_output,
    visible_reference,
)

OUT = ROOT / "outputs/residual_prediction_v1"
CELLS = [("direct", "per_group"), ("direct", "per_column"),
         ("residual", "per_group"), ("residual", "per_column")]


class MidpointHead(nn.Module):
    """Flattened-history MLP with four midpoint outputs.

    ``forward`` returns **standardised** numbers only. The reference is never added inside: decoding
    is the caller's job through :func:`decode_prediction`, which is what makes the residual
    implementation unambiguous.
    """

    def __init__(self, hidden: int = 512, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks: List[nn.Module] = []
        width = WINDOW * FEATURES + WINDOW
        for _ in range(max(1, layers - 1)):
            blocks += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        blocks += [nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 4)]
        self.trunk = nn.Sequential(*blocks)

    def forward(self, window: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([window.reshape(window.shape[0], -1), valid], dim=-1))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_spec(stats: Dict[str, np.ndarray], parameterisation: str, normalisation: str,
               name: str) -> TransformSpec:
    return fit_transform(stats["truth"], stats["mask"], stats["reference"], name=name,
                         parameterisation=parameterisation, normalisation=normalisation)


def train_cell(tokens: Tokens, panel: Dict[str, np.ndarray], spec: TransformSpec, *,
               initial_state: Dict[str, torch.Tensor], seed: int, updates: int, lr: float,
               batch_size: int = 128, dropout: float = 0.1) -> MidpointHead:
    """Train one cell from the shared initial weights, with a fixed batch order."""

    set_all_seeds(seed)
    model = MidpointHead(dropout=dropout)
    model.load_state_dict(copy.deepcopy(initial_state))
    # make a zero output mean "hold the reference" for the residual transform
    if spec.parameterisation == "residual":
        with torch.no_grad():
            last = model.trunk[-1]
            offset = hold_output(spec)
            if spec.normalisation == "per_group":
                last.bias.copy_(torch.full_like(last.bias, float(np.atleast_1d(offset)[0])))
            else:
                last.bias.copy_(torch.tensor(offset, dtype=last.bias.dtype))
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=updates,
                                                          eta_min=lr * 0.05)
    generator = np.random.default_rng(seed)          # identical batch order across cells
    order = generator.permutation(len(panel["index"]))
    truths = panel["truth"]
    references = panel["reference"]
    masks = panel["mask"]
    encoded = encode_target(truths, references, spec)
    model.train()
    cursor = 0
    for _ in range(updates):
        if cursor + batch_size > len(order):
            cursor = 0
        rows = order[cursor:cursor + batch_size]
        cursor += batch_size
        window = panel["window"][rows]
        valid = panel["token_valid"][rows]
        target = torch.tensor(encoded[rows], dtype=torch.float32)
        mask = torch.tensor(masks[rows], dtype=torch.float32)
        prediction = model(window, valid)
        loss = (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
    model.eval()
    return model


def materialise(tokens: Tokens, panel: Dict[str, np.ndarray], spec: TransformSpec,
                model: MidpointHead, batch_size: int = 512) -> Dict[str, np.ndarray]:
    """Decode a panel in physical units and report masked midpoint errors."""

    frames, references, masks = [], [], []
    for start in range(0, len(panel["index"]), batch_size):
        stop = min(start + batch_size, len(panel["index"]))
        batch = D.build_batch(tokens, panel["index"][start:stop], panel["end"][start:stop])
        with torch.no_grad():
            output = model(batch.window, batch.token_valid).numpy().astype(np.float64)
        frames.append(decode_prediction(output, batch.reference, spec))
        references.append(batch.reference)
        masks.append(batch.next_mask)
    return {"prediction": np.concatenate(frames), "reference": np.concatenate(references),
            "mask": np.concatenate(masks)}


def metrics(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray,
            speed: np.ndarray, visible_now: np.ndarray,
            roots: Sequence[str]) -> Dict[str, object]:
    """p50 / p90 / RMSE per robot, overall and on the sustained-visible-and-moving subgroup."""

    out: Dict[str, object] = {}
    roots = np.asarray(roots)
    for robot in (0, 1):
        columns = [0, 1] if robot == 0 else [2, 3]
        keep = (mask[:, columns[0]] > 0) & (mask[:, columns[1]] > 0)
        error = np.hypot(prediction[:, columns[0]] - truth[:, columns[0]],
                         prediction[:, columns[1]] - truth[:, columns[1]])
        entry = {"p50_cm": float(np.median(error[keep]) * 100) if keep.any() else None,
                 "p90_cm": float(np.quantile(error[keep], 0.9) * 100) if keep.any() else None,
                 "rmse_cm": float(np.sqrt(np.mean(error[keep] ** 2)) * 100) if keep.any() else None,
                 "windows": int(keep.sum()),
                 "roots": int(len(set(roots[keep].tolist())))}
        moving = keep & (speed[:, robot] >= 0.05) & (visible_now[:, robot] > 0.5)
        entry["sustained_visible_moving"] = {
            "p50_cm": float(np.median(error[moving]) * 100) if moving.any() else None,
            "p90_cm": float(np.quantile(error[moving], 0.9) * 100) if moving.any() else None,
            "windows": int(moving.sum()),
            "roots": int(len(set(roots[moving].tolist())))}
        out[f"r{robot}"] = entry
    return out


def hold_reference_metrics(prediction_reference: np.ndarray, truth: np.ndarray,
                           mask: np.ndarray, speed: np.ndarray, visible_now: np.ndarray,
                           roots: Sequence[str]) -> Dict[str, object]:
    """The hold baseline under the **same reference rule**, so the baseline is not quietly changed."""

    return metrics(prediction_reference, truth, mask, speed, visible_now, roots)


def run_diagnostic(tier: str = "d1", *, seeds: Sequence[int] = (0, 1, 2), updates: int = 3000,
                   lr: float = 1e-3, out_dir: Path = OUT, log: bool = True) -> dict:
    tokens = Tokens(D.TIER_SPLITS[tier])
    panels = {}
    for name in ("train", "train_diagnostic", "val"):
        panel = D.load_panel(tier, name, out_dir)
        batch = D.build_batch(tokens, panel["index"], panel["end"])
        panel = dict(panel)
        panel.update({"window": batch.window, "token_valid": batch.token_valid,
                      "speed": batch.current_speed,
                      "visible_now": batch.current_visible.astype(float),
                      "truth": batch.next_mid, "reference": batch.reference,
                      "mask": batch.next_mask})
        panels[name] = panel
    # the transform statistics are fitted on the **visible** labels of the optimisation panel only
    stats = _train_stats(tokens, panels["train"])
    result = {"tier": tier, "seeds": list(seeds), "updates": updates, "lr": lr,
              "note": "2x2 single-head diagnostic; masked loss and masked metrics",
              "cells": {}, "panels": {}}
    started = time.perf_counter()
    for seed in seeds:
        set_all_seeds(seed)
        prototype = MidpointHead()
        initial = copy.deepcopy(prototype.state_dict())
        result["cells"].setdefault("initial_state_hash", {})[str(seed)] = _state_hash(initial)
        for parameterisation, normalisation in CELLS:
            name = f"{parameterisation}_{normalisation}"
            spec = build_spec(stats, parameterisation, normalisation, name)
            model = train_cell(tokens, panels["train"], spec, initial_state=initial, seed=seed,
                               updates=updates, lr=lr)
            checkpoint = out_dir / "checkpoints" / f"single_head_{name}_seed{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "model_kind": "midpoint_head",
                        "model_kwargs": {"hidden": 512, "layers": 2, "dropout": 0.1},
                        "spec": spec.descriptor(), "seed": seed, "tier": tier,
                        "parameterisation": parameterisation, "normalisation": normalisation,
                        "panels": {k: D.panel_hash(v) for k, v in panels.items()},
                        "updates": updates, "lr": lr}, checkpoint)
            entry = {"seed": seed, "parameterisation": parameterisation,
                     "normalisation": normalisation, "spec": spec.descriptor(),
                     "parameters": parameter_count(model),
                     "checkpoint": str(checkpoint.relative_to(ROOT))}
            for panel_name in ("train", "train_diagnostic", "val"):
                panel = panels[panel_name]
                subset = {key: panel[key] for key in
                          ("index", "end", "truth", "reference", "mask")}
                decoded = materialise(tokens, subset, spec, model)
                entry[panel_name] = metrics(decoded["prediction"], panel["truth"],
                                            panel["mask"], panel["speed"],
                                            panel["visible_now"], panel["root_id"])
                entry[f"{panel_name}_hold"] = hold_reference_metrics(
                    decoded["reference"], panel["truth"], panel["mask"], panel["speed"],
                    panel["visible_now"], panel["root_id"])
            result["cells"][f"{name}_seed{seed}"] = entry
            if log:
                validation = entry["val"]["r0"]
                print(f"  seed{seed} {name:22s} val r0 p50={validation['p50_cm']:.3f} cm "
                      f"(hold {entry['val_hold']['r0']['p50_cm']:.3f}) "
                      f"moving p50={validation['sustained_visible_moving']['p50_cm']}", flush=True)
    result["seconds"] = time.perf_counter() - started
    for name, panel in panels.items():
        result["panels"][name] = {"windows": int(len(panel["index"])),
                                  "roots": int(len(set(str(r) for r in panel["root_id"]))),
                                  "hash": D.panel_hash(panel)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"single_head_diagnostic_{tier}.json").write_text(
        json.dumps(result, indent=1, sort_keys=True, default=float))
    return result


def _train_stats(tokens: Tokens, panel: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    batch = D.build_batch(tokens, panel["index"], panel["end"])
    return {"truth": batch.next_mid, "reference": batch.reference, "mask": batch.next_mask}


def _state_hash(state: Dict[str, torch.Tensor]) -> str:
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    run_diagnostic(args.tier, seeds=args.seeds, updates=args.updates, lr=args.lr,
                   out_dir=Path(args.out))
    print("SINGLE_HEAD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
