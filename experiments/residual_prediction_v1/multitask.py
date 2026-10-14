"""Phase 3: the full multi-task predictor with an optional residual midpoint head.

Everything from round 6 is kept — the same eleven heads, the same two task groups with weight 0.5,
the same visibility and exit supervision, the same parameter count. The **only** change is how the
four midpoint columns of the two geometry heads are parameterised, plus the optional per-column
scaling. A ``direct`` configuration with the round-7 engineering fixes is trained alongside the
residual one so the comparison is between parameterisations under one protocol, not against an old
report number.

The model's ``forward`` returns standardised numbers for every head. Decoding to physical units
happens in :func:`decode_geometry`, which is the same function the online API uses.
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

from experiments.night_sequence_v1.dataset import GROUPS, Tokens  # noqa: E402
from experiments.night_sequence_v1.models import (  # noqa: E402
    GROUP_NAMES,
    build_model,
    parameter_count,
)
from experiments.night_sequence_v1.train import MODEL_KWARGS  # noqa: E402
from experiments.residual_prediction_v1 import data as D  # noqa: E402
from experiments.residual_prediction_v1.targets import (  # noqa: E402
    GeometryTransform,
    TransformSpec,
    decode_geometry,
    encode_geometry,
    fit_geometry_transform,
    fit_transform,
    geometry_hold_output,
    hold_output,
)

OUT = ROOT / "outputs/residual_prediction_v1"
KIND = {name: kind for name, _w, kind in GROUPS}
GEOMETRY_COLUMNS = [0, 1, 4, 5]


def build_multitask(kind: str, **kwargs):
    """The three architectures of round 6, unchanged in size and heads.

    This round does not add a wrapper model: the round-6 classes already expose the eleven heads, and
    reusing them keeps the parameter count and the head layout identical, so the only difference
    between the ``direct`` and ``residual`` configurations is the target transform.
    """

    return build_model(kind, **kwargs)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_transforms(tokens: Tokens, panel: Dict[str, np.ndarray], *, parameterisation: str,
                   normalisation: str) -> Tuple[GeometryTransform, Dict[str, Dict[str, float]]]:
    """Fit the geometry transform and the per-group statistics on the optimisation panel only."""

    batch = D.build_batch(tokens, panel["index"], panel["end"])
    midpoint_truth = batch.next_mid
    midpoint_mask = batch.next_mask
    spec = fit_transform(midpoint_truth, midpoint_mask, batch.reference,
                         name=f"process_{parameterisation}_{normalisation}",
                         parameterisation=parameterisation, normalisation=normalisation)
    geometry_truth = batch.targets["process_geometry"].numpy()
    geometry_mask = batch.masks["process_geometry"].numpy()
    transform = fit_geometry_transform(geometry_truth, geometry_mask, batch.reference,
                                       midpoint=spec)
    normalisation_stats: Dict[str, Dict[str, float]] = {}
    for name, _w, kind in GROUPS:
        if kind != "continuous":
            continue
        values = batch.targets[name].numpy()
        mask = batch.masks[name].numpy() > 0
        selected = values[mask]
        normalisation_stats[name] = {
            "mean": float(selected.mean()) if selected.size else 0.0,
            "std": float(selected.std()) if selected.size and selected.std() > 1e-9 else 1.0}
    return transform, normalisation_stats


def encode_all(batch: D.Batch, transform: GeometryTransform,
               stats: Dict[str, Dict[str, float]]) -> Dict[str, torch.Tensor]:
    """Encode every continuous group; the discrete heads are passed through."""

    out: Dict[str, torch.Tensor] = {}
    for name, _w, kind in GROUPS:
        value = batch.targets[name]
        if kind != "continuous":
            out[name] = value
            continue
        if name.endswith("geometry"):
            raw = value.numpy()
            reference = batch.reference
            encoded = encode_geometry(raw, reference, transform)
            out[name] = torch.tensor(encoded, dtype=torch.float32)
        else:
            block = stats[name]
            out[name] = (value - block["mean"]) / block["std"]
    return out


def group_loss(name: str, output: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    kind = KIND[name]
    if kind == "continuous":
        return (((output - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
    if kind == "binary":
        loss = nn.functional.binary_cross_entropy_with_logits(output, target, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)
    loss = nn.functional.cross_entropy(output, target.view(-1), reduction="none")
    return (loss * mask.view(-1)).sum() / mask.view(-1).sum().clamp(min=1.0)


def composite(outputs, targets, masks):
    process = torch.stack([group_loss(n, outputs[n], targets[n], masks[n])
                           for n in GROUP_NAMES if n.startswith("process")]).mean()
    terminal = torch.stack([group_loss(n, outputs[n], targets[n], masks[n])
                            for n in GROUP_NAMES if n.startswith("terminal")]).mean()
    return 0.5 * process + 0.5 * terminal


def apply_geometry_hold_bias(model, transform: GeometryTransform) -> None:
    """Bias the geometry head so a zero midpoint output decodes to the reference."""

    if transform.midpoint.parameterisation != "residual":
        return
    head = model.heads.process_geometry
    terminal = model.heads.terminal_geometry
    offset = geometry_hold_output(transform)
    with torch.no_grad():
        for module in (head, terminal):
            last = module[-1]
            bias = last.bias.clone()
            for column in GEOMETRY_COLUMNS:
                bias[column] = float(offset[column])
            last.bias.copy_(bias)


def train_multitask(tokens: Tokens, tier: str, kind: str, seed: int, *, updates: int = 3200,
                    batch: int = 128, lr: float = 3e-4, parameterisation: str = "residual",
                    normalisation: str = "per_group", refresh_every: int = 64,
                    validate_every: int = 200, out_dir: Path = OUT, log: bool = True) -> dict:
    """One full run. Reference windows are materialised in chunks; targets are reused within a chunk."""

    set_all_seeds(seed)
    panel = {name: D.load_panel(tier, name, out_dir) for name in ("train", "val")}
    transform, stats = fit_transforms(tokens, panel["train"], parameterisation=parameterisation,
                                      normalisation=normalisation)
    model = build_multitask(kind, **MODEL_KWARGS[kind])
    apply_geometry_hold_bias(model, transform)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=updates,
                                                          eta_min=3e-5)
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(panel["train"]["index"]))
    # the validation panel is fixed once and reused for every validation pass
    validation = _materialise(tokens, panel["val"])
    curves: List[dict] = []
    best = {"loss": float("inf"), "update": -1, "state": None}
    started = time.perf_counter()
    cursor = 0
    model.train()
    for update in range(1, updates + 1):
        if cursor + batch > len(order):
            cursor = 0
            order = generator.permutation(len(order))
        rows = order[cursor:cursor + batch]
        cursor += batch
        panel_rows = {"index": panel["train"]["index"][rows], "end": panel["train"]["end"][rows]}
        data_batch = D.build_batch(tokens, panel_rows["index"], panel_rows["end"])
        targets = encode_all(data_batch, transform, stats)
        masks = {name: data_batch.masks[name] for name, _w, _k in GROUPS}
        outputs = model(data_batch.window, data_batch.token_valid)
        loss = composite(outputs, targets, masks)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        if update % validate_every == 0 or update == updates:
            value = _validation_loss(model, tokens, validation, transform, stats)
            curves.append({"update": update, "train_total": float(loss.detach()),
                           "val_total": value, "lr": float(schedule.get_last_lr()[0])})
            if value < best["loss"]:
                best = {"loss": value, "update": update,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            if log:
                print(f"  [{kind}|{parameterisation}|{normalisation}|s{seed}] {update}/{updates} "
                      f"train={float(loss.detach()):.4f} val={value:.4f} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{kind}_{parameterisation}_{normalisation}_seed{seed}"
    path = out_dir / "checkpoints" / f"{tag}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_kind": kind,
                "model_kwargs": MODEL_KWARGS[kind], "spec": transform.descriptor(),
                "geometry_transform": transform.descriptor(),
                "normalisation": stats, "tier": tier, "seed": seed,
                "parameterisation": parameterisation, "normalisation_mode": normalisation,
                "updates": updates, "best_val": best["loss"], "best_update": best["update"],
                "parameters": parameter_count(model),
                "panels": {k: D.panel_hash(v) for k, v in panel.items()}}, path)
    return {"tag": tag, "tier": tier, "kind": kind, "seed": seed, "updates": updates,
            "parameterisation": parameterisation, "normalisation": normalisation,
            "parameters": parameter_count(model), "best_val": best["loss"],
            "best_update": best["update"], "seconds": time.perf_counter() - started,
            "curves": curves, "checkpoint": str(path.relative_to(ROOT)),
            "transform": transform.descriptor()}


def _materialise(tokens: Tokens, panel: Dict[str, np.ndarray],
                 chunk: int = 512) -> Dict[str, np.ndarray]:
    batches = []
    for start in range(0, len(panel["index"]), chunk):
        stop = min(start + chunk, len(panel["index"]))
        batches.append(D.build_batch(tokens, panel["index"][start:stop], panel["end"][start:stop]))
    return {"batches": batches, "panel": panel}


def _validation_loss(model, tokens: Tokens, validation: Dict[str, object],
                     transform: GeometryTransform, stats) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in validation["batches"]:
            targets = encode_all(batch, transform, stats)
            masks = {name: batch.masks[name] for name, _w, _k in GROUPS}
            outputs = model(batch.window, batch.token_valid)
            total += float(composite(outputs, targets, masks)) * len(batch)
            count += len(batch)
    model.train()
    return total / max(1, count)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--kind", default="mlp_history")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parameterisation", default="residual")
    parser.add_argument("--normalisation", default="per_group")
    parser.add_argument("--updates", type=int, default=3200)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--ledger", default=str(OUT / "RUN_LEDGER.jsonl"))
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = Path(args.out).resolve()
    result = train_multitask(tokens=Tokens(D.TIER_SPLITS[args.tier]),
                             tier=args.tier, kind=args.kind, seed=args.seed,
                             updates=args.updates, parameterisation=args.parameterisation,
                             normalisation=args.normalisation, out_dir=out_dir)
    with open(args.ledger, "a") as handle:
        handle.write(json.dumps({k: v for k, v in result.items() if k != "curves"}) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "curves"}, indent=1))
    print("MULTITASK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
