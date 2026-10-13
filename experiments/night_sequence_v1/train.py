"""Round-5 training: fixed update budget, shared sampler, two heads, per-group loss curves.

Frozen by ``docs/night_sequence_v1/CONTRACT.md``: AdamW (lr 3e-4, weight decay 1e-4), gradient-norm
clip 1.0, effective batch 128, 5% linear warmup then cosine to 3e-5, validation every 200 updates,
checkpoint on the shared validation composite. All three architectures see the same examples for a
given seed.

    python experiments/night_sequence_v1/train.py --calibrate
    python experiments/night_sequence_v1/train.py --tier d0 --model mlp_history --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
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

from experiments.night_sequence_v1.dataset import (  # noqa: E402
    GROUPS,
    RootSampler,
    Tokens,
    fit_normalisation,
    to_torch,
)
from experiments.night_sequence_v1.models import (  # noqa: E402
    GROUP_NAMES,
    build_model,
    parameter_count,
)
from experiments.night_sequence_v1.sequences import EXIT_CODES  # noqa: E402

DATA = ROOT / "outputs/night_sequence_v1"
RUNS = DATA / "runs"
TIER_SPLITS = {"d0": ["d0", "val", "test"], "d1": ["d0", "d1ext", "val", "test"],
               "d2": ["d0", "d1ext", "d2ext", "val", "test"]}
PROCESS_GROUPS = [name for name in GROUP_NAMES if name.startswith("process")]
TERMINAL_GROUPS = [name for name in GROUP_NAMES if name.startswith("terminal")]
KIND = {name: kind for name, _w, kind in GROUPS}
WIDTH = {name: width for name, width, _k in GROUPS}

MODEL_KWARGS = {
    "mlp_current": {"hidden": 512, "layers": 2, "dropout": 0.1},
    "mlp_history": {"hidden": 512, "layers": 2, "dropout": 0.1},
    "transformer_history": {"d_model": 128, "heads": 4, "layers": 4, "ffn": 512, "dropout": 0.1},
}


def group_loss(name: str, output: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    kind = KIND[name]
    if kind == "continuous":
        loss = (output - target) ** 2 * mask
        return loss.sum() / mask.sum().clamp(min=1.0)
    if kind == "binary":
        loss = nn.functional.binary_cross_entropy_with_logits(output, target, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)
    loss = nn.functional.cross_entropy(output, target.view(-1), reduction="none")
    return (loss * mask.view(-1)).sum() / mask.view(-1).sum().clamp(min=1.0)


def composite_loss(outputs, targets, masks) -> Tuple[torch.Tensor, Dict[str, float]]:
    parts = {name: group_loss(name, outputs[name], targets[name], masks[name])
             for name in GROUP_NAMES}
    process = torch.stack([parts[name] for name in PROCESS_GROUPS]).mean()
    terminal = torch.stack([parts[name] for name in TERMINAL_GROUPS]).mean()
    total = 0.5 * process + 0.5 * terminal
    return total, {name: float(value.detach()) for name, value in parts.items()}


def evaluate_loss(model, tokens: Tokens, normalisation, split: str, count: int,
                  generator: np.random.Generator,
                  root_sampler: Optional[RootSampler] = None) -> Dict[str, float]:
    rows = np.where(tokens.split_of == split)[0]
    if root_sampler is None:
        root_sampler = RootSampler(tokens.steps[rows], tokens.roots[rows])
    indices, ends = root_sampler.sample(count, generator)
    indices = rows[indices]
    model.eval()
    totals: Dict[str, List[float]] = {}
    with torch.no_grad():
        for start in range(0, count, 128):
            batch = slice(start, min(start + 128, count))
            window, valid, targets, masks = to_torch(indices[batch], tokens, ends[batch],
                                                     normalisation)
            outputs = model(window, valid)
            total, parts = composite_loss(outputs, targets, masks)
            totals.setdefault("total", []).append(float(total))
            for name, value in parts.items():
                totals.setdefault(name, []).append(value)
    model.train()
    return {name: float(np.mean(values)) for name, values in totals.items()}


def train_one(tier: str, model_kind: str, seed: int, updates: int, *, batch: int = 128,
              lr: float = 3e-4, weight_decay: float = 1e-4, clip: float = 1.0,
              min_lr: float = 3e-5, warmup_fraction: float = 0.05, validate_every: int = 200,
              val_examples: int = 1024, out_dir: Path = RUNS, log: bool = True) -> dict:
    splits = TIER_SPLITS[tier]
    tokens = Tokens(splits)
    # the train pool is the tier's train roots only; val/test are never sampled for training
    train_rows = np.where(np.isin(tokens.split_of, ["d0", "d1ext", "d2ext"]))[0]
    train_sampler = RootSampler(tokens.steps[train_rows], tokens.roots[train_rows])
    validation_rows = np.where(tokens.split_of == "val")[0]
    validation_sampler = RootSampler(tokens.steps[validation_rows],
                                     tokens.roots[validation_rows])
    normalisation = fit_normalisation(tokens)
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = build_model(model_kind, **MODEL_KWARGS[model_kind])
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: _schedule(step, updates, warmup_fraction, lr, min_lr))
    curves: List[dict] = []
    best = {"loss": float("inf"), "update": -1, "state": None}
    started = time.perf_counter()
    model.train()
    for update in range(1, updates + 1):
        indices, ends = train_sampler.sample(batch, generator)
        indices = train_rows[indices]
        window, valid, targets, masks = to_torch(indices, tokens, ends, normalisation)
        outputs = model(window, valid)
        total, _parts = composite_loss(outputs, targets, masks)
        optimiser.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimiser.step()
        schedule.step()
        if update % validate_every == 0 or update == updates:
            validation = evaluate_loss(model, tokens, normalisation, "val", val_examples,
                                       np.random.default_rng(10_000 + update),
                                       root_sampler=validation_sampler)
            curves.append({"update": update, "train_total": float(total.detach()),
                           "lr": float(schedule.get_last_lr()[0]), **{
                               f"val_{k}": v for k, v in validation.items()}})
            if log:
                print(f"  [{model_kind}|{tier}|s{seed}] {update}/{updates} "
                      f"train={float(total):.4f} val={validation['total']:.4f} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
            if validation["total"] < best["loss"]:
                best = {"loss": validation["total"], "update": update,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{tier}_{model_kind}_seed{seed}"
    path = out_dir / f"{tag}.pt"
    torch.save({"state_dict": model.state_dict(), "model_kind": model_kind,
                "model_kwargs": MODEL_KWARGS[model_kind], "tier": tier, "seed": seed,
                "updates": updates, "best_val": best["loss"], "best_update": best["update"],
                "normalisation": normalisation, "exit_classes": list(EXIT_CODES),
                "groups": [{"name": n, "width": w, "kind": k} for n, w, k in GROUPS],
                "parameters": parameter_count(model)}, path)
    (out_dir / f"{tag}.curves.json").write_text(json.dumps(curves, indent=1))
    return {"tag": tag, "tier": tier, "model": model_kind, "seed": seed, "updates": updates,
            "parameters": parameter_count(model), "best_val": best["loss"],
            "best_update": best["update"], "seconds": time.perf_counter() - started,
            "checkpoint": str(path.relative_to(ROOT)), "curves": curves,
            "train_roots": int(len(set(tokens.roots[train_rows]))),
            "train_sequences": int(len(train_rows))}


def _schedule(step: int, total: int, warmup: float, lr: float, min_lr: float) -> float:
    warm = max(1, int(total * warmup))
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return max(min_lr / lr, cosine)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["d0", "d1", "d2"], default="d0")
    parser.add_argument("--model", default="mlp_history")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=6400)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--out", default=str(RUNS))
    parser.add_argument("--ledger", default=str(DATA / "run_ledger.jsonl"))
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    if args.calibrate:
        for kind in ("mlp_history", "transformer_history"):
            started = time.perf_counter()
            result = train_one("d0", kind, 0, 200, val_examples=256, out_dir=out_dir, log=False)
            print(json.dumps({"model": kind, "200_updates_seconds":
                              round(time.perf_counter() - started, 2),
                              "params": result["parameters"]}))
        return 0
    result = train_one(args.tier, args.model, args.seed, args.updates, out_dir=out_dir)
    with open(args.ledger, "a") as handle:
        handle.write(json.dumps({k: v for k, v in result.items() if k != "curves"}) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "curves"}, indent=1))
    print("TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
