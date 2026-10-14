"""Training harness for the multi-step pair and the capacity curve.

Loss protocol (identical for both multi-step models):

* a step is supervised only when a real frame exists at ``end + k``; padding past the option exit is
  masked out and no terminal state is repeated to fill it;
* the geometry target is a **residual** against the frozen prediction-start anchor, so step k never
  borrows the midpoint at ``end + k``;
* each group is averaged inside a window over its valid steps and columns, then across windows, so a
  long window cannot earn extra weight from having more labels;
* the process task is the mean of geometry / pose / velocity / validity and the direct-terminal task
  is the auxiliary head, combined 0.5 / 0.5 exactly as the previous round.
"""

from __future__ import annotations

import argparse
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

from experiments.night_sequence_v1.dataset import Tokens, build_example  # noqa: E402
from experiments.transformer_scale_latent_v1 import cache as C  # noqa: E402
from experiments.transformer_scale_latent_v1 import data as D  # noqa: E402
from experiments.transformer_scale_latent_v1.models import (  # noqa: E402
    DirectMultiStep,
    LatentAutoregressive,
    OBSERVATION_ORDER,
    OBSERVATION_WIDTH,
    parameter_breakdown,
)

OUT = ROOT / "outputs/transformer_scale_latent_v1"
#: geometry columns that carry a midpoint (the other four are the surface normal)
MIDPOINT_COLUMNS = [0, 1, 4, 5]
NORMAL_COLUMNS = [2, 3, 6, 7]
#: how a 4-vector anchor maps onto the 8-column geometry vector
ANCHOR_MAP = {0: 0, 1: 1, 4: 2, 5: 3}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _anchor_expanded(anchor: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """``(n, 4)`` anchor -> an array shaped like ``(n, ..., 8)`` holding it on the midpoint columns.

    The four normal columns stay at zero: they are absolute, not residual.
    """

    anchor = np.asarray(anchor, dtype=np.float64)
    expanded = np.zeros(shape, dtype=np.float64)
    for column, anchor_index in ANCHOR_MAP.items():
        expanded[..., column] = anchor[:, anchor_index].reshape(
            (anchor.shape[0],) + (1,) * (len(shape) - 2))
    return expanded


def geometry_residual(geometry: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    geometry = np.asarray(geometry, dtype=np.float64)
    out = np.array(geometry, copy=True)
    expanded = _anchor_expanded(anchor, geometry.shape)
    for column in ANCHOR_MAP:
        out[..., column] = geometry[..., column] - expanded[..., column]
    return out


def geometry_from_residual(residual: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    out = np.array(residual, copy=True)
    expanded = _anchor_expanded(anchor, residual.shape)
    for column in ANCHOR_MAP:
        out[..., column] = residual[..., column] + expanded[..., column]
    return out


def encode_geometry_batch(geometry: np.ndarray, anchor: np.ndarray,
                          stats: Dict[str, np.ndarray]) -> np.ndarray:
    """(n, H, 8) physical geometry -> standardised targets, residual on the midpoint columns."""

    residual = geometry_residual(geometry, anchor)
    out = np.empty_like(residual)
    mean = np.asarray(stats["geometry_mean"])
    std = np.asarray(stats["geometry_std"])
    out[..., MIDPOINT_COLUMNS] = (residual[..., MIDPOINT_COLUMNS] - mean) / std
    out[..., NORMAL_COLUMNS] = (geometry[..., NORMAL_COLUMNS] - stats["normal_mean"]) \
        / stats["normal_std"]
    return out


def decode_geometry_batch(residual: np.ndarray, anchor: np.ndarray,
                          stats: Dict[str, np.ndarray]) -> np.ndarray:
    mean = float(stats["geometry_mean"])
    std = float(stats["geometry_std"])
    residual = np.asarray(residual, dtype=np.float64)
    out = np.empty_like(residual)
    # the four midpoint columns are residual against the frozen anchor; the normals are absolute
    out[..., MIDPOINT_COLUMNS] = residual[..., MIDPOINT_COLUMNS] * std + mean
    out[..., NORMAL_COLUMNS] = residual[..., NORMAL_COLUMNS] * stats["normal_std"] \
        + stats["normal_mean"]
    return geometry_from_residual(out, anchor)


def fit_statistics(tokens: Tokens, index: np.ndarray, ends: np.ndarray,
                   horizon: int) -> Dict[str, np.ndarray]:
    """Per-group scale fitted on the **training** panel only, visible entries only."""

    batch = D.build_multistep(tokens, index, ends, horizon)
    stats: Dict[str, np.ndarray] = {}
    keep = batch.step_mask > 0
    geometry = batch.future["geometry"]
    residual = geometry_residual(geometry, batch.anchor)
    mask = (batch.geometry_mask > 0) & keep[..., None]
    selected = residual[..., MIDPOINT_COLUMNS][mask[..., MIDPOINT_COLUMNS]]
    stats["geometry_mean"] = np.asarray(selected.mean() if selected.size else 0.0)
    stats["geometry_std"] = np.asarray(selected.std() if selected.size and selected.std() > 1e-9
                                       else 1.0)
    normals = geometry[..., NORMAL_COLUMNS][mask[..., NORMAL_COLUMNS]]
    stats["normal_mean"] = np.asarray(normals.mean() if normals.size else 0.0)
    stats["normal_std"] = np.asarray(normals.std() if normals.size and normals.std() > 1e-9 else 1.0)
    for name in ("pose", "velocity"):
        values = batch.future[name][keep]
        stats[f"{name}_mean"] = np.asarray(values.mean() if values.size else 0.0)
        stats[f"{name}_std"] = np.asarray(values.std() if values.size and values.std() > 1e-9 else 1.0)
    stats["version"] = np.asarray("transformer-scale-latent-v1/stats-1")
    return stats


def encode_targets(batch: D.MultiStepBatch, stats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {"geometry": encode_geometry_batch(batch.future["geometry"], batch.anchor, stats)}
    for name in ("pose", "velocity"):
        out[name] = (batch.future[name] - stats[f"{name}_mean"]) / stats[f"{name}_std"]
    out["validity"] = batch.future["validity"]
    return out


def stepped_masks(batch: D.MultiStepBatch) -> Dict[str, np.ndarray]:
    keep = batch.step_mask[..., None]
    return {"geometry": batch.geometry_mask * keep,
            "pose": np.broadcast_to(keep, batch.future["pose"].shape).copy(),
            "velocity": np.broadcast_to(keep, batch.future["velocity"].shape).copy(),
            "validity": np.broadcast_to(keep, batch.future["validity"].shape).copy()}


def window_averaged_loss(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average inside each window over valid steps and columns, then across windows."""

    count = mask.sum(dim=(1, 2))
    per_window = (error * mask).sum(dim=(1, 2)) / count.clamp(min=1.0)
    return per_window.mean()


def process_loss(outputs: Dict[str, torch.Tensor], targets: Dict[str, np.ndarray],
                 masks: Dict[str, np.ndarray], device) -> torch.Tensor:
    total = []
    for name in OBSERVATION_ORDER:
        target = torch.tensor(targets[name], dtype=torch.float32, device=device)
        mask = torch.tensor(masks[name], dtype=torch.float32, device=device)
        total.append(window_averaged_loss((outputs[name] - target) ** 2, mask))
    return torch.stack(total).mean()


def terminal_loss(model_output, tokens: Tokens, index: np.ndarray, ends: np.ndarray,
                  anchor: np.ndarray, stats) -> torch.Tensor:
    """The auxiliary direct-terminal head: predict the option's final observation from the prefix."""

    geometry = np.zeros((len(index), 8))
    mask = np.zeros((len(index), 8))
    for row, i in enumerate(index):
        split = tokens.split_of[i]
        local = int(i) - tokens.offsets[split]
        terminal = tokens.terminal_arrays[split][local].astype(np.float64)
        geometry[row] = np.concatenate([terminal[7:11], terminal[25:29]])
        visible = tokens.terminal_visibility(split, local) if hasattr(tokens, "terminal_visibility") \
            else None
        if visible is None:
            mask[row, 0:4] = 1.0 if terminal[6] > 0.5 else 0.0
            mask[row, 4:8] = 1.0 if terminal[24] > 0.5 else 0.0
        else:
            mask[row, 0:4] = visible[0]
            mask[row, 4:8] = visible[1]
    target = torch.tensor(encode_geometry_batch(geometry, anchor, stats),
                          dtype=torch.float32)
    mask_t = torch.tensor(mask, dtype=torch.float32)
    prediction = model_output["terminal_geometry"]
    # the terminal head is a single step, so it is shaped like a one-step sequence to reuse the
    # same inside-window averaging rule as the multi-step groups
    return window_averaged_loss((prediction.unsqueeze(1) - target.unsqueeze(1)) ** 2,
                                mask_t.unsqueeze(1))


def build_model(kind: str, config: Dict[str, object]):
    size = config["size"]
    d_model, heads, layers, ffn, decoder = size
    if kind == "direct_multistep":
        return DirectMultiStep(d_model=d_model, heads=heads, layers=layers, ffn=ffn,
                               decoder_hidden=config.get("decoder_hidden", decoder))
    from experiments.transformer_scale_latent_v1.configs import TRANSITION_HIDDEN
    return LatentAutoregressive(d_model=d_model, heads=heads, layers=layers, ffn=ffn,
                                transition_hidden=TRANSITION_HIDDEN[d_model],
                                decoder_hidden=config.get("latent_decoder_hidden", decoder))


def run_batch(model, kind: str, tokens: Tokens, batch, stats,
              horizon: int, device) -> torch.Tensor:
    command = batch.command if torch.is_tensor(batch.command) \
        else torch.tensor(batch.command, dtype=torch.float32)
    elapsed = batch.elapsed if torch.is_tensor(batch.elapsed) \
        else torch.tensor(batch.elapsed, dtype=torch.float32)
    command = command.to(device)
    elapsed = elapsed.to(device)
    if kind == "latent_autoregressive":
        output = model(batch.window.to(device), batch.token_valid.to(device), command, horizon,
                       elapsed)
    else:
        output = model(batch.window.to(device), batch.token_valid.to(device), command, horizon)
    targets = encode_targets(batch, stats)
    masks = stepped_masks(batch)
    process = process_loss(output["steps"], targets, masks, device)
    terminal = terminal_loss(output, tokens, batch.index, batch.ends, batch.anchor, stats)
    return 0.5 * process + 0.5 * terminal


def _take_multistep(panel: "C.CachedMultiStep", rows: np.ndarray) -> "C.CachedMultiStep":
    selection = np.asarray(rows, dtype=np.int64)
    return C.CachedMultiStep(
        window=panel.window[torch.tensor(selection)], token_valid=panel.token_valid[torch.tensor(selection)],
        command=panel.command[torch.tensor(selection)], elapsed=panel.elapsed[torch.tensor(selection)],
        anchor=panel.anchor[selection], future={k: v[selection] for k, v in panel.future.items()},
        step_mask=panel.step_mask[selection], geometry_mask=panel.geometry_mask[selection],
        horizon=panel.horizon, index=panel.index[selection], ends=panel.ends[selection],
        root=panel.root[selection])


def evaluate(model, kind: str, tokens: Tokens, panel: "C.CachedMultiStep",
             stats, horizon: int, device, chunk: int = 256) -> Dict[str, object]:
    """Same-protocol validation: eval mode, fixed panel, per-step masked geometry error in cm."""

    model.eval()
    per_step: List[List[float]] = [[] for _ in range(horizon)]
    losses = []
    with torch.no_grad():
        for start in range(0, len(panel), chunk):
            stop = min(start + chunk, len(panel))
            batch = _take_multistep(panel, np.arange(start, stop))
            loss = run_batch(model, kind, tokens, batch, stats, horizon, device)
            losses.append(float(loss))
            prediction = decode_geometry_batch(
                _geometry_output(model, kind, batch, horizon, device), batch.anchor, stats)
            truth = batch.future["geometry"]
            keep = (batch.geometry_mask > 0) & (batch.step_mask > 0)[..., None]
            for step in range(horizon):
                columns_r0 = [0, 1]
                mask = keep[:, step, 0]
                if not mask.any():
                    continue
                error = np.hypot(prediction[:, step, 0] - truth[:, step, 0],
                                 prediction[:, step, 1] - truth[:, step, 1])
                per_step[step].extend(error[mask].tolist())
    model.train()
    return {"loss": float(np.mean(losses)) if losses else None,
            "step_p50_cm": [float(np.median(v) * 100) if v else None for v in per_step],
            "step_windows": [int(len(v)) for v in per_step]}


def _geometry_output(model, kind: str, batch: D.MultiStepBatch, horizon: int,
                     device) -> np.ndarray:
    """The standardised geometry output of one batch, in eval mode, on the CPU."""

    command = torch.tensor(batch.command, dtype=torch.float32, device=device)
    elapsed = torch.tensor(batch.elapsed, dtype=torch.float32, device=device)
    with torch.no_grad():
        if kind == "latent_autoregressive":
            output = model(batch.window.to(device), batch.token_valid.to(device), command,
                           horizon, elapsed)
        else:
            output = model(batch.window.to(device), batch.token_valid.to(device), command, horizon)
    return output["steps"]["geometry"].cpu().numpy().astype(np.float64)


def train(kind: str, config: Dict[str, object], seed: int, *, min_future_default: int = 1,
          horizon: int = 8,
          updates: int = 3200, batch_size: int = 128, lr: float = 3e-4,
          validate_every: int = 200, threads: int = 0, out_dir: Path = OUT,
          tier: str = "d1", tag: str = None, log: bool = True) -> dict:
    if threads:
        torch.set_num_threads(threads)
    out_dir = Path(out_dir).resolve()
    set_all_seeds(seed)
    device = torch.device("cpu")
    tokens = Tokens(D.TIER_SPLITS[tier])
    # repair 2.3: the corpus **layout** is indexed once, but the sample is drawn fresh. The previous
    # harness fixed 4096 windows at start-up and replayed them for every epoch, and the balanced
    # sampler took the first record of each root, so seven of eight command variants per root were
    # never seen.
    training_index = D.RootRecordIndex(tokens, ["d0", "d1ext"])
    index, ends = training_index.draw(4096, np.random.default_rng(seed + 1), horizon=horizon,
                                      min_future=min(min_future_default, horizon))
    val_index, val_ends = D.sample_windows(tokens, ["val"], 1024, seed=seed + 2,
                                           horizon=horizon, min_future=1)
    # the transform statistics come from a large fixed draw so they do not wobble epoch to epoch
    fit_index, fit_ends = D.sample_windows(tokens, ["d0", "d1ext"], 8192, seed=seed + 7,
                                           horizon=horizon, min_future=1)
    stats = fit_statistics(tokens, fit_index, fit_ends, horizon)
    train_panel = C.build_multistep_cache(tokens, index, ends, horizon)
    val_panel = C.build_multistep_cache(tokens, val_index, val_ends, horizon)
    coverage = {"epochs": [], "cumulative_windows": 0, "cumulative_unique_roots": 0,
                "cumulative_unique_records": 0, "cumulative_unique_record_prefix_pairs": 0}
    seen_roots: set = set()
    seen_records: set = set()
    seen_pairs: set = set()
    # record the initial draw too, so coverage is not understated by one epoch
    coverage["epochs"].append(D.sample_statistics(tokens, index, ends, horizon))
    coverage["cumulative_windows"] += int(len(index))
    seen_roots.update(str(tokens.roots[i]) for i in index)
    seen_records.update(str(tokens.records[i]) for i in index)
    seen_pairs.update(f"{tokens.records[i]}@{int(e)}" for i, e in zip(index, ends))
    model = build_model(kind, config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=updates,
                                                          eta_min=lr * 0.05)
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(index))
    curves: List[dict] = []
    best = {"loss": float("inf"), "update": -1, "state": None}
    checkpoint_path = out_dir / "checkpoints" / f"{(tag or kind)}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cursor = 0
    model.train()
    for update in range(1, updates + 1):
        if cursor + batch_size > len(order):
            # a new epoch: draw a **new** sample of prefixes, not a reshuffle of the same windows
            cursor = 0
            index, ends = training_index.draw(4096, generator, horizon=horizon,
                                              min_future=min(min_future_default, horizon))
            train_panel = C.build_multistep_cache(tokens, index, ends, horizon)
            order = generator.permutation(len(index))
            epoch_stats = D.sample_statistics(tokens, index, ends, horizon)
            coverage["epochs"].append(epoch_stats)
            coverage["cumulative_windows"] += epoch_stats["windows"]
            seen_roots.update(str(tokens.roots[i]) for i in index)
            seen_records.update(str(tokens.records[i]) for i in index)
            seen_pairs.update(f"{tokens.records[i]}@{int(e)}" for i, e in zip(index, ends))
        rows = order[cursor:cursor + batch_size]
        cursor += batch_size
        batch = _take_multistep(train_panel, rows)
        loss = run_batch(model, kind, tokens, batch, stats, horizon, device)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
        if update % 400 == 0 or update == updates:
            # atomic periodic write: an interrupted run must not lose the whole night
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({"state_dict": model.state_dict(), "kind": kind, "config": config,
                        "stats": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                  for k, v in stats.items()},
                        "seed": seed, "horizon": horizon, "updates": updates, "lr": lr,
                        "tier": tier, "parameters": parameter_breakdown(model),
                        "update": update, "optimiser": optimiser.state_dict(),
                        "scheduler": schedule.state_dict(),
                        "rng_numpy": generator.bit_generator.state,
                        "torch_rng": torch.get_rng_state(),
                        "sampler": "RootRecordIndex.draw per epoch",
                        "coverage": {**{k: v for k, v in coverage.items() if k != "epochs"},
                                     "epochs": len(coverage["epochs"]),
                                     "unique_roots_seen": len(seen_roots),
                                     "unique_records_seen": len(seen_records),
                                     "unique_record_prefix_seen": len(seen_pairs)},
                        "resume_kind": "optimiser + scheduler + rng + sampler state"}, temporary)
            temporary.replace(checkpoint_path)
        if update % validate_every == 0 or update == updates:
            report = evaluate(model, kind, tokens, val_panel, stats, horizon, device)
            curves.append({"update": update, "train_total": float(loss.detach()),
                           "val_total": report["loss"], "val_step_p50_cm": report["step_p50_cm"]})
            if report["loss"] is not None and report["loss"] < best["loss"]:
                best = {"loss": report["loss"], "update": update,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            if log:
                print(f"  [{kind}|s{seed}|h{horizon}] {update}/{updates} "
                      f"train={float(loss.detach()):.5f} val={report['loss']:.5f} "
                      f"step1={report['step_p50_cm'][0]} ({time.perf_counter() - started:.0f}s)",
                      flush=True)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    name = tag or f"{kind}_h{horizon}_lr{lr:g}_seed{seed}"
    path = out_dir / "checkpoints" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "kind": kind, "config": config,
                "stats": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in stats.items()},
                "seed": seed, "horizon": horizon, "updates": updates, "lr": lr,
                "tier": tier, "parameters": parameter_breakdown(model),
                "best_val": best["loss"], "best_update": best["update"],
                "train_windows": int(len(index)), "val_windows": int(len(val_index)),
                "coverage": {**{k: v for k, v in coverage.items() if k != "epochs"},
                             "epochs": len(coverage["epochs"]),
                             "unique_roots_seen": len(seen_roots),
                             "unique_records_seen": len(seen_records),
                             "unique_record_prefix_seen": len(seen_pairs)}}, path)
    return {"name": name, "kind": kind, "seed": seed, "horizon": horizon, "updates": updates,
            "lr": lr, "parameters": parameter_breakdown(model), "best_val": best["loss"],
            "best_update": best["update"], "seconds": time.perf_counter() - started,
            "curves": curves,
            "checkpoint": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT))
            else str(path),
            "config": config,
            "coverage": {**{k: v for k, v in coverage.items() if k != "epochs"},
                         "epochs": len(coverage["epochs"]),
                         "unique_roots_seen": len(seen_roots),
                         "unique_records_seen": len(seen_records),
                         "unique_record_prefix_seen": len(seen_pairs)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="direct_multistep")
    parser.add_argument("--size", default="160,4,4,640,320")
    parser.add_argument("--latent-decoder", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--updates", type=int, default=3200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    size = [int(v) for v in args.size.split(",")]
    config = {"size": size, "decoder_hidden": size[4], "latent_decoder_hidden": args.latent_decoder}
    result = train(args.kind, config, args.seed, horizon=args.horizon, updates=args.updates,
                   lr=args.lr, threads=args.threads, out_dir=Path(args.out), tag=args.tag)
    print(json.dumps({k: v for k, v in result.items() if k != "curves"}, indent=1))
    print("TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
