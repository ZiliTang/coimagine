"""Small-set capacity check, done properly.

The previous round's version had three defects this module fixes: it sampled only **one** evaluation
instant per sequence, it reported "roots" that did not match the sequences it actually used, and it
did not mask the geometry error. Here:

* 1, 16 and 64 explicit **sequences** are chosen, each from a different root unless a multi-command
  set is explicitly requested (and then it is recorded);
* training windows are drawn from those sequences only;
* evaluation covers **every legal adjacent frame** of the selected sequences, not one random instant;
* the geometry error is masked, and it is reported for the pure single midpoint task as well as for
  the full multi-task objective, so a transfer loss can be told apart from a capacity limit.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import GROUPS, Tokens  # noqa: E402
from experiments.night_sequence_v1.models import (  # noqa: E402
    build_model,
    parameter_count,
)
from experiments.night_sequence_v1.train import MODEL_KWARGS  # noqa: E402
from experiments.residual_prediction_v1 import data as D  # noqa: E402
from experiments.residual_prediction_v1.single_head import (  # noqa: E402
    MidpointHead,
    build_spec,
    set_all_seeds,
)
from experiments.residual_prediction_v1.targets import (  # noqa: E402
    decode_prediction,
    encode_target,
)

OUT = ROOT / "outputs/residual_prediction_v1"


def select_sequences(tokens: Tokens, count: int, *, seed: int,
                     allow_multiple_commands: bool = False) -> np.ndarray:
    """``count`` sequences, one per distinct root unless explicitly allowed otherwise."""

    rows = np.where(np.isin(tokens.split_of, ["d0", "d1ext"]))[0]
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(rows))
    chosen: List[int] = []
    used_roots: set = set()
    for position in order:
        row = int(rows[position])
        root = str(tokens.roots[row])
        if not allow_multiple_commands and root in used_roots:
            continue
        chosen.append(row)
        used_roots.add(root)
        if len(chosen) >= count:
            break
    return np.asarray(chosen, dtype=np.int64)


def all_adjacent_frames(tokens: Tokens, sequence_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Every legal (sequence, end) pair for those sequences: end runs over 0 .. steps-2."""

    index: List[int] = []
    ends: List[int] = []
    for row in sequence_index:
        steps = int(tokens.steps[row])
        for end in range(0, max(1, steps - 1)):
            index.append(int(row))
            ends.append(end)
    return np.asarray(index, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _group_loss(name: str, output, target, mask):
    kind = D.KIND[name]
    if kind == "continuous":
        return (((output - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
    if kind == "binary":
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target,
                                                                   reduction="none")
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)
    loss = torch.nn.functional.cross_entropy(output, target.view(-1), reduction="none")
    return (loss * mask.view(-1)).sum() / mask.view(-1).sum().clamp(min=1.0)


def panel_normalisation(tokens: Tokens, panel: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Per-group scalar mean/std from the **fixed optimisation panel**, matching the round-6 scheme.

    Fitting these on the tiny set instead would make the transfer comparison depend on the set size
    rather than on the multi-task objective.
    """

    batch = D.build_batch(tokens, panel["index"], panel["end"])
    out: Dict[str, Dict[str, float]] = {}
    for name, _w, kind in GROUPS:
        if kind != "continuous":
            continue
        values = batch.targets[name].numpy()
        mask = batch.masks[name].numpy() > 0
        selected = values[mask]
        out[name] = {"mean": float(selected.mean()) if selected.size else 0.0,
                     "std": float(selected.std()) if selected.size and selected.std() > 1e-9 else 1.0}
    return out


def train_multi(tokens: Tokens, sequences: np.ndarray, spec, *, seed: int, updates: int,
                lr: float, batch_size: int = 32, dropout: float = 0.1,
                normalisation: Dict[str, Dict[str, float]] = None):
    """The full multi-task objective on a fixed small set of sequences."""

    set_all_seeds(seed)
    model = build_model("mlp_history", **MODEL_KWARGS["mlp_history"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=updates,
                                                          eta_min=lr * 0.05)
    generator = np.random.default_rng(seed)
    steps_array = tokens.steps[sequences]
    model.train()
    for _ in range(updates):
        pick = generator.integers(0, len(sequences), size=batch_size)
        ends = np.minimum((generator.random(batch_size)
                           * np.maximum(1, steps_array[pick] - 1)).astype(np.int64),
                          np.maximum(0, steps_array[pick] - 2))
        batch = D.build_batch(tokens, sequences[pick], ends)
        targets = {}
        for name, _w, kind in GROUPS:
            value = batch.targets[name]
            if kind == "continuous" and normalisation:
                stats = normalisation[name]
                value = (value - stats["mean"]) / stats["std"]
            targets[name] = value
        outputs = model(batch.window, batch.token_valid)
        process = torch.stack([_group_loss(n, outputs[n], targets[n], batch.masks[n])
                               for n in D.KIND if n.startswith("process")]).mean()
        terminal = torch.stack([_group_loss(n, outputs[n], targets[n], batch.masks[n])
                                for n in D.KIND if n.startswith("terminal")]).mean()
        loss = 0.5 * process + 0.5 * terminal
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
    model.eval()
    return model


def train_single(tokens: Tokens, sequences: np.ndarray, spec, *, seed: int, updates: int,
                 lr: float, batch_size: int = 32, dropout: float = 0.1) -> MidpointHead:
    """The pure single midpoint task on the same fixed small set."""

    set_all_seeds(seed)
    model = MidpointHead(dropout=dropout)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=updates,
                                                          eta_min=lr * 0.05)
    generator = np.random.default_rng(seed)
    steps_array = tokens.steps[sequences]
    model.train()
    for _ in range(updates):
        pick = generator.integers(0, len(sequences), size=batch_size)
        ends = np.minimum((generator.random(batch_size)
                           * np.maximum(1, steps_array[pick] - 1)).astype(np.int64),
                          np.maximum(0, steps_array[pick] - 2))
        batch = D.build_batch(tokens, sequences[pick], ends)
        encoded = encode_target(batch.next_mid, batch.reference, spec)
        target = torch.tensor(encoded, dtype=torch.float32)
        mask = torch.tensor(batch.next_mask, dtype=torch.float32)
        prediction = model(batch.window, batch.token_valid)
        loss = (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
    model.eval()
    return model


def evaluate_single(tokens: Tokens, index: np.ndarray, ends: np.ndarray, spec,
                    model: MidpointHead) -> Dict[str, object]:
    batch = D.build_batch(tokens, index, ends)
    with torch.no_grad():
        output = model(batch.window, batch.token_valid).numpy().astype(np.float64)
    prediction = decode_prediction(output, batch.reference, spec)
    keep = batch.next_mask[:, 0] > 0
    error = np.hypot(prediction[:, 0] - batch.next_mid[:, 0],
                     prediction[:, 1] - batch.next_mid[:, 1])
    hold = np.hypot(batch.reference[:, 0] - batch.next_mid[:, 0],
                    batch.reference[:, 1] - batch.next_mid[:, 1])
    return {"windows": int(len(index)), "supervised_r0": int(keep.sum()),
            "p50_cm": float(np.median(error[keep]) * 100) if keep.any() else None,
            "p90_cm": float(np.quantile(error[keep], 0.9) * 100) if keep.any() else None,
            "rmse_cm": float(np.sqrt(np.mean(error[keep] ** 2)) * 100) if keep.any() else None,
            "hold_p50_cm": float(np.median(hold[keep]) * 100) if keep.any() else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-multiple-commands", action="store_true")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out_dir = Path(args.out)
    tokens = Tokens(D.TIER_SPLITS[args.tier])
    # statistics from the fixed optimisation panel, not from the tiny set
    panel = D.load_panel(args.tier, "train", out_dir)
    panel_batch = D.build_batch(tokens, panel["index"], panel["end"])
    stats = {"truth": panel_batch.next_mid, "reference": panel_batch.reference,
             "mask": panel_batch.next_mask}
    spec = build_spec(stats, "residual", "per_column", "residual_per_column")
    shared_normalisation = panel_normalisation(tokens, panel)
    result = {"tier": args.tier, "updates": args.updates, "lr": args.lr,
              "spec": spec.descriptor(), "tasks": {}}
    for count in args.counts:
        sequences = select_sequences(tokens, count, seed=args.seed,
                                     allow_multiple_commands=args.allow_multiple_commands)
        index, ends = all_adjacent_frames(tokens, sequences)
        entry = {"sequences": int(len(sequences)),
                 "distinct_roots": int(len(set(str(tokens.roots[s]) for s in sequences))),
                 "evaluation_windows": int(len(index)),
                 "multi_command_per_root": bool(args.allow_multiple_commands),
                 "sequence_ids": [str(tokens.records[s]) for s in sequences]}
        started = time.perf_counter()
        single = train_single(tokens, sequences, spec, seed=args.seed, updates=args.updates,
                              lr=args.lr)
        entry["single_task"] = evaluate_single(tokens, index, ends, spec, single)
        entry["single_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        multi = train_multi(tokens, sequences, spec, seed=args.seed, updates=args.updates,
                            lr=args.lr, normalisation=shared_normalisation)
        entry["multi_task_geometry"] = evaluate_multi_geometry(tokens, index, ends, multi,
                                                               shared_normalisation)
        entry["multi_seconds"] = time.perf_counter() - started
        result["tasks"][f"n{count}"] = entry
        print(json.dumps({k: v for k, v in entry.items() if k != "sequence_ids"}), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"tiny_fit_{args.tier}.json").write_text(
        json.dumps(result, indent=1, sort_keys=True, default=float))
    print("TINY_FIT_OK")
    return 0


def evaluate_multi_geometry(tokens: Tokens, index: np.ndarray, ends: np.ndarray, model,
                            normalisation: Dict[str, Dict[str, float]] = None
                            ) -> Dict[str, object]:
    """The geometry error of the multi-task model on the same windows, masked the same way."""

    frames: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    references: List[np.ndarray] = []
    for start in range(0, len(index), 256):
        stop = min(start + 256, len(index))
        batch = D.build_batch(tokens, index[start:stop], ends[start:stop])
        with torch.no_grad():
            output = model(batch.window, batch.token_valid)
        raw = output["process_geometry"].numpy()
        if normalisation:
            stats = normalisation["process_geometry"]
            raw = raw * stats["std"] + stats["mean"]
        columns = [0, 1, 4, 5]
        frames.append(raw[:, columns])
        masks.append(batch.masks["process_geometry"].numpy()[:, columns])
        truths.append(batch.next_mid)
        references.append(batch.reference)
    prediction = np.concatenate(frames).astype(np.float64)
    mask = np.concatenate(masks)
    truth = np.concatenate(truths)
    reference = np.concatenate(references)
    keep = (mask[:, 0] > 0) & (mask[:, 1] > 0)
    error = np.hypot(prediction[:, 0] - truth[:, 0], prediction[:, 1] - truth[:, 1])
    hold = np.hypot(reference[:, 0] - truth[:, 0], reference[:, 1] - truth[:, 1])
    return {"windows": int(len(index)), "supervised_r0": int(keep.sum()),
            "p50_cm": float(np.median(error[keep]) * 100) if keep.any() else None,
            "hold_p50_cm": float(np.median(hold[keep]) * 100) if keep.any() else None,
            "note": "the multi-task model keeps the round-6 direct per-group standardisation; this "
                    "is the transfer-loss reference, not the residual predictor"}


if __name__ == "__main__":
    raise SystemExit(main())
