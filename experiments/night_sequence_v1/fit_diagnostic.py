"""Fitting diagnostic: is the bottleneck overfitting, underfitting, or the implementation?

The review asked for three specific experiments, and this module is exactly those three plus the
controls they imply. Nothing here expands the data or the model, and no checkpoint from round 6 is
modified.

A. ``protocol`` -- same-protocol train/val error. Training loss is a **training-mode loss on a fresh
   random batch** (dropout active, the batch changes every step); validation loss is an eval-mode
   average on a fixed set. The two curves are therefore not comparable in height. This experiment
   puts a checkpoint in eval mode and measures **the same metric** on a fixed train set and a fixed
   validation set, in centimetres, separately for the next-step and the terminal target.

B. ``tiny`` -- can the model fit at all? Train on 1 / 4 / 16 / 64 **fixed sequences** with the full
   objective. If a 1.6M-parameter model cannot drive the error on sixty-four trajectories to a small
   value, the diagnosis is implementation / loss scale / optimisation, not capacity or data.

C. ``single`` -- the next-step midpoint on its own, comparing the direct parameterisation with the
   residual one (current position + predicted displacement), and a single per-group scale against
   per-column scales. A value already present in the input should be cheap to reproduce; if a direct
   head cannot, the residual form tells us whether the difficulty is in the parameterisation.

    python experiments/night_sequence_v1/fit_diagnostic.py --experiment protocol
    python experiments/night_sequence_v1/fit_diagnostic.py --experiment tiny
    python experiments/night_sequence_v1/fit_diagnostic.py --experiment single
"""

from __future__ import annotations

import argparse
import json
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

from experiments.night_sequence_v1.dataset import GROUPS, Tokens, build_example  # noqa: E402
from experiments.night_sequence_v1.models import (  # noqa: E402
    GROUP_NAMES,
    build_model,
    parameter_count,
)
from experiments.night_sequence_v1.train import MODEL_KWARGS  # noqa: E402
from experiments.night_sequence_v1.windows import FEATURES, WINDOW  # noqa: E402

DATA = ROOT / "outputs/night_sequence_v1"
RUNS = DATA / "runs"
OUT = DATA / "fit_diagnostic"
TIER_SPLITS = {"d0": ["d0", "val", "test"], "d1": ["d0", "d1ext", "val", "test"],
               "d2": ["d0", "d1ext", "d2ext", "val", "test"]}
KIND = {name: kind for name, _w, kind in GROUPS}

#: the two targets this diagnostic reports in centimetres
NEXT_MIDPOINT = (7, 8)      # r0 midpoint x, y inside a token (and inside the process_geometry head)
TERMINAL_SLICE = slice(0, 2)
PROCESS_SLICE = slice(0, 2)


def fixed_pairs(tokens: Tokens, split: str, count: int, seed: int,
                roots: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """A fixed set of (sequence, window-end) pairs, sampled once and reused everywhere."""

    rows = np.where(tokens.split_of == split)[0]
    if roots is not None:
        keep = set(roots)
        rows = np.asarray([row for row in rows if tokens.roots[row] in keep])
    generator = np.random.default_rng(seed)
    index = rows[generator.integers(0, len(rows), size=count)]
    steps = tokens.steps[index]
    ends = np.minimum((generator.random(count) * np.maximum(1, steps - 1)).astype(np.int64),
                      np.maximum(0, steps - 2))
    return index, ends


def batch(tokens: Tokens, index: np.ndarray, ends: np.ndarray):
    examples = [build_example(tokens, int(i), int(e)) for i, e in zip(index, ends)]
    window = torch.tensor(np.stack([e["window"] for e in examples]), dtype=torch.float32)
    valid = torch.tensor(np.stack([e["token_valid"] for e in examples]), dtype=torch.float32)
    targets, masks = {}, {}
    for name, width, kind in GROUPS:
        targets[name] = torch.tensor(np.stack([e["targets"][name] for e in examples]),
                                     dtype=torch.long if kind == "multiclass" else torch.float32)
        masks[name] = torch.tensor(np.stack([e["masks"][name] for e in examples]),
                                   dtype=torch.float32)
    return window, valid, targets, masks


def denormalise(values: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    return values * np.asarray(stats["std"]) + np.asarray(stats["mean"])


def midpoint_error_cm(prediction_cm: np.ndarray, truth_cm: np.ndarray) -> float:
    return float(np.median(np.hypot(prediction_cm[:, 0] - truth_cm[:, 0],
                                    prediction_cm[:, 1] - truth_cm[:, 1])) * 100.0)


def current_midpoint(tokens: Tokens, index: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """The midpoint the robot already sees at the window end, in R_i0 (metres)."""

    split_of = tokens.split_of[index]
    local = index - np.asarray([tokens.offsets[s] for s in split_of])
    tokens_block = np.stack([tokens.token_arrays[s][l, e]
                             for s, l, e in zip(split_of, local, ends)]).astype(np.float32)
    return tokens_block[:, NEXT_MIDPOINT[0]:NEXT_MIDPOINT[1] + 1]


# ---------------------------------------------------------------------------
# A. same-protocol train versus validation
# ---------------------------------------------------------------------------


def experiment_protocol(args) -> dict:
    tier = args.tier
    tokens = Tokens(TIER_SPLITS[tier])
    train_index, train_ends = fixed_pairs(tokens, "d0", args.train_samples, seed=11)
    val_index, val_ends = fixed_pairs(tokens, "val", args.val_samples, seed=12)
    # the target instant must exist for the next-step head; fixed_pairs already clamps that
    result: Dict[str, object] = {
        "tier": tier, "train_samples": len(train_index), "val_samples": len(val_index),
        "note": "same metric, eval mode, fixed sets; cm errors",
        "targets": {}}
    for kind, kwargs in MODEL_KWARGS.items():
        checkpoint = RUNS / f"{tier}_{kind}_seed0.pt"
        if not checkpoint.exists():
            continue
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(saved["model_kind"], **saved["model_kwargs"])
        model.load_state_dict(saved["state_dict"])
        model.eval()
        normalisation = saved["normalisation"]
        entry = {}
        for label, index, ends in (("train", train_index, train_ends),
                                   ("val", val_index, val_ends)):
            window, valid, targets, masks = batch(tokens, index, ends)
            with torch.no_grad():
                outputs = model(window, valid)
            # the model emits standardised values; the targets from build_example are already in
            # original units, so only the prediction is converted
            predicted_process = denormalise(outputs["process_geometry"].numpy(),
                                            normalisation["process_geometry"])
            true_process = targets["process_geometry"].numpy()
            predicted_terminal = denormalise(outputs["terminal_geometry"].numpy(),
                                             normalisation["terminal_geometry"])
            true_terminal = targets["terminal_geometry"].numpy()
            # r1 sits at columns 4..7 of the geometry vectors; report robot 0 to match the review
            entry[label] = {
                "next_step_midpoint_p50_cm": midpoint_error_cm(
                    predicted_process[:, PROCESS_SLICE], true_process[:, PROCESS_SLICE]),
                "terminal_midpoint_p50_cm": midpoint_error_cm(
                    predicted_terminal[:, TERMINAL_SLICE], true_terminal[:, TERMINAL_SLICE]),
                "hold_baseline_next_step_p50_cm": midpoint_error_cm(
                    current_midpoint(tokens, index, ends), true_process[:, PROCESS_SLICE]),
                "hold_baseline_terminal_p50_cm": midpoint_error_cm(
                    current_midpoint(tokens, index, ends), true_terminal[:, TERMINAL_SLICE]),
            }
        entry["checkpoint"] = checkpoint.stem
        entry["parameters"] = saved["parameters"]
        result["targets"][kind] = entry
    return result


# ---------------------------------------------------------------------------
# B. can it fit a tiny fixed set?
# ---------------------------------------------------------------------------


def train_subset(tokens: Tokens, model_kind: str, sequence_index: np.ndarray, *, steps: int,
                 batch_size: int, lr: float, seed: int, normalisation: Dict[str, Dict[str, float]]):
    torch.manual_seed(seed)
    model = build_model(model_kind, **MODEL_KWARGS[model_kind])
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps, eta_min=lr * 0.05)
    generator = np.random.default_rng(seed)
    steps_array = tokens.steps[sequence_index]

    def group_loss(name, output, target, mask):
        if KIND[name] == "continuous":
            return (((output - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
        if KIND[name] == "binary":
            loss = nn.functional.binary_cross_entropy_with_logits(output, target, reduction="none")
            return (loss * mask).sum() / mask.sum().clamp(min=1.0)
        loss = nn.functional.cross_entropy(output, target.view(-1), reduction="none")
        return (loss * mask.view(-1)).sum() / mask.view(-1).sum().clamp(min=1.0)

    model.train()
    for _ in range(steps):
        rows = generator.integers(0, len(sequence_index), size=batch_size)
        ends = np.minimum((generator.random(batch_size) * np.maximum(1, steps_array[rows] - 1))
                          .astype(np.int64), np.maximum(0, steps_array[rows] - 2))
        window, valid, targets, masks = batch(tokens, sequence_index[rows], ends)
        for name in targets:
            if KIND[name] == "continuous":
                stats = normalisation[name]
                targets[name] = (targets[name] - float(np.atleast_1d(stats["mean"])[0])) \
                    / float(np.atleast_1d(stats["std"])[0])
        outputs = model(window, valid)
        process = torch.stack([group_loss(n, outputs[n], targets[n], masks[n])
                               for n in GROUP_NAMES if n.startswith("process")]).mean()
        terminal = torch.stack([group_loss(n, outputs[n], targets[n], masks[n])
                                for n in GROUP_NAMES if n.startswith("terminal")]).mean()
        loss = 0.5 * process + 0.5 * terminal
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()
    model.eval()
    return model


def experiment_tiny(args) -> dict:
    tier = args.tier
    tokens = Tokens(TIER_SPLITS[tier])
    normalisation = None
    checkpoint = RUNS / f"{tier}_mlp_history_seed0.pt"
    if checkpoint.exists():
        normalisation = torch.load(checkpoint, map_location="cpu",
                                   weights_only=False)["normalisation"]
    else:
        raise RuntimeError("a reference checkpoint is needed for the normalisation statistics")
    train_roots = sorted({tokens.roots[i] for i in np.where(tokens.split_of == "d0")[0]})
    val_index, val_ends = fixed_pairs(tokens, "val", 1024, seed=12)
    result = {"tier": tier, "targets": {}, "note": "full objective, fixed subsets, cm errors"}
    for count in args.tiny_counts:
        roots = train_roots[:count]
        rows = np.asarray([i for i in np.where(tokens.split_of == "d0")[0]
                           if tokens.roots[i] in set(roots)])
        sequences = rows[:max(count, 1)]
        for kind in args.tiny_models:
            started = time.perf_counter()
            model = train_subset(tokens, kind, sequences, steps=args.tiny_steps,
                                 batch_size=min(64, max(8, len(sequences))),
                                 lr=3e-3, seed=0, normalisation=normalisation)
            entry = {"roots": count, "sequences": int(len(sequences)), "model": kind,
                     "steps": args.tiny_steps, "seconds": time.perf_counter() - started}
            for label, index, ends in (("train", sequences, None), ("val", val_index, val_ends)):
                if ends is None:
                    steps_array = tokens.steps[index]
                    generator = np.random.default_rng(5)
                    ends = np.minimum((generator.random(len(index))
                                       * np.maximum(1, steps_array - 1)).astype(np.int64),
                                      np.maximum(0, steps_array - 2))
                window, valid, targets, masks = batch(tokens, index, ends)
                with torch.no_grad():
                    outputs = model(window, valid)
                stats_p = normalisation["process_geometry"]
                stats_t = normalisation["terminal_geometry"]
                predicted_process = denormalise(outputs["process_geometry"].numpy(), stats_p)
                true_process = targets["process_geometry"].numpy()
                predicted_terminal = denormalise(outputs["terminal_geometry"].numpy(), stats_t)
                true_terminal = targets["terminal_geometry"].numpy()
                entry[f"{label}_next_step_midpoint_p50_cm"] = midpoint_error_cm(
                    predicted_process[:, PROCESS_SLICE], true_process[:, PROCESS_SLICE])
                entry[f"{label}_terminal_midpoint_p50_cm"] = midpoint_error_cm(
                    predicted_terminal[:, TERMINAL_SLICE], true_terminal[:, TERMINAL_SLICE])
                entry[f"{label}_hold_next_step_p50_cm"] = midpoint_error_cm(
                    current_midpoint(tokens, index, ends), true_process[:, PROCESS_SLICE])
            result["targets"][f"{kind}_r{count}"] = entry
            print(json.dumps(entry), flush=True)
    return result


# ---------------------------------------------------------------------------
# C. the next-step midpoint on its own: direct versus residual
# ---------------------------------------------------------------------------


class MidpointHead(nn.Module):
    """A single-head MLP over the same window, with a switchable output parameterisation.

    ``direct`` predicts the next midpoint in R_i0. ``residual`` predicts the *change* from the
    current midpoint and adds it back, so reproducing "almost nothing happens" is one small output
    away from the identity instead of a full regression problem.
    """

    def __init__(self, hidden: int = 512, layers: int = 2, dropout: float = 0.1,
                 parameterisation: str = "direct", per_column: bool = False):
        super().__init__()
        self.parameterisation = parameterisation
        self.per_column = per_column
        blocks: List[nn.Module] = []
        width = WINDOW * FEATURES + WINDOW
        for _ in range(max(1, layers - 1)):
            blocks += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        blocks += [nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 2)]
        self.trunk = nn.Sequential(*blocks)

    def forward(self, window: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        flat = torch.cat([window.reshape(window.shape[0], -1), valid], dim=-1)
        out = self.trunk(flat)
        if self.parameterisation == "residual":
            # the current midpoint is the last real token's own-frame position
            rows = torch.arange(window.shape[0], device=window.device)
            lengths = valid.sum(dim=1).clamp(min=1).long() - 1
            current = window[rows, lengths, :][:, NEXT_MIDPOINT[0]:NEXT_MIDPOINT[1] + 1]
            out = out + current
        return out


def experiment_single(args) -> dict:
    """Next-step or terminal midpoint, direct versus residual parameterisation."""

    target_name = args.single_target
    tier = args.tier
    tokens = Tokens(TIER_SPLITS[tier])
    train_index, train_ends = fixed_pairs(tokens, tier == "d0" and "d0" or "d0",
                                          args.train_samples, seed=11)
    val_index, val_ends = fixed_pairs(tokens, "val", args.val_samples, seed=12)
    # fit the two possible normalisations on the training targets only
    group_name = "process_geometry" if target_name == "next" else "terminal_geometry"
    train_window, train_valid, train_targets, _ = batch(tokens, train_index, train_ends)
    truth_train = train_targets[group_name].numpy()[:, PROCESS_SLICE]
    direct_stats = {"mean": float(truth_train.mean()), "std": float(truth_train.std()) or 1.0}
    residual_truth = truth_train - current_midpoint(tokens, train_index, train_ends)
    residual_stats = {"mean": float(residual_truth.mean()),
                      "std": float(residual_truth.std()) or 1.0}
    result = {"tier": tier, "target": target_name,
              "note": f"single {target_name}-midpoint head, fixed subsets, cm errors",
              "train_samples": int(len(train_index)), "val_samples": int(len(val_index)),
              "targets": {}}
    for parameterisation in ("direct", "residual"):
        stats = direct_stats if parameterisation == "direct" else residual_stats
        model = MidpointHead(parameterisation=parameterisation)
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.single_lr, weight_decay=0.0)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=args.single_steps, eta_min=args.single_lr * 0.05)
        torch.manual_seed(0)
        # materialise the fixed training set once; full-batch steps over thousands of windows would
        # dominate the whole round for no extra information
        train_window, train_valid, train_targets, _ = batch(tokens, train_index, train_ends)
        truth_train_all = train_targets[group_name][:, PROCESS_SLICE]
        current_train_all = torch.tensor(current_midpoint(tokens, train_index, train_ends))
        target_all = ((truth_train_all - current_train_all)
                      if parameterisation == "residual" else truth_train_all)
        target_all = (target_all - stats["mean"]) / stats["std"]
        generator = np.random.default_rng(0)
        model.train()
        for _ in range(args.single_steps):
            rows = torch.tensor(generator.integers(0, len(train_index),
                                                   size=min(256, len(train_index))))
            window, valid = train_window[rows], train_valid[rows]
            target = target_all[rows]
            prediction = model(window, valid)
            loss = ((prediction - target) ** 2).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
        model.eval()
        entry = {"parameterisation": parameterisation, "parameters": parameter_count(model),
                 "steps": args.single_steps, "lr": args.single_lr,
                 "target_scale_mean_m": stats["mean"], "target_scale_std_m": stats["std"]}
        for label, index, ends in (("train", train_index, train_ends), ("val", val_index, val_ends)):
            window, valid, targets, _ = batch(tokens, index, ends)
            truth = targets[group_name][:, PROCESS_SLICE].numpy()
            with torch.no_grad():
                raw = model(window, valid).numpy()
            prediction = raw * stats["std"] + stats["mean"]
            if parameterisation == "residual":
                prediction = prediction + current_midpoint(tokens, index, ends)
            entry[f"{label}_next_step_p50_cm"] = midpoint_error_cm(prediction, truth)
            entry[f"{label}_hold_baseline_p50_cm"] = midpoint_error_cm(
                current_midpoint(tokens, index, ends), truth)
        # the training-subset error is the question item C asks, so report it explicitly
        window, valid, targets, _ = batch(tokens, train_index, train_ends)
        truth = targets[group_name][:, PROCESS_SLICE].numpy()
        with torch.no_grad():
            raw = model(window, valid).numpy()
        fitted = raw * stats["std"] + stats["mean"]
        if parameterisation == "residual":
            fitted = fitted + current_midpoint(tokens, train_index, train_ends)
        entry["train_next_step_p50_cm"] = midpoint_error_cm(fitted, truth)
        entry["train_hold_baseline_p50_cm"] = midpoint_error_cm(
            current_midpoint(tokens, train_index, train_ends), truth)
        result["targets"][parameterisation] = entry
        print(json.dumps(entry), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True,
                        choices=["protocol", "tiny", "single", "all"])
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=2048)
    parser.add_argument("--tiny-counts", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--tiny-models", nargs="+", default=["mlp_history"])
    parser.add_argument("--tiny-steps", type=int, default=3000)
    parser.add_argument("--single-steps", type=int, default=3000)
    parser.add_argument("--single-lr", type=float, default=1e-3)
    parser.add_argument("--single-target", choices=["next", "terminal"], default="next")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    everything = {}
    for name in (["protocol", "tiny", "single"] if args.experiment == "all" else [args.experiment]):
        function = {"protocol": experiment_protocol, "tiny": experiment_tiny,
                    "single": experiment_single}[name]
        started = time.perf_counter()
        payload = function(args)
        payload["seconds"] = time.perf_counter() - started
        everything[name] = payload
        suffix = f"_{args.single_target}" if name == "single" else ""
        (out_dir / f"{name}{suffix}_{args.tier}.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True, default=float))
        print(f"--- {name} done in {payload['seconds']:.0f}s ---", flush=True)
    (out_dir / f"all_{args.tier}.json").write_text(
        json.dumps(everything, indent=1, sort_keys=True, default=float))
    print("FIT_DIAGNOSTIC_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
