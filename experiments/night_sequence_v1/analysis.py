"""Round-5 analysis: baselines, model predictions, metrics, per-sample predictions.

    python experiments/night_sequence_v1/analysis.py --tier d1

Produces, under ``outputs/night_sequence_v1/analysis/<tier>/``:

* ``metrics.json``            -- every required metric, separated into the pre-execution (t=0) and
                                 in-execution regimes, with sample and independent-root counts;
* ``predictions_<tag>.npz``   -- per-sample predictions keyed by (root, record, end), never just a
                                 summary;
* ``action_sensitivity.json`` -- same-root eight-action terminal differences and rank correlation
                                 with a permutation control, at t=0 only (in-execution histories
                                 already differ, so it is not a same-state counterfactual).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import GROUPS, Tokens, build_example  # noqa: E402
from experiments.night_sequence_v1.evaluate import (  # noqa: E402
    GroupedStatistics,
    fast_targets,
    build_eval_set,
    evaluate_predictions,
    grouped_predictions_batch,
    hold_predictions_batch,
)
from experiments.night_sequence_v1.models import build_model  # noqa: E402
from experiments.night_sequence_v1.sequence_metrics import (  # noqa: E402
    action_sensitivity,
    percentiles,
)
from experiments.night_sequence_v1.train import MODEL_KWARGS  # noqa: E402
from experiments.night_sequence_v1.evaluate import ELAPSED_BINS  # noqa: E402

DATA = ROOT / "outputs/night_sequence_v1"
RUNS = DATA / "runs"
ANALYSIS = DATA / "analysis"
TIER_SPLITS = {"d0": ["d0", "val", "test"],
               "d1": ["d0", "d1ext", "val", "test"],
               "d2": ["d0", "d1ext", "d2ext", "val", "test"]}
KIND = {name: k for name, _w, k in GROUPS}


def targets_and_masks(tokens: Tokens, index: np.ndarray, end: np.ndarray):
    examples = [build_example(tokens, int(i), int(e)) for i, e in zip(index, end)]
    targets, masks = {}, {}
    for name, width, kind in GROUPS:
        targets[name] = np.stack([e["targets"][name] for e in examples]).astype(
            np.int64 if kind == "multiclass" else np.float32)
        masks[name] = np.stack([e["masks"][name] for e in examples]).astype(np.float32)
    return targets, masks


def model_predictions(checkpoint: Path, tokens: Tokens, index: np.ndarray, end: np.ndarray,
                      batch: int = 512):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(saved["model_kind"], **saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    normalisation = saved["normalisation"]
    out: Dict[str, List[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(index), batch):
            stop = min(start + batch, len(index))
            examples = [build_example(tokens, int(i), int(e))
                        for i, e in zip(index[start:stop], end[start:stop])]
            window = torch.tensor(np.stack([e["window"] for e in examples]), dtype=torch.float32)
            valid = torch.tensor(np.stack([e["token_valid"] for e in examples]),
                                 dtype=torch.float32)
            outputs = model(window, valid)
            for name, _w, kind in GROUPS:
                value = outputs[name].numpy()
                if kind == "continuous":
                    stats = normalisation[name]
                    value = value * stats["std"] + stats["mean"]
                elif kind == "multiclass":
                    value = value.argmax(axis=1).astype(np.int64)
                out.setdefault(name, []).append(value)
    return {name: np.concatenate(chunks) for name, chunks in out.items()}, normalisation


def baseline_predictions(tokens: Tokens, index: np.ndarray, end: np.ndarray,
                         statistics: GroupedStatistics, mode: str,
                         remaining_table=None, exit_table=None):
    if mode == "p0":
        return hold_predictions_batch(tokens, index, end, remaining_table or {}, exit_table or {})
    return grouped_predictions_batch(statistics, tokens, index, end)


def fit_hold_tables(tokens: Tokens, train_splits, statistics: GroupedStatistics):
    """The elapsed-bin train median of the remaining duration and train majority of the exit class."""

    rows = np.where(np.isin(tokens.split_of, list(train_splits)))[0]
    generator = np.random.default_rng(7)
    count = min(len(rows) * 6, 120000)
    chosen = rows[generator.integers(0, len(rows), size=count)]
    steps = tokens.steps[chosen]
    ends = np.minimum((generator.random(count) * np.maximum(1, steps - 1)).astype(np.int64),
                      np.maximum(0, steps - 2))
    targets = fast_targets(tokens, chosen, ends)
    bins = (ends * 0.05 / 0.25).astype(np.int64)
    remaining, exit_class = {}, {}
    for value in np.unique(bins):
        selection = bins == value
        remaining[int(value)] = float(np.median(targets["terminal_remaining"][selection]))
        values, counts = np.unique(targets["terminal_exit_class"][selection], return_counts=True)
        exit_class[int(value)] = int(values[np.argmax(counts)])
    return remaining, exit_class


def grouped_keys(tokens: Tokens, index: np.ndarray, end: np.ndarray) -> Dict[str, np.ndarray]:
    elapsed = end * 0.05
    bins = [f"{ELAPSED_BINS[i]:.2f}-{ELAPSED_BINS[i + 1]:.2f}" for i in range(len(ELAPSED_BINS) - 1)]
    bin_index = np.clip((elapsed / 0.25).astype(int), 0, len(bins) - 1)
    stages = np.array([int(tokens.token_arrays[tokens.split_of[i]][i - tokens.offsets[tokens.split_of[i]], int(e), 49] * 8) for i, e in zip(index, end)])
    return {"elapsed_bin": np.asarray([bins[b] for b in bin_index]),
            "stage": np.asarray([f"stage{s}" for s in stages]),
            "regime": np.asarray(["t0" if e == 0 else "in_execution" for e in end])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--out", default=str(ANALYSIS))
    args = parser.parse_args()
    out_dir = Path(args.out) / args.tier
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = Tokens(TIER_SPLITS[args.tier])
    train_splits = {"d0": ["d0"], "d1": ["d0", "d1ext"],
                    "d2": ["d0", "d1ext", "d2ext"]}[args.tier]
    statistics = GroupedStatistics(tokens, train_splits)
    remaining_table, exit_table = fit_hold_tables(tokens, train_splits, statistics)
    payload: Dict[str, object] = {"tier": args.tier, "splits": TIER_SPLITS[args.tier],
                                 "models": {}, "baselines": {}}
    sets = {split: build_eval_set(tokens, split) for split in ("val", "test")}
    for split in ("val", "test"):
        index_all = np.concatenate([sets[split]["zero_index"], sets[split]["exec_index"]])
        end_all = np.concatenate([sets[split]["zero_end"], sets[split]["exec_end"]])
        targets, masks = targets_and_masks(tokens, index_all, end_all)
        groups = grouped_keys(tokens, index_all, end_all)
        zero = end_all == 0
        for mode, label in (("p0", "P0_hold"), ("p1", "P1_grouped")):
            predictions = baseline_predictions(tokens, index_all, end_all, statistics, mode,
                                               remaining_table, exit_table)
            roots_all = tokens.roots[index_all]
            report = {"all": evaluate_predictions(targets, masks, predictions,
                                                  binary_kind="probability", groups=groups,
                                                  roots=roots_all),
                      "t0": evaluate_predictions(
                          {k: v[zero] for k, v in targets.items()},
                          {k: v[zero] for k, v in masks.items()},
                          {k: v[zero] for k, v in predictions.items()},
                          binary_kind="probability",
                          groups={k: v[zero] for k, v in groups.items()},
                          roots=roots_all[zero])}
            payload["baselines"][f"{label}|{split}"] = report
            np.savez_compressed(out_dir / f"predictions_{label}_{split}.npz",
                                index=index_all, end=end_all, split=np.array([split]),
                                root_id=tokens.roots[index_all],
                                record_id=tokens.records[index_all], **{
                                    f"pred_{k}": v for k, v in predictions.items()})
            print(f"{label}|{split} done", flush=True)
        for checkpoint in sorted(RUNS.glob(f"{args.tier}_*_seed*.pt")):
            tag = checkpoint.stem
            predictions, normalisation = model_predictions(checkpoint, tokens, index_all, end_all)
            roots_all = tokens.roots[index_all]
            report = {"all": evaluate_predictions(targets, masks, predictions,
                                                  binary_kind="logit", groups=groups,
                                                  roots=roots_all),
                      "t0": evaluate_predictions(
                          {k: v[zero] for k, v in targets.items()},
                          {k: v[zero] for k, v in masks.items()},
                          {k: v[zero] for k, v in predictions.items()},
                          binary_kind="logit",
                          groups={k: v[zero] for k, v in groups.items()},
                          roots=roots_all[zero])}
            payload["models"][f"{tag}|{split}"] = report
            np.savez_compressed(out_dir / f"predictions_{tag}_{split}.npz",
                                index=index_all, end=end_all, split=np.array([split]),
                                root_id=tokens.roots[index_all],
                                record_id=tokens.records[index_all], **{
                                    f"pred_{k}": v for k, v in predictions.items()})
            print(f"{tag}|{split} done", flush=True)
            # t=0 same-root eight-action counterfactual on the terminal midpoint
            zero_index = sets[split]["zero_index"]
            zero_end = sets[split]["zero_end"]
            terminal = {k: v[zero] for k, v in targets.items()}
            terminal_masks = {k: v[zero] for k, v in masks.items()}
            zero_predictions = {k: v[zero] for k, v in predictions.items()}
            roots = tokens.roots[zero_index]
            truth = terminal["terminal_geometry"][:, [0, 1, 4, 5]]
            predicted = zero_predictions["terminal_geometry"][:, [0, 1, 4, 5]]
            geometry_mask = terminal_masks["terminal_geometry"][:, [0, 1, 4, 5]]
            payload["models"].setdefault("action_sensitivity", {})[tag] = action_sensitivity(
                roots, truth, predicted, geometry_mask,
                ["r0_x", "r0_y", "r1_x", "r1_y"])
            # permutation control: shuffle the action labels inside each root and recompute. The
            # spread is unchanged by construction, so any surviving rank correlation would mean the
            # metric is not measuring the action correspondence at all.
            payload["models"]["action_sensitivity"][tag]["permuted"] = action_sensitivity(
                roots, truth, predicted, geometry_mask,
                ["r0_x", "r0_y", "r1_x", "r1_y"],
                permutation=np.random.default_rng(12345))
            # per-example midpoint error, keyed by root, for the paired bootstrap
            error = np.hypot(predicted[:, 0] - truth[:, 0], predicted[:, 1] - truth[:, 1])
            valid_mask = geometry_mask[:, 0] > 0
            # both regimes are stored: the prefix-conditioned and the pre-execution one now rank the
            # architectures differently, so a single number would hide the finding
            exec_mask = (zero_end > 0) & valid_mask
            # the in-execution paired errors come from the full evaluation set restricted to end>0
            roots_all = tokens.roots[index_all]
            exec_all = (end_all > 0) & (masks["terminal_geometry"][:, 0] > 0)
            error_all = np.hypot(predictions["terminal_geometry"][:, 0]
                                 - targets["terminal_geometry"][:, 0],
                                 predictions["terminal_geometry"][:, 1]
                                 - targets["terminal_geometry"][:, 1])
            payload["models"].setdefault("paired_errors", {})[tag] = {
                "roots": [str(r) for r in roots_all], "t0_error": error.tolist(),
                "t0_valid": valid_mask.tolist(),
                "exec_error": error_all.tolist(), "exec_valid": exec_all.tolist(),
                "t0_roots": [str(r) for r in roots]}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=float))
    print("ANALYSIS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
