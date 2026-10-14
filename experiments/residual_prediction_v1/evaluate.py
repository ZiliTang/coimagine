"""Offline evaluation for the residual round.

Decodes a checkpoint with the **same** :func:`decode_geometry` the online API uses, then reports the
items the task book requires, always separated along the axes that were conflated before:

* regime: ``t=0`` (pre-execution) versus ``t>0`` (a real observed prefix);
* target: next step (one control period) versus terminal (the option's exit instant T);
* split: train / val / test, never mixed;
* robot: both robots are reported separately, never averaged into one number;
* motion: robot **translation** speed bins, so a near-static majority cannot hide a moving failure;
* counts: windows **and** independent roots for every cell.

Baselines are recomputed under the same masks and the same reference rule: P0 holds the current
observation (and when the current patch is invisible it holds the fixed origin, matching the residual
reference), P1 looks up the train statistics of (command signature, elapsed bin).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import Tokens  # noqa: E402
from experiments.night_sequence_v1.models import build_model, parameter_count  # noqa: E402
from experiments.night_sequence_v1.train import MODEL_KWARGS  # noqa: E402
from experiments.residual_prediction_v1 import data as D  # noqa: E402
from experiments.residual_prediction_v1.predict import spec_from_descriptor  # noqa: E402
from experiments.residual_prediction_v1.targets import (  # noqa: E402
    GeometryTransform,
    TransformSpec,
    decode_geometry,
    motion_bin,
    visibility_transition,
)

OUT = ROOT / "outputs/residual_prediction_v1"
COLUMNS = [0, 1, 4, 5]


def geometry_transform_from_descriptor(descriptor: Dict[str, object]) -> GeometryTransform:
    midpoint = spec_from_descriptor(descriptor["midpoint"])
    return GeometryTransform(midpoint=midpoint,
                             normal_mean=np.asarray(descriptor["normal_mean"]),
                             normal_std=np.asarray(descriptor["normal_std"]),
                             version=str(descriptor.get("version", "unknown")))


def decode_checkpoint_on(tokens: Tokens, panel: Dict[str, np.ndarray], checkpoint: Path,
                         chunk: int = 512) -> Dict[str, np.ndarray]:
    """Predict and decode a panel in physical units, for both the next-step and terminal heads."""

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    transform = geometry_transform_from_descriptor(saved["geometry_transform"])
    stats = saved["normalisation"]
    model = build_model(saved["model_kind"], **saved["model_kwargs"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    parts: Dict[str, List[np.ndarray]] = {}
    for start in range(0, len(panel["index"]), chunk):
        stop = min(start + chunk, len(panel["index"]))
        batch = D.build_batch(tokens, panel["index"][start:stop], panel["end"][start:stop])
        with torch.no_grad():
            output = model(batch.window, batch.token_valid)
        next_raw = output["process_geometry"].numpy().astype(np.float64)
        term_raw = output["terminal_geometry"].numpy().astype(np.float64)
        parts.setdefault("next", []).append(decode_geometry(next_raw, batch.reference, transform))
        parts.setdefault("term", []).append(decode_geometry(term_raw, batch.reference, transform))
        parts.setdefault("reference", []).append(batch.reference)
        parts.setdefault("end", []).append(batch.ends)
        parts.setdefault("speed", []).append(batch.current_speed)
        parts.setdefault("visible_now", []).append(batch.current_visible.astype(float))
        parts.setdefault("root", []).append(np.asarray([str(r) for r in
                                                        tokens.roots[batch.index]]))
        for target, key in (("next", "process_geometry"), ("term", "terminal_geometry")):
            parts.setdefault(f"{target}_mask", []).append(
                batch.masks[key].numpy()[:, COLUMNS].astype(float))
            parts.setdefault(f"{target}_truth", []).append(
                (batch.next_mid if target == "next" else batch.term_mid))
    return {key: np.concatenate(values) for key, values in parts.items()}


def error_table(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                speed: np.ndarray, visible_now: np.ndarray, roots: np.ndarray,
                end: np.ndarray) -> Dict[str, object]:
    """Masked p50 / p90 / RMSE per robot, split by regime and by motion bin."""

    out: Dict[str, object] = {}
    regimes = {"t=0": end == 0, "t>0": end > 0}
    for regime, selector in regimes.items():
        block: Dict[str, object] = {}
        for robot in (0, 1):
            columns = [0, 1] if robot == 0 else [2, 3]
            keep = selector & (mask[:, columns[0]] > 0) & (mask[:, columns[1]] > 0)
            error = np.hypot(prediction[:, columns[0]] - truth[:, columns[0]],
                             prediction[:, columns[1]] - truth[:, columns[1]])
            entry = _summary(error, keep, roots)
            bins = motion_bin(speed[:, robot])
            entry["by_motion"] = {name: _summary(error, keep & (bins == name), roots)
                                  for name in ("near_static(<0.01)", "slow(0.01-0.05)",
                                               "moving(>=0.05)")}
            # per-robot target visibility: the x column of each robot carries the mask
            target_visible = np.stack([mask[:, 0], mask[:, 2]], axis=1) > 0
            transitions = visibility_transition(visible_now.astype(float), target_visible, robot)
            entry["by_visibility"] = {name: _summary(error, keep & (transitions == name), roots)
                                      for name in ("sustained_visible", "became_visible",
                                                   "became_invisible", "sustained_invisible")}
            block[f"r{robot}"] = entry
        out[regime] = block
    return out


def _summary(error: np.ndarray, keep: np.ndarray, roots: np.ndarray) -> Dict[str, object]:
    if not keep.any():
        return {"p50_cm": None, "p90_cm": None, "rmse_cm": None, "windows": 0, "roots": 0}
    values = error[keep]
    return {"p50_cm": float(np.median(values) * 100), "p90_cm": float(np.quantile(values, 0.9) * 100),
            "rmse_cm": float(np.sqrt(np.mean(values ** 2)) * 100),
            "windows": int(keep.sum()), "roots": int(len(set(roots[keep].tolist())))}


def hold_baseline(reference: np.ndarray, truth: np.ndarray, mask: np.ndarray,
                  speed: np.ndarray, visible_now: np.ndarray, roots: np.ndarray,
                  end: np.ndarray) -> Dict[str, object]:
    """P0 under the residual reference rule: visible -> current midpoint, invisible -> origin."""

    return error_table(reference, truth, mask, speed, visible_now, roots, end)


def paired_root_bootstrap(roots: np.ndarray, error_a: np.ndarray, error_b: np.ndarray,
                          keep: np.ndarray, *, draws: int = 2000, seed: int = 0) -> Dict[str, object]:
    roots = roots[keep]
    difference = (error_a - error_b)[keep]
    unique = np.unique(roots)
    by_root = {root: np.where(roots == root)[0] for root in unique.tolist()}
    generator = np.random.default_rng(seed)
    values = np.empty(draws)
    for index in range(draws):
        picked = generator.choice(len(unique), size=len(unique), replace=True)
        selection = np.concatenate([by_root[unique[p]] for p in picked])
        values[index] = float(np.median(difference[selection]))
    return {"unit": "root", "draws": draws, "roots": int(len(unique)),
            "observed_median_difference_cm": float(np.median(difference) * 100),
            "ci95_low_cm": float(np.quantile(values, 0.025) * 100),
            "ci95_high_cm": float(np.quantile(values, 0.975) * 100),
            "fraction_draws_below_zero": float((values < 0).mean()),
            "interpretation": "negative favours the first model"}


def inference_cost(checkpoint: Path, *, repeats: int = 20) -> Dict[str, object]:
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(saved["model_kind"], **saved["model_kwargs"])
    model.eval()
    window = torch.zeros(1, 33, 57)
    valid = torch.ones(1, 33)
    with torch.no_grad():
        for _ in range(3):
            model(window, valid)
        started = time.perf_counter()
        for _ in range(repeats):
            model(window, valid)
        latency = (time.perf_counter() - started) / repeats
    return {"parameters": parameter_count(model), "batch1_latency_ms": latency * 1000,
            "threads": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoints", default=str(OUT / "checkpoints"))
    parser.add_argument("--pattern", default="*.pt")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out_dir = Path(args.out)
    tokens = Tokens(D.TIER_SPLITS[args.tier])
    panel = D.load_panel(args.tier, args.split, out_dir)
    result: Dict[str, object] = {"tier": args.tier, "split": args.split,
                                 "panel_hash": D.panel_hash(panel),
                                 "windows": int(len(panel["index"])),
                                 "roots": int(len(set(str(r) for r in panel["root_id"]))),
                                 "models": {}}
    checkpoints = sorted(Path(args.checkpoints).glob(args.pattern))
    for checkpoint in checkpoints:
        if "single_head" in checkpoint.name:
            continue
        decoded = decode_checkpoint_on(tokens, panel, checkpoint)
        # the decoder returns the full 8-column geometry vector; the error tables work on the
        # 4 midpoint columns, so the same COLUMNS mapping is applied here as in every other consumer
        entry = {"checkpoint": str(checkpoint.relative_to(ROOT)),
                 "next": error_table(decoded["next"][:, COLUMNS], decoded["next_truth"],
                                     decoded["next_mask"],
                                     decoded["speed"], decoded["visible_now"], decoded["root"],
                                     decoded["end"]),
                 "terminal": error_table(decoded["term"][:, COLUMNS], decoded["term_truth"],
                                         decoded["term_mask"],
                                         decoded["speed"], decoded["visible_now"], decoded["root"],
                                         decoded["end"]),
                 "hold_next": hold_baseline(decoded["reference"], decoded["next_truth"],
                                            decoded["next_mask"], decoded["speed"],
                                            decoded["visible_now"], decoded["root"],
                                            decoded["end"]),
                 # the terminal head needs its own baseline: the next-step hold is a different
                 # quantity and using it in a terminal figure would be a mislabelled baseline
                 "hold_terminal": hold_baseline(decoded["reference"], decoded["term_truth"],
                                                decoded["term_mask"], decoded["speed"],
                                                decoded["visible_now"], decoded["root"],
                                                decoded["end"]),
                 "cost": inference_cost(checkpoint)}
        np.savez_compressed(out_dir / f"predictions_{checkpoint.stem}_{args.split}.npz",
                            next=decoded["next"], term=decoded["term"],
                            reference=decoded["reference"], end=decoded["end"],
                            root=decoded["root"], next_truth=decoded["next_truth"],
                            term_truth=decoded["term_truth"], next_mask=decoded["next_mask"],
                            term_mask=decoded["term_mask"])
        result["models"][checkpoint.stem] = entry
        moving = entry["next"]["t>0"]["r0"]["by_motion"]["moving(>=0.05)"]
        print(f"{checkpoint.stem:44s} next t>0 r0 p50="
              f"{entry['next']['t>0']['r0']['p50_cm']:.3f} cm "
              f"(hold {entry['hold_next']['t>0']['r0']['p50_cm']:.3f}, moving {moving['p50_cm']:.3f})",
              flush=True)
    (out_dir / f"main_results_{args.split}.json").write_text(
        json.dumps(result, indent=1, sort_keys=True, default=float))
    print("EVALUATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
