"""Evaluation for the multi-step round.

Reported separately, never merged:

* robot 0 and robot 1;
* pre-execution ``t=0`` and a real observed prefix ``t>0``;
* the four scene classes plus the two solo classes, taken from the generator's root identifier;
* robot translation speed and patch-visibility transition;
* steps 1 / 4 / 8 (inside the trained span) and 16 (outside it).

Two comparison statistics are always given side by side because they can disagree:
``difference_of_medians`` (each model's own median, then subtracted) and
``median_of_differences`` (paired per window, then median). The previous round showed the two can
point in opposite directions against the same baseline.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import Tokens  # noqa: E402
from experiments.transformer_scale_latent_v1 import cache as C  # noqa: E402
from experiments.transformer_scale_latent_v1 import data as D  # noqa: E402
from experiments.transformer_scale_latent_v1 import panels as P  # noqa: E402
from experiments.transformer_scale_latent_v1.models import (  # noqa: E402
    DirectMultiStep,
    LatentAutoregressive,
    parameter_breakdown,
)
from experiments.transformer_scale_latent_v1.train import (  # noqa: E402
    decode_geometry_batch,
    geometry_residual,
)

OUT = ROOT / "outputs/transformer_scale_latent_v1"
STEPS = (1, 4, 8, 16)
MOTION_THRESHOLD = 0.05
#: the midpoint columns inside the 8-column geometry vector
MIDPOINTS = [0, 1, 4, 5]
#: the four-column anchor order that lines up with MIDPOINTS
ANCHOR_ORDER = [0, 1, 4, 5]


def load_model(path: Path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    kind = saved.get("kind")
    if kind is None:
        return saved, None
    config = saved["config"]
    size = config["size"]
    d_model, heads, layers, ffn, decoder = size
    if kind == "direct_multistep":
        model = DirectMultiStep(d_model=d_model, heads=heads, layers=layers, ffn=ffn,
                                decoder_hidden=config.get("decoder_hidden", decoder))
    else:
        from experiments.transformer_scale_latent_v1.configs import TRANSITION_HIDDEN
        model = LatentAutoregressive(d_model=d_model, heads=heads, layers=layers, ffn=ffn,
                                     transition_hidden=TRANSITION_HIDDEN[d_model],
                                     decoder_hidden=config.get("latent_decoder_hidden", decoder))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return saved, model


def predict_steps(saved, model, tokens: Tokens, index: np.ndarray, ends: np.ndarray,
                  horizon: int, chunk: int = 128, freeze_latent: bool = False) -> Dict[str, np.ndarray]:
    """Decoded physical geometry at every step, plus the anchor and the marginal masks."""

    frames: List[np.ndarray] = []
    anchors: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(index), chunk):
            stop = min(start + chunk, len(index))
            panel = C.build_multistep_cache(tokens, index[start:stop], ends[start:stop], horizon)
            if saved["kind"] == "latent_autoregressive":
                steps, _ = model.rollout(panel.window, panel.token_valid, panel.command, horizon,
                                         panel.elapsed, freeze_latent=freeze_latent)
                raw = steps["geometry"].numpy().astype(np.float64)
            else:
                out = model(panel.window, panel.token_valid, panel.command, horizon)
                raw = out["steps"]["geometry"].numpy().astype(np.float64)
            decoded = decode_geometry_batch(raw, panel.anchor, saved["stats"])
            frames.append(decoded)
            anchors.append(panel.anchor)
    return {"geometry": np.concatenate(frames), "anchor": np.concatenate(anchors)}


def truth_and_masks(tokens: Tokens, index: np.ndarray, ends: np.ndarray,
                    horizon: int) -> Dict[str, np.ndarray]:
    panel = C.build_multistep_cache(tokens, index, ends, horizon)
    return {"truth": panel.future["geometry"], "geometry_mask": panel.geometry_mask,
            "step_mask": panel.step_mask, "speed": panel.future["velocity"],
            "anchor": panel.anchor}


def step_error(prediction: np.ndarray, truth: np.ndarray, geometry_mask: np.ndarray,
               step_mask: np.ndarray, step: int, robot: int) -> np.ndarray:
    """Per-window 2-D midpoint error at one step, NaN where the step is not supervised."""

    first, last = (0, 2) if robot == 0 else (4, 6)
    keep = (geometry_mask[:, step, first] > 0) & (step_mask[:, step] > 0)
    error = np.hypot(prediction[:, step, first] - truth[:, step, first],
                     prediction[:, step, first + 1] - truth[:, step, first + 1])
    return np.where(keep, error, np.nan)


def summarise(values: np.ndarray) -> Dict[str, object]:
    keep = np.isfinite(values)
    if not keep.any():
        return {"p50_cm": None, "p90_cm": None, "rmse_cm": None, "windows": 0}
    selected = values[keep]
    return {"p50_cm": float(np.median(selected) * 100),
            "p90_cm": float(np.quantile(selected, 0.9) * 100),
            "rmse_cm": float(np.sqrt(np.mean(selected ** 2)) * 100),
            "windows": int(keep.sum())}


def grouped(errors: Dict[int, np.ndarray], panel: Dict[str, np.ndarray], robot: int,
            tokens: Tokens) -> Dict[str, object]:
    """Every required cut of the same per-window error arrays."""

    end = panel["end"]
    scene = panel["scene"]
    regimes = {"t=0": end == 0, "t>0": end > 0}
    speed = np.zeros(len(end))
    for position, (i, e) in enumerate(zip(panel["index"], end)):
        split = tokens.split_of[i]
        local = int(i) - tokens.offsets[split]
        token = tokens.token_arrays[split][local, int(e)]
        speed[position] = float(np.hypot(token[3], token[4])) if robot == 0 else \
            float(np.hypot(token[21], token[22]))
    motion = np.where(speed < 0.01, "near_static(<0.01)",
                      np.where(speed < 0.05, "slow(0.01-0.05)", "moving(>=0.05)"))
    out: Dict[str, object] = {"by_step": {}, "by_regime": {}, "by_scene": {}, "by_motion": {}}
    for step in STEPS:
        if step not in errors:
            continue
        values = errors[step]
        out["by_step"][f"step{step}"] = summarise(values)
        out["by_step"][f"step{step}"]["roots"] = int(len(set(
            panel["root"][np.isfinite(values)].tolist())))
        for name, selector in regimes.items():
            out["by_regime"].setdefault(name, {})[f"step{step}"] = summarise(
                np.where(selector, values, np.nan))
        for scene_name in sorted(set(scene.tolist())):
            out["by_scene"].setdefault(scene_name, {})[f"step{step}"] = summarise(
                np.where(scene == scene_name, values, np.nan))
        for motion_name in ("near_static(<0.01)", "slow(0.01-0.05)", "moving(>=0.05)"):
            out["by_motion"].setdefault(motion_name, {})[f"step{step}"] = summarise(
                np.where(motion == motion_name, values, np.nan))
    return out


def hold_baseline(anchor: np.ndarray, truth: np.ndarray, geometry_mask: np.ndarray,
                  step_mask: np.ndarray, step: int, robot: int) -> np.ndarray:
    """P0 under the frozen reference rule: hold the prediction-start midpoint."""

    first, last = (0, 2) if robot == 0 else (4, 6)
    keep = (geometry_mask[:, step, first] > 0) & (step_mask[:, step] > 0)
    anchor_columns = [0, 1] if robot == 0 else [2, 3]
    error = np.hypot(anchor[:, anchor_columns[0]] - truth[:, step, first],
                     anchor[:, anchor_columns[1]] - truth[:, step, first + 1])
    return np.where(keep, error, np.nan)


#: 0.5 s of executed time per bin; expressed in whole 0.05 s steps so the fit and the lookup use
#: integer arithmetic and cannot disagree on a float32 boundary
ELAPSED_STEPS_PER_BIN = 10


def command_signature(command: np.ndarray) -> str:
    """The P1 key. Both robots' commands **and** both active flags are included."""

    c = np.asarray(command, dtype=np.float64)
    return (f"{c[0]:.3f}|{c[1]:.3f}|{c[2]:.3f}|{c[3]:.3f}|"
            f"{int(c[4] > 0.5)}|{int(c[5] > 0.5)}")


def elapsed_bin(end: int) -> int:
    """The executed-time bin, from the integer step index; identical in fit and lookup."""

    return min(int(end) // ELAPSED_STEPS_PER_BIN, 60)


def _group_key(command: np.ndarray, end: int) -> str:
    return f"{command_signature(command)}|{elapsed_bin(end)}"


def fit_p1(tokens: Tokens, index: np.ndarray, ends: np.ndarray, horizon: int,
           min_group: int = 8) -> Dict[str, object]:
    """P1 fitted on the **training split only**, per robot, on truly visible labels.

    For every (command, elapsed bin, step, robot) group this stores the median residual
    displacement of the **visible** labels. A group with fewer than ``min_group`` labels is dropped
    and the lookup falls back to that robot's global median for that step, so a thin group cannot
    silently produce a confident number. The within-group spread is computed **inside** each group
    and then aggregated twice: equally over groups, and weighted by group size, with the rule named
    in the output.
    """

    cache = C.build_multistep_cache(tokens, index, ends, horizon)
    command = cache.command.numpy()
    keep_step = cache.step_mask > 0
    tables: Dict[int, Dict[str, Dict[int, List[np.ndarray]]]] = {0: {}, 1: {}}
    global_values: Dict[int, Dict[int, List[np.ndarray]]] = {0: {}, 1: {}}
    counts: Dict[int, Dict[int, int]] = {0: {}, 1: {}}
    for row in range(len(index)):
        key = _group_key(command[row], int(ends[row]))
        for step in range(horizon):
            if not keep_step[row, step]:
                continue
            for robot in (0, 1):
                first = 0 if robot == 0 else 4
                if cache.geometry_mask[row, step, first] <= 0:
                    continue                       # an invisible label never enters a statistic
                columns = [0, 1] if robot == 0 else [4, 5]
                anchor_columns = [0, 1] if robot == 0 else [2, 3]
                delta = cache.future["geometry"][row, step, columns] \
                    - cache.anchor[row, anchor_columns]
                tables[robot].setdefault(key, {}).setdefault(step, []).append(delta)
                global_values[robot].setdefault(step, []).append(delta)
                counts[robot][step] = counts[robot].get(step, 0) + 1
    frozen: Dict[int, Dict[str, Dict[int, np.ndarray]]] = {0: {}, 1: {}}
    spreads: Dict[int, Dict[str, Dict[str, float]]] = {0: {}, 1: {}}
    for robot in (0, 1):
        for key, steps in tables[robot].items():
            for step, values in steps.items():
                if len(values) < min_group:
                    continue
                stacked = np.stack(values)
                frozen[robot].setdefault(key, {})[step] = np.median(stacked, axis=0)
                centre = np.median(stacked, axis=0)
                spreads[robot].setdefault(key, {})[step] = float(
                    np.median(np.hypot(stacked[:, 0] - centre[0], stacked[:, 1] - centre[1])))
    global_median: Dict[int, Dict[int, np.ndarray]] = {0: {}, 1: {}}
    for robot in (0, 1):
        for step, values in global_values[robot].items():
            if values:
                global_median[robot][step] = np.median(np.stack(values), axis=0)
    # two named aggregations of the within-group spread; equal-weight is the headline because a
    # large group should not dominate a statement about "how deterministic is one group"
    equal_weight: Dict[int, Dict[int, float]] = {0: {}, 1: {}}
    size_weight: Dict[int, Dict[int, float]] = {0: {}, 1: {}}
    group_sizes: Dict[int, Dict[int, List[int]]] = {0: {}, 1: {}}
    for robot in (0, 1):
        for key, steps in spreads[robot].items():
            for step, value in steps.items():
                equal_weight[robot].setdefault(step, []).append(value)
                size = len(tables[robot][key][step])
                size_weight[robot].setdefault(step, []).append((value, size))
                group_sizes[robot].setdefault(step, []).append(size)
    summary = {}
    for robot in (0, 1):
        summary[f"r{robot}"] = {}
        for step in range(horizon):
            equal = equal_weight[robot].get(step, [])
            weighted = size_weight[robot].get(step, [])
            total = sum(size for _v, size in weighted)
            summary[f"r{robot}"][f"step{step + 1}"] = {
                "groups": len(equal),
                "within_group_spread_cm_equal_weight":
                    float(np.median(equal) * 100) if equal else None,
                "within_group_spread_cm_size_weight":
                    float(sum(v * s for v, s in weighted) / total * 100) if total else None,
                "median_group_size": float(np.median(group_sizes[robot][step]))
                if group_sizes[robot].get(step) else None,
                "visible_labels": int(counts[robot].get(step, 0))}
    return {"table": frozen, "global": global_median, "min_group": min_group,
            "groups": sum(len(frozen[r]) for r in (0, 1)),
            "groups_per_robot": {f"r{r}": len(frozen[r]) for r in (0, 1)},
            "within_group_spread": summary,
            "time_binning": f"integer step index // {ELAPSED_STEPS_PER_BIN} (0.5 s per bin)",
            "label_filter": "only labels whose own geometric visibility is true for that robot",
            "fallback": f"a group with fewer than {min_group} visible labels is dropped; the "
                        f"lookup then uses that robot's global median for the same step",
            "note": "this replaces the previous 'within-group spread', which pooled every group and "
                    "therefore measured the global spread; the 0.0517 cm figure is withdrawn"}


def p1_predict(p1: Dict[str, object], tokens: Tokens, panel: Dict[str, np.ndarray],
               horizon: int) -> np.ndarray:
    """P1's prediction, recording whether a real group statistic was actually used."""

    predictions = np.zeros((len(panel["index"]), horizon, 8))
    used_group = np.zeros((len(panel["index"]), horizon), dtype=bool)
    for position, (i, end) in enumerate(zip(panel["index"], panel["end"])):
        split = tokens.split_of[i]
        local = int(i) - tokens.offsets[split]
        token = tokens.token_arrays[split][local, int(end)]
        key = _group_key(token[51:57], int(end))
        for robot in (0, 1):
            anchor_columns = [0, 1] if robot == 0 else [2, 3]
            columns = [0, 1] if robot == 0 else [4, 5]
            anchor = np.asarray([token[7], token[8]]) if robot == 0 \
                else np.asarray([token[25], token[26]])
            for step in range(horizon):
                delta = p1["table"][robot].get(key, {}).get(step)
                if delta is not None:
                    used_group[position, step] = True
                else:
                    delta = p1["global"][robot].get(step)
                if delta is None:
                    delta = np.zeros(2)
                predictions[position, step, columns] = anchor + np.asarray(delta)
        del anchor_columns
    used = used_group.any(axis=1)
    p1["group_used_windows"] = int(used.sum())
    p1["group_hit_rate"] = float(used.mean()) if len(used) else 0.0
    p1["group_used_step_fraction"] = float(used_group.mean())
    return predictions


def compare(first: np.ndarray, second: np.ndarray) -> Dict[str, object]:
    """Both required statistics, on the windows where both models have a value."""

    keep = np.isfinite(first) & np.isfinite(second)
    if not keep.any():
        return {"difference_of_medians_cm": None, "median_of_differences_cm": None,
                "windows": 0}
    a, b = first[keep], second[keep]
    return {"difference_of_medians_cm": float((np.median(a) - np.median(b)) * 100),
            "median_of_differences_cm": float(np.median((a - b)) * 100),
            "windows": int(keep.sum()),
            "note": "negative favours the first model"}


def root_bootstrap(first: np.ndarray, second: np.ndarray, roots: np.ndarray, *,
                   draws: int = 2000, seed: int = 0) -> Dict[str, object]:
    keep = np.isfinite(first) & np.isfinite(second)
    difference = (first - second)[keep]
    root_values = roots[keep]
    unique = np.unique(root_values)
    by_root = {r: np.where(root_values == r)[0] for r in unique.tolist()}
    generator = np.random.default_rng(seed)
    medians = np.empty(draws)
    for index in range(draws):
        picked = generator.choice(len(unique), size=len(unique), replace=True)
        # the resampled statistic is the median of the *differences*; indexing ``difference`` with
        # the pooled positions is the step that was missing and produced absurd intervals
        selection = np.concatenate([by_root[unique[p]] for p in picked])
        medians[index] = float(np.median(difference[selection]))
    return {"unit": "root", "roots": int(len(unique)), "windows": int(keep.sum()), "draws": draws,
            "median_of_differences_cm": float(np.median(difference) * 100),
            "ci95_low_cm": float(np.quantile(medians, 0.025) * 100),
            "ci95_high_cm": float(np.quantile(medians, 0.975) * 100),
            "fraction_draws_below_zero": float((medians < 0).mean()),
            "note": "negative favours the first model"}


def true_t_rollout(saved, model, tokens: Tokens, panel: Dict[str, np.ndarray],
                   horizon: int = 24) -> Dict[str, object]:
    """Known-true-T diagnostic: roll to the option's real exit and score the geometry there.

    This uses the real exit instant **as an evaluation coordinate only**. It is not a deployment
    termination rule and must not be reported as autonomous stopping.
    """

    if saved["kind"] != "latent_autoregressive":
        return {"status": "only defined for the latent model"}
    index, ends = panel["index"], panel["end"]
    # only windows whose remaining life fits inside the rollout budget
    steps = np.asarray([int(tokens.steps[i]) for i in index])
    remaining = steps - 1 - ends
    keep = remaining < horizon
    errors = []
    with torch.no_grad():
        for position in np.where(keep)[0][:512]:
            i, end = int(index[position]), int(ends[position])
            need = int(remaining[position]) + 1
            cache = C.build_multistep_cache(tokens, np.asarray([i]), np.asarray([end]), need)
            steps_out, _ = model.rollout(cache.window, cache.token_valid, cache.command, need,
                                         cache.elapsed)
            raw = steps_out["geometry"].numpy().astype(np.float64)
            decoded = decode_geometry_batch(raw[0], cache.anchor, saved["stats"])
            split = tokens.split_of[i]
            local = i - tokens.offsets[split]
            terminal = tokens.terminal_arrays[split][local].astype(np.float64)
            truth = np.concatenate([terminal[7:11], terminal[25:29]])
            visible = terminal[6] > 0.5
            if not visible:
                continue
            errors.append(float(np.hypot(decoded[need - 1, 0] - truth[0],
                                         decoded[need - 1, 1] - truth[1])) * 100)
    if not errors:
        return {"status": "no evaluable windows"}
    values = np.asarray(errors)
    return {"status": "ok", "windows": int(len(values)),
            "p50_cm": float(np.median(values)), "p90_cm": float(np.quantile(values, 0.9)),
            "caveat": "uses the real exit instant as an evaluation coordinate; longer than the "
                      "trained span, so it is also a span-extrapolation result"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="test_multistep")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()
    out_dir = Path(args.out)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    panel = P.load(args.panel)
    tokens = Tokens(D.TIER_SPLITS["d1"])
    truth = truth_and_masks(tokens, panel["index"], panel["end"], args.horizon)
    # P1 is fitted on the training split only; the panel it scores is disjoint from that split
    train_index, train_ends = D.sample_windows(tokens, ["d0", "d1ext"], 4096, seed=7,
                                               horizon=args.horizon, min_future=1,
                                               t0_fraction=0.25)
    p1 = fit_p1(tokens, train_index, train_ends, args.horizon)
    p1_geometry = p1_predict(p1, tokens, panel, args.horizon)
    paths = [checkpoint_dir / f"{name}.pt" for name in args.models] \
        if args.models else sorted(checkpoint_dir.glob("*.pt"))
    result: Dict[str, object] = {"panel": args.panel, "horizon": args.horizon,
                                 "windows": int(len(panel["index"])),
                                 "roots": int(len(set(panel["root"].tolist()))),
                                 "p1_groups": p1["groups"],
                                 "p1_groups_per_robot": p1["groups_per_robot"],
                                 "p1_group_hit_rate": p1.get("group_hit_rate"),
                                 "p1_group_used_step_fraction": p1.get("group_used_step_fraction"),
                                 "p1_within_group_spread": p1["within_group_spread"],
                                 "p1_time_binning": p1["time_binning"],
                                 "p1_label_filter": p1["label_filter"],
                                 "p1_fallback": p1["fallback"],
                                 "models": {}}
    # P1 is reported like any other predictor, under the same masks
    # a second baseline with no grouping at all: it isolates how much the (command, elapsed) key
    # actually buys, and it is the floor a learned model must clear before representation claims
    global_geometry = np.zeros((len(panel["index"]), args.horizon, 8))
    for position, (i, end) in enumerate(zip(panel["index"], panel["end"])):
        split = tokens.split_of[i]
        local = int(i) - tokens.offsets[split]
        token = tokens.token_arrays[split][local, int(end)]
        for robot in (0, 1):
            columns = [0, 1] if robot == 0 else [4, 5]
            anchor = np.asarray([token[7], token[8]]) if robot == 0 \
                else np.asarray([token[25], token[26]])
            for step in range(args.horizon):
                delta = p1["global"][robot].get(step)
                if delta is None:
                    delta = np.zeros(2)
                global_geometry[position, step, columns] = anchor + np.asarray(delta)
    for label, geometry in (("P1_grouped", p1_geometry),
                            ("P1_global", global_geometry),
                            ("P0_hold", None)):
        entry: Dict[str, object] = {"kind": "baseline"}
        if label == "P0_hold":
            entry["step_p50_cm"] = {f"step{step}": {f"r{robot}": summarise(
                hold_baseline(truth["anchor"], truth["truth"], truth["geometry_mask"],
                              truth["step_mask"], step - 1, robot))["p50_cm"]
                for robot in (0, 1)} for step in STEPS}
        else:
            entry["step_p50_cm"] = {f"step{step}": {f"r{robot}": summarise(
                step_error(geometry, truth["truth"], truth["geometry_mask"], truth["step_mask"],
                           step - 1, robot))["p50_cm"] for robot in (0, 1)} for step in STEPS}
        # P0 holds the prediction-start midpoint, so its "prediction" is the anchor broadcast over
        # every step; the mask still decides which steps and robots are scored
        if geometry is None:
            hold_geometry = np.zeros((len(panel["index"]), args.horizon, 8))
            hold_geometry[..., [0, 1]] = truth["anchor"][:, None, 0:2]
            hold_geometry[..., [4, 5]] = truth["anchor"][:, None, 2:4]
        else:
            hold_geometry = geometry
        entry["r0"] = grouped({step: step_error(hold_geometry, truth["truth"],
                                                truth["geometry_mask"], truth["step_mask"],
                                                step - 1, 0)
                               for step in STEPS}, panel, 0, tokens)
        result["models"][label] = entry
    for path in paths:
        saved, model = load_model(path)
        if saved is None or saved.get("kind") is None:
            continue
        started = time.perf_counter()
        predicted = predict_steps(saved, model, tokens, panel["index"], panel["end"], args.horizon)
        entry: Dict[str, object] = {"checkpoint": str(path.relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
                                    "parameters": saved.get("parameters",
                                                            parameter_breakdown(model)),
                                    "kind": saved["kind"], "seed": saved.get("seed"),
                                    "horizon": saved.get("horizon"), "lr": saved.get("lr")}
        errors: Dict[int, Dict[int, np.ndarray]] = {}
        for robot in (0, 1):
            errors[robot] = {step: step_error(predicted["geometry"], truth["truth"],
                                              truth["geometry_mask"], truth["step_mask"], step - 1,
                                              robot) for step in STEPS if step <= args.horizon}
        entry["step_p50_cm"] = {f"step{step}": {f"r{robot}": summarise(errors[robot][step])["p50_cm"]
                                                for robot in (0, 1)}
                                for step in errors[0]}
        entry["r0"] = grouped(errors[0], panel, 0, tokens)
        entry["r1"] = grouped(errors[1], panel, 1, tokens)
        entry["hold"] = {f"step{step}": {f"r{robot}": summarise(
            hold_baseline(truth["anchor"], truth["truth"], truth["geometry_mask"],
                          truth["step_mask"], step - 1, robot))
            for robot in (0, 1)} for step in errors[0]}
        # the same-sample curve: only windows that have every step up to 16 in range, so a change in
        # sample composition cannot be mistaken for error accumulation
        available = np.asarray([int(tokens.steps[i]) - 1 - int(e)
                                for i, e in zip(panel["index"], panel["end"])])
        common = available >= 16
        entry["same_sample"] = {}
        entry["same_sample_windows"] = int(common.sum())
        for step in STEPS:
            values = errors[0][step]
            entry["same_sample"][str(step)] = summarise(np.where(common, values, np.nan))["p50_cm"]
        if saved["kind"] == "latent_autoregressive":
            frozen = predict_steps(saved, model, tokens, panel["index"][:512], panel["end"][:512],
                                   args.horizon, freeze_latent=True)
            entry["freeze_latent_ablation"] = {
                f"step{step}": summarise(step_error(frozen["geometry"],
                                                    truth["truth"][:512],
                                                    truth["geometry_mask"][:512],
                                                    truth["step_mask"][:512], step - 1, 0))["p50_cm"]
                for step in STEPS if step <= args.horizon}
            entry["true_t_rollout"] = true_t_rollout(saved, model, tokens, panel)
        entry["seconds"] = time.perf_counter() - started
        np.savez_compressed(out_dir / f"predictions_{path.stem}_{args.panel}.npz",
                            geometry=predicted["geometry"], anchor=predicted["anchor"],
                            truth=truth["truth"], geometry_mask=truth["geometry_mask"],
                            step_mask=truth["step_mask"], end=panel["end"],
                            root=panel["root"], scene=panel["scene"], index=panel["index"])
        result["models"][path.stem] = entry
        print(f"{path.stem:44s} " + " ".join(
            f"k{k}={entry['step_p50_cm'][f'step{k}']['r0']:.3f}" for k in (1, 4, 8) if f"step{k}" in entry["step_p50_cm"]),
            flush=True)
    out_path = out_dir / f"multi_step_results_{args.panel}.json"
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True, default=float))
    # the paired comparison the round exists to answer
    direct = "direct_multistep_h8_lr0.0003_seed0"
    latent = "latent_autoregressive_h8_lr0.0003_seed0"
    if direct in result["models"] and latent in result["models"]:
        comparison = {}
        predicted_direct = np.load(out_dir / f"predictions_{direct}_{args.panel}.npz")
        predicted_latent = np.load(out_dir / f"predictions_{latent}_{args.panel}.npz")
        for step in STEPS:
            if step > args.horizon:
                continue
            for robot in (0, 1):
                a = step_error(predicted_direct["geometry"], truth["truth"],
                               truth["geometry_mask"], truth["step_mask"], step - 1, robot)
                b = step_error(predicted_latent["geometry"], truth["truth"],
                               truth["geometry_mask"], truth["step_mask"], step - 1, robot)
                comparison[f"step{step}_r{robot}"] = {
                    "statistics": compare(a, b),
                    "bootstrap": root_bootstrap(a, b, panel["root"])}
        (out_dir / f"pair_comparison_{args.panel}.json").write_text(
            json.dumps(comparison, indent=1, sort_keys=True, default=float))
        result["pair_comparison"] = comparison
    print("EVALUATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
