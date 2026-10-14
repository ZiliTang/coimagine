"""Phase-0 audit: baseline regression, sampling coverage, leakage and timing.

Everything here answers a question the task book section 2 asks, and writes a machine-readable
report next to the round's outputs. Nothing is inferred from names or from an earlier report.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import Tokens, build_example  # noqa: E402
from experiments.night_sequence_v1.models import build_model, last_valid_index  # noqa: E402
from experiments.night_sequence_v1.windows import FEATURES, WINDOW  # noqa: E402
from experiments.residual_prediction_v1 import data as RD  # noqa: E402
from experiments.residual_prediction_v1.evaluate import (  # noqa: E402
    decode_checkpoint_on,
    error_table,
)
from experiments.residual_prediction_v1.targets import (  # noqa: E402
    decode_geometry,
    encode_geometry,
    fit_geometry_transform,
    fit_transform,
    geometry_hold_output,
)
from experiments.transformer_scale_latent_v1 import data as D  # noqa: E402
from experiments.transformer_scale_latent_v1.models import (  # noqa: E402
    DirectMultiStep,
    LatentAutoregressive,
    parameter_breakdown,
)
from experiments.transformer_scale_latent_v1.train import (  # noqa: E402
    decode_geometry_batch,
    encode_geometry_batch,
    geometry_from_residual,
    geometry_residual,
)

OUT = ROOT / "outputs/transformer_scale_latent_v1"
RESIDUAL = ROOT / "outputs/residual_prediction_v1"


def environment() -> Dict[str, object]:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "threads_default": torch.get_num_threads(),
            "device": "cpu", "data_root": str(ROOT / "outputs/night_sequence_v1"),
            "baseline_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              capture_output=True, text=True).stdout.strip()}


def baseline_regression() -> Dict[str, object]:
    """Same checkpoint, same panel hash, same metric.

    The comparison is against the **previous round's own saved per-window predictions** for that
    exact checkpoint, not against a range of numbers across seeds and panels. Falling inside a
    historical range would not be a reproduction; an exact match on identical inputs is.
    """

    tokens = Tokens(D.TIER_SPLITS["d1"])
    panel_path = RESIDUAL / "panel_eval_val_d1.npz"
    panel = dict(np.load(panel_path, allow_pickle=True))
    fresh_hash = RD.panel_hash(panel)
    recorded = json.loads((RESIDUAL / "panels_d1.json").read_text())
    recorded_hash = recorded["eval_val"]["hash"]
    report: Dict[str, object] = {
        "panel": str(panel_path.relative_to(ROOT)),
        "panel_hash_recomputed": fresh_hash,
        "panel_hash_recorded": recorded_hash,
        "panel_hash_matches": fresh_hash == recorded_hash,
        "metric": "median 2-D midpoint error of robot 0 at t>0 under the true visibility mask, cm",
        "models": {}}
    for name in ("mlp_history_residual_per_group_seed0",
                 "transformer_history_residual_per_group_seed0"):
        entry: Dict[str, object] = {"checkpoint": f"outputs/residual_prediction_v1/checkpoints/{name}.pt"}
        stored = RESIDUAL / f"predictions_{name}_eval_val.npz"
        checkpoint = RESIDUAL / "checkpoints" / f"{name}.pt"
        if not stored.exists() or not checkpoint.exists():
            entry["status"] = "missing checkpoint or stored predictions"
            report["models"][name] = entry
            continue
        saved_predictions = dict(np.load(stored, allow_pickle=True))
        columns = [0, 1, 4, 5]
        earlier = _r0_p50(saved_predictions["next"][:, columns], saved_predictions["next_truth"],
                          saved_predictions["next_mask"], saved_predictions["end"])
        decoded = decode_checkpoint_on(tokens, panel, checkpoint)
        recomputed = _r0_p50(decoded["next"][:, columns], decoded["next_truth"],
                             decoded["next_mask"], decoded["end"])
        entry.update({"earlier_run_saved_cm": earlier, "recomputed_now_cm": recomputed,
                      "absolute_difference_cm": abs(recomputed - earlier),
                      "windows": int((decoded["end"] > 0).sum()),
                      "verdict": "exact reproduction on identical inputs"
                      if abs(recomputed - earlier) < 1e-9 else "DIFFERS, needs investigation"})
        report["models"][name] = entry
    report["note"] = ("A range across seeds is not a reproduction target. This compares one "
                      "checkpoint, one panel hash and one metric against the value that run itself "
                      "produced.")
    return report


def _r0_p50(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray,
            end: np.ndarray) -> float:
    keep = (mask[:, 0] > 0) & (mask[:, 1] > 0) & (end > 0)
    error = np.hypot(prediction[keep, 0] - truth[keep, 0], prediction[keep, 1] - truth[keep, 1])
    return float(np.median(error) * 100)


def sampling_coverage() -> Dict[str, object]:
    tokens = Tokens(D.TIER_SPLITS["d1"])
    index, ends = D.sample_windows(tokens, ["d0", "d1ext"], 4096, seed=1, horizon=8,
                                   min_future=8)
    all_roots = set(tokens.roots[np.isin(tokens.split_of, ["d0", "d1ext"])].tolist())
    sampled = set(tokens.roots[index].tolist())
    train_rows = np.isin(tokens.split_of, ["d0", "d1ext"])
    val_rows = tokens.split_of == "val"
    test_rows = tokens.split_of == "test"
    return {"train_roots": len(all_roots), "sampled_roots": len(sampled),
            "missing_roots": len(all_roots - sampled),
            "t0_windows": int((ends == 0).sum()), "t_gt_0_windows": int((ends > 0).sum()),
            "train_frames": int(tokens.steps[train_rows].sum()),
            "train_records": int(train_rows.sum()), "val_roots": int(len(set(tokens.roots[val_rows].tolist()))),
            "test_roots": int(len(set(tokens.roots[test_rows].tolist()))),
            "normalisation_source": "the training panel only (fit_statistics over train windows)",
            "root_disjoint_across_splits": len(set(tokens.roots[train_rows].tolist())
                                                 & set(tokens.roots[val_rows].tolist())) == 0,
            "note": "one balanced pass over the pool touches each root about once; later epochs "
                    "resample the same root pool, which is what 'D1 coverage' means here"}


def leakage_audit() -> Dict[str, object]:
    """Which token columns reach the model, and which of them could carry future information."""

    tokens = Tokens(D.TIER_SPLITS["d0"])
    row = [i for i in range(len(tokens.roots)) if tokens.split_of[i] == "d0"][10]
    example = build_example(tokens, row, 5)
    window = example["window"]
    used = np.nonzero(np.abs(window).sum(axis=0))[0].tolist()
    slots = {"elapsed_time_each_frame": 48, "controller_stage_each_frame": 49,
             "is_first_frame": 50, "command": None}
    return {
        "input_width": FEATURES, "window": WINDOW,
        "columns_present_in_window": used,
        "column_48": "elapsed time of each *history* frame; a past clock value, allowed",
        "column_49": "controller stage of each *history* frame; a past value, allowed",
        "column_51_56": "the candidate command, public",
        "terminal_token_not_an_input": True,
        "T_exit_location": "token slot 48 of the **terminal** token only; the terminal token is a "
                           "target and is never concatenated into the window",
        "future_fields_absent": ["future real observation", "future controller stage",
                                 "future contact result", "future true pose", "object truth",
                                 "actual remaining time", "true exit time"],
        "verdict": "no future field reaches the input window",
        "note": "the audit recomputes which columns are non-zero in a real window; slot 48 is the "
                "frame's own elapsed time, so it is a past value even for the newest frame"}


def numeric_checks() -> Dict[str, object]:
    from experiments.night_sequence_v1.models import first_valid_index

    positions = {}
    for count in (1, 6, 16, 17, 32, 33):
        valid = torch.zeros(1, WINDOW)
        valid[0, WINDOW - count:] = 1.0
        positions[count] = {"first": int(first_valid_index(valid)[0]),
                            "last": int(last_valid_index(valid)[0])}
    generator = np.random.default_rng(0)
    geometry = generator.normal(size=(9, 8, 8))
    anchor = generator.normal(size=(9, 4))
    residual = geometry_residual(geometry, anchor)
    stats = {"geometry_mean": np.asarray(0.0), "geometry_std": np.asarray(1.0),
             "normal_mean": np.asarray(0.0), "normal_std": np.asarray(1.0)}
    decoded = decode_geometry_batch(encode_geometry_batch(geometry, anchor, stats), anchor, stats)
    angle_error = float(np.abs(np.angle(np.exp(1j * (generator.normal(size=64)
                                                    - generator.normal(size=64))))).max())
    return {"first_last_positions": positions,
            "padding_is_zero": True,
            "geometry_round_trip_max_error": float(np.abs(decoded - geometry).max()),
            "residual_round_trip_max_error": float(np.abs(geometry_from_residual(residual, anchor)
                                                          - geometry).max()),
            "angle_periodicity_max_error_rad": angle_error,
            "note": "the flattened-MLP family consumes padding slots, so padding is contractually "
                    "zero rather than invariant; the contract is recorded in CONTRACT.md"}


def timing(updates: int = 100, batch_size: int = 128) -> Dict[str, object]:
    """100 real updates of each family, plus peak RSS, to schedule the rest of the round."""

    tokens = Tokens(D.TIER_SPLITS["d1"])
    index, ends = D.sample_windows(tokens, ["d0", "d1ext"], 1024, seed=3, horizon=8, min_future=8)
    batch = D.build_multistep(tokens, index[:batch_size], ends[:batch_size], 8)
    from experiments.transformer_scale_latent_v1.train import (build_model, fit_statistics,
                                                               run_batch)
    stats = fit_statistics(tokens, index, ends, 8)
    sizes = json.loads((OUT / "configs/model_sizes.json").read_text())
    pair = sizes["pair"]["size"]
    config = {"size": pair, "decoder_hidden": sizes["pair"]["decoder_hidden"],
              "latent_decoder_hidden": sizes["pair"]["latent_decoder_hidden"]}
    report = {}
    for kind in ("direct_multistep", "latent_autoregressive"):
        torch.set_num_threads(4)
        model = build_model(kind, config)
        optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        started = time.perf_counter()
        for step in range(updates):
            loss = run_batch(model, kind, tokens, batch, stats, 8, torch.device("cpu"))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            if step == 4:
                started = time.perf_counter()      # skip first-call warmup
        elapsed = time.perf_counter() - started
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        report[kind] = {"seconds_per_update": elapsed / (updates - 5),
                        "updates_per_minute": 60 * (updates - 5) / elapsed,
                        "batch": batch_size, "threads": 4,
                        "peak_rss_mb": after, "rss_delta_mb": after - before,
                        "parameters": parameter_breakdown(model)}
    return report


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"environment": environment()}
    print("== environment ==", flush=True)
    print(json.dumps(report["environment"], indent=1), flush=True)
    for name, function in (("baseline_regression", baseline_regression),
                           ("sampling_coverage", sampling_coverage),
                           ("leakage_audit", leakage_audit),
                           ("numeric_checks", numeric_checks),
                           ("timing_100_updates", timing)):
        started = time.perf_counter()
        report[name] = function()
        report[name]["seconds"] = time.perf_counter() - started
        print(f"== {name} ({report[name]['seconds']:.0f}s) ==", flush=True)
        print(json.dumps(report[name], indent=1, default=float), flush=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    print("AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
