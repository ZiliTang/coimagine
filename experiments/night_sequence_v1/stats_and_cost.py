"""Round-5 statistics and resource accounting: paired root bootstrap and inference cost.

    python experiments/night_sequence_v1/stats_and_cost.py --tier d1

Two things the task book requires that the plain metric tables do not carry:

* **paired root bootstrap** for the MLP-history versus Transformer-history comparison. Both models
  are scored on exactly the same sampled examples, so the comparison is paired *per example*; the
  unit of resampling is the **root**, because frames and actions of one root are not independent
  samples. Training-seed spread is reported separately and never folded into the bootstrap.
* **inference cost**: parameter count, peak process memory during load plus a forward pass,
  batch-size-1 latency and batched throughput. Same-updates is not same-FLOPs, so this is reported
  alongside the learning curves rather than instead of them.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import GROUPS, Tokens, build_example  # noqa: E402
from experiments.night_sequence_v1.models import build_model, parameter_count  # noqa: E402
from experiments.night_sequence_v1.windows import FEATURES, WINDOW  # noqa: E402

DATA = ROOT / "outputs/night_sequence_v1"
TIER_SPLITS = {"d0": ["d0", "val", "test"], "d1": ["d0", "d1ext", "val", "test"],
               "d2": ["d0", "d1ext", "d2ext", "val", "test"]}
#: both questions the review flagged: the architecture pair and the history-versus-single-frame pair.
#: Round 5 trained MLP-current only on D1, so that pair is registered per tier and skipped when a
#: checkpoint is absent.
PAIRS = [("mlp_history", "transformer_history"), ("mlp_current", "mlp_history")]


def root_bootstrap(roots: Sequence[str], error_a: np.ndarray, error_b: np.ndarray,
                   valid: np.ndarray, *, draws: int = 2000, seed: int = 0) -> Dict[str, object]:
    """Bootstrap the median-error difference over **roots**, paired per example.

    ``error_a`` and ``error_b`` come from two models scored on the same examples, so the difference
    is computed example by example and only then resampled. Negative means model A is better.
    """

    roots = np.asarray(roots)
    keep = np.asarray(valid, dtype=bool)
    roots = roots[keep]
    difference = np.asarray(error_a)[keep] - np.asarray(error_b)[keep]
    unique = np.unique(roots)
    by_root = {root: np.where(roots == root)[0] for root in unique.tolist()}
    generator = np.random.default_rng(seed)
    draws_values = np.empty(draws, dtype=float)
    for index in range(draws):
        picked = generator.choice(len(unique), size=len(unique), replace=True)
        selection = np.concatenate([by_root[unique[p]] for p in picked])
        draws_values[index] = float(np.median(difference[selection]))
    return {
        "unit": "root (frames and actions of one root are not independent samples)",
        "draws": draws, "roots": int(len(unique)), "examples": int(keep.sum()),
        "observed_median_difference_m": float(np.median(difference)),
        "ci95_low_m": float(np.quantile(draws_values, 0.025)),
        "ci95_high_m": float(np.quantile(draws_values, 0.975)),
        "fraction_draws_below_zero": float((draws_values < 0).mean()),
        "interpretation": "negative favours the first model of the pair",
    }


def measure_cost(model_kind: str, kwargs: dict, *, device: str = "cpu",
                 batch: int = 128, repeats: int = 20) -> Dict[str, object]:
    torch.set_num_threads(1)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    model = build_model(model_kind, **kwargs)
    model.eval()
    window = torch.randn(1, WINDOW, FEATURES)
    valid = torch.ones(1, WINDOW)
    with torch.no_grad():
        model(window, valid)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # batch-size-1 latency
    with torch.no_grad():
        for _ in range(3):
            model(window, valid)
        started = time.perf_counter()
        for _ in range(repeats):
            model(window, valid)
        latency = (time.perf_counter() - started) / repeats
    # batched throughput
    big_window = torch.randn(batch, WINDOW, FEATURES)
    big_valid = torch.ones(batch, WINDOW)
    with torch.no_grad():
        model(big_window, big_valid)
        started = time.perf_counter()
        rounds = max(1, repeats // 2)
        for _ in range(rounds):
            model(big_window, big_valid)
        batched = (time.perf_counter() - started) / rounds
    cpu_seconds = sum(p.numel() for p in model.parameters()) * 0  # placeholder for clarity
    del cpu_seconds
    return {
        "model": model_kind, "parameters": parameter_count(model),
        "peak_rss_delta_mb": (after - before) / 1024.0,
        "batch1_latency_ms": latency * 1000.0,
        "batch128_latency_ms": batched * 1000.0,
        "batch128_throughput_per_s": batch / batched,
        "threads": 1,
        "note": "peak RSS delta covers model construction plus one forward pass; "
                "latency measured with torch threads pinned to 1",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--out", default=str(DATA / "analysis"))
    args = parser.parse_args()
    out_dir = Path(args.out) / args.tier
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"no metrics at {metrics_path}")
        return 1
    payload = json.loads(metrics_path.read_text())

    # ---- paired root bootstrap on the per-example errors written by analysis.py
    bootstrap: Dict[str, object] = {}
    paired = payload.get("models", {}).get("paired_errors", {})
    for kind_a, kind_b in PAIRS:
        tag_a = f"{args.tier}_{kind_a}_seed0"
        tag_b = f"{args.tier}_{kind_b}_seed0"
        if tag_a not in paired or tag_b not in paired:
            continue
        if not all((DATA / "runs" / f"{tag}.pt").exists() for tag in (tag_a, tag_b)):
            continue
        entry_a, entry_b = paired[tag_a], paired[tag_b]
        bootstrap[f"{tag_a}_vs_{tag_b}|t0"] = root_bootstrap(
            entry_a.get("t0_roots", entry_a["roots"]),
            np.asarray(entry_a["t0_error"]), np.asarray(entry_b["t0_error"]),
            np.asarray(entry_a["t0_valid"]), draws=args.draws)
        if "exec_valid" in entry_a:
            bootstrap[f"{tag_a}_vs_{tag_b}|in_execution"] = root_bootstrap(
                entry_a["roots"], np.asarray(entry_a["exec_error"]),
                np.asarray(entry_b["exec_error"]), np.asarray(entry_a["exec_valid"]),
                draws=args.draws)
    # training-seed spread is reported on its own axis, never inside the bootstrap
    seed_spread: Dict[str, List[float]] = {}
    for label, entry in payload["models"].items():
        if not label.endswith("|test") or not isinstance(entry, dict) or "t0" not in entry:
            continue
        family = label.split("|")[0].rsplit("_seed", 1)[0]
        seed_spread.setdefault(family, []).append(
            entry["t0"]["terminal_geometry"]["r0_mid_2d"]["p50"])
    spread = {family: {"seeds": len(values), "min": min(values), "max": max(values),
                       "range": max(values) - min(values)}
              for family, values in sorted(seed_spread.items()) if len(values) > 1}

    # ---- inference cost
    from experiments.night_sequence_v1.train import MODEL_KWARGS
    cost = [measure_cost(kind, kwargs) for kind, kwargs in MODEL_KWARGS.items()]

    result = {"tier": args.tier, "paired_root_bootstrap": bootstrap,
              "training_seed_spread_m": spread, "inference_cost": cost}
    (out_dir / "stats_and_cost.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1))
    print("STATS_AND_COST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
