"""Fixed train-panel versus validation evaluation for a world-model checkpoint.

The previous round only logged the random training-batch loss, which cannot answer under- versus
over-fitting. This script loads a checkpoint and scores a **fixed train panel** and the **fixed
validation panel** in eval mode with the identical metric.
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

from experiments.night_sequence_v1.dataset import Tokens  # noqa: E402
from experiments.transformer_scale_latent_v1 import cache as C  # noqa: E402
from experiments.transformer_scale_latent_v1 import data as D  # noqa: E402
from experiments.transformer_scale_latent_v1.evaluate import (  # noqa: E402
    load_model,
    predict_steps,
    step_error,
)

OUT = ROOT / "outputs/night_repair2_v1/world_models"
STEP_KEYS = (1, 4, 8, 16)


def score(checkpoint: Path, tokens: Tokens, index: np.ndarray, ends: np.ndarray,
          horizon: int = 16) -> Dict[str, object]:
    saved, model = load_model(checkpoint)
    predicted = predict_steps(saved, model, tokens, index, ends, horizon)
    cache = C.build_multistep_cache(tokens, index, ends, horizon)
    entry: Dict[str, object] = {"kind": saved["kind"], "windows": int(len(index)),
                                "roots": int(len(set(cache.root.tolist())))}
    for step in STEP_KEYS:
        error = step_error(predicted["geometry"], cache.future["geometry"], cache.geometry_mask,
                           cache.step_mask, step - 1, 0)
        keep = np.isfinite(error)
        entry[f"step{step}_r0_p50_cm"] = float(np.median(error[keep]) * 100) if keep.any() else None
        entry[f"step{step}_windows"] = int(keep.sum())
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="d1")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--panels", type=int, default=1024)
    args = parser.parse_args()
    tokens = Tokens(D.TIER_SPLITS[args.tier])
    train_index, train_ends = D.sample_windows(tokens, ["d0", "d1ext"], args.panels, seed=10_003,
                                               horizon=16, min_future=1, t0_fraction=0.25)
    val_index, val_ends = D.sample_windows(tokens, ["val"], args.panels, seed=10_002,
                                           horizon=16, min_future=1, t0_fraction=0.25)
    overlap = set(tokens.roots[train_index].tolist()) & set(tokens.roots[val_index].tolist())
    report: Dict[str, object] = {
        "note": "both panels are scored in eval mode with the same metric; the random training-batch "
                "loss is not a substitute",
        "train_panel_seed": 10_003, "val_panel_seed": 10_002,
        "root_overlap": len(overlap), "models": {}}
    checkpoints = ROOT / "outputs/night_repair_optionrl_v1/world_models/checkpoints"
    for name in ("repair_direct_multistep_seed0", "repair_latent_autoregressive_seed0"):
        checkpoint = checkpoints / f"{name}.pt"
        if not checkpoint.exists():
            continue
        report["models"][name] = {
            "fixed_train": score(checkpoint, tokens, train_index, train_ends),
            "validation": score(checkpoint, tokens, val_index, val_ends)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "world_train_eval.json").write_text(json.dumps(report, indent=1, default=float))
    for name, entry in report["models"].items():
        train = entry["fixed_train"]
        val = entry["validation"]
        print(f"{name}")
        print("  split      " + " ".join(f"k{k:>2}" for k in STEP_KEYS))
        print("  fixed train" + " ".join(f"{train[f'step{k}_r0_p50_cm']:6.3f}" for k in STEP_KEYS))
        print("  validation " + " ".join(f"{val[f'step{k}_r0_p50_cm']:6.3f}" for k in STEP_KEYS))
    print("WORLD_TRAIN_EVAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
