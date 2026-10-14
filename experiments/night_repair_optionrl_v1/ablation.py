"""Same-window latent ablation: normal recurrence versus a frozen initial latent.

The previous round compared the normal full-panel ``t>0`` result against a frozen result computed on
the first 512 windows with mixed regimes, which is not a comparison. Here both arms use:

* the **same checkpoint**;
* the **same panel and the same window IDs**;
* the **same prediction span** and the **same per-step visibility mask**;
* the **same execution regime** (only ``t>0``, since that is where a recurrence can matter).

Per-sample errors are written out so the paired differences can be recomputed by anyone.
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
from experiments.transformer_scale_latent_v1 import data as D  # noqa: E402
from experiments.transformer_scale_latent_v1 import panels as P  # noqa: E402
from experiments.transformer_scale_latent_v1.evaluate import (  # noqa: E402
    compare,
    load_model,
    predict_steps,
    root_bootstrap,
    step_error,
)

OUT = ROOT / "outputs/night_repair_optionrl_v1/world_models"
STEPS = (1, 4, 8, 16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(OUT / "checkpoints/repair_latent_autoregressive_seed0.pt"))
    parser.add_argument("--panel", default="test_multistep")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    tokens = Tokens(D.TIER_SPLITS["d1"])
    panel = P.load(args.panel)
    # only t>0, so both arms are scored on the regime where a recurrence can act at all
    keep_rows = np.flatnonzero(panel["end"] > 0)
    index, ends = panel["index"][keep_rows], panel["end"][keep_rows]
    roots = panel["root"][keep_rows]
    saved, model = load_model(Path(args.checkpoint))
    normal = predict_steps(saved, model, tokens, index, ends, args.horizon)
    frozen = predict_steps(saved, model, tokens, index, ends, args.horizon, freeze_latent=True)
    from experiments.transformer_scale_latent_v1.cache import build_multistep_cache
    cache = build_multistep_cache(tokens, index, ends, args.horizon)
    report: Dict[str, object] = {
        "checkpoint": args.checkpoint, "panel": args.panel,
        "windows": int(len(index)), "roots": int(len(set(roots.tolist()))),
        "regime": "t>0 only", "horizon": args.horizon,
        "same_checkpoint": True, "same_windows": True, "same_mask": True,
        "per_step": {}}
    for step in STEPS:
        for robot in (0, 1):
            a = step_error(normal["geometry"], cache.future["geometry"], cache.geometry_mask,
                           cache.step_mask, step - 1, robot)
            b = step_error(frozen["geometry"], cache.future["geometry"], cache.geometry_mask,
                           cache.step_mask, step - 1, robot)
            entry = {"statistics": compare(a, b),
                     "bootstrap": root_bootstrap(a, b, roots),
                     "normal_p50_cm": float(np.nanmedian(a) * 100),
                     "frozen_p50_cm": float(np.nanmedian(b) * 100),
                     "windows": int(np.isfinite(a).sum())}
            report["per_step"][f"step{step}_r{robot}"] = entry
    np.savez_compressed(Path(args.out) / f"ablation_per_sample_{args.panel}.npz",
                        normal=normal["geometry"], frozen=frozen["geometry"],
                        truth=cache.future["geometry"], geometry_mask=cache.geometry_mask,
                        step_mask=cache.step_mask, end=panel["end"][keep_rows], root=roots,
                        index=index)
    (Path(args.out) / f"ablation_{args.panel}.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, default=float))
    for key, entry in report["per_step"].items():
        print(f"  {key:10s} normal={entry['normal_p50_cm']:.4f} frozen={entry['frozen_p50_cm']:.4f} "
              f"median_of_diffs={entry['statistics']['median_of_differences_cm']:+.4f} "
              f"CI=[{entry['bootstrap']['ci95_low_cm']:+.4f},{entry['bootstrap']['ci95_high_cm']:+.4f}]")
    print("ABLATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
