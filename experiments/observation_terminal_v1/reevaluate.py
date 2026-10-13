"""Re-score the **frozen** checkpoints after the two evaluation fixes. Nothing is retrained.

Fixes applied before this run:
* every baseline statistic is computed from ``y_raw`` (original units). P1 previously read the
  standardised labels and handed them to an evaluator expecting metres and seconds;
* binary predictions declare their type. Baselines emit probabilities in [0, 1]; the network emits
  logits. Applying a sigmoid to a 0/1 baseline put both 0 and 1 above 0.5, so every sample was
  called positive and the visibility/contact comparison was meaningless.

    python experiments/observation_terminal_v1/reevaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.observation_terminal_v1 import baselines as BL  # noqa: E402
from experiments.observation_terminal_v1.evaluate_terminal import (  # noqa: E402
    BINARY_LOGIT,
    BINARY_PROBABILITY,
    evaluate,
)
from experiments.observation_terminal_v1.train_terminal import (  # noqa: E402
    load_all,
    render,
    start_patch_valid,
)
from experiments.option_prediction_mechanism_v1.phase_b.model import (  # noqa: E402
    MultiHeadMLP,
    _standardise,
    predict,
    unstandardise_predictions,
)

OUT = ROOT / "outputs/observation_terminal_v1"
SEEDS = (0, 1, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out_dir = Path(args.out)
    data, records, observations, meta = load_all()
    starts = start_patch_valid(records)
    _standardise(data)
    payload = {"stage": "reevaluate_frozen_checkpoints",
               "fixes": ["baselines use y_raw (original units)",
                         "binary predictions declare probability vs logit",
                         "groupings report sample and independent-root counts"],
               "data": {"records": len(records), **meta,
                        "train": int((data.split == "train").sum()),
                        "val": int((data.split == "val").sum()),
                        "test": int((data.split == "test").sum())},
               "baselines": {}, "runs": {}}
    for split in ("val", "test"):
        for name, function in (("P0_persistence", BL.persistence_predictions),
                               ("P1_command_grouped", BL.command_grouped_predictions)):
            predictions, info = function(data, split)
            entry = evaluate(data, observations, predictions, split, name, permutation_seed=0,
                             start_patch_valid=starts, binary_kind=BINARY_PROBABILITY)
            entry["baseline_info"] = {k: v for k, v in info.items()
                                      if k != "common_subset_index"}
            payload["baselines"][f"{name}|{split}"] = entry
            geometry = entry["terminal_observation_geometry"]
            print(f"{name}|{split}: mid2d p50 {geometry['r0_mid_2d_m']['p50']:.4f} m | "
                  f"angle p50 {geometry['r0_normal_angle_deg']['p50']:.3f} deg | "
                  f"duration p50 {entry['duration']['duration_s']['p50']:.4f} s", flush=True)
    for seed in SEEDS:
        path = out_dir / "checkpoints" / f"deployable_seed{seed}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = MultiHeadMLP(saved["input_dim"], data.specs)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        model_input = np.hstack([data.x, np.zeros_like(data.x_priv)])
        predictions = unstandardise_predictions(data, predict(model, model_input, data.specs))
        key = f"deployable_seed{seed}"
        payload["runs"][key] = {
            "parameters": saved["parameters"], "input_dim": saved["input_dim"],
            "best_epoch": saved.get("best_epoch"), "checkpoint": str(path),
            "val": evaluate(data, observations, predictions, "val", key, permutation_seed=0,
                            start_patch_valid=starts, binary_kind=BINARY_LOGIT),
            "test": evaluate(data, observations, predictions, "test", key, permutation_seed=0,
                             start_patch_valid=starts, binary_kind=BINARY_LOGIT),
        }
        geometry = payload["runs"][key]["test"]["terminal_observation_geometry"]
        print(f"{key}|test: mid2d p50 {geometry['r0_mid_2d_m']['p50']:.4f} m | "
              f"angle p50 {geometry['r0_normal_angle_deg']['p50']:.3f} deg", flush=True)
    (out_dir / "terminal_report_reeval.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float))
    (out_dir / "terminal_report_reeval.md").write_text(render(payload))
    print(render(payload))
    print("REEVALUATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
