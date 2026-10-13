"""Phase B2: train the three independent seeds of the single default model.

    python experiments/forecast_baseline/train_models.py --run-id <id>
"""

from __future__ import annotations

import argparse

import torch

# Keep the training reduction order fixed so a rerun reproduces the same weights.
torch.set_num_threads(1)
import json
import time
from pathlib import Path

from coimagine.continuous2d_learning.model import (
    ModelConfig,
    load_window_records,
    train_model,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="forecast-20261012")
    parser.add_argument("--data-root", default="outputs/forecast_baseline_v1")
    parser.add_argument("--model-seeds", default="0,1,2")
    args = parser.parse_args()

    run_dir = ROOT / args.data_root / args.run_id
    windows_path = run_dir / "dataset" / "windows.jsonl"
    train_records = load_window_records(windows_path, "train")
    val_records = load_window_records(windows_path, "val")
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = ModelConfig()
    summary = {
        "model_config": config.as_dict(),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "seeds": {},
    }
    started = time.time()
    for seed in (int(value) for value in args.model_seeds.split(",") if value):
        model, log = train_model(
            train_records,
            val_records,
            seed=seed,
            config=config,
            checkpoint_path=model_dir / f"seed{seed}.pt",
        )
        (model_dir / f"seed{seed}_training.json").write_text(json.dumps(log.as_dict(), indent=2))
        summary["seeds"][str(seed)] = {
            "best_epoch": log.best_epoch,
            "best_val_loss": log.best_val_loss,
            "epochs_run": len(log.epochs),
            "parameter_count": log.parameter_count,
            "wall_clock": log.wall_clock,
            "checkpoint_sha256": log.checkpoint_sha256,
            "final_train_loss": log.epochs[-1]["train_loss"] if log.epochs else None,
            "final_val_loss": log.epochs[-1]["val_loss"] if log.epochs else None,
        }
        print(
            f"seed {seed}: best_epoch={log.best_epoch} best_val={log.best_val_loss:.5f} "
            f"epochs={len(log.epochs)} params={log.parameter_count} wall={log.wall_clock:.1f}s"
        )
    summary["wall_clock"] = time.time() - started
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("TRAINING_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
