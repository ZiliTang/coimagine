"""B2 training: train the point model for seeds 0/1/2 and report single-step metrics.

    python experiments/deterministic_grid/run_training.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coimagine.deterministic_grid.contract import default_maps
from coimagine.deterministic_grid.data import DataConfig, generate_dataset
from coimagine.deterministic_grid.model import ModelConfig, single_step_metrics, train_model

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--out-root", default="outputs/deterministic_grid_v1")
    parser.add_argument("--budget", type=int, default=30_000)
    parser.add_argument("--data-seed", type=int, default=202_610_12)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--max-epochs", type=int, default=50)
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    maps = {grid_map.name: grid_map for grid_map in default_maps()}
    dataset = generate_dataset(default_maps(), DataConfig(budget=args.budget, seed=args.data_seed))
    config = ModelConfig(max_epochs=args.max_epochs)
    summary = {"model_config": config.as_dict(), "data_seed": args.data_seed, "seeds": {}}
    for seed in [int(value) for value in args.seeds.split(",") if value]:
        checkpoint = run_dir / "models" / f"seed{seed}.pt"
        model, log = train_model(dataset, maps, seed=seed, config=config, checkpoint_path=checkpoint)
        metrics = {split: single_step_metrics(model, dataset, maps, split).as_dict()
                   for split in ("train", "val", "test")}
        (run_dir / "training").mkdir(parents=True, exist_ok=True)
        (run_dir / "training" / f"seed{seed}.json").write_text(
            json.dumps({"log": log.as_dict(), "single_step": metrics}, indent=2)
        )
        summary["seeds"][str(seed)] = {
            "best_epoch": log.best_epoch,
            "best_val_loss": log.best_val_loss,
            "wall_clock": log.wall_clock,
            "checkpoint_sha256": log.checkpoint_sha256,
            "test": metrics["test"],
        }
        print(json.dumps({"seed": seed, **summary["seeds"][str(seed)]}, indent=2))
    (run_dir / "training" / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
