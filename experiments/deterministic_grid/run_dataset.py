"""B2 dataset: generate the frozen 30k transition set and save its statistics/splits.

    python experiments/deterministic_grid/run_dataset.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coimagine.deterministic_grid.contract import default_maps
from coimagine.deterministic_grid.data import DataConfig, generate_dataset, save_dataset

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--out-root", default="outputs/deterministic_grid_v1")
    parser.add_argument("--budget", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=202_610_12)
    args = parser.parse_args()
    out_dir = ROOT / args.out_root / args.run_id / "dataset"
    config = DataConfig(budget=args.budget, seed=args.seed)
    dataset = generate_dataset(default_maps(), config)
    hashes = save_dataset(dataset, out_dir)
    print(json.dumps({
        "samples": len(dataset.samples),
        "distinct_states": len(dataset.state_split),
        "stats": dataset.stats,
        "hashes": hashes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
