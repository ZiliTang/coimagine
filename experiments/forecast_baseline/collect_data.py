"""Phase B3: collect the 200-episode continuous dataset.

    python experiments/forecast_baseline/collect_data.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coimagine.continuous2d_learning.records import CollectionConfig, collect_dataset

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="forecast-20261012")
    parser.add_argument("--out-root", default="outputs/forecast_baseline_v1")
    parser.add_argument("--episodes-per-scene", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--full-step-episodes", type=int, default=1)
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    data_dir = run_dir / "dataset"
    collection = CollectionConfig(
        episodes_per_scene=args.episodes_per_scene,
        max_windows=args.max_windows,
        full_step_episodes=args.full_step_episodes,
    )
    manifest = collect_dataset(data_dir, collection=collection)
    (run_dir / "dataset_summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({key: manifest[key] for key in ("episodes", "windows", "wall_clock")}, indent=2))
    for split, row in manifest["coverage"].items():
        print(
            f"{split}: windows={row['windows']} eps={row['episodes']} "
            f"translation={row['object_translation']} rotation={row['object_rotation']} "
            f"stationary={row['stationary']} contact={row['contact']} "
            f"success={row['success']} failure={row['failure']}"
        )
        print(f"  combinations={row['combinations']}")
        print(f"  terminations={row['terminations']}")
    print("DATA_COLLECTION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
