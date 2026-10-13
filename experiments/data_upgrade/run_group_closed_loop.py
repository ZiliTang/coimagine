"""Phase D3: closed loop for the nine group checkpoints (H=3, S0_v4 / S1_v4).

    python experiments/data_upgrade/run_group_closed_loop.py --run-id <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d_learning.closed_loop import METHOD_LEARNED, ClosedLoopConfig, run_episode
from coimagine.continuous2d_learning.model import load_model
from coimagine.data_upgrade.training import GROUP_BRANCH, GROUP_OLD, GROUP_RANDOM

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--depths", default="3")
    parser.add_argument("--model-seeds", default="0,1,2")
    parser.add_argument("--model-subdir", default="models")
    parser.add_argument("--tag", default="combined")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / f"closed_loop_groups_{args.tag}"
    model_dir = run_dir / args.model_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    config = default_config()
    closed = ClosedLoopConfig(depths=tuple(int(v) for v in args.depths.split(",") if v))
    rows = []
    started = time.time()
    for group in (GROUP_OLD, GROUP_RANDOM, GROUP_BRANCH):
        for seed in (int(v) for v in args.model_seeds.split(",") if v):
            model = load_model(model_dir / f"{group.replace('+', '_')}_seed{seed}.pt")
            for scene in closed.scenes:
                for episode_seed in closed.episode_seeds:
                    for depth in closed.depths:
                        row = run_episode(
                            METHOD_LEARNED, scene, episode_seed, depth, config,
                            closed=closed, model=model, model_seed=seed,
                        )
                        row["data_group"] = f"{group}@{args.tag}"
                        rows.append(row)
            done = [r for r in rows if r["data_group"] == f"{group}@{args.tag}" and r["model_seed"] == seed]
            print(
                f"{group} seed {seed}: episodes={len(done)} success={sum(r['success'] for r in done)} "
                f"no_feasible={sum(r['no_feasible_plan'] for r in done)}",
                flush=True,
            )
    with (out_dir / "episodes.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (out_dir / "windows.jsonl").open("w") as handle:
        for row in rows:
            for decision in row["decisions"]:
                handle.write(json.dumps({"episode": {k: row[k] for k in ("data_group", "scene_id", "episode_seed", "H", "model_seed")}, **decision}) + "\n")
    aggregate = []
    for group in (GROUP_OLD, GROUP_RANDOM, GROUP_BRANCH):
        subset = [r for r in rows if r["data_group"] == f"{group}@{args.tag}"]
        aggregate.append(
            {
                "data_group": f"{group}@{args.tag}",
                "episodes": len(subset),
                "success_rate": sum(r["success"] for r in subset) / max(1, len(subset)),
                "mean_windows": sum(r["windows"] for r in subset) / max(1, len(subset)),
                "mean_terminal_distance": sum(r["terminal_distance"] for r in subset) / max(1, len(subset)),
                "mean_J": sum(r["J"] for r in subset) / max(1, len(subset)),
                "no_feasible_plan": sum(r["no_feasible_plan"] for r in subset),
                "physical_failures": sum(r["physical_failure"] for r in subset),
                "prediction_nodes": sum(r["prediction_nodes"] for r in subset),
            }
        )
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": closed.as_dict(),
                "episodes": len(rows),
                "groups": [GROUP_OLD, GROUP_RANDOM, GROUP_BRANCH],
                "wall_clock": time.time() - started,
                "checkpoint_sha256": {
                    f"{group.replace('+', '_')}_seed{seed}": hashlib.sha256(
                        (model_dir / f"{group.replace('+', '_')}_seed{seed}.pt").read_bytes()
                    ).hexdigest()
                    for group in (GROUP_OLD, GROUP_RANDOM, GROUP_BRANCH)
                    for seed in (int(v) for v in args.model_seeds.split(",") if v)
                },
            },
            indent=2, sort_keys=True,
        )
    )
    for row in aggregate:
        print(json.dumps(row))
    print("GROUP_CLOSED_LOOP_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
