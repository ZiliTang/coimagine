"""Phase D: closed-loop baselines under the frozen scalar protocol.

    python experiments/forecast_baseline/run_closed_loop.py --run-id <id> [--methods ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import torch

# Single-sample inference: extra BLAS threads cost far more than they save here.
torch.set_num_threads(1)

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d_learning.closed_loop import (
    METHOD_EXACT,
    METHOD_LEARNED,
    METHOD_UNIFORM,
    ClosedLoopConfig,
    run_episode,
)
from coimagine.continuous2d_learning.model import load_model

ROOT = Path(__file__).resolve().parents[2]


def _aggregate(rows):
    out = []
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["H"], row["model_seed"]), []).append(row)
    for (method, depth, model_seed), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        out.append(
            {
                "method": method,
                "H": depth,
                "model_seed": model_seed,
                "episodes": len(group),
                "success_rate": float(np.mean([row["success"] for row in group])),
                "mean_completion_time": float(np.mean([row["completion_time"] for row in group])),
                "mean_windows": float(np.mean([row["windows"] for row in group])),
                "mean_terminal_distance": float(np.mean([row["terminal_distance"] for row in group])),
                "physical_failures": int(np.sum([row["physical_failure"] for row in group])),
                "mean_J": float(np.mean([row["J"] for row in group])),
                "no_feasible_plan": int(np.sum([row["no_feasible_plan"] for row in group])),
                "prediction_nodes": int(np.sum([row["prediction_nodes"] for row in group])),
                "exact_simulation_steps": int(np.sum([row["exact_simulation_steps"] for row in group])),
                "prefix_replay_steps": int(np.sum([row["prefix_replay_steps"] for row in group])),
                "wall_clock": float(np.sum([row["wall_clock"] for row in group])),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="forecast-20261012")
    parser.add_argument("--out-root", default="outputs/forecast_baseline_v1")
    parser.add_argument("--model-run", default="", help="run dir holding models/seed*.pt (default: the output run dir)")
    parser.add_argument("--methods", default=f"{METHOD_EXACT},{METHOD_LEARNED},{METHOD_UNIFORM}")
    parser.add_argument("--model-seeds", default="0,1,2")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 = the full protocol")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    model_run = ROOT / args.model_run if args.model_run else run_dir
    out_dir = run_dir / "closed_loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = default_config()
    closed = ClosedLoopConfig()
    methods = [value for value in args.methods.split(",") if value]
    model_seeds = [int(value) for value in args.model_seeds.split(",") if value]
    models = {seed: load_model(model_run / "models" / f"seed{seed}.pt") for seed in model_seeds}

    rows = []
    budget = args.max_episodes
    for method in methods:
        # ``uniform_root`` draws one window from the root library and does not depend on a
        # pool depth, so it runs once per (scene, episode seed): the 6 protocol episodes.
        depths = [None] if method == METHOD_UNIFORM else list(closed.depths)
        for scene in closed.scenes:
            for depth in depths:
                for episode_seed in closed.episode_seeds:
                    seeds = model_seeds if method == METHOD_LEARNED else [None]
                    for model_seed in seeds:
                        if budget and len(rows) >= budget:
                            break
                        row = run_episode(
                            method, scene, episode_seed, depth, config,
                            closed=closed,
                            model=models[model_seed] if model_seed is not None else None,
                            model_seed=model_seed,
                        )
                        rows.append(row)
                        print(
                            f"{method} {scene} H={depth} ep_seed={episode_seed} "
                            f"model={model_seed} success={row['success']} windows={row['windows']} "
                            f"J={row['J']:+.3f} wall={row['wall_clock']:.1f}s"
                        )
    with (out_dir / "episodes.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with (out_dir / "windows.jsonl").open("w") as handle:
        for row in rows:
            for decision in row["decisions"]:
                handle.write(json.dumps({"episode": {key: row[key] for key in ("method", "scene_id", "episode_seed", "H", "model_seed")}, **decision}) + "\n")
    aggregate = _aggregate(rows)
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True))
    checkpoints = {
        str(seed): hashlib.sha256((model_run / "models" / f"seed{seed}.pt").read_bytes()).hexdigest()
        for seed in model_seeds
    }
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"config": closed.as_dict(), "methods": methods, "episodes": len(rows),
             "model_run": str(model_run.relative_to(ROOT)), "checkpoint_sha256": checkpoints},
            indent=2, sort_keys=True,
        )
    )
    for row in aggregate:
        print(json.dumps(row))
    print("CLOSED_LOOP_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
