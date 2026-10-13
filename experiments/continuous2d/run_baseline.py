"""Protocol-fix baseline: random control, depth-1 MPC and depth-3 MPC on the ten scenes.

    python experiments/continuous2d/run_baseline.py --run-id <id> --out-root outputs/protocol_fix_v1
    python experiments/continuous2d/run_baseline.py --quick

Uses the fixed protocol: one persistent live environment per episode and a separate planner
environment; exact prefix-replay restore in planning; success absorption and terminal
failure handling. Writes ``results.csv`` / ``results.json`` / ``summary.json``, schema-v2
``windows.jsonl`` / ``episodes.jsonl`` and one rendered S0 and S1 success each.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from coimagine.continuous2d import Continuous2DEnv, default_config, default_scenes
from coimagine.continuous2d.legality import scene_content_hash
from coimagine.continuous2d.data import code_version
from coimagine.continuous2d.data import episode_record, window_records, write_jsonl
from coimagine.continuous2d.mpc import EpisodeResult, run_mpc_episode, run_random_episode
from coimagine.continuous2d.render import render_episode

ROOT = Path(__file__).resolve().parents[2]


def _row(condition: str, scene: str, seed: int, result: EpisodeResult, spec) -> Dict[str, object]:
    return {
        "condition": condition,
        "scene": scene,
        "scene_version": spec.scene_version,
        "scene_content_hash": scene_content_hash(spec),
        "code_version": code_version(),
        "category": scene[:2],
        "seed": seed,
        "success": int(result.success),
        "windows": result.windows,
        "termination": result.termination,
        "no_feasible_plan": int(result.no_feasible_plan),
        "success_time": result.success_time,
        "control_usage": result.control_usage,
        "terminal_distance": result.terminal_distance,
        "plan_nodes": result.plan_nodes,
        "candidates": result.candidates,
        "live_physics_steps": result.live_physics_steps,
        "candidate_simulated_steps": result.candidate_simulated_steps,
        "prefix_replay_steps": result.prefix_replay_steps,
        "plan_wall_clock": result.plan_wall_clock,
        "wall_clock": result.wall_clock,
        "failure_distribution": json.dumps(result.failure_distribution),
        "chosen": " ".join(str(value) for value in result.chosen_candidate_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--out-root", default="outputs/protocol_fix_v1")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--depths", default="1,3")
    parser.add_argument("--scenes", default="", help="comma-separated scene names; empty = all 10")
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--no-render", dest="render", action="store_false")
    args = parser.parse_args()

    config = default_config()
    scenes = default_scenes()
    if args.quick:
        scenes = (scenes[0], scenes[5])
    if args.scenes:
        wanted = [name for name in args.scenes.split(",") if name]
        scenes = tuple(scene for scene in scenes if scene.name in wanted)
    depths = [int(value) for value in args.depths.split(",")]

    out_dir = ROOT / args.out_root / args.run_id / "continuous2d"
    out_dir.mkdir(parents=True, exist_ok=True)

    live = Continuous2DEnv(config)
    planner = Continuous2DEnv(config)
    rows: List[Dict[str, object]] = []
    windows: List[Dict[str, object]] = []
    episodes: List[Dict[str, object]] = []
    rendered_episodes: Dict[str, EpisodeResult] = {}
    rendered_scene: Dict[str, str] = {}
    started = time.time()

    def record(condition: str, scene: str, seed: int, result: EpisodeResult, spec) -> None:
        rows.append(_row(condition, scene, seed, result, spec))
        episode_id = f"{condition}-{scene}-{seed}"
        windows.extend(window_records(episode_id, scene, seed, result, config))
        episodes.append(episode_record(episode_id, scene, seed, result, config))
        print(
            f"{condition} {scene} seed={seed} success={result.success} term={result.termination} "
            f"windows={result.windows} nodes={result.plan_nodes} "
            f"cand_steps={result.candidate_simulated_steps} prefix_steps={result.prefix_replay_steps} "
            f"live_steps={result.live_physics_steps} t={result.wall_clock:.1f}s",
            flush=True,
        )

    for scene in scenes:
        for seed in config.seeds:
            record("random", scene.name, seed, run_random_episode(live, scene.name, config, seed=seed), scene)

        for depth in depths:
            result = run_mpc_episode(
                live, scene.name, config, depth=depth, beam=config.mpc.beam, seed=0,
                planner_env=planner,
            )
            record(f"mpc_depth{depth}", scene.name, 0, result, scene)
            if result.success and scene.category not in rendered_episodes:
                rendered_episodes[scene.category] = result
                rendered_scene[scene.category] = scene.name

    results_path = out_dir / "results.csv"
    fieldnames = [
        "condition", "scene", "category", "seed", "scene_version", "scene_content_hash",
        "code_version", "success", "windows", "termination",
        "no_feasible_plan", "success_time", "control_usage", "terminal_distance", "plan_nodes",
        "candidates", "live_physics_steps", "candidate_simulated_steps", "prefix_replay_steps",
        "plan_wall_clock", "wall_clock", "failure_distribution", "chosen",
    ]
    with results_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "config_hash": config.config_hash(),
                "schema_version": 2,
                "protocol": "exact-restore, live/planner separated, success-absorbing",
                "scene_versions": {scene.name: scene.scene_version for scene in scenes},
                "scene_hashes": {scene.name: scene_content_hash(scene) for scene in scenes},
                "rows": rows,
                "wall_clock": time.time() - started,
            },
            indent=2,
        )
    )
    write_jsonl(out_dir / "windows.jsonl", windows)
    write_jsonl(out_dir / "episodes.jsonl", episodes)

    rendered: List[str] = []
    if args.render:
        for category in ("S0", "S1"):
            episode = rendered_episodes.get(category)
            scene_name = rendered_scene.get(category)
            if episode is None or scene_name is None:
                continue
            rendered.extend(
                render_episode(
                    episode, scene_name, config, str(out_dir / "media"),
                    stem=f"{scene_name}_mpc_success", every=4,
                )
            )

    aggregate: Dict[str, Dict[str, float]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        aggregate[condition] = {
            "n": len(subset),
            "success_rate": sum(int(row["success"]) for row in subset) / len(subset),
            "mean_windows": sum(int(row["windows"]) for row in subset) / len(subset),
            "mean_wall_clock": sum(float(row["wall_clock"]) for row in subset) / len(subset),
            "mean_plan_wall_clock": sum(float(row["plan_wall_clock"]) for row in subset) / len(subset),
            "mean_live_physics_steps": sum(int(row["live_physics_steps"]) for row in subset) / len(subset),
            "mean_candidate_simulated_steps": sum(int(row["candidate_simulated_steps"]) for row in subset) / len(subset),
            "mean_prefix_replay_steps": sum(int(row["prefix_replay_steps"]) for row in subset) / len(subset),
            "mean_plan_nodes": sum(int(row["plan_nodes"]) for row in subset) / len(subset),
        }
    (out_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))
    print(f"rendered {len(rendered)} files; output={out_dir}; total={time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
