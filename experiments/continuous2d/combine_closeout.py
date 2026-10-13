"""Closeout A2: combine unchanged-scene results with the targeted rerun of the modified scenes.

    python experiments/continuous2d/combine_closeout.py \
        --previous outputs/protocol_fix_v1/<run_id>/continuous2d \
        --targeted outputs/closeout_v1/<run_id>/targeted/continuous2d \
        --out      outputs/closeout_v1/<run_id>/combined

Every combined row records ``source_run`` (which run produced it), ``rerun_this_round``,
``scene_version``, ``scene_content_hash`` and ``code_version``. The table is labelled as a
mix: unchanged scenes reuse the previous run, the three modified scenes were rerun. It is
not presented as a full rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from coimagine.continuous2d.legality import scene_content_hash
from coimagine.continuous2d.scenes import default_scenes

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"random": "随机(3 种子)", "mpc_depth1": "精确模型 depth1", "mpc_depth3": "精确模型 depth3"}
ORDER = {"random": 0, "mpc_depth1": 1, "mpc_depth3": 2}


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--targeted", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    previous_dir = ROOT / args.previous
    targeted_dir = ROOT / args.targeted
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = {scene.name: scene for scene in default_scenes()}
    modified = {name for name, scene in specs.items() if scene.scene_version != "v1"}
    previous = {(row["condition"], row["scene"], row["seed"]): row for row in _read(previous_dir / "results.csv")}
    targeted = {(row["condition"], row["scene"], row["seed"]): row for row in _read(targeted_dir / "results.csv")}

    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for scene in specs:
        for condition in ("random", "mpc_depth1", "mpc_depth3"):
            seeds = sorted({seed for (cond, sc, seed) in previous if cond == condition and sc == scene})
            if scene in modified:
                seeds = sorted({seed for (cond, sc, seed) in targeted if cond == condition and sc == scene})
            for seed in seeds:
                key = (condition, scene, seed)
                if scene in modified:
                    source = targeted.get(key)
                    source_run = args.targeted
                else:
                    source = previous.get(key)
                    source_run = args.previous
                if source is None:
                    missing.append(f"{condition}/{scene}/{seed}")
                    continue
                row = dict(source)
                row.update({
                    "scene_version": specs[scene].scene_version,
                    "scene_content_hash": scene_content_hash(specs[scene]),
                    "source_run": source_run,
                    "rerun_this_round": int(scene in modified),
                })
                rows.append(row)

    fieldnames = [
        "condition", "scene", "category", "seed", "scene_version", "scene_content_hash",
        "source_run", "rerun_this_round", "success", "windows", "termination",
        "no_feasible_plan", "success_time", "control_usage", "terminal_distance",
        "plan_nodes", "candidates", "live_physics_steps", "candidate_simulated_steps",
        "prefix_replay_steps", "plan_wall_clock", "wall_clock", "failure_distribution", "chosen",
    ]
    with (out_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # per-scene table with the source column
    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["scene"], row["category"])].append(row)
    lines = [
        "# 连续平台收尾汇总（未变场景沿用上一轮、修改场景本轮定向复跑）",
        "",
        f"修改场景（scene_version=closeout_v2）：{', '.join(sorted(modified))}；其余 7 个场景几何未变，"
        f"沿用 `{args.previous}`。每行 source 注明来源。",
        "",
        "| 条件 | 场景 | source | 成功 | termination | 窗口 | 实际时间(s) | 控制用量 | 终态距离 | 节点 | 候选模拟步 | 精确重放步 | 规划墙钟(s) |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (condition, scene, category), group in sorted(
        grouped.items(), key=lambda item: (item[0][1], ORDER.get(item[0][0], 9))
    ):
        successes = sum(int(row["success"]) for row in group)
        terms = Counter(str(row["termination"]) for row in group)
        source = str(group[0]["source_run"])
        rerun = "本轮" if int(group[0]["rerun_this_round"]) else "沿用"
        def mean(key: str) -> float:
            return sum(float(row[key]) for row in group) / len(group)
        lines.append(
            f"| {LABELS[condition]} | {scene}（{category}） | {rerun} | {successes}/{len(group)} | "
            f"{', '.join(f'{k}:{v}' for k, v in sorted(terms.items()))} | {mean('windows'):.1f} | "
            f"{mean('success_time'):.1f} | {mean('control_usage'):.1f} | {mean('terminal_distance'):.2f} | "
            f"{mean('plan_nodes'):.0f} | {mean('candidate_simulated_steps'):.0f} | "
            f"{mean('prefix_replay_steps'):.0f} | {mean('plan_wall_clock'):.1f} |"
        )
    lines += ["", "注：混合表，不是本轮全表重跑；场景版本与内容哈希见 results.csv。", ""]
    (out_dir / "results_table.md").write_text("\n".join(lines), encoding="utf-8")

    aggregate: Dict[str, Dict[str, float]] = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        aggregate[condition] = {
            "n": len(subset),
            "success_rate": sum(int(row["success"]) for row in subset) / len(subset),
            "mean_windows": sum(int(row["windows"]) for row in subset) / len(subset),
            "mean_wall_clock": sum(float(row["wall_clock"]) for row in subset) / len(subset),
        }
    summary = {
        "label": "unchanged scenes reused, modified scenes rerun this round",
        "modified_scenes": sorted(modified),
        "previous_run": args.previous,
        "targeted_run": args.targeted,
        "aggregate": aggregate,
        "missing_rows": missing,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))
    print(json.dumps({"aggregate": aggregate, "missing_rows": missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
