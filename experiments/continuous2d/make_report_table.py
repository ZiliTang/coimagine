"""Render a results table from ``results.csv`` (all scenes, no mixed denominators).

    python experiments/continuous2d/make_report_table.py --run <dir>

Cells are per (condition, scene); random rows aggregate the three fixed seeds and the
success column is successes/3; MPC rows are single deterministic runs. Reports success,
terminations, executed windows, real time, control usage, terminal distance, search nodes,
candidate simulated steps, exact prefix replay steps and planning wall clock.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"random": "随机(3 种子)", "mpc_depth1": "精确模型 depth1", "mpc_depth3": "精确模型 depth3"}
ORDER = {"random": 0, "mpc_depth1": 1, "mpc_depth3": 2}


def build(run: Path) -> str:
    with (run / "results.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["scene"], row["category"])].append(row)

    lines = [
        "| 条件 | 场景 | 成功 | termination | 窗口 | 实际时间(s) | 控制用量 | 终态距离 | 搜索节点 | 候选模拟步 | 精确重放步 | 规划墙钟(s) |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (condition, scene, category), group in sorted(
        grouped.items(), key=lambda item: (item[0][1], ORDER.get(item[0][0], 9))
    ):
        successes = sum(int(row["success"]) for row in group)
        terms = Counter(row["termination"] for row in group)
        term_text = ", ".join(f"{name}:{count}" for name, count in sorted(terms.items()))
        def mean(key: str) -> float:
            return sum(float(row[key]) for row in group) / len(group)
        lines.append(
            f"| {LABELS[condition]} | {scene}（{category}） | {successes}/{len(group)} | {term_text} | "
            f"{mean('windows'):.1f} | {mean('success_time'):.1f} | {mean('control_usage'):.1f} | "
            f"{mean('terminal_distance'):.2f} | {mean('plan_nodes'):.0f} | "
            f"{mean('candidate_simulated_steps'):.0f} | {mean('prefix_replay_steps'):.0f} | "
            f"{mean('plan_wall_clock'):.1f} |"
        )
    lines += [
        "",
        "注：随机行为 3 个固定种子；MPC 每场景一次确定性运行。depth3 为 beam=3 的宽度受限搜索，"
        "不是全局最优。精确重放步为按前缀重放的物理步数，不计入候选模拟步。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    table = build(ROOT / args.run)
    target = ROOT / (args.out or (Path(args.run) / "results_table.md"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(table, encoding="utf-8")
    print(table)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
