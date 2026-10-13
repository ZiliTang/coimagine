"""Write the Grid errata from existing raw outputs (read-only; no model re-run).

    python experiments/grid_selection_audit/write_errata.py

Reads ``summary.csv`` and ``candidates.jsonl`` from the existing phase-B run and writes
``docs/protocol_fix_v1/errata.md``. It never recomputes the model, never changes the old
raw data and never overwrites the old ``conclusion.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]


def _read_summary(path: Path) -> List[Dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _aggregate(rows: List[Dict[str, str]], depth: int, size: int) -> Dict[str, float]:
    subset = [
        row for row in rows
        if int(row["H"]) == depth and int(row["K"]) == size and row["scope"] == "all"
    ]
    keys = [
        "selected_bias", "selection_excess", "candidate_regret",
        "chosen_true_return", "uniform_candidate_return",
    ]
    return {key: sum(float(row[key]) for row in subset) / len(subset) for key in keys}


def _representative_case(candidates: Path, scene_id: int, seed: int, depth: int, chosen_id: int):
    """Model-transition counts for the representative case, from the saved raw file."""

    lengths: Counter = Counter()
    chosen_length = None
    chosen_values = None
    with candidates.open() as handle:
        for line in handle:
            record = json.loads(line)
            if (
                int(record["scene_id"]) != scene_id
                or int(record["seed"]) != seed
                or int(record["H"]) != depth
                or int(record["candidate_id"]) >= 128
            ):
                continue
            length = len(record["model_trace"])
            lengths[length] += 1
            if int(record["candidate_id"]) == chosen_id:
                chosen_length = length
                chosen_values = {
                    "J_hat": record["J_hat"],
                    "J_ref": record["J_ref"],
                    "action_sequence": record["action_sequence"],
                    "n_consistent_worlds": record["information_condition"]["n_consistent_worlds"],
                }
    return lengths, chosen_length, chosen_values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        default="outputs/migration_v1/phaseB-full/grid_selection_audit",
    )
    parser.add_argument("--out", default="docs/protocol_fix_v1/errata.md")
    parser.add_argument("--scene-id", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    audit = ROOT / args.audit_dir
    rows = _read_summary(audit / "summary.csv")
    scene_rows = [
        row for row in rows
        if row["scope"] == "scene"
        and int(row["scene_id"]) == args.scene_id
        and int(row["seed"]) == args.seed
        and int(row["H"]) == args.depth
        and int(row["K"]) == 128
    ]
    chosen_id = int(scene_rows[0]["chosen_candidate_id"]) if scene_rows else -1
    chosen_regret = float(scene_rows[0]["candidate_regret"]) if scene_rows else float("nan")
    lengths, chosen_length, chosen_values = _representative_case(
        audit / "candidates.jsonl", args.scene_id, args.seed, args.depth, chosen_id
    )

    sizes = sorted({int(row["K"]) for row in rows})
    lines = [
        "# Grid 诊断 errata（只更正解释，不重算旧模型）",
        "",
        "本文件由 `experiments/grid_selection_audit/write_errata.py` 从已有 "
        "`phaseB-full/grid_selection_audit/summary.csv` 与 `candidates.jsonl` **只读**生成；"
        "旧模型、旧动力学、旧评分、旧候选与旧原始数据均未改动，旧 `conclusion.md` 未覆盖。",
        "",
        "## H=3 固定深度：K=16/64/128 的事实",
        "",
        "| K | selected_bias | selection_excess | candidate_regret | chosen_true_return | uniform_candidate_return |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size in sizes:
        aggregate = _aggregate(rows, args.depth, size)
        lines.append(
            f"| {size} | {aggregate['selected_bias']:+.3f} | {aggregate['selection_excess']:+.3f} | "
            f"{aggregate['candidate_regret']:.3f} | {aggregate['chosen_true_return']:+.3f} | "
            f"{aggregate['uniform_candidate_return']:+.3f} |"
        )
    lines += [
        "",
        "## Errata 声明",
        "",
        "1. 当前诊断仍使用**隐藏参数、学习信念与原边缘状态传播**"
        "（`learning/belief_model.py:step_actions` 把同一 41 维边缘动态复制 16 份后按 b(θ) 混合）；"
        "未使用 `diagnostics/conditional_planner.py` 的逐参数条件状态。",
        "2. **成功概率阈值化导致两边收益语义不一致**：模型侧把 `P(in goal) ≥ 0.5` 当作 100% "
        "成功并截断后续代价（`diagnostics/grid_selection_audit.py:model_outcome`），"
        "真实侧是与观测一致世界上的精确期望（同文件 `true_outcome`）。",
        f"3. 代表 H=3 案例（scene={args.scene_id}, seed={args.seed}, H={args.depth}, K=128，"
        f"选中 candidate_id={chosen_id}）的模型分**只推演 {chosen_length} 步**后触发阈值；"
        f"该场景 128 条候选中模型推演步数分布为 {dict(sorted(lengths.items()))}。"
        + (f" 选中候选 J_hat={chosen_values['J_hat']:+.2f}，J_ref={chosen_values['J_ref']:+.2f}，"
           f"一致世界数={chosen_values['n_consistent_worlds']}。" if chosen_values else ""),
        f"4. 固定 H=3 下**实际选中收益随 K 增加而提高**"
        f"（chosen_true_return："
        + " → ".join(f"K={size} {_aggregate(rows, args.depth, size)['chosen_true_return']:+.3f}" for size in sizes)
        + "）；不能据此声称“搜索越宽越差”。",
        "5. 旧结论应限定为**当前管线的现象**；本轮未隔离纯动力学误差，也未修复概率版吸收传播。",
        "6. 结论生成器 `_conclusion`/`conclusion_from_rows` 已改为只输出事实表与"
        "“评分协议待修正，暂不作机制判断”，不再写死“支持”或固定数字；本轮不重算旧数据。",
        "",
        f"（代表案例 regret={chosen_regret:.3f}；selection_excess 等逐场景值见原 `summary.csv`。）",
        "",
    ]
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {target}")
    print(f"representative case model-trace length={chosen_length}, distribution={dict(sorted(lengths.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
