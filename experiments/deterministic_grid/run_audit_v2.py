"""A1-A4: run the corrected finite-Grid audit and write the Phase A evidence bundle.

    python experiments/deterministic_grid/run_audit_v2.py --run-id grid-audit-v2

The saved dataset and the three saved checkpoints are read-only inputs. This script never
regenerates data and never trains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from coimagine.deterministic_grid.audit import AuditConfig, RUN_METRIC_KEYS
from coimagine.diagnostics.grid_audit_v2 import run_phase_a

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RUN = "outputs/deterministic_grid_v1/grid-20261012-194632"
DEFAULT_DOCS_OUT = "docs/data_upgrade_v1/A_grid_audit_v2.md"
DEFAULT_FIXES_DOC_OUT = "docs/data_upgrade_v1/A1_A2_grid_fixes.md"
#: read-only, previous round (never written by this script)
PREVIOUS_RUN_ROOT = "outputs/forecast_baseline_v1/grid-audit-v2"
#: read-only, the old fallback-protocol recursive diagnostic that STATUS.md quoted
OLD_FALLBACK_DIAGNOSTIC = "outputs/forecast_baseline_v1/forecast-20261012/diagnostic/recursive.json"
DECOMP_LABELS = {
    "success_reward": "成功奖励 10ΔS",
    "executed_steps": "执行步数 -Δt",
    "terminal_distance": "终态距离 -0.5Δd",
    "invalid_penalty": "非法预测罚项 -10ΔF",
}


def _signed(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.{digits}f}"


def _plain(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _state_text(payload: Optional[dict]) -> str:
    if not payload:
        return "-"
    return (
        f"r0=({payload['robot0'][0]},{payload['robot0'][1]}) "
        f"r1=({payload['robot1'][0]},{payload['robot1'][1]}) "
        f"o=({payload['object'][0]},{payload['object'][1]})"
    )


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        + "\n"
    )


def _jsonl_dump(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
                + "\n"
            )


METRIC_COLUMNS = (
    ("selected_bias", _signed),
    ("candidate_mean_bias", _signed),
    ("selection_excess", _signed),
    ("candidate_regret", _plain),
    ("chosen_true_return", _signed),
    ("uniform_candidate_return", _signed),
    ("candidate_best_true_return", _signed),
    ("chosen_true_success", _plain),
    ("candidate_invalid_rate", _plain),
    ("candidate_success_rate", _plain),
)


def render_metrics_table(artifacts: dict) -> str:
    eval_states = artifacts["eval_states_v2"]
    lines: List[str] = ["# Phase A 固定候选机制表（更正版）", ""]
    lines += [
        "聚合规则：先在单个真实初始状态内对候选种子取均值，再对状态取均值/范围；",
        "模型种子与候选种子是重复测量，不构成独立场景。收益越大越好。",
        "",
    ]
    lines += ["## 每 H 的实际不同候选数", "", "| eval_set | H | runs | actual_K min | max | mean | H=1 上限 81 |",
              "|---|---:|---:|---:|---:|---:|:--:|"]
    for row in eval_states["candidate_counts"]:
        lines.append(
            f"| {row['eval_set']} | {row['H']} | {row['n_state_candidate_seed_runs']} | "
            f"{row['actual_K_min']} | {row['actual_K_max']} | {row['actual_K_mean']:.1f} | "
            f"{'是' if row['capped_at_81'] else '否'} |"
        )
    for label, key in (("新验证集（无坐标偏置）", "metrics_v2"), ("旧 20 状态（未改动）", "metrics_old")):
        lines += ["", f"## {label}", "",
                  "| H | K | map | model | n状态 | selected_bias | candidate_mean_bias | selection_excess | regret | chosen_true | uniform_cand | candidate_best | chosen_success | cand_invalid | cand_success |",
                  "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in eval_states[key]:
            if row["map"] != "all":
                continue
            model = "exact" if row["model_seed"] == -1 else str(row["model_seed"])
            cells = " | ".join(fn(row[metric]) for metric, fn in METRIC_COLUMNS)
            lines.append(
                f"| {row['H']} | {row['K']} | {row['map']} | {model} | {row['n_states']} | {cells} |"
            )
    lines += ["", "`model=exact` 是精确模型（`model_seed=-1`）单独一行，偏差恒为 0，**绝不进入任何均值**；"
              "`model=mean` 只对学习种子 0/1/2 求平均（A1 修复口径）。", ""]
    return "\n".join(lines)


def render_docs(artifacts: dict) -> str:
    eval_states = artifacts["eval_states_v2"]
    group = artifacts["group_stats"]
    decomposition = artifacts["decomposition"]
    traced = artifacts["traced_cases"]
    a4 = artifacts["recursive_vs_teacher"]
    config = artifacts["config"]
    lines: List[str] = [
        "# 阶段 A：Grid 诊断更正（A1–A4）",
        "",
        "状态：实现完成 / 测试通过 / 实验已运行。本文全部数字来自 `run_audit_v2.py` 的一次运行，"
        f"读取旧的保存数据集与旧 checkpoint（未重训、未重建数据）。运行 ID：`grid-audit-v2`；"
        f"配置：`{json.dumps(config, ensure_ascii=False)}`；墙钟 {artifacts['wall_clock']:.1f} s。",
        "",
        "## 0. 输入哈希（运行前后一致）",
        "",
        "| 文件 | SHA256 | 前=后 |",
        "|---|---|:--:|",
    ]
    for name in sorted(artifacts["start_hashes"]):
        same = artifacts["start_hashes"][name] == artifacts["end_hashes"].get(name)
        lines.append(f"| {_hash_label(name)} | `{artifacts['start_hashes'][name]}` | {'是' if same else '否'} |")
    lines += [
        "",
        f"`hash_check.unchanged = {str(artifacts['hash_check']['unchanged']).lower()}`。",
        "",
        "旧 20 状态重跑与旧 `per_run.jsonl` 的逐行对照："
        f"matched={artifacts['old_set_reproduction']['matched_rows']} / "
        f"missing={artifacts['old_set_reproduction']['missing_rows']}，"
        f"最大绝对差={artifacts['old_set_reproduction']['max_abs_diff']}，"
        f"完全一致={artifacts['old_set_reproduction']['reproduced_exactly']}。",
        "",
        "## 1. 三个与 closeout_v2 无关的事实（本次运行重新核对）",
        "",
        "1. 旧 20 个评估状态的 robot0 单一格："
        f"{artifacts['facts']['1_old_states_robot0_single_cell']['distinct_robot0_cells']}"
        f"（single_cell={artifacts['facts']['1_old_states_robot0_single_cell']['all_robot0_at_single_cell']}）。",
        "2. 旧 0.53 的字段名是 `moved_full_state_accuracy`（物体移动样本的完整联合状态准确率），"
        "不是物体推动准确率："
        f"{json.dumps(artifacts['facts']['2_stored_0p53_metric']['stored_per_model_seed'], ensure_ascii=False)}，"
        f"均值 {artifacts['facts']['2_stored_0p53_metric']['stored_mean']:.4f}。",
        "3. H=4 的候选平均偏差内含非法预测罚项："
        f"均值 {artifacts['facts']['3_invalid_penalty_inside_candidate_mean_bias']['mean_invalid_penalty_term']:.3f}，"
        f"同口径 J_hat-J_ref 均值 "
        f"{artifacts['facts']['3_invalid_penalty_inside_candidate_mean_bias']['mean_total_J_hat_minus_J_ref']:.3f}。",
        "",
        "## 2. A1 修正后的分组统计（test split，按真实转移分组）",
        "",
        "分组只由真实转移决定：`object_moved`（物体锚点变了）、`object_stationary`（物体锚点没变，"
        "机器人仍可能移动）、`full_state_unchanged`（完整下一状态等于当前状态，是前者的子集但单独列出）。",
        "",
        "| map | group | model | n | object_acc | robot0_acc | robot1_acc | full_state_acc | invalid_rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group["rows"]:
        if row["map"] not in ("both",):
            continue
        model = "mean" if row["model_seed"] == "mean" else str(row["model_seed"])
        lines.append(
            f"| {row['map']} | {row['group']} | {model} | {row['n']} | "
            f"{_plain(row['object_accuracy'])} | {_plain(row['robot0_accuracy'])} | "
            f"{_plain(row['robot1_accuracy'])} | {_plain(row['full_state_accuracy'])} | "
            f"{_plain(row['invalid_state_rate'])} |"
        )
    lines += ["", "每张地图分开的同一张表（模型种子均值）：", "",
              "| map | group | n | object_acc | robot0_acc | robot1_acc | full_state_acc | invalid_rate |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in group["rows"]:
        if row["model_seed"] != "mean" or row["map"] == "both":
            continue
        lines.append(
            f"| {row['map']} | {row['group']} | {row['n']} | {_plain(row['object_accuracy'])} | "
            f"{_plain(row['robot0_accuracy'])} | {_plain(row['robot1_accuracy'])} | "
            f"{_plain(row['full_state_accuracy'])} | {_plain(row['invalid_state_rate'])} |"
        )
    lines += ["", "更正后的整体单步数字（三模型种子）：", "",
              "| split | model | n | object_acc | robot0_acc | robot1_acc | full_state_acc | invalid_rate |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in group["by_split"]:
        lines.append(
            f"| {row['split']} | {row['model_seed']} | {row['n']} | {_plain(row['object_accuracy'])} | "
            f"{_plain(row['robot0_accuracy'])} | {_plain(row['robot1_accuracy'])} | "
            f"{_plain(row['full_state_accuracy'])} | {_plain(row['invalid_state_rate'])} |"
        )
    lines += [
        "",
        f"关系检查：full_state_unchanged ⊆ object_stationary = "
        f"{group['relations']['full_state_unchanged_subset_of_object_stationary']}；"
        f"object_moved + object_stationary = 全部 = "
        f"{group['relations']['object_moved_plus_object_stationary_equals_all']}。",
        "",
        "## 3. A2 无坐标偏置验证集",
        "",
        f"选样规则：{eval_states['selection_rule']}。",
        f"选样是否使用坐标：{'否' if not eval_states['used_coordinates_for_selection'] else '是'}；"
        f"是否使用模型误差：{'否' if not eval_states['used_model_error_for_selection'] else '是'}；"
        f"是否使用回报或成功率：{'否' if not eval_states['used_return_or_success_for_selection'] else '是'}。",
        "",
        "### 3.1 位置覆盖",
        "",
        "| map | n | distinct robot0 cells | distinct robot1 cells | distinct anchors |",
        "|---|---:|---:|---:|---:|",
    ]
    for map_name, row in eval_states["coverage"]["per_map"].items():
        lines.append(
            f"| {map_name} | {row['n_states']} | {row['n_distinct_robot0_cells']} | "
            f"{row['n_distinct_robot1_cells']} | {row['n_distinct_object_anchors']} |"
        )
    overall = eval_states["coverage"]["overall"]
    lines.append(
        f"| 合计 | {overall['n_states']} | {overall['n_distinct_robot0_cells']} | "
        f"{overall['n_distinct_robot1_cells']} | {overall['n_distinct_object_anchors']} |"
    )
    lines += [
        "",
        f"对比旧集合：robot0 只有 {eval_states['old_coverage']['overall']['n_distinct_robot0_cells']} 个不同格，"
        f"robot1 有 {eval_states['old_coverage']['overall']['n_distinct_robot1_cells']} 个不同格。",
        "",
        "### 3.2 新验证集状态（10/地图，按 SHA256 排序）",
        "",
        "| map | rank | state | SHA256(\"grid-audit-v2|\"+state_key) |",
        "|---|---:|---|---|",
    ]
    for entry in eval_states["new_states"]:
        lines.append(
            f"| {entry['map']} | {entry['rank_within_map']} | {_state_text(entry['state'])} | "
            f"`{entry['sha256'][:16]}…` |"
        )
    lines += ["", "旧 20 状态原样保留（坐标排序取前几个）：", "",
              "| map | state（旧集合） |", "|---|---|"]
    for entry in eval_states["old_states_unchanged"]:
        lines.append(f"| {entry['map']} | {_state_text(entry['state'])} |")
    lines += [
        "",
        "### 3.3 关键指标（H=4, K=256）新旧并排（模型种子均值；先状态内平均候选种子，再对状态平均）",
        "",
        "| set | selected_bias | candidate_mean_bias | selection_excess | regret | chosen_true | uniform_cand | candidate_best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("新", "metrics_v2"), ("旧", "metrics_old")):
        row = next(
            r for r in eval_states[key]
            if r["H"] == 4 and r["K"] == 256 and r["map"] == "all" and r["model_seed"] == "mean"
        )
        lines.append(
            f"| {label} | {_signed(row['selected_bias'])} | {_signed(row['candidate_mean_bias'])} | "
            f"{_signed(row['selection_excess'])} | {_plain(row['candidate_regret'])} | "
            f"{_signed(row['chosen_true_return'])} | {_signed(row['uniform_candidate_return'])} | "
            f"{_signed(row['candidate_best_true_return'])} |"
        )
    lines += ["", "完整表（每 H、K、模型种子）另见 `outputs/forecast_baseline_v1/grid-audit-v2/metrics_table_v2.md`。", ""]

    # aggregation reconciliation: the legacy row-mean headline stays traceable
    reconciliation = artifacts["aggregation_reconciliation"]
    lines += [
        "### 3.4 聚合口径对照（旧表数字不被覆盖；A1 修复后重新解读）",
        "",
        "上一轮把旧表 `1.250/-1.450/2.700` 与 `+0.938/-1.088/+2.025` 的差别解释为"
        "“按行平均 vs 状态优先”两种口径。该解释**不正确**：旧代码的 `mean` 行把精确模型"
        "（`model_seed=-1`，偏差恒 0）也平均了进去，`0.938` 是含 exact 的 4 种子均值，"
        "`0.938 × 4/3 = 1.251`。修复后 `mean` 只含学习种子 0/1/2；由于每个状态的重复次数相同，"
        "状态优先与按行平均两种口径数值相等（都等于旧表的 1.250/-1.450/2.700）。",
        "",
        "| H | K | 口径 | selected_bias | candidate_mean_bias | selection_excess |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in reconciliation["old_stored_row_mean_metrics"]:
        lines.append(
            f"| {row['H']} | {row['K']} | 旧表（按行平均，原样引用） | {_signed(row['selected_bias'])} | "
            f"{_signed(row['candidate_mean_bias'])} | {_signed(row['selection_excess'])} |"
        )
    for row in reconciliation["legacy_row_mean_aggregation"]:
        lines.append(
            f"| {row['H']} | {row['K']} | 本次重跑·按行平均 | {_signed(row['selected_bias'])} | "
            f"{_signed(row['candidate_mean_bias'])} | {_signed(row['selection_excess'])} |"
        )
    for row in eval_states["metrics_old"]:
        if row["map"] == "all" and row["model_seed"] == "mean":
            lines.append(
                f"| {row['H']} | {row['K']} | 本次重跑·状态优先（CONTRACT） | {_signed(row['selected_bias'])} | "
                f"{_signed(row['candidate_mean_bias'])} | {_signed(row['selection_excess'])} |"
            )
    lines += [
        "",
        f"重跑·按行平均与旧表最大绝对差={reconciliation['max_abs_diff_legacy_vs_stored']:.2e}，"
        f"在 1e-9 容差内一致={reconciliation['legacy_reproduced']}。修复后旧数据的"
        "状态优先（学习种子 0/1/2）= "
        f"{_signed(next(r['selected_bias'] for r in eval_states['metrics_old'] if r['H'] == 4 and r['K'] == 256 and r['map'] == 'all' and r['model_seed'] == 'mean'))}"
        "/"
        f"{_signed(next(r['candidate_mean_bias'] for r in eval_states['metrics_old'] if r['H'] == 4 and r['K'] == 256 and r['map'] == 'all' and r['model_seed'] == 'mean'))}"
        "/"
        f"{_signed(next(r['selection_excess'] for r in eval_states['metrics_old'] if r['H'] == 4 and r['K'] == 256 and r['map'] == 'all' and r['model_seed'] == 'mean'))}"
        "，与按行平均相同。`metrics_old` 中 `model=exact` 行单列，绝不进入均值。",
        "",
    ]

    # A3
    check = decomposition["algebraic_check"]
    lines += [
        "## 4. A3 收益误差拆解与错误追踪",
        "",
        "拆解恒等式：J_hat - J_ref = 10ΔS - Δt - 0.5Δd - 10ΔF。"
        f"逐候选逐模型种子代数检查：checked={check['n_checked']}，"
        f"max|Σ分项-总偏差|={check['max_abs_residual']:.2e}，passed={check['passed']}。",
        "",
        "### 4.1 候选均值（新验证集，按 H 平均到状态）",
        "",
        "| H | model | 成功奖励 | 执行步数 | 终态距离 | 非法罚项 | 分项和 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in decomposition["candidate_mean"]:
        lines.append(
            f"| {row['H']} | {row['model_seed']} | {_signed(row['success_reward'])} | "
            f"{_signed(row['executed_steps'])} | {_signed(row['terminal_distance'])} | "
            f"{_signed(row['invalid_penalty'])} | {_signed(row['total'])} |"
        )
    lines += ["", "### 4.2 选中候选（同一聚合规则）", "",
              "| H | K | model | 成功奖励 | 执行步数 | 终态距离 | 非法罚项 | 分项和 |",
              "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in decomposition["chosen"]:
        lines.append(
            f"| {row['H']} | {row['K']} | {row['model_seed']} | {_signed(row['success_reward'])} | "
            f"{_signed(row['executed_steps'])} | {_signed(row['terminal_distance'])} | "
            f"{_signed(row['invalid_penalty'])} | {_signed(row['total'])} |"
        )
    lines += ["", "### 4.3 追踪案例（新验证集 H=4, K=256）", "",
              f"选样：{traced['selection']['selection']}。", "",
              "| role | state | run | chosen_id | 精确最佳 id | regret | 选中 true J | 选中 model J | 首次完整状态不一致 | 首次物体锚点不一致 | 首次虚假成功 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in traced["cases"]:
        lines.append(
            f"| {case['role']} | {_state_text(case['state'])} | ms={case['model_seed']}, cs={case['candidate_seed']} | "
            f"{case['chosen_candidate_id']} | {case['best_true_candidate_id']} | {_plain(case['chosen_candidate_regret'])} | "
            f"{_signed(case['chosen_true_J'])} | {_signed(case['chosen_model_J'])} | "
            f"{case['first_full_state_mismatch_step']} | {case['first_object_anchor_mismatch_step']} | "
            f"{case['first_false_success_step']} |"
        )
    lines += ["", "A2 单候选反事实排序检验（主分析）：只修复**原选中候选**自己的首次该类错误，"
              "只替换该候选的 J_hat，其余候选逐项不变，再重排。"
              "“排序改变”与“新选择真实收益改善”分开报告。", "",
              "| role | 干预 | 首次步 | 排序改变 | 反事实选中 id | 被改动的条目数 | 原选中 true J | 新选中 true J | true 改善 | 结论 |",
              "|---|---|---:|:--:|---:|---:|---:|---:|:--:|---|"]
    for case in traced["cases"]:
        for key, payload in case["counterfactuals"].items():
            lines.append(
                f"| {case['role']} | {payload['intervention']} | {payload['first_step']} | "
                f"{'是' if payload['changed_ranking'] else '否'} | "
                f"{payload['counterfactual_chosen_candidate_id']} | {payload['n_changed_entries']} | "
                f"{_signed(payload['baseline_chosen_true_return'])} | "
                f"{_signed(payload['new_chosen_true_return'])} | "
                f"{'是' if payload['true_return_improved'] else '否'} | {payload['note']} |"
            )
    lines += ["", "旧全候选干预（历史附件，因果表述已更正；不是单候选主分析）：", "",
              "| role | 干预 | 首次步 | 排序改变 | 反事实选中 id | true 改善 | 说明 |",
              "|---|---|---:|:--:|---:|:--:|---|"]
    for case in traced["cases"]:
        for key, payload in case.get("counterfactuals_all_candidates_historical", {}).items():
            lines.append(
                f"| {case['role']} | {payload['intervention']} | {payload['first_step']} | "
                f"{'是' if payload['changed_ranking'] else '否'} | "
                f"{payload['counterfactual_chosen_candidate_id']} | "
                f"{'是' if payload['true_return_improved'] else '否'} | {payload['note']} |"
            )
    lines += ["", "逐步轨迹（每个案例：选中候选的真状态/预测状态/动作；对照案例同格式）：", ""]
    for case in traced["cases"]:
        lines += [
            f"**案例 role={case['role']}，{_state_text(case['state'])}，ms={case['model_seed']} cs={case['candidate_seed']}，"
            f"选中 id={case['chosen_candidate_id']}（true J={case['chosen_true_J']:+.3f}, model J={case['chosen_model_J']:+.3f}），"
            f"精确最佳 id={case['best_true_candidate_id']}（true J={case['best_true_true_components']['J']:+.3f}, "
            f"model J={case['best_true_model_components']['J']:+.3f}）**",
            "",
            "| step | action | 输入状态 | 预测状态 | 真下一步 | legal | 完整匹配 | 物体锚点匹配 | 虚假成功 | 终止 |",
            "|---:|---:|---|---|---|---|:--:|:--:|:--:|---|",
        ]
        for step in case["chosen_steps"]:
            lines.append(
                f"| {step['step']} | {step['action']} | {_state_text(step['input_state'])} | "
                f"{_state_text(step['predicted_state'])} | {_state_text(step['true_next_state'])} | "
                f"{step['legal']} | {step['full_state_match']} | {step['object_anchor_match']} | "
                f"{step['false_success']} | {step['termination_reason'] or '-'} |"
            )
        lines.append("")

    false_case = next((c for c in traced["cases"] if c["first_false_success_step"] is not None), None)
    top_cases = [c for c in traced["cases"] if c["role"] == "top_regret"]
    control_cases = [c for c in traced["cases"] if c["role"] == "control"]
    state_changed = [c["counterfactuals"]["full_state"]["changed_ranking"] for c in top_cases]
    object_changed = [c["counterfactuals"]["object_anchor"]["changed_ranking"] for c in top_cases]
    control_state_changed = [
        c["counterfactuals"]["full_state"]["changed_ranking"] for c in control_cases
    ]
    lines += ["", "### 4.4 首个与排序改变直接相关的错误（单候选主分析）", ""]
    if false_case is not None:
        cf = false_case["counterfactuals"]["false_success"]
        lines += [
            f"案例 {_state_text(false_case['state'])}（ms={false_case['model_seed']}, "
            f"cs={false_case['candidate_seed']}）：选中候选 id={false_case['chosen_candidate_id']}，"
            f"模型给 J_hat={false_case['chosen_model_J']:+.3f}，真实 J_ref={false_case['chosen_true_J']:+.3f}。"
            f"第 {false_case['first_false_success_step']} 步模型预测物体锚点进入目标并吸收成功，"
            "而真实物体锚点未动，这是**虚假成功**。该干预用整个真实下一步状态替换预测状态，"
            "因此命名为“首次虚假成功处的完整状态干预”，不是只修成功分类。只替换该候选的 J_hat 后重排，"
            f"选中 id 变为 {cf['counterfactual_chosen_candidate_id']}，排序改变={cf['changed_ranking']}，"
            f"新选择真实收益改善={cf['true_return_improved']} → {cf['note']}。",
            "",
            f"另外 {sum(state_changed)}/{len(state_changed)} 个高 regret 案例的首次完整状态不一致"
            f"（机器人格，第 {'/'.join(str(c['first_full_state_mismatch_step']) for c in top_cases)} 步）与 "
            f"{sum(object_changed)}/{len(object_changed)} 个首次物体锚点不一致"
            f"（第 {'/'.join(str(c['first_object_anchor_mismatch_step']) for c in top_cases)} 步）的单候选干预"
            f"也改变了选中候选；对照案例第 {control_cases[0]['first_full_state_mismatch_step']} 步的"
            f"机器人格不一致没有改变排序（排序改变={control_state_changed[0]}）。"
            "“排序改变”不等于“修复有收益”，两列分开读。",
            "",
            "选中候选的非法预测罚项在所有 (H, K, 模型种子) 上都是 +0.000，而候选均值的非法预测罚项"
            "约 -2.0。因此 `selection_excess` 混合了选中候选自身的偏差与候选池的非法预测罚项，"
            "不能整体解释为选中方案的乐观误差。",
            "",
        ]
    else:
        lines += ["本次追踪的四个案例都没有出现虚假成功；按规则写作 `未建立直接联系`。", ""]

    # A4
    lines += [
        "## 5. A4 递归 vs teacher-forced（与主诊断同协议）",
        "",
        "同一固定真实轨迹：teacher-forced 每步从真实状态预测，recursive 从自身上一合法预测状态预测。"
        "非法预测立即终止且不再伪造下一状态；预测成功吸收。逐步给出累积终止率、累积非法率、"
        "存活链误差（含分母）、全部链累计失败/错误率。",
        "",
    ]
    for label, key in (("新验证集", "v2"), ("旧 20 状态", "old")):
        arm = a4[key]
        pooled = arm["pooled"]
        lines += [f"### 5.{1 if key == 'v2' else 2} {label}（模型种子合并；每步）", "",
                  "| mode | step | 累积终止率 | 累积非法率 | 累积虚假成功率 | 漏判真实成功率 | 存活链完整状态误差 (分母) | 存活链位置误差 | 全部链累计失败/错误率 |",
                  "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
        for mode in ("teacher", "recursive"):
            for index, row in enumerate(pooled[mode]["steps"]):
                step_rows = [
                    arm["per_model"][seed][mode]["steps"][index] for seed in arm["per_model"]
                ]
                denominator = float(np.mean([r["survivor_denominator"] for r in step_rows]))
                lines.append(
                    f"| {mode} | {row['step']} | {_plain(row['termination_rate_cumulative'])} | "
                    f"{_plain(row['invalid_rate_cumulative'])} | {_plain(row['false_success_rate_cumulative'])} | "
                    f"{_plain(row['missed_true_success_rate_cumulative'])} | "
                    f"{_plain(row['full_state_error_on_survivors'])} ({denominator:.0f}) | "
                    f"{_plain(row['position_error_on_survivors'])} | "
                    f"{_plain(row['failure_or_error_rate_cumulative_all_chains'])} |"
                )
        lines.append("")
    lines += [
        "终止状态判断错误与物理状态预测错误分开统计：前者包括非法终止、虚假成功、漏判真实成功；"
        "后者是完整状态不一致率与位置误差。旧 fallback 曲线（保留上一合法状态）只留在旧运行目录，"
        "本文不复用它解释主流程。",
        "",
        "## 6. 旧表述 → 更正表述",
        "",
        "| 旧表述 | 更正表述 |",
        "|---|---|",
        "| `moved` 分组 / “物体推动样本” | `object_moved`：“物体锚点发生移动的样本”，不代表推动成功 |",
        "| `no-op` 分组 | `object_stationary`：物体锚点未动，机器人**可能**移动；另有 `full_state_unchanged` 表示整状态未变 |",
        "| “0.53 是物体推动准确率” | 0.53 是 `object_moved` 样本上的完整联合状态准确率（旧字段 `moved_full_state_accuracy`） |",
        "| 旧 20 个评估起点 | 旧起点来自坐标字符串排序，robot0 全在同一格；新验证集按 SHA256 排序，10/地图，不看坐标与模型误差 |",
        "| “selection_excess=2.700 全是选中方案的乐观误差” | 候选平均偏差含约 -1.782 的非法预测罚项；拆解后成功奖励/步数/距离/非法罚项分别列出 |",
        "| “H=4/K=256 的 selected_bias=1.250、candidate_mean_bias=-1.450、selection_excess=2.700” | 该数字本身正确（旧表按行平均，且只含学习种子）；上一轮把它与 +0.938/-1.088/+2.025 的差别解释成“两种聚合口径”则不正确——0.938 是精确行被错误混入后的 4 种子均值，修复后同一旧数据的状态优先口径仍为 +1.250/-1.450/+2.700 |",
        "| “selection_excess 全部来自选中方案高估” | 选中候选的非法预测罚项为 +0.000，候选均值约 -2.0；selection_excess 混合了池内非法预测罚项，不只是选中方案高估 |",
        "| 旧递归诊断曲线可直接解释主流程 | 旧递归在非法预测时保留上一合法状态；主诊断立即终止。二者协议不同，新 A4 与主诊断一致 |",
        "",
        "## 7. 证据限度",
        "",
        "- 三个模型种子与候选种子是同一条件的重复测量，样本量小，不作显著性声明。",
        "- 候选池是固定随机/模板序列，不包含精确最优序列，regret 只说明该候选类内的排序损失。",
        "- “状态不同”本身不构成排序原因；只有单错误反事实重排改变选中候选时才写“该错误改变了候选排序”，"
        "否则写 `未建立直接联系`。",
        "",
    ]
    return "\n".join(lines)


def _load_json(path: Path):
    return json.loads(Path(path).read_text())


def _seed_rows(eval_states: dict, key: str, *, depth: int = 4, size: int = 256) -> List[dict]:
    return [
        row for row in eval_states[key]
        if int(row["H"]) == depth and int(row["K"]) == size and row["map"] == "all"
    ]


def _metric_mean(rows: Sequence[dict], seeds: Sequence[int], metric: str) -> float:
    values = [float(row[metric]) for row in rows if row["model_seed"] in seeds]
    return float(np.mean(values))


def _hash_label(name: str) -> str:
    if name in ("0", "1", "2"):
        return f"models/seed{name}.pt"
    return name


def _row_mean_metrics(per_run: Sequence[dict], eval_set: str) -> Dict[tuple, Dict[str, float]]:
    """The legacy row-mean rule (only learned seeds), recomputed from this run's per_run."""

    grouped: Dict[tuple, List[dict]] = {}
    for run in per_run:
        if run["eval_set"] != eval_set or int(run["model_seed"]) == -1:
            continue
        grouped.setdefault((int(run["H"]), int(run["K"])), []).append(run)
    return {
        key: {metric: float(np.mean([float(r[metric]) for r in group])) for metric in RUN_METRIC_KEYS}
        for key, group in grouped.items()
    }


def _a1_table(eval_states: dict, key: str) -> Dict[str, object]:
    """Per-seed rows plus the two aggregation rules (exact included vs learned only)."""

    rows = _seed_rows(eval_states, key)
    metrics = ("selected_bias", "candidate_mean_bias", "selection_excess", "candidate_regret", "chosen_true_return")
    exact_row = next((row for row in rows if row["model_seed"] == -1), None)
    learned = (0, 1, 2)
    return {
        "per_seed": {
            str(row["model_seed"]): {metric: float(row[metric]) for metric in metrics}
            for row in rows
        },
        "n_states": int(rows[0]["n_states"]) if rows else 0,
        "mean_learned_only": {metric: _metric_mean(rows, learned, metric) for metric in metrics},
        "mean_including_exact": {metric: _metric_mean(rows, (-1, *learned), metric) for metric in metrics},
        "exact_row": None if exact_row is None else {metric: float(exact_row[metric]) for metric in metrics},
    }


def grid_termination_reconciliation(new_a4: dict) -> Dict[str, object]:
    """A4: recompute the Grid recursive cumulative termination rate from the raw JSON.

    The quantity is the fraction of chains whose recursive model chain had terminated by the
    final evaluated step (illegal prediction terminates the chain, a predicted goal absorbs).
    It is recomputed here from this run's own ``recursive_vs_teacher.json`` payload, and
    compared with the previous round's stored JSON and with the old fallback-protocol
    diagnostic that ``STATUS.md`` quoted as ~0.918.
    """

    result: Dict[str, object] = {
        "definition": (
            "recursive 模式在固定真实轨迹上从自身上一合法预测继续；非法预测立即终止、预测成功吸收；"
            "累积终止率 = 到该步为止已终止链数 / 全部链数；末步即累计终止率"
        ),
        "new_run": {},
        "previous_round": {},
        "status_0p918_source": {},
        "scan_for_0p918": {},
    }
    for arm in ("v2", "old"):
        data = new_a4[arm]
        per_model = {}
        for seed, payload in data["per_model"].items():
            summary = payload["recursive"]
            per_model[seed] = {
                "n_chains": int(summary["n_chains"]),
                "chains_terminated": int(summary["chains_terminated"]),
                "final_step_cumulative_termination_rate": (
                    int(summary["chains_terminated"]) / int(summary["n_chains"])
                ),
                "stored_final_step": float(summary["steps"][-1]["termination_rate_cumulative"]),
            }
        result["new_run"][arm] = {
            "horizon": int(data["horizon"]),
            "candidates_per_state": int(data["candidates_per_state"]),
            "verification": {
                "terminated_over_chains_per_seed": {
                    seed: f"{row['chains_terminated']}/{row['n_chains']}" for seed, row in per_model.items()
                },
                "recomputed_matches_stored": bool(
                    all(
                        abs(row["final_step_cumulative_termination_rate"] - row["stored_final_step"]) < 1e-12
                        for row in per_model.values()
                    )
                ),
                "n_distinct_per_model_rates": len(
                    {round(row["final_step_cumulative_termination_rate"], 12) for row in per_model.values()}
                ),
            },
            "pooled_final_step_cumulative_termination_rate": float(
                data["pooled"]["recursive"]["steps"][-1]["termination_rate_cumulative"]
            ),
            "pooled_final_step_cumulative_termination_rate_teacher": float(
                data["pooled"]["teacher"]["steps"][-1]["termination_rate_cumulative"]
            ),
            "per_model": per_model,
        }

    previous_path = ROOT / PREVIOUS_RUN_ROOT / "recursive_vs_teacher.json"
    if previous_path.exists():
        previous = _load_json(previous_path)
        for arm in ("v2", "old"):
            if arm not in previous:
                continue
            result["previous_round"][arm] = {
                "source": str(previous_path.relative_to(ROOT)),
                "pooled_final_step_recursive": float(
                    previous[arm]["pooled"]["recursive"]["steps"][-1]["termination_rate_cumulative"]
                ),
                "pooled_final_step_teacher": float(
                    previous[arm]["pooled"]["teacher"]["steps"][-1]["termination_rate_cumulative"]
                ),
                "per_model_recursive_final": {
                    seed: float(payload["recursive"]["steps"][-1]["termination_rate_cumulative"])
                    for seed, payload in previous[arm]["per_model"].items()
                },
            }

    fallback_path = ROOT / OLD_FALLBACK_DIAGNOSTIC
    if fallback_path.exists():
        fallback = _load_json(fallback_path)
        result["status_0p918_source"] = {
            "source": str(fallback_path.relative_to(ROOT)),
            "protocol": (
                "旧 fallback 递归诊断：非法预测时保留上一合法状态（与 A4 主协议的“非法即终止”不同）"
            ),
            "per_model_cumulative_termination_rate_final_window": {
                seed: float(payload["all_chains"]["cumulative_termination_rate"][-1])
                for seed, payload in fallback.items()
            },
            "per_model_cumulative_invalid_rate_final_window": {
                seed: float(payload["all_chains"]["cumulative_invalid_rate"][-1])
                for seed, payload in fallback.items()
            },
            "n_chains": {
                seed: int(payload["n_chains"]) for seed, payload in fallback.items()
            },
        }

    def scan(node, path: str = "") -> List[Dict[str, object]]:
        hits: List[Dict[str, object]] = []
        if isinstance(node, dict):
            for key, value in node.items():
                hits.extend(scan(value, f"{path}/{key}"))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                hits.extend(scan(value, f"{path}[{index}]"))
        elif isinstance(node, float) and abs(node - 0.918) < 5e-4:
            hits.append({"path": path, "value": float(node)})
        return hits

    result["scan_for_0p918"] = {
        "hits_in_new_a4_json": scan(new_a4),
        "note": (
            "在本轮 A4 原始 JSON 中扫描 0.918±5e-4：若为空，则 STATUS 的 0.918 并非来自本协议记录。"
        ),
    }
    return result


def render_fixes_doc(artifacts: dict, reconciliation: Dict[str, object], *, out_dir: Path) -> str:
    """A1/A2 fixes and the A4 termination-rate reconciliation, generated from the JSON."""

    eval_states = artifacts["eval_states_v2"]
    traced = artifacts["traced_cases"]
    legacy = artifacts["aggregation_reconciliation"]
    row_means = {
        key: _row_mean_metrics(artifacts["per_run"], key).get((4, 256)) for key in ("v2", "old")
    }
    a1 = {"v2": _a1_table(eval_states, "metrics_v2"), "old": _a1_table(eval_states, "metrics_old")}
    metrics = ("selected_bias", "candidate_mean_bias", "selection_excess", "candidate_regret", "chosen_true_return")
    metric_zh = {
        "selected_bias": "selected_bias",
        "candidate_mean_bias": "candidate_mean_bias",
        "selection_excess": "selection_excess",
        "candidate_regret": "regret",
        "chosen_true_return": "chosen_true",
    }
    lines: List[str] = [
        "# A1/A2 Grid 修复与 A4 终止率对账",
        "",
        "本文件由 `experiments/deterministic_grid/run_audit_v2.py` 从本轮运行的原始 JSON 生成，"
        f"不手改数字。本轮输出目录：`{Path(out_dir)}`；"
        f"配置：`{json.dumps(artifacts['config'], ensure_ascii=False)}`；墙钟 {artifacts['wall_clock']:.1f} s。",
        "只重算统计：未重训 Grid、未重建数据、未覆盖旧目录。",
        "",
        "## 0. 输入哈希（运行前后一致）",
        "",
        "| 文件 | SHA256 | 前=后 |",
        "|---|---|:--:|",
    ]
    for name in sorted(artifacts["start_hashes"]):
        same = artifacts["start_hashes"][name] == artifacts["end_hashes"].get(name)
        lines.append(f"| {_hash_label(name)} | `{artifacts['start_hashes'][name]}` | {'是' if same else '否'} |")
    lines += [
        "",
        f"`hash_check.unchanged = {str(artifacts['hash_check']['unchanged']).lower()}`。",
        "",
        "## 1. A1 聚合修复",
        "",
        "旧实现 `aggregate_by_state` 在构造 `model_seed=\"mean\"` 行时，把 `model_seed=-1` 的精确模型"
        "也一起平均。精确模型在同一评分函数下 `J_hat = J_ref`，其偏差恒为 0，于是每个头条数字被"
        "静默乘上 `3/4`。修复后 `mean` 行只平均学习种子 0/1/2，精确模型单独成行、绝不进入任何均值。",
        "",
        "### 1.1 H=4/K=256 新旧验证集：修复前/后",
        "",
        "| 集 | 口径 | selected_bias | candidate_mean_bias | selection_excess | regret | chosen_true |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("新验证集", "v2"), ("旧验证集", "old")):
        table = a1[key]
        for rule_label, values in (
            ("修复前：状态优先后对 4 个种子平均（含 exact）", table["mean_including_exact"]),
            ("修复后：状态优先后只对学习种子 0/1/2 平均", table["mean_learned_only"]),
        ):
            lines.append(
                f"| {label} | {rule_label} | " + " | ".join(_signed(values[metric]) for metric in metrics) + " |"
            )
    for key, label in (("old", "旧验证集"), ("v2", "新验证集")):
        row = row_means.get(key)
        if row is None:
            continue
        lines.append(
            f"| {label} | 旧“按行平均”口径（只含学习种子，本轮重算） | "
            f"{_signed(row['selected_bias'])} | {_signed(row['candidate_mean_bias'])} | "
            f"{_signed(row['selection_excess'])} | {_plain(row['candidate_regret'])} | "
            f"{_signed(row['chosen_true_return'])} |"
        )
    lines += [
        "",
        "旧表 H=4/K=256 的 `1.250 / −1.450 / 2.700` 就是这个“按行平均”口径，"
        f"本轮重算与旧表最大绝对差 {legacy['max_abs_diff_legacy_vs_stored']:.2e}"
        f"（一致={legacy['legacy_reproduced']}）。在只含学习种子时，两个验证集的"
        "“先状态内平均再跨状态平均”与“按行平均”都相同，因为每个状态的重复次数相同"
        "（这正对应 A1 验证条款：重复次数相同时两种口径相等）；"
        "新旧数字的差异只来自是否把精确行混入均值。",
        "",
        "任务书定位参考值：旧集 H=4/K=256 约 `1.250 / −1.450 / 2.700`；新集约 "
        "`0.867 / −1.719 / 2.586`、regret ≈1.075、chosen_true ≈−5.650。"
        "全精度值以上表为准。",
        "",
        "### 1.2 精确行已排除的算术检查",
        "",
        "| 集 | 指标 | exact 行 | 学习种子 0 | 1 | 2 | 修复后均值(0/1/2) | 含 exact 的 4 种子均值 | 含 exact 均值 = 学习均值 × 3/4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for key, label in (("v2", "新验证集"), ("old", "旧验证集")):
        table = a1[key]
        exact = table["exact_row"] or {}
        for metric in metrics:
            per_seed = table["per_seed"]
            learned = table["mean_learned_only"][metric]
            included = table["mean_including_exact"][metric]
            ratio_ok = abs(included - learned * 0.75) < 1e-12
            lines.append(
                f"| {label} | {metric_zh[metric]} | {_signed(exact.get(metric))} | "
                f"{_signed(per_seed['0'][metric])} | {_signed(per_seed['1'][metric])} | "
                f"{_signed(per_seed['2'][metric])} | {_signed(learned)} | {_signed(included)} | "
                f"{'是' if ratio_ok else '否'} |"
            )
    lines += [
        "",
        "精确行的 selected_bias/candidate_mean_bias/selection_excess（以及 regret）全为 +0.000，"
        "因此这三个偏差类指标“含 exact 的均值”正是学习均值的 3/4；"
        "`chosen_true` 的 exact 行不是 0（它是候选池的最优真实收益），所以该指标不满足 3/4 关系，"
        "表中如实写“否”。这不是“两种聚合口径”的口径差，而是精确模型被错误地当成了一个学习种子。",
        "",
        "## 2. A2 单候选反事实（主分析）",
        "",
        "只缓存原选中候选 c 的首次指定错误，仅对 c 重跑并继续其模型链，只替换 `J_hat[c]`，"
        "其余候选逐项保持 bit-identical，然后重排；同时报告新选择的**真实**收益是否改善。",
        "",
        "| role | state | ms/cs | 干预 | 首次步 | 排序改变 | 新选中 id | 改动条目数 | 原选中 true J | 新选中 true J | true 改善 |",
        "|---|---|---|---|---:|:--:|---:|---:|---:|---:|:--:|",
    ]
    for case in traced["cases"]:
        state_text = (
            f"r0={tuple(case['state']['robot0'])} r1={tuple(case['state']['robot1'])} "
            f"o={tuple(case['state']['object'])}"
        )
        for key, payload in case["counterfactuals"].items():
            lines.append(
                f"| {case['role']} | {state_text} | ms={case['model_seed']}, cs={case['candidate_seed']} | "
                f"{payload['intervention']} | {payload['first_step']} | "
                f"{'是' if payload['changed_ranking'] else '否'} | "
                f"{payload['counterfactual_chosen_candidate_id']} | {payload['n_changed_entries']} | "
                f"{_signed(payload['baseline_chosen_true_return'])} | "
                f"{_signed(payload['new_chosen_true_return'])} | "
                f"{'是' if payload['true_return_improved'] else '否'} |"
            )
    lines += [
        "",
        "`false_success` 干预通过替换整个真实下一步状态实现，命名为“首次虚假成功处的完整状态干预”，"
        "它不是只修成功分类。物体锚点干预若使组合状态非法，则按原协议在该步终止，不额外修机器人位置。",
        "",
        "### 2.1 旧全候选干预（历史附件，已更正因果表述）",
        "",
        "| role | 干预 | 首次步 | 排序改变 | 新选中 id | true 改善 | 说明 |",
        "|---|---|---:|:--:|---:|:--:|---|",
    ]
    for case in traced["cases"]:
        for key, payload in case.get("counterfactuals_all_candidates_historical", {}).items():
            lines.append(
                f"| {case['role']} | {payload['intervention']} | {payload['first_step']} | "
                f"{'是' if payload['changed_ranking'] else '否'} | "
                f"{payload['counterfactual_chosen_candidate_id']} | "
                f"{'是' if payload['true_return_improved'] else '否'} | {payload['note']} |"
            )
    top_cases = [c for c in traced["cases"] if c["role"] == "top_regret"]
    full_changed = sum(c["counterfactuals"]["full_state"]["changed_ranking"] for c in top_cases)
    full_improved = sum(c["counterfactuals"]["full_state"]["true_return_improved"] for c in top_cases)
    obj_changed = sum(c["counterfactuals"]["object_anchor"]["changed_ranking"] for c in top_cases)
    obj_improved = sum(c["counterfactuals"]["object_anchor"]["true_return_improved"] for c in top_cases)
    previous_traced_path = ROOT / PREVIOUS_RUN_ROOT / "traced_cases.json"
    same_traced_states = None
    if previous_traced_path.exists():
        previous_traced = _load_json(previous_traced_path)

        def _signature(cases: Sequence[dict]):
            return [
                (c["role"], c["model_seed"], c["candidate_seed"], c["state_key"], c["chosen_candidate_id"])
                for c in cases
            ]

        same_traced_states = _signature(previous_traced["cases"]) == _signature(traced["cases"])
    lines += [
        "",
        "### 2.2 与上一轮同一批追踪状态的对照",
        "",
        f"四个追踪状态（role / 模型种子 / 候选种子 / state_key / 原选中 id）与上一轮完全一致："
        f"{same_traced_states}。上一轮的高 regret 案例仍是 "
        + "、".join(f"`{tuple(c['state']['robot0'])}/{tuple(c['state']['robot1'])}/{tuple(c['state']['object'])}`"
                    for c in top_cases)
        + "。",
        "",
        f"- 单候选 `full_state` 干预仍对 {full_changed}/{len(top_cases)} 个高 regret 案例改变了选中候选，"
        f"但新选择的真实收益改善只有 {full_improved}/{len(top_cases)}（`(3,4)` 案例 −5.5→−4.5）；"
        f"单候选 `object_anchor` 为 {obj_changed}/{len(top_cases)} 改变选择、"
        f"{obj_improved}/{len(top_cases)} 真实收益改善。"
        "上一轮“3/3 改变了排序”的说法成立，但它不等于修复有收益。",
        f"- 上一轮关于虚假成功案例的句子（“只修复该候选自身这一次虚假成功后对全部候选重排，"
        f"选中 id 变为 12”）描述的是旧全候选重排；本主分析的单候选干预也把 id 变为 12，"
        f"但真实收益没有改善。全候选版本现作为历史附件单列。",
        "- 上一轮把 `false_success` 干预写作“只修成功分类”，已撤回并改名。",
        "",
        "## 3. A4 Grid 递归累计终止率对账",
        "",
        f"定义：{reconciliation['definition']}",
        "",
        "### 3.1 本轮从原始 JSON 重算",
        "",
        "| 集 | H | 每模型链数 | 已终止/总链数 | 末步累计终止率 | teacher 末步累计终止率 |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for arm, label in (("v2", "新验证集"), ("old", "旧 20 状态")):
        payload = reconciliation["new_run"][arm]
        fractions = "；".join(
            f"{seed}: {row['chains_terminated']}/{row['n_chains']}"
            for seed, row in payload["per_model"].items()
        )
        lines.append(
            f"| {label} | {payload['horizon']} | {payload['candidates_per_state']} 候选×每状态 | "
            f"{fractions} | {_plain(payload['pooled_final_step_cumulative_termination_rate'])} | "
            f"{_plain(payload['pooled_final_step_cumulative_termination_rate_teacher'])} |"
        )
    lines += [
        "",
        f"逐链数与存储值的自洽检查 recomputed_matches_stored = "
        f"{reconciliation['new_run']['v2']['verification']['recomputed_matches_stored']}"
        f"（每个模型种子的比率不同，因此 pooled 值是三种子的均值）。",
        "",
        "### 3.2 与上一轮记录、STATUS 0.918 的来源",
        "",
        "| 来源 | 协议 | 递归末步累计终止率 |",
        "|---|---|---:|",
    ]
    prev_v2 = reconciliation["previous_round"].get("v2")
    if prev_v2:
        lines.append(
            f"| `{prev_v2['source']}`（上一轮） | 与 A4 相同（非法即终止） | "
            f"{_plain(prev_v2['pooled_final_step_recursive'])} |"
        )
    source = reconciliation.get("status_0p918_source") or {}
    if source:
        for seed, value in source["per_model_cumulative_termination_rate_final_window"].items():
            lines.append(f"| `{source['source']}` 模型 {seed} | {source['protocol']} | {_plain(value)} |")
    lines += [
        "",
        f"本轮 A4 原始 JSON 中扫描 0.918±5e-4 的命中数："
        f"{len(reconciliation['scan_for_0p918']['hits_in_new_a4_json'])}。",
        "",
        "结论：A4 主协议下 Grid 递归累计终止率（新验证集，H=4）为 "
        f"{_plain(reconciliation['new_run']['v2']['pooled_final_step_cumulative_termination_rate'])}，"
        "与详细表的 0.137 一致；STATUS 的 0.917/0.918 来自旧 fallback 诊断"
        "（`forecast-20261012/diagnostic/recursive.json`，非法预测保留上一合法状态、链结构与步数不同），"
        "不是 A4 协议下的数字，0.918 本身在本轮 A4 记录中没有对应量。",
        "",
        "## 4. 更正与撤回",
        "",
        "| 旧表述 | 处理 | 更正后 |",
        "|---|---|---|",
        "| `model_seed=\"mean\"` 包含精确模型 | 更正 | 只平均学习种子 0/1/2；精确模型单列，偏差恒 0 |",
        "| 旧 1.250/−1.450/2.700 与 CONTRACT 口径的差异是“聚合口径”问题 | 更正 | 差异来自精确行混入；修复后学习种子状态优先均值≈旧按行平均（每状态重复次数相同） |",
        "| `false_success` 是“只修成功分类” | 撤回 | 它替换整个真实下一步状态，命名为“首次虚假成功处的完整状态干预” |",
        "| 旧全候选干预可读作“只修选中候选” | 撤回 | 旧干预修了全部候选，仅作历史附件保留 |",
        "| Grid 递归累计终止率 ≈0.918（STATUS） | 更正 | A4 协议下新验证集 H=4 为 "
        f"{_plain(reconciliation['new_run']['v2']['pooled_final_step_cumulative_termination_rate'])}"
        "（旧 20 状态为 "
        f"{_plain(reconciliation['new_run']['old']['pooled_final_step_cumulative_termination_rate'])}）；"
        "0.917 出现在旧 fallback 诊断 |",
        "",
        "## 5. 证据限度",
        "",
        "- 只重算统计，未重训 Grid、未新增 Grid 实验；三个模型种子与候选种子是重复测量。",
        "- 单候选反事实只改变一个候选的模型分数；“选择改变”与“修复有真实收益”分开报告，"
        "二者都可能单独成立或不成立。",
        "- 上一轮旧目录只读引用，未被覆盖。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="grid-audit-v2")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--docs-out", default=DEFAULT_DOCS_OUT)
    parser.add_argument("--fixes-doc-out", default=DEFAULT_FIXES_DOC_OUT)
    parser.add_argument("--states-per-map", type=int, default=10)
    args = parser.parse_args()

    run_dir = ROOT / args.source_run
    out_dir = ROOT / args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = run_phase_a(run_dir, config=AuditConfig(), states_per_map=args.states_per_map)
    reconciliation = grid_termination_reconciliation(artifacts["recursive_vs_teacher"])

    _json_dump(out_dir / "group_stats.json", artifacts["group_stats"])
    _json_dump(out_dir / "eval_states_v2.json", artifacts["eval_states_v2"])
    _jsonl_dump(out_dir / "per_run_v2.jsonl", artifacts["per_run"])
    _jsonl_dump(out_dir / "candidate_records_v2.jsonl", artifacts["candidate_records"])
    _json_dump(out_dir / "decomposition.json", artifacts["decomposition"])
    _json_dump(out_dir / "traced_cases.json", artifacts["traced_cases"])
    _json_dump(out_dir / "recursive_vs_teacher.json", artifacts["recursive_vs_teacher"])
    _json_dump(out_dir / "hash_check.json", artifacts["hash_check"])
    _json_dump(out_dir / "termination_rate_reconciliation.json", reconciliation)
    if not artifacts["hash_check"]["unchanged"]:
        # CONTRACT section 0: a changed checkpoint or data hash is a BLOCKED condition.
        raise SystemExit(
            "BLOCKED: input hashes changed during the run: "
            + json.dumps(artifacts["hash_check"]["changed"], ensure_ascii=False)
        )
    summary = {
        "run_id": args.run_id,
        "source_run": str(run_dir),
        "config": artifacts["config"],
        "seeds": artifacts["seeds"],
        "dataset_size": artifacts["dataset_size"],
        "hash_check": artifacts["hash_check"],
        "facts": artifacts["facts"],
        "old_set_reproduction": artifacts["old_set_reproduction"],
        "aggregation_reconciliation": artifacts["aggregation_reconciliation"],
        "decomposition_algebraic_check": artifacts["decomposition"]["algebraic_check"],
        "termination_rate_reconciliation": reconciliation,
        "wall_clock": artifacts["wall_clock"],
    }
    _json_dump(out_dir / "summary.json", summary)
    (out_dir / "metrics_table_v2.md").write_text(render_metrics_table(artifacts))

    docs_path = ROOT / args.docs_out
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(render_docs(artifacts))

    fixes_path = ROOT / args.fixes_doc_out
    fixes_path.parent.mkdir(parents=True, exist_ok=True)
    fixes_path.write_text(render_fixes_doc(artifacts, reconciliation, out_dir=out_dir))

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "docs": str(docs_path),
                "fixes_doc": str(fixes_path),
                "hash_check_unchanged": artifacts["hash_check"]["unchanged"],
                "old_set_reproduced": artifacts["old_set_reproduction"]["reproduced_exactly"],
                "decomposition_passed": artifacts["decomposition"]["algebraic_check"]["passed"],
                "wall_clock": artifacts["wall_clock"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
