"""Render the numeric tables used by REPORT.md from saved artefacts only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
import long_train as lt  # noqa: E402
import analyze_stability as analysis  # noqa: E402


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def run_dir(root, stage, arm):
    return root / lt.STABILITY_RUN_DIRS[(stage, arm)]


def evaluation_table(rows, initial):
    lines = ["| 更新 | 原8确定 | 原8随机 | 训练32确定 | val64确定 | val64随机 |",
             "|---:|---:|---:|---:|---:|---:|"]
    lines.append("| 0 | " + " | ".join(
        fmt(initial[key]["success"]) for key in
        ("deterministic", "stochastic", "train_panel", "validation", "validation_stochastic"))
        + " |")
    for row in rows:
        if "evaluation" not in row:
            continue
        evaluation = row["evaluation"]
        lines.append(f"| {row['update']} | " + " | ".join(
            fmt(evaluation[key]["success"]) for key in
            ("deterministic", "stochastic", "train_panel", "validation")) +
            f" | {fmt(evaluation.get('validation_stochastic', {}).get('success'))} |")
    return lines


def tracking_heat_table(rows, initial=None):
    first = next((row["evaluation"]["deterministic"]["per_instance"] for row in rows
                  if "evaluation" in row), None)
    instances = sorted(first.keys()) if first else []
    if not instances:
        return ["（无评价点）"]
    header = "| 更新 | " + " | ".join(name.replace("L0_train_", "") for name in instances) + " |"
    lines = [header, "|---:|" + "---:|" * len(instances)]
    if initial is not None:
        values = initial["deterministic"]["per_instance"]
        lines.append("| 0 | " + " | ".join(
            "1" if values.get(name, 0) >= 0.5 else "0" for name in instances) + " |")
    for row in rows:
        if "evaluation" not in row:
            continue
        values = row["evaluation"]["deterministic"]["per_instance"]
        lines.append(f"| {row['update']} | " + " | ".join(
            "1" if values.get(name, 0) >= 0.5 else "0" for name in instances) + " |")
    return lines


def short_loop_table(rows):
    lines = ["| 窗口 | 采样成功 | 失败短技能决策占比 | 失败短技能物理时间占比 | 成功短技能时间占比 | safety_abort/更新 | 决策/更新 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    window = max(5, min(100, len(rows) // 4))
    for label, subset in (("前 %d 更新" % window, rows[:window]),
                          ("后 %d 更新" % window, rows[-window:])):
        def avg(path):
            values = []
            for row in subset:
                node = row
                for key in path:
                    node = node.get(key) if isinstance(node, dict) else None
                if isinstance(node, (int, float)):
                    values.append(float(node))
            return sum(values) / len(values) if values else None

        def safety(subset):
            total = 0
            for row in subset:
                for group in ("success", "failure"):
                    total += (row.get("diagnostics", {}).get(group) or {}).get(
                        "exits", {}).get("safety_abort", 0)
            return total / len(subset)

        def decisions(subset):
            values = [row.get("decisions", 0) for row in subset]
            return sum(values) / len(values) if values else None
        lines.append(f"| {label} | {fmt(avg(('success',)))} | "
                     f"{fmt(avg(('diagnostics', 'failure', 'short_decision_fraction')))} | "
                     f"{fmt(avg(('diagnostics', 'failure', 'short_time_fraction')))} | "
                     f"{fmt(avg(('diagnostics', 'success', 'short_time_fraction')))} | "
                     f"{fmt(safety(subset), 1)} | {fmt(decisions(subset), 1)} |")
    return lines


def audit_table(rows):
    lines = ["| 更新 | 共享 actor 范数 | 共享 critic 范数 | 余弦 | KL(决策平均) | KL(轨迹平均) | 0017 logp变化 | 0056 logp变化 | 0063 logp变化 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        audit = row.get("gradient_audit")
        if not audit:
            continue
        fixed = {entry["instance"]: entry for entry in row.get("fixed_history", [])}
        cells = [fmt(fixed.get(f"L0_train_{name}", {}).get("mean_logp_change"), 3)
                 for name in ("0017", "0056", "0063")]
        lines.append(f"| {row['update']} | {fmt(audit['shared_actor_norm'])} | "
                     f"{fmt(audit['shared_critic_norm'])} | "
                     f"{fmt(audit['shared_actor_critic_cosine'], 3)} | "
                     f"{fmt(row['joint_kl']['by_decision_mean'])} | "
                     f"{fmt(row['joint_kl']['by_trajectory_mean'])} | " +
                     " | ".join(cells) + " |")
    return lines


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else lt.STABILITY_ROOT
    lines = ["# 数值表（自动生成，供 REPORT.md 引用）", ""]
    for stage in ("pilot", "long"):
        for arm in ("shared", "stopped"):
            directory = run_dir(root, stage, arm)
            if not (directory / "updates.jsonl").exists():
                continue
            rows = analysis.read_updates(directory)
            initial = json.loads((directory / "initial.json").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            lines += [f"## {stage} / {arm} ({manifest['config']['critic_mode']})", "",
                      f"- 状态：{manifest.get('status')}；完成 {rows[-1]['update'] if rows else 0}/"
                      f"{manifest['config']['updates']}；"
                      f"墙钟 {((manifest.get('finished_unix') or manifest['started_unix']) - manifest['started_unix'])/3600:.2f} h",
                      ""]
            lines += evaluation_table(rows, initial)
            lines += ["", "保留热图（原8例确定性，1=成功）：", ""]
            lines += tracking_heat_table(rows, initial)
            lines += ["", "短技能与退出：", ""]
            lines += short_loop_table(rows)
            lines += ["", "梯度/KL/固定历史遗忘：", ""]
            lines += audit_table(rows)
            lines += [""]
    paired = analysis.read_json(root / "comparison" / "test_paired.json")
    if paired:
        lines += ["## 独立 test64 配对差（实例级）", "",
                  "| 比较 | 均值 | 95% 区间 | 正/负/零 |", "|---|---:|---|---|"]
        for run, entry in pair_rows(paired):
            lines.append(f"| {run} | {fmt(entry.get('mean'))} | "
                         f"[{fmt(entry.get('lo'))}, {fmt(entry.get('hi'))}] | "
                         f"{entry.get('positive')}/{entry.get('negative')}/{entry.get('zero')} |")
        lines.append("")
    audit_summary = analysis.read_json(root / "pilot_audit" / "summary.json")
    if audit_summary:
        lines += ["## S1 原始 batch 审计", "",
                  f"- 并行回放：{audit_summary['parallel_replay']}",
                  f"- 失败 episode 周期：{audit_summary['failure_periods']}",
                  f"- 批：{audit_summary['batch']}",
                  f"- 更新前两臂一致：{audit_summary['before_update_identical']}",
                  ""]
        for arm, entry in audit_summary["zero_actor_advantage_one_step"].items():
            lines.append(f"- 零 actor 优势 / {arm}：shared={entry['shared_changed']}，"
                         f"actor_heads={entry['actor_heads_changed']}，"
                         f"value_head={entry['value_head_changed']}")
        lines.append("")
    (root / "comparison").mkdir(parents=True, exist_ok=True)
    (root / "comparison" / "TABLES.md").write_text("\n".join(lines) + "\n")
    print("wrote", root / "comparison" / "TABLES.md")


def pair_rows(paired):
    rows = []
    for run, entry in paired.get("runs", {}).items():
        for key in ("best_minus_update0_deterministic", "best_minus_update0_stochastic",
                    "final_at_budget_minus_update0_deterministic",
                    "final_at_budget_minus_update0_stochastic"):
            if entry.get(key):
                rows.append((f"{run}/{key}", entry[key]))
    for group in ("A_minus_B", "A_minus_B_final_at_budget"):
        for key, entry in paired.get(group, {}).items():
            rows.append((f"{group}/{key}", entry))
    return rows


if __name__ == "__main__":
    main()
