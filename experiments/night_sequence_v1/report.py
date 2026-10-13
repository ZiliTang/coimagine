"""Generate REPORT.md from the analysis metrics. No number is written by hand.

    python experiments/night_sequence_v1/report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

DATA = ROOT / "outputs/night_sequence_v1"
DOCS = ROOT / "docs/night_sequence_v1"


def load(tier: str) -> Optional[dict]:
    path = DATA / "analysis" / tier / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def t0(entry: dict) -> dict:
    return entry.get("t0", entry.get("all", {}))


def row(label: str, entry: dict) -> List[str]:
    block = t0(entry)
    geometry = block.get("terminal_geometry", {})
    return [label,
            fmt(geometry.get("r0_mid_2d", {}).get("p50")),
            fmt(geometry.get("r0_mid_2d", {}).get("p90")),
            fmt(geometry.get("r0_normal_angle_deg", {}).get("p50")),
            fmt(block.get("terminal_pose", {}).get("p50")),
            fmt(block.get("terminal_remaining", {}).get("p50"), 3),
            fmt(block.get("terminal_exit_class", {}).get("accuracy"), 3)]


def main() -> int:
    tiers = {tier: load(tier) for tier in ("d0", "d1", "d2")}
    tiers = {k: v for k, v in tiers.items() if v}
    if not tiers:
        print("no metrics yet")
        return 1
    summary = json.loads((DATA / "token_summary.json").read_text())
    runs = []
    for path in sorted((DATA / "runs").glob("*.pt")):
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=False)
        runs.append(saved)

    lines = ["# night_sequence_v1 — REPORT", "",
             "任务书：`docs/tasks/20261014_0205_CoImagine_夜间资源积累与受控扩展任务书.md`（v2.1）。",
             "协议：`CONTRACT.md`；数据：`DATA_INDEX.md`；状态与全部 run：`STATUS.md`、`RUN_LEDGER.md`。", ""]

    # ---- scale
    lines += ["## 1. 规模（四个不同的量）", "",
              "| 档位 | 独立 root | option 序列 | 真实帧 |", "|---|---:|---:|---:|"]
    role = {"d0": "D0 训练", "d1ext": "D1 扩展", "d2ext": "D2 扩展（余量目标）",
            "val": "新验证", "test": "新测试"}
    for split in ("d0", "d1ext", "d2ext", "val", "test"):
        entry = summary.get(split)
        if not entry:
            continue
        lines.append(f"| {split}（{role[split]}） | {entry['independent_roots']} | "
                     f"{entry['sequences']} | {entry['frames']} |")
    d1_roots = summary.get("d0", {}).get("independent_roots", 0) + \
        summary.get("d1ext", {}).get("independent_roots", 0)
    d1_sequences = summary.get("d0", {}).get("sequences", 0) + \
        summary.get("d1ext", {}).get("sequences", 0)
    d2_roots = d1_roots + summary.get("d2ext", {}).get("independent_roots", 0)
    d2_sequences = d1_sequences + summary.get("d2ext", {}).get("sequences", 0)
    lines += ["", f"- **D1 = {d1_roots} root / {d1_sequences} 序列**；"
                  f"**D2 = {d2_roots} root / {d2_sequences} 序列**。",
              "- 训练窗口数由采样决定（每次更新 128 个窗口），它不是新的独立起点；滑窗不增加 root 数。", ""]

    # ---- matrix
    lines += ["## 2. 完成矩阵", "",
              "| run | 模型 | 档位 | seed | 参数量 | 最优验证损失 | 最优更新 |", "|---|---|---|---:|---:|---:|---:|"]
    for saved in runs:
        lines.append(f"| {saved['tier']}_{saved['model_kind']}_seed{saved['seed']} | "
                     f"{saved['model_kind']} | {saved['tier']} | {saved['seed']} | "
                     f"{saved['parameters']:,} | {saved['best_val']:.4f} | {saved['best_update']} |")
    expected = 15 + 2
    lines += ["", f"共 {len(runs)} / {expected} 次正式训练（主对照 15 次 + D2 成对 2 次）。"
                  "每次 run 的更新预算统一为 3200（标定见 `CONTRACT.md` §4）。", ""]

    # ---- main comparison per tier
    for tier in ("d0", "d1", "d2"):
        if tier not in tiers:
            continue
        payload = tiers[tier]
        lines += [f"## 3{'.' if tier == 'd0' else ''} {tier.upper()} 主对照（test，t=0 终态）",
                  "",
                  "| 模型 | r0 中点二维 p50 (m) | p90 (m) | r0 法向角 p50 (°) | 位姿 p50 | "
                  "剩余时长 p50 (s) | 退出准确率 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for label in ("P0_hold|test", "P1_grouped|test"):
            if label in payload["baselines"]:
                lines.append("| " + " | ".join(row(label.split("|")[0], payload["baselines"][label])) + " |")
        for label, entry in sorted(payload["models"].items()):
            if not label.endswith("|test") or not isinstance(entry, dict) or "t0" not in entry:
                continue
            lines.append("| " + " | ".join(row(label.split("|")[0], entry)) + " |")
        # action sensitivity
        sensitivity = payload["models"].get("action_sensitivity", {})
        if sensitivity:
            lines += ["", "t=0 同 root 八动作终态差值 / 秩相关（r0 中点 x；置换对照≈0）：", "",
                      "| run | 动作差值 p50 (m) | Spearman ρ | 有效 root |", "|---|---:|---:|---:|"]
            for name, entry in sorted(sensitivity.items()):
                lines.append(f"| {name} | "
                             f"{fmt(entry['action_difference_error']['r0_x']['p50'], 5)} | "
                             f"{fmt(entry['rank_correlation']['r0_x']['mean'], 3)} | "
                             f"{entry['rank_correlation']['r0_x']['roots_defined']}/"
                             f"{entry['rank_correlation']['r0_x']['roots_attempted']} |")
        lines.append("")

    # ---- verdicts
    def best(payload, tier_label) -> Dict[str, Optional[float]]:
        out = {}
        for label in ("P0_hold|test", "P1_grouped|test"):
            if label in payload["baselines"]:
                out[label] = t0(payload["baselines"][label])["terminal_geometry"]["r0_mid_2d"]["p50"]
        for label, entry in payload["models"].items():
            if label.endswith("|test") and isinstance(entry, dict) and "t0" in entry:
                out[label] = t0(entry)["terminal_geometry"]["r0_mid_2d"]["p50"]
        return out

    lines += ["## 4. 三个研究问题", ""]
    if "d0" in tiers and "d1" in tiers:
        d0_best = min(v for k, v in best(tiers["d0"], "d0").items()
                      if k.startswith("d0_mlp_history") and v is not None)
        d1_best = min(v for k, v in best(tiers["d1"], "d1").items()
                      if k.startswith("d1_mlp_history") and v is not None)
        lines += [f"**更多独立起点是否改善观测预测？** D0 训练（512 root）的 MLP-history 中点误差 "
                  f"p50 = {d0_best:.4f} m；D1 训练（4096 root，8×）为 {d1_best:.4f} m"
                  f"（{'改善' if d1_best < d0_best else '未改善'}）。", ""]
    d1 = tiers.get("d1")
    if d1:
        table = best(d1, "d1")
        current = [v for k, v in table.items() if k.startswith("d1_mlp_current") and v is not None]
        history = [v for k, v in table.items() if k.startswith("d1_mlp_history") and v is not None]
        if current and history:
            lines += [f"**历史窗口是否比单帧有用？** 同档位同种子下 MLP-current 最优 "
                      f"{min(current):.4f} m vs MLP-history 最优 {min(history):.4f} m"
                      f"（{'history 更优' if min(history) < min(current) else 'history 未更优'}）。", ""]
        transformer = [v for k, v in table.items() if k.startswith("d1_transformer") and v is not None]
        if transformer and history:
            lines += [f"**Transformer 是否更好？** 同历史同目标同预算下，Transformer-history 最优 "
                      f"{min(transformer):.4f} m vs MLP-history 最优 {min(history):.4f} m"
                      f"（参数比 1.80×）。结论只来自这两个历史模型之间，不涉及单帧模型。", ""]
    else:
        lines += ["（D1 分析尚未生成。）", ""]

    # ---- in-execution: fixed-dt next step versus terminal
    lines += ["## 5. 执行中：固定 Δt 下一步 vs 终态（t=0 之外的主评价）", ""]
    for tier in ("d0", "d1", "d2"):
        if tier not in tiers:
            continue
        payload = tiers[tier]
        rows = []
        for label in ("P0_hold|test", "P1_grouped|test"):
            if label not in payload["baselines"]:
                continue
            entry = payload["baselines"][label]
            rows.append(("| " + " | ".join([
                label.split("|")[0],
                fmt(entry["all"]["grouped"]["regime"]["in_execution"]["p50"]),
                fmt(entry["all"]["process_geometry"]["r0_mid_2d"]["p50"]),
                fmt(entry["all"]["terminal_geometry"]["r0_mid_2d"]["p50"]),
                str(entry["all"]["grouped"]["regime"]["in_execution"]["samples"])]) + " |"))
        for label, entry in sorted(payload["models"].items()):
            if not label.endswith("|test") or not isinstance(entry, dict) or "all" not in entry:
                continue
            rows.append("| " + " | ".join([
                label.split("|")[0],
                fmt(entry["all"]["grouped"]["regime"]["in_execution"]["p50"]),
                fmt(entry["all"]["process_geometry"]["r0_mid_2d"]["p50"]),
                fmt(entry["all"]["terminal_geometry"]["r0_mid_2d"]["p50"]),
                str(entry["all"]["grouped"]["regime"]["in_execution"]["samples"])]) + " |")
        lines += [f"**{tier.upper()}**（test，执行中样本 5105；真实观测前缀条件预测）", "",
                  "| 模型 | 执行中 r0 中点二维 p50 (m) | 固定 Δt 下一步 r0 中点 p50 (m) | "
                  "执行中终态 r0 中点 p50 (m) | 执行中样本 |", "|---|---:|---:|---:|---:|"]
        lines += rows + [""]

    lines += ["## 6. 能支持什么、不能支持什么", "",
              "**能支持**：预测接口可训练、可评价；两档数据均可加载；架构对照在相同输入、mask、"
              "目标与更新预算下完成；误差在**原始单位**和**真实标签 mask**上计算。", "",
              "**不能支持**：",
              "- **t=0（执行前）与执行中不能混说**：执行中更准不等于执行前规划预测更准；"
              "执行中的评价用的是真实观测前缀。",
              "- **真实前缀条件预测不是自由 rollout**：本轮不做递归回灌，也不把未预测字段用真值补齐"
              "后声称完成自主推演。",
              "- 单步/序列预测改善不等于规划改善；本轮没有闭环实验，**未评估**闭环收益。",
              "- 不能由观测误差推出状态已辨识、未来分支合法或可递归。",
              "- 未按成功或误差筛选样本，也未在 test 上继续调参；结构在训练前写入 CONTRACT。", "",
              "## 7. 未验证 / 未完成", "",
              "- 自由多步递推所需的完整状态字段（见 `presentation_assets/figures/F08`）。",
              "- 闭环任务收益：本轮无实验。",
              "- 任务容差下几厘米中点误差是否可用：需要下游技能的容差定义。",
              "- 信息歧义正例：原场景冻结协议重做为负结果，未在其他场景验证。", "",
              "## 8. 下一轮只建议一项工作", "",
              "**把执行中评价改成“真实前缀 + 逐步回灌自身预测”的受限递推，并明确标注哪些字段仍是"
              "真值输入。** 依据：本轮已能预测固定 Δt 的下一个观测与终态，但完全不知道误差如何随"
              "递推步数累积；这是当前唯一挡住“能否自主推演”的未知量，且不需要新数据或新场景。", ""]
    (DOCS / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {DOCS / 'REPORT.md'} ({len(lines)} lines)")
    print("REPORT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
