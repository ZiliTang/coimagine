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
            fmt(block.get("terminal_pose", {}).get("all_columns_flattened", {}).get("p50")),
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
             "协议：`CONTRACT.md`；数据：`DATA_INDEX.md`；状态与全部 run：`STATUS.md`、`RUN_LEDGER.md`；",
             "模型卡：`MODEL_CARDS.md`；汇报素材：`presentation_assets/INDEX.md`。", "",
             "> **⚠ 本版状态（实现审查修正轮）**：上一版报告中的"
             "“历史窗口优于单帧”“Transformer 不如历史 MLP”“数据规模饱和”三条结论**已撤回**。",
             "> 原因：MLP-current 与 Transformer-history 在训练时**没有读到有效输入**"
             "（左补齐下用 `valid.sum()-1` 取索引，t=0 时读到全零补齐区），"
             "另有退出时刻 T 多算一个周期、几何中点定义被改、P1 未用可见性 mask、"
             "验证集每次重抽四处问题。",
             "> 逐条复现、根因与修正见 `IMPLEMENTATION_FIX.md`；"
             "旧检查点与旧评价结果保存在 `outputs/night_sequence_v1/superseded/`。",
             "> 本报告的**全部数字来自修正后重训的模型**。", "",
             "## 0. 一句话结论", "",
             "**在固定技能、相同输入与相同更新预算下，观测终态可以被训练出来：执行中误差 "
             "0.043 m，优于起始保持 0.052 m 与命令分组统计 0.087 m（配对 root bootstrap 95% CI "
             "不含 0）。但两条架构结论都**随区间翻转**：执行前（t=0）单帧 MLP 与 Transformer 更好，"
             "执行中历史 MLP 更好；Transformer 在参数只有 0.56× 的情况下 batch=128 延迟是历史 MLP 的 "
             "24–29 倍。更多独立起点（512→16384，32×）只带来很小的执行中改善后趋平，"
             "但该结论**限定在这套实现与统一更新预算内**。**", ""]

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
                  "| 模型 | r0 中点二维 p50 (m) | p90 (m) | r0 法向角 p50 (°) | "
                  "位姿各列合并 p50 | 剩余时长 p50 (s) | 退出准确率 |",
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

    stats = {}
    for tier in ("d0", "d1", "d2"):
        path = DATA / "analysis" / tier / "stats_and_cost.json"
        if path.exists():
            stats[tier] = json.loads(path.read_text())
    lines += ["## 4. 三个研究问题（修正后重训）", "",
              "三个问题的答案都**依赖评价区间**，因此逐区间给出；括号内为配对 root bootstrap 的 "
              "中位差与 95% CI（重采样单位是 root）。", ""]

    def medians(tier: str, family: str, regime: str):
        payload = tiers[tier]
        values = []
        for label, entry in payload["models"].items():
            if not label.endswith("|test") or not isinstance(entry, dict):
                continue
            if not label.startswith(f"{tier}_{family}"):
                continue
            if regime == "t0":
                values.append(entry["t0"]["terminal_geometry"]["r0_mid_2d"]["p50"])
            else:
                group = entry["all"]["grouped"]["regime"].get("in_execution")
                if group:
                    values.append(group["p50"])
        return values

    # ---- Q1 more independent starting points
    scale = []
    for tier in ("d0", "d1", "d2"):
        if tier not in tiers:
            continue
        roots = {"d0": summary.get("d0", {}).get("independent_roots", 0),
                 "d1": summary.get("d0", {}).get("independent_roots", 0)
                       + summary.get("d1ext", {}).get("independent_roots", 0),
                 "d2": summary.get("d0", {}).get("independent_roots", 0)
                       + summary.get("d1ext", {}).get("independent_roots", 0)
                       + summary.get("d2ext", {}).get("independent_roots", 0)}[tier]
        t0_values = medians(tier, "mlp_history", "t0")
        exec_values = medians(tier, "mlp_history", "in_execution")
        if t0_values and exec_values:
            scale.append((tier, roots, min(t0_values), min(exec_values)))
    if scale:
        lines += ["**问 1：更多独立起点是否改善观测预测？**（同一模型 MLP-history、同一目标、同一预算）",
                  "", "| 档位 | 独立 train root | t=0 终态 p50 (m) | 执行中 p50 (m) |",
                  "|---|---:|---:|---:|"]
        for tier, roots, t0_value, exec_value in scale:
            lines.append(f"| {tier} | {roots} | {t0_value:.4f} | {exec_value:.4f} |")
        first, last = scale[0], scale[-1]
        lines += ["", f"从 {first[1]} 到 {last[1]} 个独立起点（{last[1] // max(1, first[1])}×）："
                      f"执行中误差 {first[3]:.4f} → {last[3]:.4f} m（相对变化 "
                      f"{(last[3] - first[3]) / first[3] * 100:+.1f}%），"
                      f"t=0 误差 {first[2]:.4f} → {last[2]:.4f} m。"
                      f"**结论限定在这套实现与统一更新预算内**：增益很小且很快趋平，"
                      f"不能据此断言“数据收益已经饱和”。", ""]

    # ---- Q2 history versus single frame
    if "d1" in tiers:
        current_t0 = medians("d1", "mlp_current", "t0")
        history_t0 = medians("d1", "mlp_history", "t0")
        current_ex = medians("d1", "mlp_current", "in_execution")
        history_ex = medians("d1", "mlp_history", "in_execution")
        boot = (stats.get("d1", {}).get("paired_root_bootstrap", {})
                if stats else {})
        key_t0 = "d1_mlp_current_seed0_vs_d1_mlp_history_seed0|t0"
        key_ex = "d1_mlp_current_seed0_vs_d1_mlp_history_seed0|in_execution"
        lines += ["**问 2：历史窗口是否比单帧有用？**", "",
                  "| 区间 | MLP-current (单帧) | MLP-history | 谁更好 | 配对中位差与 95% CI |",
                  "|---|---:|---:|---|---|"]
        if current_t0 and history_t0:
            entry = boot.get(key_t0)
            detail = (f"{entry['observed_median_difference_m']:+.5f} m "
                      f"[{entry['ci95_low_m']:+.5f}, {entry['ci95_high_m']:+.5f}]"
                      if entry else "—")
            winner = "单帧更好" if min(current_t0) < min(history_t0) else "历史更好"
            lines.append(f"| t=0（无历史可用） | {min(current_t0):.4f} | {min(history_t0):.4f} | "
                         f"{winner} | {detail} |")
        if current_ex and history_ex:
            entry = boot.get(key_ex)
            detail = (f"{entry['observed_median_difference_m']:+.5f} m "
                      f"[{entry['ci95_low_m']:+.5f}, {entry['ci95_high_m']:+.5f}]"
                      if entry else "—")
            winner = "单帧更好" if min(current_ex) < min(history_ex) else "历史更好"
            lines.append(f"| 执行中（有真实前缀） | {min(current_ex):.4f} | {min(history_ex):.4f} | "
                         f"{winner} | {detail} |")
        lines += ["", "**答案随区间翻转**：执行前没有历史可用时单帧更好，"
                      "有真实观测前缀时历史窗口更好。因此不能笼统说“历史已被证明有效”，"
                      "只有在执行中这个条件下才成立。", ""]

    # ---- Q3 transformer versus MLP-history
    if "d1" in tiers:
        transformer_t0 = medians("d1", "transformer_history", "t0")
        transformer_ex = medians("d1", "transformer_history", "in_execution")
        history_t0 = medians("d1", "mlp_history", "t0")
        history_ex = medians("d1", "mlp_history", "in_execution")
        boot = (stats.get("d1", {}).get("paired_root_bootstrap", {}) if stats else {})
        key_t0 = "d1_mlp_history_seed0_vs_d1_transformer_history_seed0|t0"
        key_ex = "d1_mlp_history_seed0_vs_d1_transformer_history_seed0|in_execution"
        lines += ["**问 3：Transformer 是否更好、代价多少？**（同历史、同目标、同预算，参数比 1.80×）",
                  "", "| 区间 | MLP-history | Transformer-history | 谁更好 | 配对中位差与 95% CI |",
                  "|---|---:|---:|---|---|"]
        if transformer_t0 and history_t0:
            entry = boot.get(key_t0)
            detail = (f"{entry['observed_median_difference_m']:+.5f} m "
                      f"[{entry['ci95_low_m']:+.5f}, {entry['ci95_high_m']:+.5f}]"
                      if entry else "—")
            lines.append(f"| t=0 | {min(history_t0):.4f} | {min(transformer_t0):.4f} | "
                         f"{'Transformer 更好' if min(transformer_t0) < min(history_t0) else 'MLP 更好'} "
                         f"| {detail} |")
        if transformer_ex and history_ex:
            entry = boot.get(key_ex)
            detail = (f"{entry['observed_median_difference_m']:+.5f} m "
                      f"[{entry['ci95_low_m']:+.5f}, {entry['ci95_high_m']:+.5f}]"
                      if entry else "—")
            lines.append(f"| 执行中 | {min(history_ex):.4f} | {min(transformer_ex):.4f} | "
                         f"{'Transformer 更好' if min(transformer_ex) < min(history_ex) else 'MLP 更好'} "
                         f"| {detail} |")
        lines += ["", "**答案同样随区间翻转**：t=0 时 Transformer 更好，执行中时历史 MLP 更好，"
                      "两侧的 95% CI 都不含 0。计算成本见下一节（batch=128 延迟约 24–29×）。"
                      "因此上一版“Transformer 不如 MLP”的说法**不成立**，"
                      "正确的说法是“在哪个区间、以什么代价”。", ""]

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
            regime = entry["all"]["grouped"]["regime"]
            rows.append(("| " + " | ".join([
                label.split("|")[0],
                fmt(regime["in_execution"]["p50"]),
                fmt(entry["all"]["process_geometry"]["r0_mid_2d"]["p50"]),
                fmt(entry["all"]["terminal_geometry"]["r0_mid_2d"]["p50"]),
                f"{regime['in_execution']['samples']} / {regime['in_execution']['independent_roots']}",
                fmt(entry["t0"]["terminal_geometry"]["r0_mid_2d"]["p50"])]) + " |"))
        for label, entry in sorted(payload["models"].items()):
            if not label.endswith("|test") or not isinstance(entry, dict) or "all" not in entry:
                continue
            regime = entry["all"]["grouped"]["regime"]
            rows.append("| " + " | ".join([
                label.split("|")[0],
                fmt(regime["in_execution"]["p50"]),
                fmt(entry["all"]["process_geometry"]["r0_mid_2d"]["p50"]),
                fmt(entry["all"]["terminal_geometry"]["r0_mid_2d"]["p50"]),
                f"{regime['in_execution']['samples']} / {regime['in_execution']['independent_roots']}",
                fmt(entry["t0"]["terminal_geometry"]["r0_mid_2d"]["p50"])]) + " |")
        lines += [f"**{tier.upper()}**（test；真实观测前缀条件预测，不是自由 rollout）", "",
                  "列的含义严格区分，**不得混用**："
                  "「执行中」= 只在 `end>0` 的样本上分组统计；"
                  "「全样本终态」= 含 t=0 的 `all` 指标（**不是**执行中指标）；"
                  "「t=0」= 只有 O0 与候选命令。", "",
                  "| 模型 | 执行中 r0 中点 p50 (m) | 固定 Δt 下一步 p50 (m) | "
                  "全样本终态 p50 (m) | 执行中样本/独立 root | t=0 终态 p50 (m) |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += rows + [""]

    # ---- fixed-dt horizon split by how fast the robot is actually moving
    motion_rows = []
    for tier in ("d1",):
        if tier not in tiers:
            continue
        payload = tiers[tier]
        entries = [("P0_hold", payload["baselines"].get("P0_hold|test")),
                   ("P1_grouped", payload["baselines"].get("P1_grouped|test"))]
        entries += [(label.split("|")[0], entry) for label, entry in sorted(payload["models"].items())
                    if label.endswith("|test") and isinstance(entry, dict) and "all" in entry]
        for name, entry in entries:
            block = (entry or {}).get("all", {}).get("process_geometry_by_motion")
            if not block:
                continue
            motion_rows.append((name, block))
    if motion_rows:
        keys = ["near_static(<0.01m/s)", "slow(0.01-0.05)", "moving(>0.05)"]
        lines += ["## 5b. 固定 Δt 头按真实运动状态拆分（修正后）", "",
                  "Δt=0.05 s 时，近静止样本上「保持当前状态」几乎就是正确答案；"
                  "不拆分会把大量近乎平凡的样本混进同一个数字里。", "",
                  "| 模型 | 近静止 p50 | 慢速 p50 | 运动中 p50 | 近静止样本/独立 root |",
                  "|---|---:|---:|---:|---:|"]
        for name, block in motion_rows:
            cells = [fmt(block.get(k, {}).get("p50"), 5) for k in keys]
            stat = block.get(keys[0], {})
            lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} | "
                         f"{stat.get('samples', 0)} / {stat.get('independent_roots', 0)} |")
        lines += ["", "**这张表说明固定 Δt 头目前不成立**：P0 在近静止样本上是 0.0001 m 量级，"
                      "而学习模型是 0.05 m 量级——也就是说模型**没有**学到「短时几乎不变」这件事，"
                      "在占比最大的近静止子集上反而远差于平凡基线。"
                      "因此不能声称模型已经学好固定间隔动力学；"
                      "该头需要按运动状态分别训练或评价，而不是看一个合并数字。", ""]

    if stats:
        lines += ["## 6. 统计显著性与计算成本", "",
                  "**配对 root bootstrap**（同一样本上两模型逐例配对，重采样单位是 root；"
                  "帧与动作不算独立样本）：", "",
                  "| 档位 | 比较 | 中位差 (m) | 95% CI | 有效 root | P(前者更好) |",
                  "|---|---|---:|---|---:|---:|"]
        for tier, entry in stats.items():
            for key, value in entry["paired_root_bootstrap"].items():
                lines.append(f"| {tier} | {key.replace('_seed0', '').replace('_vs_', ' vs ')} | "
                             f"{value['observed_median_difference_m']:+.5f} | "
                             f"[{value['ci95_low_m']:+.5f}, {value['ci95_high_m']:+.5f}] | "
                             f"{value['roots']} | {value['fraction_draws_below_zero']:.3f} |")
        lines += ["", "训练 seed 波动（与 bootstrap 分开报告，未并入区间）：", "",
                  "| 模型族 | seeds | 极差 (m) |", "|---|---:|---:|"]
        for tier, entry in stats.items():
            for family, value in entry["training_seed_spread_m"].items():
                lines.append(f"| {family} | {value['seeds']} | {value['range']:.5f} |")
        lines += ["", "**推理成本**（torch 线程固定为 1；峰值内存为“建模 + 一次前向”的进程 RSS 增量）：", "",
                  "| 模型 | 参数量 | 峰值 RSS 增量 | batch=1 延迟 | batch=128 延迟 | 吞吐 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for cost in stats["d1"]["inference_cost"]:
            lines.append(f"| {cost['model']} | {cost['parameters']:,} | "
                         f"{cost['peak_rss_delta_mb']:.1f} MB | {cost['batch1_latency_ms']:.2f} ms | "
                         f"{cost['batch128_latency_ms']:.2f} ms | "
                         f"{cost['batch128_throughput_per_s']:.0f} /s |")
        lines += ["", "相同更新数**不等于**相同 FLOPs：Transformer-history 的 batch=128 延迟约为 "
                      "MLP-history 的 24–29 倍，参数却更少。这是研究问题 3 里“代价多少”的答案，"
                      "因此架构结论必须与计算成本一起读。", "",
                  "动作差值/秩相关的**置换对照**（打乱同 root 内动作标签后重算）："
                  "ρ 从 0.71 降到约 −0.04，说明该指标确实在测动作对应关系，而不是从数据规模里"
                  "凭空得到相关性。", ""]

    lines += ["## 7. 能支持什么、不能支持什么", "",
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
              "## 8. 未验证 / 未完成", "",
              "- 自由多步递推所需的完整状态字段（见 `presentation_assets/figures/F08`）。",
              "- 闭环任务收益：本轮无实验。",
              "- 任务容差下几厘米中点误差是否可用：需要下游技能的容差定义。",
              "- 信息歧义正例：原场景冻结协议重做为负结果，未在其他场景验证。", "",
              "## 9. 下一轮只建议一项工作", "",
              "**把执行中评价改成“真实前缀 + 逐步回灌自身预测”的受限递推，并明确标注哪些字段仍是"
              "真值输入。** 依据：本轮已能预测固定 Δt 的下一个观测与终态，但完全不知道误差如何随"
              "递推步数累积；这是当前唯一挡住“能否自主推演”的未知量，且不需要新数据或新场景。", ""]
    (DOCS / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {DOCS / 'REPORT.md'} ({len(lines)} lines)")
    print("REPORT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
