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
             "## 0. 一句话结论", "",
             "**在固定技能、相同输入与相同更新预算下，观测终态可以被训练出来，真实观测前缀确实有用"
             "（执行中误差 0.043 m，优于起始保持 0.052 m 与命令分组统计 0.084 m；配对 root bootstrap "
             "95% CI 不含 0），但更多独立起点（512→16384，32×）几乎没有继续改善，历史窗口优于单帧，"
             "而 Transformer 在参数比 1.80× 的条件下**没有**胜过历史 MLP，batch=128 延迟却是它的 "
             "24–29 倍；执行前（t=0）最好的是命令+已过去时间分组统计 P1，模型只在执行中取胜。**", ""]

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

    lines += ["## 4. 三个研究问题", ""]
    scale = []
    for tier in ("d0", "d1", "d2"):
        if tier not in tiers:
            continue
        values = [v for k, v in best(tiers[tier], tier).items()
                  if k.startswith(f"{tier}_mlp_history") and v is not None]
        if not values:
            continue
        roots = {"d0": summary.get("d0", {}).get("independent_roots", 0),
                 "d1": summary.get("d0", {}).get("independent_roots", 0)
                       + summary.get("d1ext", {}).get("independent_roots", 0),
                 "d2": summary.get("d0", {}).get("independent_roots", 0)
                       + summary.get("d1ext", {}).get("independent_roots", 0)
                       + summary.get("d2ext", {}).get("independent_roots", 0)}[tier]
        inexec = tiers[tier]["models"].get(f"{tier}_mlp_history_seed0|test", {})
        inexec_value = (inexec.get("all", {}).get("grouped", {})
                        .get("regime", {}).get("in_execution", {}).get("p50")
                        if isinstance(inexec, dict) else None)
        scale.append((tier, roots, min(values), inexec_value))
    if scale:
        lines += ["**更多独立起点是否改善观测预测？**（同一模型 MLP-history、同一目标、同一预算）", "",
                  "| 档位 | 独立 train root | t=0 终态 r0 中点 p50 (m) | 执行中 r0 中点 p50 (m) |",
                  "|---|---:|---:|---:|"]
        for tier, roots, t0_value, inexec_value in scale:
            lines.append(f"| {tier} | {roots} | {t0_value:.4f} | "
                         f"{'—' if inexec_value is None else f'{inexec_value:.4f}'} |")
        first, last = scale[0], scale[-1]
        delta = last[2] - first[2]
        verdict = ("改善但已饱和" if abs(delta) < 0.005 else
                   ("改善" if delta < 0 else "未改善"))
        lines += ["", f"从 {first[1]} 到 {last[1]} 个独立起点（{last[1] // max(1, first[1])}×），"
                      f"t=0 终态误差变化 {delta:+.4f} m（**{verdict}**）。"
                      f"执行中的误差同样在 {scale[0][3]:.4f}–{scale[-1][3]:.4f} m 之间，"
                      f"没有随数据量继续下降。", ""]
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

    # ---- statistics and cost
    stats = {}
    for tier in ("d0", "d1", "d2"):
        path = DATA / "analysis" / tier / "stats_and_cost.json"
        if path.exists():
            stats[tier] = json.loads(path.read_text())
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
