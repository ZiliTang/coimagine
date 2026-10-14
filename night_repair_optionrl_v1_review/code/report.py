"""Assemble the morning report tables and figures from the saved run outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

for candidate in ("/home/tzl/.local/share/fonts/wqy.ttc", "/tmp/wqy.ttc", "/tmp/cjk.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        matplotlib.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=candidate).get_name(), "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = ROOT / "outputs/night_repair_optionrl_v1"
WM = BASE / "world_models"
RL = BASE / "option_rl"
COLLECT = BASE / "collection"
DOCS = ROOT / "docs/night_repair_optionrl_v1"
STEPS = (1, 4, 8, 16)


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def model_table(results: Optional[dict]) -> List[str]:
    if not results:
        return ["_模型评估结果缺失_"]
    lines = ["| 模型 | 参数量 | 区间 | k=1 | k=4 | k=8 | k=16 |", "|---|---:|---|---:|---:|---:|---:|"]
    for name, entry in sorted(results["models"].items()):
        parameters = entry.get("parameters")
        parameters = parameters.get("total") if isinstance(parameters, dict) else None
        for regime in ("t>0", "t=0"):
            block = entry.get("r0", {}).get("by_regime", {}).get(regime, {})
            cells = []
            for step in STEPS:
                value = block.get(f"step{step}", {}).get("p50_cm")
                cells.append("—" if value is None else f"{value:.3f}")
            lines.append(f"| {name} | {parameters:,} | {regime} | " + " | ".join(cells) + " |"
                         if parameters else f"| {name} | — | {regime} | " + " | ".join(cells) + " |")
    spread: Dict[str, Dict[int, List[float]]] = {}
    for name, entry in results["models"].items():
        if not isinstance(entry.get("parameters"), dict):
            continue
        family = "直接多步" if entry["kind"] == "direct_multistep" else "latent 自回归"
        for step in STEPS:
            value = entry.get("r0", {}).get("by_regime", {}).get("t>0", {}).get(
                f"step{step}", {}).get("p50_cm")
            if value is not None:
                spread.setdefault(family, {}).setdefault(step, []).append(value)
    if spread:
        lines += ["", "| 模型族 | 跨度 | seed 值 | 极差 cm |", "|---|---:|---|---:|"]
        for family, steps in sorted(spread.items()):
            for step, values in sorted(steps.items()):
                values = sorted(values)
                lines.append(f"| {family} | k={step} | " +
                             " / ".join(f"{v:.3f}" for v in values) +
                             f" | {max(values) - min(values):.3f} |")
    return lines


def ablation_table(ablation: Optional[dict]) -> List[str]:
    if not ablation:
        return ["_消融结果缺失_"]
    lines = ["| 跨度/机器人 | 正常 p50 | 冻结 p50 | 逐样本差值中位数 | 95% CI | 判读 |",
             "|---|---:|---:|---:|---|---|"]
    for key, entry in sorted(ablation["per_step"].items()):
        low = entry["bootstrap"]["ci95_low_cm"]
        high = entry["bootstrap"]["ci95_high_cm"]
        if high < 0:
            verdict = "递推有用（冻结更差）"
        elif low > 0:
            verdict = "冻结更好"
        else:
            verdict = "分不出"
        lines.append(f"| {key} | {entry['normal_p50_cm']:.4f} | {entry['frozen_p50_cm']:.4f} | "
                     f"{entry['statistics']['median_of_differences_cm']:+.4f} | "
                     f"[{low:+.4f}, {high:+.4f}] | {verdict} |")
    lines += ["", f"同样是本：{ablation['windows']} 个窗口 / {ablation['roots']} 个 root，"
                  f"区间 {ablation['regime']}，同一检查点、同一跨度、同一可见性 mask。"]
    return lines


def rl_table(random_baseline: Optional[dict], policy: Optional[dict]) -> List[str]:
    lines = ["| 策略 | 划分 | 成功率 | 净推进 (m) | 中位技能数 | 平均回报 |",
             "|---|---|---:|---:|---:|---:|"]
    if random_baseline:
        for split in ("validation", "test"):
            entry = random_baseline.get(split)
            if entry:
                lines.append(f"| 均匀随机（同规则） | {split} | {entry['success_rate']:.4f} | "
                             f"{entry['mean_net_progress_m']:+.4f} | {entry['median_skills']:.0f} | "
                             f"{entry['mean_return']:+.4f} |")
    if policy:
        for split in ("val", "test"):
            entry = policy.get(split)
            if entry:
                lines.append(f"| 训练后策略（确定性） | {split} | {entry['success_rate']:.4f} | "
                             f"{entry['mean_net_progress_m']:+.4f} | {entry['median_skills']:.0f} | "
                             f"{entry['mean_return']:+.4f} |")
    return lines


def collection_table(summary: Optional[dict]) -> List[str]:
    if not summary:
        return ["_试采结果缺失_"]
    lines = ["| 来源 | episode | root | 技能数 | 原始帧 | 成功率 | 截断率 | 净推进 (m) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, entry in summary["settings"].items():
        lines.append(f"| {name} | {entry['episodes']} | {entry['roots']} | {entry['skills']} | "
                     f"{entry['raw_frames']} | {entry['success_rate']:.3f} | "
                     f"{entry['truncated_rate']:.3f} | {entry['mean_net_progress_m']:+.4f} |")
    lines += ["", f"合计：{summary['episodes_total']} episode、{summary['independent_roots']} 个独立 root、"
                  f"{summary['skills_total']} 次技能、{summary['raw_frames_total']} 原始帧；"
                  f"分支族 {summary['branch_families']} 个。"]
    return lines


def figure_model(results: Optional[dict], ablation: Optional[dict]) -> Optional[Path]:
    if not results:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    families = {"direct_multistep": ("-", "o", "直接多步 Transformer"),
                "latent_autoregressive": ("--", "s", "latent 自回归")}
    for name, entry in sorted(results["models"].items()):
        if not isinstance(entry.get("parameters"), dict):
            continue
        style = families.get(entry["kind"], (":", "^", entry["kind"]))
        for robot, colour in ((0, "#1f4e79"), (1, "#b03030")):
            values = [entry.get("r0" if robot == 0 else "r1", {}).get("by_regime", {})
                      .get("t>0", {}).get(f"step{step}", {}).get("p50_cm") for step in STEPS]
            if any(v is not None for v in values):
                axes[0].plot(STEPS, [v if v is not None else np.nan for v in values], style[0],
                             marker=style[1], color=colour,
                             label=f"{style[2]} r{robot}", linewidth=1.3)
    for name, colour, linestyle in (("P0_hold", "#b03030", ":"), ("P1_global", "#e08214", "-."),
                                    ("P1_grouped", "#7a1010", "--")):
        block = results["models"].get(name, {}).get("r0", {}).get("by_regime", {}).get("t>0", {})
        values = [block.get(f"step{step}", {}).get("p50_cm") for step in STEPS]
        if any(v is not None for v in values):
            axes[0].plot(STEPS, [v if v is not None else np.nan for v in values], linestyle,
                         color=colour, linewidth=1.4, label=f"{name}（基线）")
    axes[0].axvline(8, color="#999999", linestyle="-.", linewidth=1.0)
    axes[0].text(8.15, axes[0].get_ylim()[1] * 0.92, "训练跨度边界", fontsize=8)
    axes[0].set_xlabel("预测跨度 k（步，1 步 = 0.05 s）", fontsize=10)
    axes[0].set_ylabel("t>0 中点 p50 (cm)", fontsize=10)
    axes[0].set_title("(a) 修复后误差–跨度（k=16 为跨度外推）", fontsize=11)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    if ablation:
        for key in ("step1_r0", "step4_r0", "step8_r0", "step16_r0"):
            entry = ablation["per_step"].get(key)
            if entry:
                axes[1].plot([entry["normal_p50_cm"], entry["frozen_p50_cm"]],
                             [int(key.replace("step", "").split("_")[0])] * 2, "o-",
                             color="#1f4e79", alpha=0.6)
        steps = [int(k.replace("step", "").split("_")[0]) for k in
                 ("step1_r0", "step4_r0", "step8_r0", "step16_r0") if k in ablation["per_step"]]
        axes[1].plot(steps, [ablation["per_step"][f"step{s}_r0"]["normal_p50_cm"] for s in steps],
                     "o-", color="#1f4e79", label="正常递推（r0）")
        axes[1].plot(steps, [ablation["per_step"][f"step{s}_r0"]["frozen_p50_cm"] for s in steps],
                     "s--", color="#b03030", label="冻结 latent（r0）")
    axes[1].set_xlabel("预测跨度 k", fontsize=10)
    axes[1].set_ylabel("t>0 中点 p50 (cm)", fontsize=10)
    axes[1].set_title("(b) 同样本 latent 消融", fontsize=11)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("F1 修复后的世界模型对照与 latent 消融", fontsize=12)
    figure.tight_layout()
    path = DOCS / "F1_repair_and_ablation.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_rl(policy: Optional[dict], random_baseline: Optional[dict]) -> Optional[Path]:
    if not policy:
        return None
    history = policy.get("history", [])
    if not history:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    iterations = [row["iteration"] for row in history]
    axes[0].plot(iterations, [row["val_net_progress_m"] for row in history], "o-",
                 label="训练后策略（确定性，固定验证 root）")
    axes[0].plot(iterations, [row["explore_return"] for row in history], "s--", alpha=0.6,
                 label="训练探索回报（含 ε 随机技能）")
    if random_baseline:
        entry = random_baseline.get("validation")
        if entry:
            axes[0].axhline(entry["mean_net_progress_m"], color="#b03030", linestyle=":",
                            label=f"均匀随机基线 {entry['mean_net_progress_m']:+.3f} m")
    axes[0].set_xlabel("PPO 迭代", fontsize=10)
    axes[0].set_ylabel("净目标推进 (m) / 回报", fontsize=10)
    axes[0].set_title("(a) 学习曲线与随机基线", fontsize=11)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(iterations, [row["val_success_rate"] for row in history], "o-", color="#2ca02c",
                 label="训练后策略成功率")
    if random_baseline:
        entry = random_baseline.get("validation")
        if entry:
            axes[1].axhline(entry["success_rate"], color="#b03030", linestyle=":",
                            label=f"随机成功率 {entry['success_rate']:.3f}")
    axes[1].set_xlabel("PPO 迭代", fontsize=10)
    axes[1].set_ylabel("成功率", fontsize=10)
    axes[1].set_title("(b) 成功率", fontsize=11)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("F2 option 级 RL：训练后策略与同规则随机基线", fontsize=12)
    figure.tight_layout()
    path = DOCS / "F2_rl_learning.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DOCS))
    args = parser.parse_args()
    DOCS.mkdir(parents=True, exist_ok=True)
    results = load(WM / "multi_step_results_test_multistep.json")
    ablation = load(WM / "ablation_test_multistep.json")
    random_baseline = load(RL / "random_baseline.json")
    policy = load(RL / "train_result_seed0.json")
    collection = load(COLLECT / "TRIAL_SUMMARY.json")
    sections = {
        "model_table": model_table(results),
        "ablation_table": ablation_table(ablation),
        "rl_table": rl_table(random_baseline, policy),
        "collection_table": collection_table(collection),
    }
    (Path(args.out) / "TABLES.md").write_text(
        "\n\n".join(f"## {name}\n\n" + "\n".join(lines) for name, lines in sections.items()) + "\n")
    for function in (lambda: figure_model(results, ablation), lambda: figure_rl(policy, random_baseline)):
        path = function()
        print("wrote", path)
    print(json.dumps({key: len(value) for key, value in sections.items()}, indent=1))
    print("REPORT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
