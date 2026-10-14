"""The five core figures required by the task book section 10.6.

F1  parameters against error and inference cost
F2  same-protocol train/validation curves
F3  multi-step error against the prediction span
F4  direct terminal prediction, the known-true-T rollout diagnostic, and the latent ablation
F5  typical trajectories: a median case, a tail failure, and a visibility change, chosen by a fixed
    rule rather than by hand
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/transformer_scale_latent_v1"
DOCS = ROOT / "docs/transformer_scale_latent_v1"

for candidate in ("/home/tzl/.local/share/fonts/wqy.ttc", "/tmp/wqy.ttc", "/tmp/cjk.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        matplotlib.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=candidate).get_name(), "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

STEPS = (1, 4, 8, 16)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def figure_one(results: Dict[str, object], audit: Dict[str, object]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    capacity = _load(OUT / "capacity_results_test_multistep.json") or {"models": {}}
    for name, entry in sorted(capacity["models"].items(), key=lambda kv: kv[1]["parameters"]):
        axes[0].plot(entry["parameters"] / 1e6, entry["t>0_r0"]["p50_cm"], "o",
                     color="#1f4e79", markersize=9)
        axes[0].annotate(name.replace("capacity_", "").replace("_seed0", ""),
                         (entry["parameters"] / 1e6, entry["t>0_r0"]["p50_cm"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=7)
    for name, entry in results["models"].items():
        parameters = entry.get("parameters")
        if not isinstance(parameters, dict):
            continue
        value = entry.get("r0", {}).get("by_regime", {}).get("t>0", {}).get("step1", {}).get("p50_cm")
        if value is None:
            continue
        axes[0].plot(parameters["total"] / 1e6, value, "s", color="#2ca02c", markersize=9)
        axes[0].annotate(name.split("_h8")[0], (parameters["total"] / 1e6, value),
                         textcoords="offset points", xytext=(6, -10), fontsize=7)
    # the floors every learned model has to clear
    for name, colour, style in (("P0_hold", "#b03030", ":"), ("P1_global", "#e08214", "-."),
                                ("P1_grouped", "#7a1010", "--")):
        block = results["models"].get(name, {}).get("r0", {}).get("by_regime", {}).get("t>0", {})
        value = block.get("step1", {}).get("p50_cm")
        if value is not None:
            axes[0].axhline(value, color=colour, linestyle=style, linewidth=1.2,
                            label=f"{name} {value:.3f} cm")
    axes[0].set_xlabel("总可训练参数量（百万）", fontsize=10)
    axes[0].set_ylabel("t>0 下一步 r0 中点 p50 (cm)", fontsize=10)
    axes[0].set_title("(a) 参数量–误差，与基线地板对照", fontsize=11)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    timing = {name: value for name, value in
              (audit or {}).get("timing_100_updates", {}).items()
              if isinstance(value, dict)}
    if timing:
        names = [name for name in timing]
        axes[1].bar(np.arange(len(names)),
                    [timing[name]["seconds_per_update"] for name in names],
                    color=["#2ca02c", "#1f4e79"][:len(names)])
        axes[1].set_xticks(np.arange(len(names)))
        axes[1].set_xticklabels(["直接多步", "latent 递推"][:len(names)], fontsize=9)
        axes[1].set_ylabel("每次更新秒数（4 线程，CPU）", fontsize=10)
        axes[1].set_title("(b) 同规模配对的训练成本", fontsize=11)
        axes[1].grid(alpha=0.25, axis="y")
    figure.suptitle("F1 容量曲线与配对成本：同参数量不等于同计算量", fontsize=12)
    figure.tight_layout()
    path = DOCS / "F1_capacity_and_cost.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_two(curves_by_run: Dict[str, List[dict]]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, entry in sorted(curves_by_run.items()):
        if not entry:
            continue
        updates = [row["update"] for row in entry]
        axes[0].plot(updates, [row["val_total"] for row in entry], label=name, linewidth=1.2)
    axes[0].set_xlabel("优化更新数", fontsize=10)
    axes[0].set_ylabel("固定验证面板总损失", fontsize=10)
    axes[0].set_title("(a) 同口径验证曲线（eval 模式、同一面板）", fontsize=11)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    for name, entry in sorted(curves_by_run.items()):
        if not entry:
            continue
        for step in (1, 8):
            values = [row["val_step_p50_cm"][step - 1] for row in entry
                      if row.get("val_step_p50_cm") and row["val_step_p50_cm"][step - 1] is not None]
            if values:
                updates = [row["update"] for row in entry
                           if row.get("val_step_p50_cm")
                           and row["val_step_p50_cm"][step - 1] is not None]
                axes[1].plot(updates, values, linewidth=1.2,
                             linestyle="-" if step == 1 else "--",
                             label=f"{name} k={step}")
    axes[1].set_xlabel("优化更新数", fontsize=10)
    axes[1].set_ylabel("第 k 步 r0 中点 p50 (cm)", fontsize=10)
    axes[1].set_title("(b) 分跨度验证误差", fontsize=11)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.suptitle("F2 同口径学习曲线（训练批次损失只作优化日志，不用于比较）", fontsize=12)
    figure.tight_layout()
    path = DOCS / "F2_same_protocol_curves.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_three(results: Dict[str, object]) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {"direct_multistep": ("-", "o", "直接多步 Transformer"),
              "latent_autoregressive": ("--", "s", "latent 自回归"),
              "transformer_history": (":", "^", "直接预测 Transformer")}
    for name, entry in sorted(results["models"].items()):
        if not isinstance(entry.get("parameters"), dict):
            continue                      # baselines get their own lines below
        family = entry["kind"]
        style = styles.get(family, ("-", "x", family))
        steps = [step for step in STEPS if f"step{step}" in entry.get("step_p50_cm", {})]
        for robot, colour in ((0, "#1f4e79"), (1, "#b03030")):
            values = [entry["step_p50_cm"][f"step{step}"][f"r{robot}"] for step in steps]
            axes[0].plot(steps, values, style[0], marker=style[1], color=colour,
                         label=f"{style[2]} r{robot}")
    for name, colour, style in (("P0_hold", "#b03030", ":"), ("P1_global", "#e08214", "-."),
                                ("P1_grouped", "#7a1010", "--")):
        block = results["models"].get(name, {}).get("r0", {}).get("by_regime", {}).get("t>0", {})
        steps = [step for step in STEPS if block.get(f"step{step}", {}).get("p50_cm") is not None]
        if steps:
            axes[0].plot(steps, [block[f"step{step}"]["p50_cm"] for step in steps], style,
                         color=colour, linewidth=1.3, label=f"{name}（基线）")
    axes[0].axvline(8, color="#999999", linestyle="-.", linewidth=1.0)
    axes[0].text(8.2, axes[0].get_ylim()[1] * 0.9, "训练跨度边界", fontsize=8)
    axes[0].set_xlabel("预测跨度 k（步，1 步 = 0.05 s）", fontsize=10)
    axes[0].set_ylabel("中点 p50 (cm)", fontsize=10)
    axes[0].set_title("(a) 误差–跨度（k=16 在训练跨度之外）", fontsize=11)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    # the same-sample curve: only windows that have all 16 real frames
    for name, entry in sorted(results["models"].items()):
        if not isinstance(entry.get("parameters"), dict):
            continue
        sample = entry.get("same_sample") or {}
        steps = sorted(int(key) for key in sample if sample.get(key) is not None)
        if steps:
            axes[1].plot(steps, [sample[str(step)] for step in steps], marker="o",
                         label=name, linewidth=1.3)
    floors = results["models"].get("P1_grouped", {}).get("r0", {}).get("by_regime", {}).get("t>0", {})
    steps = sorted(step for step in STEPS if floors.get(f"step{step}", {}).get("p50_cm") is not None)
    if steps:
        axes[1].plot(steps, [floors[f"step{step}"]["p50_cm"] for step in steps], "--",
                     color="#7a1010", label="P1 命令+时间分组（基线）")
    axes[1].set_xlabel("预测跨度 k", fontsize=10)
    axes[1].set_ylabel("中点 p50 (cm)", fontsize=10)
    axes[1].set_title("(b) 固定同样本（k≤16 全有真实帧）的误差增长", fontsize=11)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.suptitle("F3 多步误差随跨度的变化：区分误差累积与样本构成变化", fontsize=12)
    figure.tight_layout()
    path = DOCS / "F3_horizon_error.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_four(results: Dict[str, object]) -> Path:
    """Direct terminal behaviour and the latent recurrence: what is implemented and what is not."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    entries = [(name, entry) for name, entry in sorted(results["models"].items())
               if isinstance(entry.get("parameters"), dict)]
    labels = [name.split("_h8")[0] for name, _e in entries]
    width = 0.35
    for offset, (step, alpha) in enumerate(((1, 1.0), (8, 0.55))):
        values = [entry["r0"]["by_regime"].get("t>0", {}).get(f"step{step}", {}).get("p50_cm")
                  for _n, entry in entries]
        axes[0].bar(np.arange(len(entries)) + (offset - 0.5) * width,
                    [v if v is not None else 0.0 for v in values], width, alpha=alpha,
                    color="#1f4e79", label=f"k={step}")
    hold = results["models"].get("P0_hold", {}).get("r0", {}).get("by_regime", {}).get("t>0", {})
    if hold.get("step1", {}).get("p50_cm") is not None:
        axes[0].axhline(hold["step1"]["p50_cm"], color="#b03030", linestyle="--", linewidth=1.3,
                        label=f"P0 保持 k=1（{hold['step1']['p50_cm']:.3f} cm）")
    axes[0].set_xticks(np.arange(len(entries)))
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("t>0 r0 中点 p50 (cm)", fontsize=10)
    axes[0].set_title("(a) 多步模型与保持基线", fontsize=11)
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8)
    latent = [entry for name, entry in entries if name.startswith("latent")]
    if latent:
        entry = latent[0]
        ablation = entry.get("freeze_latent_ablation", {})
        steps = [step for step in STEPS if f"step{step}" in ablation]
        if steps:
            axes[1].plot(steps, [ablation[f"step{step}"] for step in steps], "o-",
                         label="把递推 latent 置回初值（消融）")
        main_steps = [step for step in STEPS
                      if entry["r0"]["by_regime"].get("t>0", {}).get(f"step{step}", {})
                      .get("p50_cm") is not None]
        axes[1].plot(main_steps, [entry["r0"]["by_regime"]["t>0"][f"step{step}"]["p50_cm"]
                                  for step in main_steps], "s--", label="正常自由递推")
        rollout = entry.get("true_t_rollout", {})
        if rollout.get("status") == "ok":
            axes[1].axhline(rollout["p50_cm"], color="#7a1010", linestyle=":",
                            label=f"递推到真实 T 的诊断（{rollout['p50_cm']:.2f} cm，用了真实 T）")
    axes[1].text(0.02, 0.04, "自主退出（按模型自身预测退出）本轮未实现；\n"
                             "真实 T 只作评估坐标，不是部署停止规则",
                 transform=axes[1].transAxes, fontsize=8)
    axes[1].set_xlabel("预测跨度 k", fontsize=10)
    axes[1].set_ylabel("t>0 r0 中点 p50 (cm)", fontsize=10)
    axes[1].set_title("(b) latent 递推消融与真实 T 诊断", fontsize=11)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.suptitle("F4 终态相关结果：自主退出尚未实现，真实 T 只作评估坐标", fontsize=12)
    figure.tight_layout()
    path = DOCS / "F4_terminal_and_rollout.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_five(results: Dict[str, object], panel_name: str = "test_multistep") -> Path:
    """Chosen by a fixed rule: the median-error window, the worst window, a visibility change."""

    direct = "direct_multistep_h8_lr0.0003_seed0"
    path = OUT / f"predictions_{direct}_{panel_name}.npz"
    if not path.exists():
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.text(0.5, 0.5, "逐样本预测尚未生成", ha="center")
        out = DOCS / "F5_trajectories.png"
        figure.savefig(out, dpi=150)
        plt.close(figure)
        return out
    data = dict(np.load(path, allow_pickle=True))
    truth, prediction = data["truth"], data["geometry"]
    keep = (data["geometry_mask"][:, :, 0] > 0) & (data["step_mask"] > 0)
    error8 = np.where(keep[:, 7], np.hypot(prediction[:, 7, 0] - truth[:, 7, 0],
                                           prediction[:, 7, 1] - truth[:, 7, 1]), np.nan)
    finite = np.flatnonzero(np.isfinite(error8))
    median_row = finite[np.argsort(error8[finite])[len(finite) // 2]]
    worst_row = finite[np.argmax(error8[finite])]
    # a genuine visibility transition: r0 visible at the anchor but not at step 8, or the reverse
    first_visible = data["geometry_mask"][:, 0, 0] > 0
    last_visible = data["geometry_mask"][:, 7, 0] > 0
    changed = np.flatnonzero((first_visible != last_visible) & np.isfinite(error8))
    change_row = int(changed[np.argmin(np.abs(error8[changed] - np.median(error8[finite])))]
                      if changed.size else median_row)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, row, title in ((axes[0], median_row, "(a) 中位案例（固定规则选中位）"),
                             (axes[1], worst_row, "(b) 尾部失败（固定规则选最大误差）"),
                             (axes[2], change_row,
                              "(c) 可见性变化案例（锚点与末端可见性不同）")):
        steps = np.arange(1, 9)
        axis.plot(truth[row, :8, 0] * 100, truth[row, :8, 1] * 100, "o-", label="真值 r0 中点")
        axis.plot(prediction[row, :8, 0] * 100, prediction[row, :8, 1] * 100, "s--",
                  label="模型预测")
        axis.plot(data["anchor"][row, 0] * 100, data["anchor"][row, 1] * 100, "k*",
                  markersize=11, label="残差锚点")
        axis.set_xlabel("R_i0 x (cm)", fontsize=9)
        axis.set_ylabel("R_i0 y (cm)", fontsize=9)
        axis.set_title(f"{title}  k=8 误差 {error8[row] * 100:.2f} cm", fontsize=10)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_aspect("equal", adjustable="datalim")
    figure.suptitle("F5 典型轨迹真值/预测叠图（选择规则固定，不挑好例子）", fontsize=12)
    figure.tight_layout()
    out = DOCS / "F5_trajectories.png"
    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    results = _load(OUT / "multi_step_results_test_multistep.json")
    audit = _load(OUT / "audit.json")
    if results is None:
        print("no evaluation results yet")
        return 1
    curves = {}
    for path in sorted((OUT / "curves").glob("*.json")) if (OUT / "curves").exists() else []:
        curves[path.stem] = json.loads(path.read_text())
    for name, entry in results["models"].items():
        checkpoint = OUT / "checkpoints" / f"{name}.pt"
        if checkpoint.exists():
            import torch
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("curves"):
                curves[name] = saved["curves"]
    for function in (lambda: figure_one(results, audit), lambda: figure_two(curves),
                     lambda: figure_three(results), lambda: figure_four(results),
                     lambda: figure_five(results)):
        print("wrote", function())
    print("FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
