"""The three diagnostic figures required by the task book section 8.

F1  the 2x2 single-head diagnostic: four engineering configurations, three seeds, against the hold
    baseline under the same reference rule.
F2  the small-set refit: the pure midpoint task against the full multi-task objective, evaluated on
    every legal adjacent frame.
F3  the main migration: residual versus the corrected direct control, per architecture, with the
    hold baseline, on the frozen test panel.
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

OUT = ROOT / "outputs/residual_prediction_v1"
DOCS = ROOT / "docs/residual_prediction_v1"

for candidate in ("/tmp/wqy.ttc", "/tmp/cjk.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        name = font_manager.FontProperties(fname=candidate).get_name()
        matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

CELL_ORDER = ["direct_per_group", "direct_per_column", "residual_per_group",
              "residual_per_column"]
CELL_LABEL = {"direct_per_group": "直接 + 单标量", "direct_per_column": "直接 + 逐列",
              "residual_per_group": "残差 + 单标量", "residual_per_column": "残差 + 逐列"}


def figure_one() -> Path:
    payload = json.loads((OUT / "single_head_diagnostic_d1.json").read_text())
    cells: Dict[str, Dict[str, dict]] = {}
    for name, entry in payload["cells"].items():
        if name == "initial_state_hash":
            continue
        cell, seed = name.rsplit("_seed", 1)
        cells.setdefault(cell, {})[seed] = entry
    hold = cells[CELL_ORDER[0]]["0"]["val_hold"]["r0"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    seeds = ["0", "1", "2"]
    width = 0.25
    for position, (axis, mode) in enumerate(zip(axes, ("overall", "moving"))):
        for offset, seed in enumerate(seeds):
            values = []
            for cell in CELL_ORDER:
                entry = cells[cell][seed]["val"]["r0"]
                values.append(entry["p50_cm"] if mode == "overall"
                              else entry["sustained_visible_moving"]["p50_cm"])
            axis.bar(np.arange(len(CELL_ORDER)) + (offset - 1) * width, values, width,
                     label=f"seed {seed}")
        baseline = hold["p50_cm"] if mode == "overall" \
            else hold["sustained_visible_moving"]["p50_cm"]
        axis.axhline(baseline, color="#b03030", linestyle="--", linewidth=1.4,
                     label="保持当前（同一 reference 规则）")
        axis.set_xticks(np.arange(len(CELL_ORDER)))
        axis.set_xticklabels([CELL_LABEL[c] for c in CELL_ORDER], fontsize=9)
        axis.set_ylabel("下一步 r0 中点 p50 (cm)", fontsize=10)
        axis.set_title("(a) 验证集总体" if mode == "overall" else "(b) 持续可见且运动中",
                       fontsize=11)
        axis.grid(alpha=0.25, axis="y")
        axis.legend(fontsize=8)
    figure.suptitle("F1 单头 2×2 对照：直接/残差 × 单标量/逐列（3 seeds，同一初始权重与批次顺序）",
                    fontsize=12)
    figure.tight_layout()
    path = DOCS / "F1_single_head_diagnostic.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_two() -> Path:
    payload = json.loads((OUT / "tiny_fit_d1.json").read_text())
    tasks = payload["tasks"]
    counts = sorted(tasks, key=lambda key: tasks[key]["sequences"])
    labels = [str(tasks[key]["sequences"]) for key in counts]
    single = [tasks[key]["single_task"]["p50_cm"] for key in counts]
    multi = [tasks[key]["multi_task_geometry"]["p50_cm"] for key in counts]
    hold = [tasks[key]["single_task"]["hold_p50_cm"] for key in counts]
    windows = [tasks[key]["evaluation_windows"] for key in counts]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(counts))
    axes[0].bar(x - 0.25, single, 0.25, label="纯中点单任务（残差）")
    axes[0].bar(x, multi, 0.25, label="完整多任务（六头，直接规范化）")
    axes[0].bar(x + 0.25, hold, 0.25, label="保持当前")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{label} 条序列\n({n} 窗口)" for label, n in zip(labels, windows)],
                            fontsize=9)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("下一步 r0 中点 p50 (cm，对数轴)", fontsize=10)
    axes[0].set_title("(a) 拟合能力与迁移损失", fontsize=11)
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, single, "o-", label="纯中点单任务")
    axes[1].plot(x, multi, "s-", label="完整多任务")
    axes[1].plot(x, hold, "^--", label="保持当前")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("固定序列数", fontsize=10)
    axes[1].set_ylabel("p50 (cm)", fontsize=10)
    axes[1].set_title("(b) 评价覆盖每条序列的全部合法相邻帧", fontsize=11)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("F2 小集合拟合：单任务能把 1 条序列压到 0.027 cm；共享多任务目标拿不到这个解",
                    fontsize=12)
    figure.tight_layout()
    path = DOCS / "F2_tiny_set_fit.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def figure_three() -> Path:
    payload = json.loads((OUT / "main_results_eval_test.json").read_text())
    models = payload["models"]
    hold = models["mlp_history_residual_per_group_seed0"]["hold_next"]
    families = [("mlp_current", "单帧 MLP"), ("mlp_history", "历史 MLP"),
                ("transformer_history", "历史 Transformer")]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    configurations = [("单帧 MLP\n(残差)", "mlp_current", "residual", "#2ca02c"),
                      ("历史 MLP\n(残差)", "mlp_history", "residual", "#1f77b4"),
                      ("Transformer\n(残差)", "transformer_history", "residual", "#17a2b8"),
                      ("历史 MLP\n(直接对照)", "mlp_history", "direct", "#d62728")]
    for axis, (head, regime, robot, title) in zip(
            axes, [("next", "t>0", "r0", "(a) 下一步 r0，t>0"),
                   ("next", "t>0", "r1", "(b) 下一步 r1，t>0"),
                   ("terminal", "t>0", "r0", "(c) 终态 r0（t>0）")]):
        positions = np.arange(len(configurations))
        for offset, seed in enumerate((0, 1, 2)):
            values = []
            for _label, prefix, parameterisation, _colour in configurations:
                name = f"{prefix}_{parameterisation}_per_group_seed{seed}"
                values.append(models[name][head][regime][robot]["p50_cm"])
            axis.bar(positions + (offset - 1) * 0.26, values, 0.26,
                     color=[c for _l, _p, _q, c in configurations],
                     alpha=1.0 - 0.22 * offset,
                     label="seed 0/1/2" if offset == 0 else None,
                     edgecolor="white", linewidth=0.5)
        baseline_source = payload["models"]["mlp_history_residual_per_group_seed0"][
            "hold_terminal" if head == "terminal" else "hold_next"]
        baseline = baseline_source[regime][robot]["p50_cm"]
        axis.axhline(baseline, color="#7a1010", linestyle="--", linewidth=1.5,
                     label=f"保持当前 {baseline:.3f} cm")
        axis.set_xticks(positions)
        axis.set_xticklabels([label for label, _p, _q, _c in configurations], fontsize=9)
        axis.set_ylabel("中点 p50 (cm)", fontsize=10)
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=0.25, axis="y")
        axis.legend(fontsize=8)
    figure.suptitle("F3 迁移到完整多任务模型（冻结测试面板）：残差把下一步误差从 2.2 cm 降到 0.14 cm，"
                    "并优于保持基线", fontsize=12)
    figure.tight_layout()
    path = DOCS / "F3_main_migration.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    for function in (figure_one, figure_two, figure_three):
        print("wrote", function())
    print("FIGURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
