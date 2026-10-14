"""Generate the result tables and CSVs from the saved JSON, so nothing is transcribed by hand."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs/transformer_scale_latent_v1"
STEPS = (1, 4, 8, 16)
PRETTY = {"P0_hold": "P0 保持当前", "P1_global": "P1 全局位移（不分组）",
          "P1_grouped": "P1 命令+时间分组",
          "direct_multistep_h8_lr0.0003_seed0": "直接多步（1.6M 配对）",
          "latent_autoregressive_h8_lr0.0003_seed0": "latent 自回归（1.6M 配对）"}


def value(entry: dict, regime: str, step: int, robot: str = "r0") -> Optional[float]:
    block = entry.get(robot, {}).get("by_regime", {}).get(regime, {}).get(f"step{step}", {})
    return block.get("p50_cm")


def main() -> int:
    results = json.loads((OUT / "multi_step_results_test_multistep.json").read_text())
    rows: List[List[object]] = []
    header = ["model", "kind", "parameters", "regime", "step", "p50_cm", "p90_cm", "rmse_cm",
              "windows", "roots"]
    models = results["models"]
    for name, entry in models.items():
        regime_block = entry.get("r0", {}).get("by_regime", {})
        for regime in ("t=0", "t>0"):
            for step in STEPS:
                block = regime_block.get(regime, {}).get(f"step{step}")
                if not block:
                    continue
                rows.append([name, entry.get("kind", "baseline"),
                             entry.get("parameters", {}).get("total") if isinstance(
                                 entry.get("parameters"), dict) else None,
                             regime, step, block.get("p50_cm"), block.get("p90_cm"),
                             block.get("rmse_cm"), block.get("windows"), block.get("roots")])
    with open(OUT / "multi_step_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    markdown = ["| 模型 | 参数量 | 区间 | t>0 k=1 | t>0 k=4 | t>0 k=8 | t>0 k=16 | t=0 k=1 |",
                "|---|---:|---|---:|---:|---:|---:|---:|"]
    for name, entry in models.items():
        parameters = entry.get("parameters")
        if isinstance(parameters, dict):
            parameters = parameters.get("total")
        cells = []
        for step in STEPS:
            cells.append(value(entry, "t>0", step))
        zero = value(entry, "t=0", 1)
        formatted = " | ".join("—" if c is None else f"{c:.3f}" for c in cells)
        zero_text = "—" if zero is None else f"{zero:.3f}"
        markdown.append(f"| {PRETTY.get(name, name)} | {parameters:,} | "
                        f"{entry.get('kind', 'baseline')} | {formatted} | {zero_text} |"
                        if parameters else
                        f"| {PRETTY.get(name, name)} | — | {entry.get('kind', 'baseline')} | "
                        f"{formatted} | {zero_text} |")
    capacity = json.loads((OUT / "capacity_results_test_multistep.json").read_text())
    markdown += ["", "| 直接预测 Transformer | 参数量 | t>0 k=1 | t>0 p90 | t=0 k=1 | batch1 ms | batch128 ms |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
    for name, entry in sorted(capacity["models"].items(), key=lambda kv: kv[1]["parameters"]):
        markdown.append(f"| {name} | {entry['parameters']:,} | {entry['t>0_r0']['p50_cm']:.3f} | "
                        f"{entry['t>0_r0']['p90_cm']:.3f} | {entry['t=0_r0']['p50_cm']:.3f} | "
                        f"{entry['batch1_latency_ms']:.2f} | {entry['batch128_latency_ms']:.2f} |")
    # seed spread: three seeds per family, reported separately, never pooled as extra independent
    # roots; the spread is what tells us whether a difference is larger than run-to-run variation
    spread: Dict[str, Dict[str, List[float]]] = {}
    for name, entry in models.items():
        if not isinstance(entry.get("parameters"), dict):
            continue
        family = "直接多步" if entry["kind"] == "direct_multistep" else "latent 自回归"
        for step in STEPS:
            cell = entry.get("r0", {}).get("by_regime", {}).get("t>0", {}).get(
                f"step{step}", {}).get("p50_cm")
            if cell is not None:
                spread.setdefault(family, {}).setdefault(f"step{step}", []).append(cell)
    if spread:
        markdown += ["", "| 模型族 | 跨度 | seed0 | seed1 | seed2 | 极差 cm |",
                     "|---|---|---:|---:|---:|---:|"]
        for family, steps in sorted(spread.items()):
            for step, values in sorted(steps.items(), key=lambda kv: int(kv[0][4:])):
                values = sorted(values)
                markdown.append(f"| {family} | {step} | " +
                                " | ".join(f"{v:.3f}" for v in values) +
                                f" | {max(values) - min(values):.3f} |")
    comparison = OUT / "pair_comparison_test_multistep.json"
    if comparison.exists():
        data = json.loads(comparison.read_text())
        markdown += ["", "| 比较（直接多步 − latent） | 误差中位数之差 cm | 逐样本差值中位数 cm | 95% CI | root |",
                     "|---|---:|---:|---|---:|"]
        for key, entry in sorted(data.items()):
            stats, boot = entry["statistics"], entry["bootstrap"]
            markdown.append(f"| {key} | {stats['difference_of_medians_cm']:+.4f} | "
                            f"{stats['median_of_differences_cm']:+.4f} | "
                            f"[{boot['ci95_low_cm']:+.4f}, {boot['ci95_high_cm']:+.4f}] | "
                            f"{boot['roots']} |")
    (OUT / "TABLES.md").write_text("\n".join(markdown) + "\n")
    print("\n".join(markdown))
    print("TABLES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
