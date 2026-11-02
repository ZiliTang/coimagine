"""Saved-log audit of preparation time and nearby-initial-pose effect variation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def bounds(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return [min(values), max(values)] if values else None


def main(out):
    manifest = json.loads((out / "manifest.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    families = {}
    for case in manifest["cases"]:
        perturbation = case.get("perturbation")
        if perturbation:
            families.setdefault(perturbation["base"], set()).update((case["name"], perturbation["base"]))
    results = []
    for name, members in families.items():
        for enabled in (False, True):
            rows = [m for m in metrics if m["case"] in members and m["variant"] == enabled]
            completed = [r for r in rows if r["status"] == "completed"]
            results.append({"family": name, "fast_approach": enabled,
                            "requested": len(rows), "executed": len(completed),
                            "controller_complete": sum(r["exit"] == "effect_complete" for r in completed),
                            "fulfilled": sum(r["fulfilled"] for r in completed),
                            "actual_effect": bounds(completed, "actual_effect"),
                            "sideways_m": bounds(completed, "sideways"),
                            "rotation_deg": bounds(completed, "rotation_deg"),
                            "preparation_seconds": bounds(completed, "preparation_seconds"),
                            "total_seconds": bounds(completed, "seconds"),
                            "members": sorted(members)})
    regressions = []
    for case in manifest["cases"]:
        pair = [m for m in metrics if m["case"] == case["name"]]
        if len(pair) == 2 and all(r["status"] == "completed" for r in pair):
            old, new = sorted(pair, key=lambda r: r["variant"])
            if old["fulfilled"] and not new["fulfilled"]:
                regressions.append(case["name"])
    report = {"version": "approach-stability-audit-v1", "families": results,
              "effect_regressions": regressions,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "fixed development perturbations; all exits retained, not an RL or learned-model result"}
    (out / "stability.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 相邻初态效果稳定性", "", "包含全部执行结果，不因退出或未兑现筛选样本。范围为样本最小/最大值。",
             "", "| 场景族 | 接近加速 | 样本数 | 声明完成 | 离线兑现 | 效果范围 | 侧偏 m | 总耗时 s |",
             "|---|---|---:|---:|---:|---|---|---|"]
    for r in results:
        lines.append(f"| {r['family']} | {r['fast_approach']} | {r['executed']}/{r['requested']} | "
                     f"{r['controller_complete']} | {r['fulfilled']} | {r['actual_effect']} | "
                     f"{r['sideways_m']} | {r['total_seconds']} |")
    lines += ["", f"原兑现而新未兑现：{regressions}", "", "固定请求、物理、奖励、预算和既有离线阈值均不变。",
              "横向/角度变化仍须作为模型目标分布建模；小面板不证明神经网络拟合精度或 RL 学习效率。"]
    (out / "STABILITY.md").write_text("\n".join(lines) + "\n")
    plot(out, metrics, families)
    print(json.dumps(report, indent=2))


def plot(out, metrics, families):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    names = ("two_faces_N", "rotate_CCW_0.1", "tilt_+8", "cooperative_N")
    old, new = [], []
    for name in names:
        pair = [m for m in metrics if m["case"] == name and m["status"] == "completed"]
        old.append(next((m["preparation_seconds"] for m in pair if not m["variant"]), float("nan")))
        new.append(next((m["preparation_seconds"] for m in pair if m["variant"]), float("nan")))
    xs = np.arange(len(names))
    axes[0, 0].bar(xs - .18, old, .36, color="#d38b32", label="previous approach")
    axes[0, 0].bar(xs + .18, new, .36, color="#218d70", label="faster approach")
    axes[0, 0].set_xticks(xs, ("two faces", "rotate 0.1", "tilt +8", "cooperative"))
    axes[0, 0].set_ylabel("Preparation time / s")
    axes[0, 0].legend(fontsize=8)
    for ax, name in zip((axes[0, 1], axes[1, 0], axes[1, 1]), ("two_faces_N", "tilt_+8", "cooperative_N")):
        members = families.get(name, {name})
        for enabled, color in ((False, "#d38b32"), (True, "#218d70")):
            rows = [m for m in metrics if m["case"] in members and m["variant"] == enabled
                    and m["status"] == "completed"]
            ax.scatter([m["sideways"] for m in rows], [m["actual_effect"] for m in rows],
                       label="previous" if not enabled else "faster", color=color, s=45,
                       marker="x" if not enabled else "o")
        ax.axhline(.3, color="#6b7890", linestyle="--", linewidth=.8)
        ax.set_title(f"{name}: nearby starts, all outcomes")
        ax.set_xlabel("Final cross-track / m")
        ax.set_ylabel("Forward effect / m")
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.suptitle("Preparation speed and initial-pose sensitivity (same requests and budgets)")
    fig.savefig(out / "stability.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
