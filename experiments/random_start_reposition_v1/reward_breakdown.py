"""Offline reward accounting for an active frozen run; never modifies training state.

Training did not retain reward components. Progress, time and terminal events are exactly
recoverable; distance cost is an explicitly labelled accounting residual, NOT an independent
physical-log recomputation. Job selection is reconstructed from the frozen seed/update rule.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ("progress", "distance_cost_residual", "time_cost", "success_bonus", "failure_penalty")


def decompose(total, start, end, periods, success, config):
    if config["gamma"] != 1:
        raise ValueError("This accounting requires dense-reward-v2, gamma=1")
    progress = config["progress_weight"] * (start - end)
    time_cost = -config["time_cost"] * periods
    success_bonus = config["success_bonus"] * success
    failure_penalty = config["failure_penalty"] * (1 - success)
    distance_cost = total - progress - time_cost - success_bonus - failure_penalty
    if distance_cost > 1e-7:
        raise ValueError("Positive inferred distance cost: accounting assumptions do not hold")
    return dict(total=total, progress=progress, distance_cost_residual=distance_cost,
                time_cost=time_cost, success_bonus=success_bonus, failure_penalty=failure_penalty,
                initial_distance_mean=start, final_distance_mean=end, periods_mean=periods,
                success_rate=success,
                distance_cycle_weighted_mean=-distance_cost / (config["distance_cost"] * .05 * periods))


def selected_initial_mean(config, update, initial_distances):
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], update]))
    distances = []
    for _ in range(config["episodes_per_update"]):
        name = config["instances"][int(rng.integers(len(config["instances"]))) ]
        rng.integers(2 ** 31)  # rollout behaviour seed: must consume the same draw as training
        distances.append(initial_distances[name])
    return float(np.mean(distances))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/random_start_dense_reward_v2_long")
    args = parser.parse_args()
    out = args.run / "reward_breakdown"
    out.mkdir(exist_ok=True)
    manifest = json.loads((args.run / "manifest.json").read_text())
    config = manifest["config"]
    env = config["environment"]
    distances = {row["instance"]: row["start_goal_distance"] for row in manifest["subset"]["rows"]}
    history = []
    lines = (args.run / "updates.jsonl").read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            history.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    if not history:
        raise ValueError("No complete updates")
    training = []
    evaluation = []
    for row in history:
        if row["success_trajectories"] + row["failure_trajectories"] != config["episodes_per_update"]:
            raise ValueError("Unexpected rollout count")
        training.append(dict(update=row["update"], **decompose(
            row["return_mean"], selected_initial_mean(config, row["update"], distances),
            row["final_distance_mean"], row["periods_mean"], row["success"], env)))
        if "evaluation" in row:
            for mode, panel in row["evaluation"].items():
                components = [decompose(ep["return"], distances[ep["instance"]],
                                         ep["final_goal_distance"], ep["periods"], float(ep["success"]), env)
                              for ep in panel["rows"]]
                evaluation.append(dict(update=row["update"], mode=mode,
                                       **{key: float(np.mean([ep[key] for ep in components]))
                                          for key in components[0]}))
    for name, rows in (("training.csv", training), ("evaluation.csv", evaluation)):
        if rows:
            with (out / name).open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for axis, key in zip(axes.flat, ("total", *COMPONENTS)):
        x = [row["update"] for row in training]
        y = np.asarray([row[key] for row in training])
        n = min(50, len(y))
        axis.plot(x, y, alpha=.2, label="training batch")
        axis.plot(x[n - 1:], np.convolve(y, np.ones(n) / n, mode="valid"), label=f"training mean {n}")
        for mode in ("deterministic", "stochastic"):
            rows = [row for row in evaluation if row["mode"] == mode]
            axis.plot([row["update"] for row in rows], [row[key] for row in rows],
                      linestyle="--", label=f"fixed-panel {mode}")
        axis.set_title(key); axis.grid(alpha=.2); axis.legend(fontsize=7)
        axis.set_xlabel("update"); axis.set_ylabel("reward / episode")
    fig.suptitle(f"Reward accounting through update {training[-1]['update']} (distance cost inferred by residual)")
    fig.tight_layout()
    temporary = out / "components.tmp.png"
    fig.savefig(temporary, dpi=150); plt.close(fig)
    temporary.replace(out / "components.png")
    n = min(50, len(training))
    first, last = training[:n], training[-n:]
    means = lambda rows: {key: float(np.mean([row[key] for row in rows]))
                         for key in ("total", *COMPONENTS, "distance_cycle_weighted_mean")}
    early, late = means(first), means(last)
    delta = {key: late[key] - early[key] for key in early}
    summary = dict(through_update=training[-1]["update"],
                   early_updates=[first[0]["update"], first[-1]["update"]],
                   late_updates=[last[0]["update"], last[-1]["update"]],
                   early=early, late=late, delta=delta,
                   method="exact aggregate accounting; distance cost is residual, not independently measured",
                   assumptions="frozen v2 gamma=1; no anomalies admitted by collector; original job seed rule",
                   causal_limit="duration and geometry both affect integrated distance cost; contributions are not causal attribution")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    table = "\n".join(f"| {key} | {early[key]:.4f} | {late[key]:.4f} | {delta[key]:+.4f} |"
                       for key in ("total", *COMPONENTS))
    (out / "README.md").write_text(f"""# 回报分解（截至更新 {training[-1]['update']}）

![子项曲线](components.png)

前{n}更新与最近{n}更新比较。正的变化表示该项使总回报上升。

| 分量 | 早期 | 近期 | 变化 |
|---|---:|---:|---:|
{table}

## 数据来源与限制

训练日志没有直接存储各分量。按冻结的seed/update规则重建每批实例初始距离，
从初末距离还原进展，从周期数还原时间成本，从成功比例还原成功/失败事件。
distance_cost_residual = total - 其他分量，是余额推算，**不是独立物理日志复算**。
采样器拒绝模拟异常，故有效episode仅成功或预算失败；本分析不改变训练。
积分距离成本同时受执行时长和几何距离影响，不能将其全部称为“更接近目标”。
训练批次的实例配比不同；固定面板分解见evaluation.csv中的逐评价点结果。
该图为离线快照，不是训练程序原有自动刷新的曲线；重新运行下方命令即可更新。

```bash
.venv/bin/python experiments/random_start_reposition_v1/reward_breakdown.py
```
""")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
