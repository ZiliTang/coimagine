"""One preregistered, bounded rerun after the training-semantics repairs.

120 updates x 8 episodes, seed 0, original five-instance diagnostic subset. No automatic
extensions, S5 collection, reward changes or checkpoint selection on the test split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/random_start_rl_retrain_v1"
SUBSET = ROOT / "outputs/random_start_reposition_v1/results/diagnostic_subset.json"


def summarise():
    result_path = OUT / "results/rl_diagnostic.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    history = result["history"] if result is not None else [
        json.loads(line) for line in (OUT / "results/rl_updates.jsonl").read_text().splitlines()
        if line.strip()]
    fields = ["update", "rollout_return_mean", "rollout_success", "rollout_periods_mean",
              "rollout_final_distance_mean", "dev_success_deterministic", "dev_return_mean"]
    with (OUT / "results/curve.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in history)
    updates = [row["update"] for row in history]
    returns = np.asarray([row["rollout_return_mean"] for row in history])
    points = [row for row in history if "dev_success_deterministic" in row]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(updates, returns, alpha=0.3, label="training batch return")
    if len(returns) >= 10:
        axes[0].plot(updates[9:], np.convolve(returns, np.ones(10) / 10, mode="valid"),
                     label="training 10-update mean")
    axes[0].plot([row["update"] for row in points], [row["dev_return_mean"] for row in points],
                 label="fixed-panel deterministic return")
    axes[0].set_ylabel("discounted return")
    axes[1].plot(updates, [row["rollout_success"] for row in history], alpha=0.3,
                 label="training batch success")
    axes[1].plot([row["update"] for row in points],
                 [row["dev_success_deterministic"] for row in points], label="fixed-panel deterministic")
    axes[1].plot([row["update"] for row in points],
                 [row["dev_stochastic_evaluation"]["success"] for row in points],
                 label="fixed-panel stochastic, 3 seeds/instance")
    axes[1].axhline(0.75, linestyle="--", color="grey", label="gate threshold")
    axes[1].set_ylabel("success fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].plot(updates, [row["critic_before"]["explained_variance"] for row in history],
                 label="critic pre-update EV")
    axes[2].plot(updates, [row["critic_after"]["explained_variance"] for row in history],
                 label="critic post-update EV (same rollout batch)")
    axes[2].set_ylabel("explained variance")
    axes[2].set_xlabel("update")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "results/curves.png", dpi=150)
    plt.close(fig)
    blocks = []
    for index in range(0, len(history), 10):
        block = history[index:index + 10]
        blocks.append(f"| {block[0]['update']}–{block[-1]['update']} | "
                      f"{np.mean([r['rollout_return_mean'] for r in block]):.4f} | "
                      f"{np.mean([r['rollout_success'] for r in block]):.3f} |")
    if result is None:
        import torch
        checkpoint = torch.load(OUT / "checkpoints/rl_resume.pt", map_location="cpu",
                                weights_only=False)
        last_eval = points[-1]
        best = max(row["dev_success_deterministic"] for row in points)
        successes = sum(row["success_trajectories"] for row in history)
        manifest = json.loads((OUT / "manifest.json").read_text())
        (OUT / "README.md").write_text(f"""# 修复后有界重训 — 时间预算耗尽（incomplete）

预设 45 分钟硬上限已触发，未发生已报告的数值崩溃；不延长训练。
计划 120 次更新，完整记录到 **{history[-1]['update']} 次**，
共 **{len(history) * 8} 条已完成更新所用训练轨迹**，其中 {successes} 次成功。
被中断更新的部分采样不计入该数。最后可恢复检查点在 **第 {checkpoint['update']} 次更新**；
检查点之后的逐更新指标仍保留，但没有对应可恢复模型。

## 已有评价，而非完整最终评价

- 原冻结 5 实例，仅作开发诊断，不声称泛化。
- 最新完整评价：第 **{last_eval['update']}** 次，确定性成功率 **{last_eval['dev_success_deterministic']:.3f}**，
  随机执行（每实例 3 个固定行为种子）成功率 **{last_eval['dev_stochastic_evaluation']['success']:.3f}**。
- 历史最高确定性成功率 **{best:.3f}**；没有达到连续三点 ≥75% 的门槛。
- 没有完成计划末尾的最终验证评价；没有测试集结果，不授予 S4 通过。
- 训练状态见 `manifest.json`：`{manifest['status']}`。未自动续训或进入 S5。

![已完成更新的曲线](results/curves.png)

## 回报与采样成功率

| 更新区间 | 平均 episode 折扣回报 | 训练采样成功率 |
|---|---:|---:|
{chr(10).join(blocks)}

回报没有稳定上升，后期采样成功消失，不能把延长训练本身视为学习改善。
critic EV 是当前 rollout 同批拟合指标，不是验证集精度。
奖励/折扣、物理/技能/网络和 324 维输入均未改；动作/阶段历史缺失仍存在。
相比历史训练同时变更了三项修复、RNG 使用和执行预算，不是单因素因果消融。

## 文件

- `results/rl_updates.jsonl`：每次完整更新与逐实例评价。
- `results/curve.csv`、`results/curves.png`：简表及回报/成功率/critic 曲线。
- `checkpoints/rl_resume.pt`：真实恢复状态；`rl_update_{last_eval['update']:04d}.pt`：最近完整评价模型。
- `manifest.json`、`code/`、`configs/`：运行范围、源码快照、精确实例清单及哈希。
- `train.log`：控制台日志。当前没有完整的 `rl_diagnostic.json`，不能据此伪造最终结果。
""")
        return
    final = result["final_deterministic"]
    stochastic = result["final_stochastic"]
    text = f"""# 修复后有界重训结果

## 范围与结论

- 120 更新 × 8 episode，单种子 0；原冻结 5 实例诊断子集，不是泛化实验。
- 只带入 critic 时刻、循环梯度截断和局部 RNG 修复；奖励/折扣/物理/技能/网络/324 维输入不变。
- 初始化确定性成功率：{result['initial_deterministic']['success']:.3f}。
- 最终确定性成功率：{final['success']:.3f}；随机执行成功率：{stochastic['success']:.3f}。
- 最终确定性平均折扣回报：{final['return_mean']:.4f}；最终距离中位数：{final['median_final_distance']:.4f} m。
- 连续最近三点评价门槛通过：**{result['pass_signal']}**。
- 固定验证面板（8 实例）最终确定性成功率：{result['validation_deterministic']['success']:.3f}；未读取测试集。

![回报、成功率与 critic 曲线](results/curves.png)

## 训练采样分块统计

| 更新区间 | 平均 episode 折扣回报 | 训练采样成功率 |
|---|---:|---:|
{chr(10).join(blocks)}

## 限制

回报不等同于成功率；critic EV 是同批数据拟合指标。随机评价以实例为统计单位，
三行为种子不是三倍独立实例。仅 5 个有成功证据的实例，少于原任务要求的 8–16 个。
新旧实验预算和 RNG 使用不同，不是隔离每项修复的严格因果对照。
缺动作/阶段历史的输入问题仍未改变。未自动扩训、扩采或进入 S5。

配置/源码哈希见 `manifest.json` 和 `code/`；逐更新记录见 `results/rl_updates.jsonl`，
全量评价见 `results/rl_diagnostic.json`，简表见 `results/curve.csv`。
"""
    (OUT / "README.md").write_text(text)


def main():
    import rl
    if rl.ENV_VERSION != "random-start-reposition-v1":
        raise RuntimeError("Historical v1 retrain runner: use its recorded code snapshot, not the v2 reward environment")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "code").mkdir()
    (OUT / "configs").mkdir()
    subset = json.loads(SUBSET.read_text())
    historical_path = ROOT / "outputs/random_start_reposition_v1/results/rl_diagnostic.json"
    historical = json.loads(historical_path.read_text())
    original_selected = subset["selected"]
    subset["selected"] = historical["dev_subset"]["instances"]
    subset["retrain_selection_note"] = (
        "Freeze the five instances actually used by the old training, not its eight-entry "
        "source list which was silently filtered by --scan 32")
    subset["source_selected"] = original_selected
    frozen_subset = OUT / "configs/diagnostic_subset.json"
    frozen_subset.write_text(json.dumps(subset, indent=2))
    files = [Path(__file__), ROOT / "experiments/random_start_reposition_v1/rl.py",
             ROOT / "tests/test_random_start_rl.py"]
    files += list((ROOT / "src/coimagine/random_start").glob("*.py"))
    files += list((ROOT / "src/coimagine/continuous2d").glob("*.py"))
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        target = OUT / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    command = [sys.executable, str(ROOT / "experiments/random_start_reposition_v1/rl.py"),
               "--out", str(OUT), "--updates", "120", "--episodes-per-update", "8",
               "--seed", "0", "--threads", "1", "--eval-every", "5", "--eval-stochastic",
               "--save-every", "5", "--subset-file", str(OUT / "configs/diagnostic_subset.json")]
    manifest = {"command": command, "training_episode_cap": 960,
                "wall_time_cap_seconds": 2700, "source_sha256": hashes,
                "source_subset_sha256": hashlib.sha256(SUBSET.read_bytes()).hexdigest(),
                "frozen_subset_sha256": hashlib.sha256(frozen_subset.read_bytes()).hexdigest(),
                "historical_result_sha256": hashlib.sha256(historical_path.read_bytes()).hexdigest(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                "status": "running", "started_unix": time.time()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "README.md").write_text("# 修复后有界重训\n\n运行中。上限 120×8 episode、45 分钟。\n"
                                  "逐更新日志：`results/rl_updates.jsonl`；控制台日志：`train.log`。\n")
    try:
        with (OUT / "train.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=2700)
        summarise()
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"] = "incomplete"
        manifest["error"] = repr(error)
        with (OUT / "README.md").open("a") as stream:
            stream.write(f"\n运行未完整结束：`{error!r}`。保留日志与检查点；不授予通过门槛。\n")
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if isinstance(error, subprocess.TimeoutExpired) and (OUT / "results/rl_updates.jsonl").exists():
            summarise()
        raise
    finally:
        manifest["finished_unix"] = time.time()
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Completed: {OUT / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
