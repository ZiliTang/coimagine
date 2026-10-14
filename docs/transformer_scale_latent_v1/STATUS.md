# STATUS — 本轮完成度（截至提交时刻）

计时以系统时钟与日志累计耗时为准。

| Run | 状态 | 日志累计耗时 | 检查点 |
|---|---|---|---|
| `capacity_0p9M_seed0` (895,937) | **完成** 3200/3200 | 见 `logs/` | 有 |
| `capacity_2p0M_seed0` (2,009,569) | **完成** 3200/3200 | 见 `logs/` | 有 |
| `direct_multistep_h8_lr0.0003_seed0` (1,469,952) | **完成** 3200/3200 | 见 `logs/` | 有 |
| `latent_autoregressive_h8_lr0.0003_seed0` (1,461,600) | **完成** 3200/3200 | 见 `logs/` | 有 |
| `direct_multistep_..._seed1` | 中断后重启，进行中 | 一次外部中断终止了进程 | 无（检查点只在结束时写出） |
| `direct_multistep_..._seed2` | 中断后重启，进行中 | 同上 | 无 |
| `latent_autoregressive_..._seed1` | 中断后重启，进行中 | 同上 | 无 |
| `latent_autoregressive_..._seed2` | 中断后重启，进行中 | 同上 | 无 |
| `capacity_1p6M_seed0` (1,635,569) | 中断后重启，进行中 | 同上 | 无 |
| `capacity_4p0M_seed0` (4,149,569) | 中断时 600/3200，**未重启** | — | 无 |

**外部中断说明**：一次会话中断终止了全部训练进程，并吞掉了约 4.5 小时机时。
因此任务书要求的"1.6M 配对补全三个种子"与"约 4.0M 扩展"未完成。
已完成部分（seed 0 配对、两个容量点、全部基线、审计、五张图、可复算审查包）不受影响，
因为它们的检查点与逐样本预测在中断前已落盘。

**断点续跑**：

```bash
PY=.venv/bin/python
$PY experiments/transformer_scale_latent_v1/run_matrix.py --stage c --updates 3200 --jobs 4 --threads 3
$PY experiments/transformer_scale_latent_v1/run_matrix.py --stage a --updates 3200 --jobs 2 --threads 4
```

已存在的检查点会被跳过；补全后重跑 `evaluate.py`、`capacity_eval.py`、`tables.py`、`figures.py`
即可刷新全部结论。
