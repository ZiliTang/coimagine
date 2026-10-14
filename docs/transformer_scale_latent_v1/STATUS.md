# STATUS — 本轮完成度（截至提交时刻）

计时以系统时钟与日志累计耗时为准。

| Run | 状态 | 日志累计耗时 | 检查点 |
|---|---|---|---|
| `capacity_0p9M_seed0` (895,937) | **完成** 3200/3200 | 见 `logs/` | 有 |
| `capacity_2p0M_seed0` (2,009,569) | **完成** 3200/3200 | 见 `logs/` | 有 |
| `direct_multistep_h8_lr0.0003_seed0` (1,469,952) | **完成** 3200/3200 | 见 `logs/` | 有 |
| `latent_autoregressive_h8_lr0.0003_seed0` (1,461,600) | **完成** 3200/3200 | 见 `logs/` | 有 |
| `direct_multistep_..._seed1` | **完成** 3200/3200（中断后重启） | 见 `logs/` | 有 |
| `direct_multistep_..._seed2` | **完成** 3200/3200（中断后重启） | 见 `logs/` | 有 |
| `latent_autoregressive_..._seed1` | **完成** 3200/3200（中断后重启） | 见 `logs/` | 有 |
| `latent_autoregressive_..._seed2` | **完成** 3200/3200（中断后重启） | 见 `logs/` | 有 |
| `capacity_1p6M_seed0` (1,635,569) | **完成** 3200/3200（中断后重启） | 见 `logs/` | 有 |
| `capacity_4p0M_seed0` (4,149,569) | 中断时 600/3200，**未重启** | — | 无 |

**外部中断说明**：一次会话中断终止了全部训练进程，并吞掉了约 4.5 小时机时。
重启后已补齐 **1.6M 配对的全部三个种子**（直接多步与 latent 各 3 个）与
**正确的约 1.64M 容量点**。仍然缺少的只有**约 4.15M 容量点**与**自主退出评估**。

**断点续跑**：

```bash
PY=.venv/bin/python
$PY experiments/transformer_scale_latent_v1/run_matrix.py --stage c --updates 3200 --jobs 4 --threads 3
$PY experiments/transformer_scale_latent_v1/run_matrix.py --stage a --updates 3200 --jobs 2 --threads 4
```

已存在的检查点会被跳过；补全后重跑 `evaluate.py`、`capacity_eval.py`、`tables.py`、`figures.py`
即可刷新全部结论。
