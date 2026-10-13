# STATUS — night_sequence_v1

生成时间：2026-10-14 04:39 (Asia/Shanghai)

## 起点与工作区

- 本轮实际起点 HEAD：`d7731ca`（observation_terminal_v1 评价修正版）；生成时 HEAD：`8b0593a1326079478128ee7a01739d131c8cd669`。
- 起点工作区差异：仅新增任务书 `docs/tasks/20261014_0205_...md`（未跟踪），未改动任何已冻结配置。
- 生成时未提交改动条数：6。

## 环境

- Python 3.10.12 / Linux x86_64
- 线程与核心：os.cpu_count()=24

## 采集

| stage | 序列 | 耗时 (s) | workers |
|---|---:|---:|---:|
| d0 | 4096 | 39 | 12 |
| eval | 6144 | 59 | 12 |
| d1 | 28672 | 265 | 12 |
| d2 | 98304 | 1588 | 8 |

## 数据规模

| 档位 | 独立 root | option 序列 | 真实帧 |
|---|---:|---:|---:|
| d0 | 512 | 4096 | 315257 |
| d1ext | 3584 | 28672 | 2201416 |
| d2ext | 12288 | 98304 | 7542453 |
| val | 256 | 2048 | 157162 |
| test | 512 | 4096 | 313359 |

## 训练矩阵

| run | 模型 | 档位 | seed | 参数量 | 更新 | 最优更新 | 最优验证损失 |
|---|---|---|---:|---:|---:|---:|---:|
| d0_mlp_history_seed0 | mlp_history | d0 | 0 | 1,608,513 | 3200 | 2400 | 0.1137 |
| d0_mlp_history_seed1 | mlp_history | d0 | 1 | 1,608,513 | 3200 | 2400 | 0.1119 |
| d0_mlp_history_seed2 | mlp_history | d0 | 2 | 1,608,513 | 3200 | 2400 | 0.1138 |
| d0_transformer_history_seed0 | transformer_history | d0 | 0 | 895,937 | 3200 | 2800 | 0.2028 |
| d0_transformer_history_seed1 | transformer_history | d0 | 1 | 895,937 | 3200 | 2800 | 0.2133 |
| d0_transformer_history_seed2 | transformer_history | d0 | 2 | 895,937 | 3200 | 2800 | 0.2128 |
| d1_mlp_current_seed0 | mlp_current | d1 | 0 | 686,913 | 3200 | 2800 | 0.2135 |
| d1_mlp_current_seed1 | mlp_current | d1 | 1 | 686,913 | 3200 | 2800 | 0.2122 |
| d1_mlp_current_seed2 | mlp_current | d1 | 2 | 686,913 | 3200 | 2800 | 0.2140 |
| d1_mlp_history_seed0 | mlp_history | d1 | 0 | 1,608,513 | 3200 | 2400 | 0.1117 |
| d1_mlp_history_seed1 | mlp_history | d1 | 1 | 1,608,513 | 3200 | 2400 | 0.1124 |
| d1_mlp_history_seed2 | mlp_history | d1 | 2 | 1,608,513 | 3200 | 2400 | 0.1136 |
| d1_transformer_history_seed0 | transformer_history | d1 | 0 | 895,937 | 3200 | 2800 | 0.2010 |
| d1_transformer_history_seed1 | transformer_history | d1 | 1 | 895,937 | 3200 | 2800 | 0.2076 |
| d1_transformer_history_seed2 | transformer_history | d1 | 2 | 895,937 | 3200 | 2800 | 0.2085 |
| d2_mlp_history_seed0 | mlp_history | d2 | 0 | 1,608,513 | 3200 | 2400 | 0.1138 |
| d2_transformer_history_seed0 | transformer_history | d2 | 0 | 895,937 | 3200 | 2800 | 0.2027 |

合计 17 次正式训练。全部 run 使用统一更新预算 3200（标定见 CONTRACT §4）。

## 未完成 / 未验证

- D2 的成对训练：视 D2 采集完成时间与预算决定；未完成则在此列出。
- 自由多步递推、闭环收益：本轮不实现、不评估。
