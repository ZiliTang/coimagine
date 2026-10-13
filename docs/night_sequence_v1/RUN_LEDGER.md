# RUN_LEDGER — night_sequence_v1

每行一次正式训练；失败记录同样保留。

| run | model | tier | seed | params | updates | best_update | best_val | checkpoint |
|---|---|---|---:|---:|---:|---:|---:|---|
| d0_mlp_history_seed0 | mlp_history | d0 | 0 | 1,608,513 | 3200 | 2400 | 0.1137 | `outputs/night_sequence_v1/runs/d0_mlp_history_seed0.pt` |
| d0_mlp_history_seed1 | mlp_history | d0 | 1 | 1,608,513 | 3200 | 2400 | 0.1119 | `outputs/night_sequence_v1/runs/d0_mlp_history_seed1.pt` |
| d0_mlp_history_seed2 | mlp_history | d0 | 2 | 1,608,513 | 3200 | 2400 | 0.1138 | `outputs/night_sequence_v1/runs/d0_mlp_history_seed2.pt` |
| d0_transformer_history_seed0 | transformer_history | d0 | 0 | 895,937 | 3200 | 2800 | 0.2028 | `outputs/night_sequence_v1/runs/d0_transformer_history_seed0.pt` |
| d0_transformer_history_seed1 | transformer_history | d0 | 1 | 895,937 | 3200 | 2800 | 0.2133 | `outputs/night_sequence_v1/runs/d0_transformer_history_seed1.pt` |
| d0_transformer_history_seed2 | transformer_history | d0 | 2 | 895,937 | 3200 | 2800 | 0.2128 | `outputs/night_sequence_v1/runs/d0_transformer_history_seed2.pt` |
| d1_mlp_current_seed0 | mlp_current | d1 | 0 | 686,913 | 3200 | 2800 | 0.2135 | `outputs/night_sequence_v1/runs/d1_mlp_current_seed0.pt` |
| d1_mlp_current_seed1 | mlp_current | d1 | 1 | 686,913 | 3200 | 2800 | 0.2122 | `outputs/night_sequence_v1/runs/d1_mlp_current_seed1.pt` |
| d1_mlp_current_seed2 | mlp_current | d1 | 2 | 686,913 | 3200 | 2800 | 0.2140 | `outputs/night_sequence_v1/runs/d1_mlp_current_seed2.pt` |
| d1_mlp_history_seed0 | mlp_history | d1 | 0 | 1,608,513 | 3200 | 2400 | 0.1117 | `outputs/night_sequence_v1/runs/d1_mlp_history_seed0.pt` |
| d1_mlp_history_seed1 | mlp_history | d1 | 1 | 1,608,513 | 3200 | 2400 | 0.1124 | `outputs/night_sequence_v1/runs/d1_mlp_history_seed1.pt` |
| d1_mlp_history_seed2 | mlp_history | d1 | 2 | 1,608,513 | 3200 | 2400 | 0.1136 | `outputs/night_sequence_v1/runs/d1_mlp_history_seed2.pt` |
| d1_transformer_history_seed0 | transformer_history | d1 | 0 | 895,937 | 3200 | 2800 | 0.2010 | `outputs/night_sequence_v1/runs/d1_transformer_history_seed0.pt` |
| d1_transformer_history_seed1 | transformer_history | d1 | 1 | 895,937 | 3200 | 2800 | 0.2076 | `outputs/night_sequence_v1/runs/d1_transformer_history_seed1.pt` |
| d1_transformer_history_seed2 | transformer_history | d1 | 2 | 895,937 | 3200 | 2800 | 0.2085 | `outputs/night_sequence_v1/runs/d1_transformer_history_seed2.pt` |
| d2_mlp_history_seed0 | mlp_history | d2 | 0 | 1,608,513 | 3200 | 2400 | 0.1138 | `outputs/night_sequence_v1/runs/d2_mlp_history_seed0.pt` |
| d2_transformer_history_seed0 | transformer_history | d2 | 0 | 895,937 | 3200 | 2800 | 0.2006 | `outputs/night_sequence_v1/runs/d2_transformer_history_seed0.pt` |

## 失败记录

无。

## 已知实现问题与修复（保留复现，不隐藏）

1. Transformer 首版把因果掩码与 key-padding 掩码分开传入，全掩码行 softmax 得到 NaN，残差流又把 NaN 经 `0*NaN` 泄漏到真实行 → 验证损失 NaN。改为单个加法掩码（用 -1e9 而非 -inf），并把 padded 行清零；现已有测试固定“改变 padding 内容不改变输出”。
2. 退出类别头宽度曾固定为 12，而完整 Exit 词表有 15 类，D1 上出现 `Target 13 out of bounds`。改为 `len(EXIT_CODES)`，并重新采集受影响的 6 条 d1ext 序列。
3. `GroupedStatistics` 首版对每条 train 序列枚举全部时刻（>1.5M 次 build_example），无法在预算内完成；改为对 (序列, 时刻) 对做有界均匀采样并在 token 数组上向量化。
