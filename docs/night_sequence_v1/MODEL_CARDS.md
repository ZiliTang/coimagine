# MODEL_CARDS — night_sequence_v1

每张卡片的数字来自实际 checkpoint 与分析 JSON。

## 共同部署输入白名单（所有新模型一致）

- 两机器人自身位姿与速度（各自 R_i0 下）
- 自身可见片段的：valid、中点、法向、距离、点数、置信度
- 自身接触标志与接触点
- 允许共享的队友同类量（相对自身 R_i0）
- 当前联合命令（alpha、speed、active mask）
- 已过去时间、阶段码、O0 锚点标志

## 明确禁止的输入通路

- 物体真值位姿/速度
- 真实面编号
- root id / 采样 seed / split
- 未来观测
- 未来控制力/力序列
- 真实终止时间
- 真实退出原因

说明：真值 `object_world` / `robots_world` 在数据里单独分区，token 构造器不读取；
`force` 列被保存用于审计，但**没有**进入任何 token。这一条由
`tests/test_night_sequence_v1.py::test_both_history_models_receive_identical_inputs_and_no_forbidden_channel`
与 F05 的输入白名单共同固定。

## d0_mlp_history_seed0

- 架构：`mlp_history`，参数 1,608,513；训练档位 d0，seed 0
- 训练：3200 次优化更新，最优验证损失 0.1137（第 2400 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d0_mlp_history_seed0.pt`
- test t=0 终态 r0 中点二维 p50：0.1121 m；执行中 p50：0.0444 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d0_mlp_history_seed1

- 架构：`mlp_history`，参数 1,608,513；训练档位 d0，seed 1
- 训练：3200 次优化更新，最优验证损失 0.1119（第 2400 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d0_mlp_history_seed1.pt`
- test t=0 终态 r0 中点二维 p50：0.1066 m；执行中 p50：0.0467 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d0_mlp_history_seed2

- 架构：`mlp_history`，参数 1,608,513；训练档位 d0，seed 2
- 训练：3200 次优化更新，最优验证损失 0.1138（第 2400 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d0_mlp_history_seed2.pt`
- test t=0 终态 r0 中点二维 p50：0.1104 m；执行中 p50：0.0470 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d0_transformer_history_seed0

- 架构：`transformer_history`，参数 895,937；训练档位 d0，seed 0
- 训练：3200 次优化更新，最优验证损失 0.2028（第 2800 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d0_transformer_history_seed0.pt`
- test t=0 终态 r0 中点二维 p50：0.1255 m；执行中 p50：0.0616 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d0_transformer_history_seed1

- 架构：`transformer_history`，参数 895,937；训练档位 d0，seed 1
- 训练：3200 次优化更新，最优验证损失 0.2133（第 2800 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d0_transformer_history_seed1.pt`
- test t=0 终态 r0 中点二维 p50：0.1251 m；执行中 p50：0.0592 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d0_transformer_history_seed2

- 架构：`transformer_history`，参数 895,937；训练档位 d0，seed 2
- 训练：3200 次优化更新，最优验证损失 0.2128（第 2800 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d0_transformer_history_seed2.pt`
- test t=0 终态 r0 中点二维 p50：0.1257 m；执行中 p50：0.0619 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d1_mlp_current_seed0

- 架构：`mlp_current`，参数 686,913；训练档位 d1，seed 0
- 训练：3200 次优化更新，最优验证损失 0.2135（第 2800 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d1_mlp_current_seed0.pt`
- test t=0 终态 r0 中点二维 p50：0.1258 m；执行中 p50：0.0543 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d1_mlp_current_seed1

- 架构：`mlp_current`，参数 686,913；训练档位 d1，seed 1
- 训练：3200 次优化更新，最优验证损失 0.2122（第 2800 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d1_mlp_current_seed1.pt`
- test t=0 终态 r0 中点二维 p50：0.1241 m；执行中 p50：0.0590 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d1_mlp_current_seed2

- 架构：`mlp_current`，参数 686,913；训练档位 d1，seed 2
- 训练：3200 次优化更新，最优验证损失 0.2140（第 2800 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d1_mlp_current_seed2.pt`
- test t=0 终态 r0 中点二维 p50：0.1244 m；执行中 p50：0.0597 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d1_mlp_history_seed0

- 架构：`mlp_history`，参数 1,608,513；训练档位 d1，seed 0
- 训练：3200 次优化更新，最优验证损失 0.1117（第 2400 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d1_mlp_history_seed0.pt`
- test t=0 终态 r0 中点二维 p50：0.1063 m；执行中 p50：0.0434 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d1_mlp_history_seed1

- 架构：`mlp_history`，参数 1,608,513；训练档位 d1，seed 1
- 训练：3200 次优化更新，最优验证损失 0.1124（第 2400 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d1_mlp_history_seed1.pt`
- test t=0 终态 r0 中点二维 p50：0.1086 m；执行中 p50：0.0437 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d1_mlp_history_seed2

- 架构：`mlp_history`，参数 1,608,513；训练档位 d1，seed 2
- 训练：3200 次优化更新，最优验证损失 0.1136（第 2400 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d1_mlp_history_seed2.pt`
- test t=0 终态 r0 中点二维 p50：0.1094 m；执行中 p50：0.0426 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d1_transformer_history_seed0

- 架构：`transformer_history`，参数 895,937；训练档位 d1，seed 0
- 训练：3200 次优化更新，最优验证损失 0.2010（第 2800 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d1_transformer_history_seed0.pt`
- test t=0 终态 r0 中点二维 p50：0.1242 m；执行中 p50：0.0616 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d1_transformer_history_seed1

- 架构：`transformer_history`，参数 895,937；训练档位 d1，seed 1
- 训练：3200 次优化更新，最优验证损失 0.2076（第 2800 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d1_transformer_history_seed1.pt`
- test t=0 终态 r0 中点二维 p50：0.1238 m；执行中 p50：0.0598 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d1_transformer_history_seed2

- 架构：`transformer_history`，参数 895,937；训练档位 d1，seed 2
- 训练：3200 次优化更新，最优验证损失 0.2085（第 2800 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d1_transformer_history_seed2.pt`
- test t=0 终态 r0 中点二维 p50：0.1240 m；执行中 p50：0.0611 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d2_mlp_history_seed0

- 架构：`mlp_history`，参数 1,608,513；训练档位 d2，seed 0
- 训练：3200 次优化更新，最优验证损失 0.1137（第 2400 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d2_mlp_history_seed0.pt`
- test t=0 终态 r0 中点二维 p50：0.1082 m；执行中 p50：0.0436 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积

## d2_transformer_history_seed0

- 架构：`transformer_history`，参数 895,937；训练档位 d2，seed 0
- 训练：3200 次优化更新，最优验证损失 0.2027（第 2800 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0
- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）
- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位
- 检查点：`outputs/night_sequence_v1/runs/d2_transformer_history_seed0.pt`
- test t=0 终态 r0 中点二维 p50：0.1244 m；执行中 p50：0.0608 m
- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout
- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；闭式前缀条件预测，未验证递推误差累积
