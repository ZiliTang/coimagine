# FIELDS — observation_terminal_v1

版本 `observation-terminal-v1`。每个字段：名称、维数、单位、坐标系、采样时刻、来源函数、
是否依赖物体真值、缺失规则、归一化、评价单位。

约定：**T** = option 实际退出的物理时刻；**R_i0** = 机器人 i 在 option 开始时（t=0）的自身位姿
`(p_i0, θ_i0)` 定义的固定参考系。**x** = 部署输入；**x_priv** = 特权槽（本轮恒零）；
**y/mask** = 标签。

## 1. 部署输入 x（79 维，沿用上一轮）

| # | 名称 | 维 | 单位 | 坐标系 | 时刻 | 来源 | 依赖物体真值 | 缺失规则 |
|---|---|---:|---|---|---|---|---|---|
| 1 | `command` | 10 | rad / m·s⁻¹ / s / N / 0-1 | 世界（推方向为 α 相对观测法向） | 技能参数 | `record.input.command`（`alpha_rad`、`speed`、`active_duration`、`force_limit`、`active_robots`） | 否 | 无 |
| 2 | `scene` | 25 | m / 计数 | 相对机器人 0 起始位姿 | 静态地图 | `record.input.scene`（arena、障碍、物体尺寸、机器人半径、扫描范围/射线数、控制周期） | 否（物体**尺寸**是公开场景描述） | 缺失障碍补零并置存在位 0 |
| 3-4 | `robot{i}` 起始观测 | 22 ×2 | 混合 | 各自 R_i0 | **t=0** | `record.input.robot{i}`：起始速度、片段距离/点数/置信度、法向 sin/cos、质心、线段两端点、接触标志与接触点、三个 mask | 否 | 片段无效时相应项为 0 且 `mask_patch_valid=0` |
| 5 | `teammate_relation` | 4 | m, 无量纲 | 机器人 0 起始系 | t=0 | 由双方起始世界位姿导出（公开可共享） | 否 | 无 |

起始输入严格要求 **t=0**；本轮数据无观测延迟，逐条核对 t=0 有观测。
**不含**物体真值、真实面编号、未来观测、退出类型、真实时长。

## 2. 特权槽 x_priv（7 维，本轮**恒为零**）

物体真值起始位姿（R_0 下 x、y、cos、sin）与起始速度（vx、vy、ω）。
保留槽位是为了与上一轮**输入维数 86、参数量 97,573** 一致；**不参与任何预测**。

## 3. 标签（T 时刻，来自精确重放的退出现测）

| 组 | 名称 | 维 | 单位 | 坐标系 | 时刻 | 来源函数 | 依赖物体真值 | 缺失/mask | 评价单位 |
|---|---|---:|---|---|---|---|---|---|---|
| 1 | `terminal_observation_geometry` | 8 | m, 无量纲 | **各自 R_i0** | **T** | `dataset_terminal.terminal_geometry()`：`observation_summary.patch_segment_robot` 与 `patch_normal_robot` 经 §4.1 变换 | **否**（只用自己的传感片段与自身位姿） | 该机器人 `patch_valid=0` → 四维 `mask=0`，**不填零参与损失** | 中点 x/y：m；法向：`atan2` 包裹角误差（度） |
| 2 | `robot_pose_delta` | 6 | m, rad | R_i0 | T | `robot_motion()`：`p_iT − p_i0` 与 `wrap(θ_iT − θ_i0)` | 否（自身位姿） | 恒有效 | m / rad（报告可转度） |
| 3 | `robot_residual_velocity` | 6 | m·s⁻¹, rad·s⁻¹ | R_i0 | T | `robot_motion()`：终态速度旋转到 R_i0 与 ω | 否 | 恒有效 | m/s、rad/s |
| 4 | `duration` | 1 | s | — | T | 重放的 `wall_clock_steps × 0.05` | 否 | 恒有效 | s |
| 5 | `observation_validity` | 4 | 0/1 | — | T | 该机器人 `patch_valid`、`contacting` | 否 | 恒有效（**接触与可见各自独立**） | 混淆/召回 |
| 6 | `exit_class` | 1 | 类别（12） | — | T | 重放的 `exit_reason`（沿用旧类别表） | 否 | 恒有效 | 混淆/召回 |

`terminal_observation_geometry` 的 4 列/机器人顺序：
`[mid_x, mid_y, normal_sin, normal_cos]`。

## 4. 归一化（按组、展平，**不是逐列**）

对每个连续组：取 train split 中 `mask>0` 的项**展平**后求**单个** mean/std，
shape 为 `(1,)` 并广播到该组各列；缺失与 NaN 不参与。
统计量随检查点交付，评价加载同一份，不重新拟合。网络输出**只反标准化一次**。

## 5. 基线的字段口径

- **P0**：组 1 复制起始片段（同一 R_i0）→ 起始 `patch_segment_own_frame` 中点与
  `patch_normal_own_frame`；组 2 全 0；组 3 起始速度旋转到 R_i0；组 5 复制起始标志；
  组 4 = train 中位数；组 6 = train 多数类。起始片段无效时组 1 置 `mask=0` 并记
  `geometry_fallback`，数值预测用 train 有效目标的**逐列中位数**，只在共同样本上计分。
- **P1**：按完整公开命令签名（含 active mask）分组，连续目标取组内 train 中位数；
  法向对单位向量取均值后归一化（退化时用固定回退规则）；离散用组内频率/多数类；
  未见命令回退整体 train 统计并计数。

## 6. 数据索引与哈希

| split | 记录数 | roots | 来源文件 |
|---|---:|---:|---|
| train | 4096 | 512 | `phase_b_tier2/executions_train.jsonl` |
| val | 512 | 64 | `phase_b_tier2/executions_val.jsonl` |
| test | 512 | 64 | `phase_b_tier2/executions_test.jsonl` |

T 时刻观测由 `outputs/observation_terminal_v1/terminal_observations_<split>.jsonl` 给出，
重放一致性 5120/5120；审计总账 `label_audit_ledger.json`；
各文件 sha256 由构包脚本写入 `MANIFEST.md`。
