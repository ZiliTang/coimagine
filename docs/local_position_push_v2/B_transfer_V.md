# B. 可观测终态转移（留一 root）

臂 `V`，split `test`，run `local-position-push-v1`。
目标：signed_distance, tangential_offset, normal_angle_deg（表达在机器人自身起始坐标系）。
验证：leave-one-root-out; all 8 variants of a root held out together。

| 特征集 | 目标 | 单位 | n | 中位绝对误差 | 平均绝对误差 | R²（辅助） |
|---|---|---|---:|---:|---:|---:|
| skill_only | signed_distance | m | 958 | 0.0113 | 0.0125 | 0.111 |
| skill_only | tangential_offset | m | 958 | 0.0137 | 0.0216 | -0.045 |
| skill_only | normal_angle_deg | deg | 958 | 3.8625 | 8.6313 | 0.020 |
| observable | signed_distance | m | 958 | 0.0072 | 0.0093 | 0.377 |
| observable | tangential_offset | m | 958 | 0.0122 | 0.0143 | 0.620 |
| observable | normal_angle_deg | deg | 958 | 4.6271 | 7.5839 | 0.415 |
| truth_reference | signed_distance | m | 958 | 0.0074 | 0.0098 | 0.320 |
| truth_reference | tangential_offset | m | 958 | 0.0169 | 0.0207 | 0.207 |
| truth_reference | normal_angle_deg | deg | 958 | 2.5511 | 4.5858 | 0.768 |
| observable_plus_truth | signed_distance | m | 958 | 0.0071 | 0.0091 | 0.394 |
| observable_plus_truth | tangential_offset | m | 958 | 0.0123 | 0.0146 | 0.617 |
| observable_plus_truth | normal_angle_deg | deg | 958 | 2.6572 | 4.7325 | 0.754 |

## 离散结局（全部执行都保留，不丢样本）

- 执行数 960；终态片段可观测 958，不可观测 2（保留为离散结果）。
- 终态仍在接触 125；观测面发生变化 15。
- 退出原因：{'settled_observed': 647, 'settle_timeout': 190, 'contact_lost': 78, 'settle_unobserved': 13, 'patch_lost': 32}。

## 可部署模型的残差 vs **隐藏**物体朝向

| 目标 | n | 四分层残差均值 | 层间极差 | Spearman ρ |
|---|---:|---|---:|---:|
| signed_distance | 958 | -0.0027, 0.0009, 0.0016, 0.0001 | 0.0043 | 0.073 |
| tangential_offset | 958 | -0.0000, -0.0020, -0.0028, 0.0051 | 0.0079 | 0.069 |
| normal_angle_deg | 958 | -1.9704, 1.5659, 0.2862, 0.0924 | 3.5363 | 0.075 |

隐藏角 = 物体真值朝向相对机器人起始朝向的夹角，本地扫描原理上不可观测；若残差随它系统变化，说明该终态目标仍依赖隐藏状态。
