# B. 可观测终态转移（留一 root）

臂 `X`，split `test`，run `local-position-push-v1`。
目标：signed_distance, tangential_offset, normal_angle_deg（表达在机器人自身起始坐标系）。
验证：leave-one-root-out; all 8 variants of a root held out together。

| 特征集 | 目标 | 单位 | n | 中位绝对误差 | 平均绝对误差 | R²（辅助） |
|---|---|---|---:|---:|---:|---:|
| skill_only | signed_distance | m | 957 | 0.0093 | 0.0117 | 0.160 |
| skill_only | tangential_offset | m | 957 | 0.0140 | 0.0216 | -0.051 |
| skill_only | normal_angle_deg | deg | 957 | 3.0227 | 7.2517 | -0.013 |
| observable | signed_distance | m | 957 | 0.0072 | 0.0092 | 0.375 |
| observable | tangential_offset | m | 957 | 0.0108 | 0.0132 | 0.658 |
| observable | normal_angle_deg | deg | 957 | 3.9826 | 6.4326 | 0.421 |
| truth_reference | signed_distance | m | 957 | 0.0078 | 0.0100 | 0.314 |
| truth_reference | tangential_offset | m | 957 | 0.0170 | 0.0206 | 0.198 |
| truth_reference | normal_angle_deg | deg | 957 | 2.5147 | 4.4872 | 0.713 |
| observable_plus_truth | signed_distance | m | 957 | 0.0068 | 0.0091 | 0.391 |
| observable_plus_truth | tangential_offset | m | 957 | 0.0107 | 0.0134 | 0.651 |
| observable_plus_truth | normal_angle_deg | deg | 957 | 2.9746 | 4.8495 | 0.676 |

## 离散结局（全部执行都保留，不丢样本）

- 执行数 960；终态片段可观测 957，不可观测 3（保留为离散结果）。
- 终态仍在接触 124；观测面发生变化 8。
- 退出原因：{'settled_observed': 592, 'settle_timeout': 171, 'contact_lost': 146, 'settle_unobserved': 8, 'patch_lost': 31, 'patch_support_lost': 12}。

## 可部署模型的残差 vs **隐藏**物体朝向

| 目标 | n | 四分层残差均值 | 层间极差 | Spearman ρ |
|---|---:|---|---:|---:|
| signed_distance | 957 | -0.0023, 0.0003, 0.0017, 0.0001 | 0.0041 | 0.036 |
| tangential_offset | 957 | -0.0001, -0.0018, -0.0025, 0.0047 | 0.0072 | 0.080 |
| normal_angle_deg | 957 | -1.5672, 1.7517, 0.1657, -0.3736 | 3.3189 | 0.062 |

隐藏角 = 物体真值朝向相对机器人起始朝向的夹角，本地扫描原理上不可观测；若残差随它系统变化，说明该终态目标仍依赖隐藏状态。
