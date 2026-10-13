# A1 分类审计

split `test`，run `local-position-push-v1`。

## 0. 估计量坐标系核对

- 坐标系：**world**。two scans of a static surface from two robot poses 0.01 m apart; world_points() maps returns with the robot pose before matching, so the ego-motion cancels and the world points are identical。
- 静止表面（机器人向表面靠近 0.01 m、物体不动）估计 = 0.0000 m/s（期望 0.0），匹配点 4。
- 对照：真实平移 0.004 m / 0.05 s → 估计 0.0800 m/s（期望 0.0800）。
- 判定：robot motion does not enter the estimate。

## 臂 V

- 执行 960；有效推动完成 `push_completed` 850/960；进入有效推动 `reached_effective_push` 850。
- 局部判稳（`settled_observed`）：647 → 判稳率 0.674。
- 退出原因：{'contact_lost': 78, 'patch_lost': 32, 'settle_timeout': 190, 'settle_unobserved': 13, 'settled_observed': 647}。

- 未判稳 313 例中：退出瞬间真值低速 313；完整 0.30 s 保持窗真值低速 313；退出时局部表面速度不可测 15。

## 臂 X

- 执行 960；有效推动完成 `push_completed` 772/960；进入有效推动 `reached_effective_push` 772。
- 局部判稳（`settled_observed`）：592 → 判稳率 0.617。
- 退出原因：{'contact_lost': 146, 'patch_lost': 31, 'patch_support_lost': 12, 'settle_timeout': 171, 'settle_unobserved': 8, 'settled_observed': 592}。

- 未判稳 368 例中：退出瞬间真值低速 368；完整 0.30 s 保持窗真值低速 368；退出时局部表面速度不可测 21。

## 退出瞬间真值 vs 完整窗口真值（交叉表）

| 臂 | (瞬间低速, 窗口低速) | 执行数 |
|---|---|---:|
| V | instant_low=True,window_low=False | 24 |
| V | instant_low=True,window_low=True | 936 |
| X | instant_low=True,window_low=False | 24 |
| X | instant_low=True,window_low=True | 936 |

## 中止后判稳

| 臂 | 中止执行 | 记录 `settled_after_abort` | 重算判据成立 |
|---|---:|---:|---:|
| V | 110 | 0 | 0 |
| X | 189 | 0 | 0 |

记录的 `settled_after_abort` 结构性恒为 False（`exit_reason` 被中止原因覆盖），所以上表第 3 列才是真实的中止后判稳数。

## 未判稳执行的退出真值

| 臂 | 指标 | n | p50 | p90 | p95 | max |
|---|---|---:|---:|---:|---:|---:|
| V | 退出瞬间物体线速度 m/s | 313 | 0.0000 | 0.0077 | 0.0093 | 0.0125 |
| V | 退出时真值表面法向速度（含旋转）m/s | 313 | 0.0000 | 0.0091 | 0.0093 | 0.0104 |
| X | 退出瞬间物体线速度 m/s | 368 | 0.0000 | 0.0087 | 0.0093 | 0.0137 |
| X | 退出时真值表面法向速度（含旋转）m/s | 368 | 0.0000 | 0.0093 | 0.0095 | 0.0105 |

## 估计量的虚警地板（真值确静止的连续两周期；含旋转的真值参考）

| 臂 | 指标 | n | p50 | p90 | max |
|---|---|---:|---:|---:|---:|
| V | 估计量 m/s | 12139 | 0.0294 | 0.0470 | 0.1076 |
| V | 真值表面法向速度 m/s | 12139 | 0.0000 | 0.0019 | 0.0045 |
| X | 估计量 m/s | 12129 | 0.0300 | 0.0473 | 0.0903 |
| X | 真值表面法向速度 m/s | 12129 | 0.0000 | 0.0019 | 0.0049 |

## 真值确实在动时，估计量是否看见

| 臂 | 指标 | n | p50 | p90 | max |
|---|---|---:|---:|---:|---:|
| V | 估计量 m/s | 5524 | 0.0478 | 0.0888 | 0.1609 |
| V | 真值表面法向速度 m/s | 5524 | 0.0363 | 0.0681 | 0.1076 |
| X | 估计量 m/s | 5117 | 0.0460 | 0.0873 | 0.1650 |
| X | 真值表面法向速度 m/s | 5117 | 0.0351 | 0.0676 | 0.1099 |

## 估计量虚警地板的**逐变体**分解（真值确静止的两周期）

| 臂 | 变体 | n | p50 | p90 | max |
|---|---|---:|---:|---:|---:|
| V | baseline | 89 | 0.0034 | 0.0058 | 0.0100 |
| V | delay | 4081 | 0.0001 | 0.0032 | 0.0364 |
| V | noise | 7528 | 0.0370 | 0.0502 | 0.1076 |
| V | param | 86 | 0.0035 | 0.0058 | 0.0120 |
| V | pos_10mm | 93 | 0.0033 | 0.0057 | 0.0137 |
| V | pos_2mm | 89 | 0.0033 | 0.0058 | 0.0101 |
| V | vel_0.02 | 81 | 0.0034 | 0.0060 | 0.0083 |
| V | vel_0.05 | 92 | 0.0033 | 0.0061 | 0.0093 |
| X | baseline | 88 | 0.0033 | 0.0057 | 0.0080 |
| X | delay | 4011 | 0.0001 | 0.0032 | 0.0363 |
| X | noise | 7598 | 0.0372 | 0.0505 | 0.0903 |
| X | param | 79 | 0.0035 | 0.0057 | 0.0114 |
| X | pos_10mm | 93 | 0.0033 | 0.0058 | 0.0156 |
| X | pos_2mm | 90 | 0.0032 | 0.0058 | 0.0155 |
| X | vel_0.02 | 83 | 0.0036 | 0.0056 | 0.0079 |
| X | vel_0.05 | 87 | 0.0034 | 0.0057 | 0.0200 |

## 逐变体退出原因

### V

| 变体 | 计数 |
|---|---|
| baseline | contact_lost=7, patch_lost=4, push_completed=109, reached_effective_push=109, settled_observed=109 |
| delay | contact_lost=22, patch_lost=4, push_completed=94, reached_effective_push=94, settle_timeout=82, settle_unobserved=12 |
| noise | contact_lost=8, patch_lost=4, push_completed=108, reached_effective_push=108, settle_timeout=108 |
| param | contact_lost=8, patch_lost=4, push_completed=108, reached_effective_push=108, settled_observed=108 |
| pos_10mm | contact_lost=9, patch_lost=4, push_completed=107, reached_effective_push=107, settled_observed=107 |
| pos_2mm | contact_lost=9, patch_lost=4, push_completed=107, reached_effective_push=107, settled_observed=107 |
| vel_0.02 | contact_lost=8, patch_lost=4, push_completed=108, reached_effective_push=108, settle_unobserved=1, settled_observed=107 |
| vel_0.05 | contact_lost=7, patch_lost=4, push_completed=109, reached_effective_push=109, settled_observed=109 |

### X

| 变体 | 计数 |
|---|---|
| baseline | contact_lost=14, patch_lost=4, patch_support_lost=1, push_completed=101, reached_effective_push=101, settled_observed=101 |
| delay | contact_lost=35, patch_lost=4, patch_support_lost=2, push_completed=79, reached_effective_push=79, settle_timeout=73, settle_unobserved=6 |
| noise | contact_lost=17, patch_lost=4, patch_support_lost=1, push_completed=98, reached_effective_push=98, settle_timeout=98 |
| param | contact_lost=15, patch_lost=3, patch_support_lost=4, push_completed=99, reached_effective_push=99, settle_unobserved=1, settled_observed=97 |
| pos_10mm | contact_lost=17, patch_lost=4, patch_support_lost=1, push_completed=98, reached_effective_push=98, settled_observed=98 |
| pos_2mm | contact_lost=17, patch_lost=4, patch_support_lost=1, push_completed=98, reached_effective_push=98, settled_observed=98 |
| vel_0.02 | contact_lost=15, patch_lost=4, patch_support_lost=1, push_completed=100, reached_effective_push=100, settle_unobserved=1, settled_observed=99 |
| vel_0.05 | contact_lost=16, patch_lost=4, patch_support_lost=1, push_completed=99, reached_effective_push=99, settled_observed=99 |
