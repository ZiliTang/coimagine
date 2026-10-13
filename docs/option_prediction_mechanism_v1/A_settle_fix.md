# A. 判稳测量与记账的小范围修复

数据：V 臂，dev（24 root × 8 变体 = 192）选型；test（960）在选定候选后重跑并与**既有回归集**
（`outputs/local_position_push_v1/` 的 V test 日志）按 `(root_id, variant)` 配对。
脚本：`experiments/option_prediction_mechanism_v1/phase_a/{run_a.py,analyse_a3.py}`；
表：`a3_dev.json`、`a3_test_compare.json`。控制律、门槛（0.02 m/s）、保持窗（0.30 s）、
机器人低速门槛、力上限、阶段结构**全部未改**。

## 1. 修了什么

### 1.1 记账（A1）

`ExecutionResult` 现在分开记录 `primary_exit_reason`（首次中止原因）、`locally_settled`
（判据是否曾经成立）、`settled_observed`、`settled_after_abort`。旧实现里
`settled = exit_reason == "settled_observed"` 而中止路径又把 `exit_reason` 覆盖成中止原因，
所以 `settled_after_abort = settled and abort_exit` **结构性恒为 False**（死代码）。

修复后两条赋值路径都可达，并有真实轨迹测试：
`tests/test_option_prediction_mechanism_a.py::test_settled_after_abort_is_reachable_on_a_real_abort_then_settle_trajectory`
（`dev/dual_rotate/002 baseline`：PUSH 期 `contact_lost`，SETTLE 期间达到判据 →
`primary_exit_reason="contact_lost"`、`locally_settled=True`、`settled_after_abort=True`、
`settled_observed=False`）。**中止后判稳不计入正常完成。**

`metrics.py` 另加真值审计标签：`exit_truth_low_speed`、`window_truth_low_speed`,
以及把"估计器错"和"可观测性限制"分开的 `evidence_truth_surface_low` / `evidence_truth_surface_max`。

### 1.2 时间戳（A2）

1. **`now` 用当前控制周期时间，不再用帧自带时间。** 旧代码把 `observation.time` 同时当
   `sample_time` 和 `now`，于是 `now - sample_time ≡ 0`，`SURFACE_FRESHNESS` 这道检查**形同虚设**、
   延迟帧永远"新鲜"。现在 `now` 是控制周期时间，帧的 `sample_time` 是传感器采样时间，
   两者分开。
2. **同一帧只算一份证据。** `sample_time` 重复时不入缓冲；判据通过这些周期**冻结**保持窗
   （不累加、也不清空）。
3. **不可用必须标记。** 过旧（`> SURFACE_FRESHNESS = 0.20 s`）、真丢观测、匹配不足、
   **采样间隔 `> SURFACE_MAX_SAMPLE_GAP = 0.15 s`** 都返回 `reason` 并清零保持窗。
4. **窗口化估计量**（只比较合同里冻结的 **0.20 s 与 0.30 s** 两个长度）：
   先在窗口基线上对每个匹配点求**有符号**法向速度，再做空间聚合；主统计量是
   `median_i |v_i|`——取有符号中位数会让旋转引起的 ± 运动互相抵消，取"有符号量的中位数绝对值"
   则保留旋转。噪声靠**加长基线**压低（0.05 s → 0.20 s 使逐点噪声速度降到约 1/4），
   **不是**靠"先取绝对值再平均"。
5. 每个周期记录 `settle_evidence_span`（判据覆盖的采样时段）与 `settle_evidence_lag`
   （相对当前时刻的滞后），确认时同时报告。

## 2. 候选选型（dev，安全优先）

| | old（window=0，重录） | **w=0.20** | w=0.30 |
|---|---:|---:|---:|
| 正常完成 | 138/192 = 0.719 | **182/192 = 0.948** | 182/192 = 0.948 |
| 中止后判稳 | 7 | 8 | 8 |
| 局部判稳 | 145/192 = 0.755 | **190/192 = 0.990** | 190/192 = 0.990 |
| 超时 | 41 | **0** | 0 |
| **估计器误判**（真值局部表面速度仍超门槛） | 0/145 | **0/190** | **3/190** |
| 整体未静止但局部正常（盲区） | 6 | 3 | 3 |
| 陈旧数据误判 | 0 | 0 | 0 |
| 确认延迟 p50 / p90 (s) | 0.900 / 1.100 | **0.950 / 1.250** | 1.000 / 1.250 |

顺序按合同：先看运动误判与陈旧数据误判，再看确认率与延迟。

- 陈旧数据误判两者都是 0（新鲜度检查现在真正生效）。
- **0.30 s 在 baseline / pos_2mm / vel_0.02 上各出现 1 例估计器误判（共 3），0.20 s 为 0。**
  这与"窗口越长滞后越大、运动重新开始时旧静止证据占比越高"一致——正是合同 A2 第 5 项
  要求的检查（`test_longer_window_lags_more_when_motion_restarts`）。
- 确认率两者相同（0.948 / 0.990），0.20 s 延迟略低。

**选定 `settle_window = 0.20 s`**（`min_baseline = 0.10`、`max_sample_gap = 0.15`、
`freshness = 0.20`），其余冻结量不变。

## 3. test 960 复核与配对

| | old（window=0 重录） | **冻结 w=0.20** |
|---|---:|---:|
| 正常完成 | 647/960 = 0.674 | **849/960 = 0.884** |
| 中止后判稳 | 85 | **109** |
| 局部判稳 | 732/960 = 0.762 | **958/960 = 0.998** |
| 超时 | 190 | **0** |
| 不可观测 | 13 | **1** |
| 中止类退出 | 25 | 1 |
| 估计器误判 | 18/732 = 0.025 | **16/958 = 0.017** |
| 盲区（整体未静止） | 14 | **9** |
| 陈旧数据误判 | 0 | 0 |
| 确认延迟 p50 / p90 (s) | 0.900 / 1.100 | 0.950 / 1.200 |

与**既有回归集**（旧 test 日志）逐条配对（960 条）：

```
normal->normal 647 | unsettled->normal 202 | abort->after_abort 109 | abort->abort 1 | unsettled->unsettled 1
```

即：旧协议下 202 条超时/不可观测的执行现在正常完成；109 条中止执行现在被正确记为
"中止后判稳"；只剩 1 条仍未判稳。**旧 test 已被多次查看，这是既有回归集，不是全新泛化验证。**

## 4. 哪些误判被消除、哪些限制仍在

**消除（或大幅减少）：**

- `noise` 与 `delay` 两个压力变体的判稳率从 **0/24 各自**回到 dev 上 23/24（noise 23 正常 + 1
  中止后，delay 21 正常 + 2 中止后）；这是估计量分辨力与时间戳处理的直接结果。
- 超时从 190 降到 **0**。
- 陈旧数据误判 0：延迟帧不再被当成"新鲜"证据。
- 估计器误判（真值局部表面速度仍超门槛）比例 0.025 → 0.017；绝对数 18 → 16。

**仍然存在的限制（必须随结果一起报告）：**

1. **局部传感盲区。** 9 例在 test 上"局部法向速度确实低、但物体整体未静止"——物体在
   大于 `SETTLE_ANGULAR` 的角速度下绕接触区附近转动时，被观察到的法向位移很小。
   本地法向扫描**原理上**看不见纯切向运动（有断言测试
   `test_pure_tangential_motion_is_a_declared_blind_spot`）。**本修复不保证物体全状态静止。**
2. **仍有 16/958 例估计器误判**：确认窗内至少一个周期的 patch 级真值表面速度超过 0.02 m/s，
   幅度 0.0202–0.0428 m/s（整体分布 p50 = 0.0136）。它们出现在减速进入静止的起始段，
   是估计量的噪声/对应误差，不是记账问题。
3. 证据滞后最大 0.15 s（100 ms 延迟变体）：判据证明的是**哪个采样时段**的静止，
   与该时段相对当前时刻的滞后都已随记录保存。

## 5. 测试

`tests/test_option_prediction_mechanism_a.py`（10 项，全过）：自身运动抵消、已知法向匀速
（0.015 与 0.030 两档）、旋转不被空间平均抵消、2 mm 噪声、重复帧只算一份证据、过旧与
大间隔标记不可用、缺失观测不假装证据、静止→运动的窗口滞后、纯切向盲区、以及
`settled_after_abort` 可达性。
