# D007 执行回报（GPT 复核后订正版）

- 状态：执行已完成；**复核后订正**。C 候选仍未通过门槛，不推广；本轮存在分析错误、预算超限和证据缺口，不能记作全部验收通过。
- 本文件订正 [原 D007 回报](D007_Lateral_Transfer_Calibration.md) 的坐标分析、probe 统计、机器人 1 结论、预算和图表；原文件保留为历史。
- GPT 复核：[D007_Lateral_Transfer_Review.md](../reviews/D007_Lateral_Transfer_Review.md)；独立复算：`outputs/agent_exchange/D007_20261030_002/gpt_review/`（`recompute.py`、`corrected_probes.json/csv`）。
- 第二轮 36 条原始 run 是最终证据：`outputs/agent_exchange/D007_20261030_002/`；首轮 `_001` 保留但不采用。
- 没有新增物理运行、没有训练、没有默认替换、没有 D008。

## 先回答

**去掉外环/额外阻尼后，机器人更能跟随目标了吗？** 没有。B/C 在 adjacent/镜像上让角偏从 A 的 0.111 rad 增大到 0.405/0.460 rad，推进从 0.966 m 降到 0.929/0.899 m；C 的 G 增益改善只有 **+0.04665**（要求 ≥0.10），C/B RMS 比 **1.11251**（要求 ≤0.90）。**C 不获支持。**

**偏移有没有进入 target？** 有。全部 16 条 probe 的 `target_before_visibility → target_before_reservation → target_final` 差为 0；直接 offset 对目标的归一化响应约为 **1.000**。因此之前“robot1 偏移没进入 target / 下一步修偏移入口”的结论**撤回**。

**真正的问题是什么？** 跟踪参考点（reference_without_offset）随闭环轨迹/估计历史改变，抵消甚至反转 net target 差；可见性/reservation 不是 probe 的阻断层。下一步应检查端点/参考估计在物体坐标系中的变化和误差，而不是改偏移入口、放开 reservation 或继续调 gain。

## 1. 复核修正的坐标错误

旧 `analyze_d007.py::body_t_at` 读取 `effect.n`；实际字段是 `effect.face_normal`。缺失时旧代码静默返回备用方向，最终两台机器人都投影到世界 Y：

- robot0 面法向约世界 X，切向约 Y，错误被掩盖；
- robot1 面法向约世界 Y，切向约负 X，旧分析主要测了法向；
- 镜像的 `+t` 被固定，G 正负抵消，不能称“符合镜像”。

旧 `probes.csv`/`analysis.json`/link 图保留在 `probes_old_wrong_coordinates.csv`、`analysis_old_wrong_coordinates.json`、`chains_old_wrong_coordinates/`，不再作为结论。

## 2. 修正后的 probe 统计

GPT 独立复算方法：用 1.95 s 基线所见面的切线，转换到物体坐标系并固定；每个 tick 的机器人/target 减本轨迹物体位置、按本轨迹物体角度旋转；target 与机器人用 P4.before 对齐控制记录；support 按请求方向累计，contact 只统计 2–3 s；检查 anchor_switched、法向突跳与异常，不设缺字段备用轴。

| 指标 | 原分析 | 修正后 | 门槛 |
|---|---:|---:|---:|
| B/C 共同有效配对 | 4/8 | **6/8** | ≥6 |
| B 的 G_robot 中位数 | — | 0.78835 | — |
| C 的 G_robot 中位数 | — | 0.83500 | — |
| C−B | 不可用 | **+0.04665** | ≥+0.10 |
| B 的 RMS e_t 中位数 | .02395 | 0.024024 | — |
| C 的 RMS e_t 中位数 | .02545 | 0.026727 | — |
| C/B RMS 比 | 1.06 | **1.11251** | ≤0.90 |

- 有效数量门槛通过；改善门槛和 baseline 非退化门槛仍失败。
- robot1 的 B 组两种方向和 C 的负方向有有效支持；只有 C 的另一方向及其镜像各 5 tick，未达 .5 s。**不能再说 robot1 support 一律为 0。**
- 旧 `G≈1.36–2.69` 是世界坐标差，不是物体坐标系相对增益，不能沿用。

## 3. 链路分解

在物体坐标系中把未投影 target 拆为：

```text
target = reference_without_offset + t*(point_offset + role_offset + q_rot)
```

- 全部 16 条 probe 的可见性/reservation 裁剪均为 **0**；
- 直接 offset 对目标的归一化响应约 **1.000**，偏移正常写入；
- 参考项 `reference_without_offset` 随轨迹变化，后段抵消/反转净目标差。例如 adjacent/B/robot1/−4 cm 在 2.8–3.0 s：直接 offset ≈ +1.000，跟踪参考项 ≈ −1.547，净 target ≈ −0.547；C 对应 +1.000−1.163≈−0.163。
- 这是两条闭环轨迹之间的分解，不是同一瞬间状态的开环导数；不能把所有参考变化直接定为 endpoint 算法错误，也不能从命令力推断实际接触力矩。

## 4. Baseline 与 C 门槛

| adjacent | A | B | C |
|---|---:|---:|---:|
| 推进 m | 0.966489 | 0.928824 | 0.899343 |
| 最大角偏 rad | 0.110584 | 0.404669 | 0.460170 |

镜像结论一致。B 相对 A 的角偏明显增大；C 相对 B 再损失约 .02948 m 推进。C 门槛：

- 共同有效 6/8：通过；
- G 改善 +.04665：**失败**；
- RMS 比 1.11251：**失败**；
- adjacent/镜像推进损失 ~6.7 cm、角偏增加 ~.35 rad：**失败**。

W_N/W_E A/B/C 均为 129 tick success；没有新增异常。C 的删除阻尼候选仍不推广。

## 5. 预算与执行记录订正

- 首轮 `_001` 36 条（manifest 20.9876 s）+ 第二轮 `_002` 36 条（manifest 21.6265 s）；实际至少 **72 条运行尝试**，正式物理阶段合计约 **42.6141 s**。仅第二轮 36 条作为最终证据；首轮仍计入预算，不能用“不作为证据”从预算中消除。
- 任务书正式上限是 36 条；本轮数量超限，未见单独追加授权记录。测试中的物理实例另计，未并入 36/72。
- 缺少改动前基线源码快照；第二轮 12 个候选快照哈希匹配，但 manifest 的 `source_hashes` 是冻结基线，`candidate_hashes` 是候选，不能混读。
- 全 0.05 s tick 非干扰验证仍缺：record=False 只在 0.2 s decision 边界采样；这些点误差为 0，但不能补足全 tick 证明。
- 执行命令/进程组硬截止证据不完整；不能确认所有保障已实现。
- 67 项测试通过是 DS 记录；GPT 未重跑。纯数学测试没有覆盖真实记录 schema 或真实 profile 工厂，因此没有发现本次坐标错误。

## 6. 缩略的 probe 结果

| id | case | prof | side | off | G_robot | support/contact | valid |
|---|---|---|---:|---:|---:|---:|---|
| 013 | adjacent | B | 0 | −.04 | 1.128 | 18/20 | yes |
| 014 | adjacent | B | 0 | +.04 | 0.788 | 18/20 | yes |
| 015 | adjacent | B | 1 | −.04 | 0.477 | 10/20 | yes |
| 016 | adjacent | B | 1 | +.04 | 0.312 | 10/20 | yes |
| 017 | adjacent | C | 0 | −.04 | 1.241 | 18/20 | yes |
| 018 | adjacent | C | 0 | +.04 | 0.364 | 18/20 | yes |
| 019 | adjacent | C | 1 | −.04 | 0.835 | 13/20 | yes |
| 020 | adjacent | C | 1 | +.04 | 0.234 | 5/20 | **no** |
| 021–028 | mirror | B/C | 0/1 | ±.04 | 镜像对应值 | 10–18/20 | 1 个 no |

完整 16 行见 `gpt_review/corrected_probes.csv` / `analysis_corrected.json`。

## 7. 偏差与未完成

- 坐标分析错误已订正；旧分析/旧图表保留但不再引用。
- 预算超限（至少 72 条尝试）；任务书 36 条上限未满足。
- 基线源码快照缺口、全 tick 非干扰缺口、命令/硬截止证据缺口。
- B/C 的有效样本数量现在通过，但改善门槛失败；C 仍是 not-supported。
- 没有训练、没有 D008、没有默认替换；D002 B 冻结；完整 D005 8-case 当前版本验收仍未补齐。

## 8. 交回 GPT / 下一步

建议先做 **D007 离线订正收尾**（已完成本轮）：

- 用 `gpt_review/recompute.py` 的坐标/窗口/有效规则替代旧分析；
- 保留旧 `analysis_old_wrong_coordinates.json`、`probes_old_wrong_coordinates.csv`、`chains_old_wrong_coordinates/`；
- 新图表在 `chains/`（物体坐标系 target/robot tangent、e_t/lateral、object angle）；
- 不再按旧结论改 robot1 偏移入口、放开 reservation 或调 gain。

若之后另开控制器实验，优先问题是：**跟踪参考点为什么在物体坐标系中明显移动，以及 endpoint measurement/anchor 的误差/可观测性如何影响目标**。先从现有 endpoint/anchor 记录与真值的离线误差分解开始，再决定是否修改估计器；新实验需独立任务书，本次不发布 D008。

## 9. 产物

- `outputs/agent_exchange/D007_20261030_002/`：
  - `runs/`（36 条最终证据）
  - `gpt_review/corrected_probes.json/csv`、`recompute.py`
  - `analysis_corrected.json`、`probes_corrected.csv`
  - `analysis_old_wrong_coordinates.json`、`probes_old_wrong_coordinates.csv`、`metrics_old_wrong_coordinates.csv`
  - `chains/`（6 张修正图）、`chains_old_wrong_coordinates/`（旧图）
  - `matrix.csv`、`summary.json`、`manifest.json`、`source_snapshot/`
- 旧 `D007_20261030_001/` 保留为首轮尝试，计入总预算，不作为最终证据。
