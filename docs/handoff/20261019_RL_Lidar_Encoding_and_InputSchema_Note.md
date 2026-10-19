# 备忘：RL 324 维输入里的雷达量程外编码与输入 schema 建议

日期：2026-10-19｜来源：option 测台 Round 000 回放讨论｜状态：**仅记录，未改动任何代码**

本文供规划端在完成多轮 option 调优后，整理为下一版实现方案。它不改变当前冻结的
324 维 RL 契约，也不属于 option 测台任务。

## 1. 现状：超出 1.20 m 无回波编码为 (0.0, 0.0)

`src/coimagine/random_start/lidar.py`：量程 `LIDAR_RANGE = 1.20 m`；1.20 m 内无命中时
返回 `distance = 0.0`、`valid = False`、`category = CATEGORY_NONE(0)`。

`src/coimagine/random_start/env.py::observation_vector` 把它写进 RL 输入：

```python
values += [((scan.distances[index] / LIDAR_RANGE) if scan.valid[index] else 0.0)
           for index in range(len(scan.valid))]
values += [scan.category[index] / 3.0 for index in range(len(scan.valid))]
```

因此每束雷达对应两个通道：

| 情况 | 原始 distance | valid | category | RL `lidar_XX` | RL `class_XX` |
|---|---:|---|---:|---:|---:|
| 命中物体 | d | True | 1 | d/1.2 ∈ (0,1] | 1/3 |
| 命中静态（墙/障碍） | d | True | 2 | d/1.2 | 2/3 |
| 命中队友 | d | True | 3 | d/1.2 | 1.0 |
| **超出 1.20 m 无回波** | 0.0 | False | 0 | **0.0** | **0.0** |

索引布局（每机器人先 72 束 lidar、再 72 束 class）：
`r0 lidar` 36–107，`r0 class` 108–179，`r1 lidar` 180–251，`r1 class` 252–323。
（前面 36 维：双方 own 12 + teammate 4 + patch 8 + goal 3 + contact 6 + time 3。）

真实记录核对：`outputs/random_start_rl_diagnosis_v4/replay_initial.pt` 的 324 维
`boundary_frame` 中，无回波射线恰好 `lidar == 0.0` 且 `class == 0.0`；有回波射线
`class ∈ {1/3, 2/3, 1.0}`。用当前 v3.1 环境重建 `face` 初始态结果一致（r0 46 条有回波、
26 条无回波）。

## 2. 为什么 1.0（量程上限）在连续性上更合理

雷达测的是**截断变量**：真实距离 `r ∈ [0.18, ∞)`（机器人半径 0.18 m），量程内 `d = r`，
超出表示 `r ≥ 1.20`。

- **单调性**：用 1.0 时通道随真实距离单调：命中 1.19 m → 0.99，无回波（真实 ≥1.20）
  → 1.00。用 0.0 时在量程边界反折：1.19 m → 0.99，1.21 m → 0.00，**2 cm 物理变化造成
  0.99 的输入跳变**。
- **不可达区间**：归一化后真实命中最小约 0.15（0.18/1.2）；0.0 哨兵让模型在
  `(0, 0.15)` 这段物理不可能的区间里收到信号，还要学“低端是特例”。
- **沿射线环不平滑**：相邻两条射线在量程边界会出现 0.99 vs 0.00 的突跳；改成 1.0 后是
  0.99 vs 1.00。若以后用 ray-ring 1D 卷积/attention 或任何沿角度平滑的结构，收益更大。
- **常规做法**：range sensor 通常把 max-range 编码为归一化上限，并配一个 hit/invalid
  标志；0.0 当哨兵较少见。

## 3. 但这不是正确性 bug

`class` 通道就是 hit/no-hit 标志：无回波 `class = 0`，真实命中 `class ≥ 1/3`。所以
`(lidar=0, class=0)` 唯一对应无回波，信息完整；0.0 只是让**距离通道**多了一个不必要的
不连续。它是 **inductive bias / 表示质量**问题，不是错误。

## 4. 更值得一起考虑的两点

1. **class 是任意序数**：`none=0, object=1, static=2, teammate=3` 再除以 3，隐含
   “object < static < teammate”的序关系，但该序无物理意义。更合理是 **one-hot（4 通道）**
   或小 embedding。
2. **一致性**：patch 通道也是“无 patch 给 0.0 + 单独 `patch_valid` 位”。若要统一，距离类
   特征都用 `min(d, R)/R`，有效性单独一位。

候选 schema：每束 `(min(d, R)/R, hit_flag)` + 每束 one-hot 类别
（object/static/teammate/none），一次解决连续性、无效标志与类别序数三个问题。

## 5. 代价与边界

改 `lidar` 编码会改变 **324 维输入 schema**，使现有契约、warm start 与所有 RL
checkpoint 失效，必须新开契约、重训并做配对比较。不能混入 option 测台调优轮次，也不能
用隐式全局 monkeypatch 切换。测试台 HTML 用原始 `distance/valid/category`，显示层不受影响。

## 6. 建议的下一步（交给规划端）

- 作为**独立的“RL 输入表示”预注册实验**：对比 `0.0 哨兵` 与 `1.0 截断 + 显式 valid +
  one-hot 类别`，固定网络容量、数据与在线预算，做配对比较。
- 可选的有界 probe（非长训练）：同一小预测/控制任务，两种编码各训一个同容量小网络，
  比较收敛速度与最终误差；单独输出，不碰冻结契约。
- 在决定前不要改动现有 324 维契约或训练/恢复校验。
