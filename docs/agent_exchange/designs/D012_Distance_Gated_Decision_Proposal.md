# D012 提案：距离触发的 option 决策点

- 状态：`proposal_for_planning`；未授权实现/运行，不改当前默认，不训练。
- 目的：让 RL 的一个决策对应大致相同的**有效位移**，降低机器人速度增益变化造成的策略状态分布漂移。
- 不是：不替代接触/速度/冲量稳定性，不改变物理、成功判据、奖励或动作 ID；不是完整 D012 任务书。

## 1. 问题

当前固定时间决策（例如 0.2/1s）下，速度增益一变，一个决策覆盖的位移就变：
- 快机器人一个决策推得远，慢机器人一个决策几乎没动；
- 同一离散动作在不同速度下对应不同物理效果，策略需要重新适应；
- 固定 1s 尤其差，慢机器人长期到不了 0.25m，fallback 会变成“卡住样本”。

## 2. 建议语义

一次 option 执行的目标不是“持续 T 秒”，而是：

> **沿期望效果方向累计 0.25m，或达到 Tmax 为止；Tmax 按支持的最慢速度计算。**

- `s_trigger = 0.25m`（可调，规划侧定）。
- `v_min = 支持的最小有效速度`；`Tmax = s_trigger / v_min`（例如 v_min=.10m/s → 2.5s），不是固定 1s。
- 另设最小决策间隔（如 ≥0.2s / ≥4 个控制 tick），避免快机器人过于频繁。
- 累计进度应用**有效进度**：
  - PUSH：物体/接触点沿请求方向的位移，或机器人沿动作方向的 signed 位移；
  - REPOSITION：沿边界的 signed 路径长度；
  - 不用纯路径长度，避免绕行/抖动触发。
- 到达 `s_trigger` → 正常决策点；
- 到 `Tmax` 仍未到 → 标记 `distance_timeout`（可能是阻塞/低速），作为单独样本，不混同正常到达；
- 接触丢失/安全/BRAKE/新意图仍可抢占，不改变原生命周期。

## 3. 需要同时改的观测/接口

若只改决策触发，策略仍看不到速度/时间：
- observation 增加 `since_decision_s`、`progress_since_decision_m`、`actual_speed`、`contact`、`distance_timeout`；
- option/action 的语义写成“推 0.25m / 最多 Tmax”，而不是“推 1s”；
- 记录每次决策的 tick、累计位移、实际时间、接触/力、是否超时；
- 评测同时报告：每米决策数、每决策位移分布、超时率、接触时序、转角/推进、成功率。

## 4. 实施边界（下一轮规划用）

- 只做**一个**距离触发接口实验，不和 EKF/FF/候选/heading 同时改。
- 保留固定时间决策作为对照；同一物理脚本/初态/成功判据。
- 至少比较速度档 0.12/0.20/0.30 m/s 的同一策略/option；固定时间 vs 距离触发。
- 不因“每决策位移相同”就宣称跨速度泛化；接触冲量、转角、稳定时间仍需单独评估。
- 环境当前是固定 `decision_ticks=4`，事件驱动决策需要改 `OptionSystemEnv`/runner/collector，不能只改 skill；先做离线/短轨迹验证，再跑正式矩阵。

## 5. 预期收益与风险

**可能收益**
- 同一离散动作跨速度对应更接近的空间效果；
- 快机器人更早产生决策/样本，慢机器人不会被固定时间强行切碎；
- 策略少依赖“速度×时间”的隐式尺度。

**风险**
- 变长/变短决策改变 credit assignment 和样本时间分布；
- 快机器人每决策时间短、接触冲量可能更大，距离触发不解决；
- 慢机器人大量 timeout，样本类别不平衡；
- 需要观测显式携带 `dt/progress`，否则策略仍无法区分。

## 6. 建议

值得进入下一轮规划，优先级在“option 效果速度/稳定性”之后、正式 RL 训练之前。先作为**决策接口实验**，不要直接替换默认 5Hz。


## 7. 补充：goal 只作为可选事件，不做基础 PUSH 的强制 terminal

- 基础 `PUSH` 保持 goal-free；不把 testbed/task 的 `obs.goal_xy` 自动注入 PUSH。
- 若上层额外提供 `object_goal`，只允许作为**可选复合 option / wrapper**（例如 `PushToGoal(goal)`），默认关闭；没有 goal 时完全走基础 PUSH。
- 不做 PUSH 内强制 terminal；RL 可以在进度决策点上选择 STOP/BRAKE。
- 若采用 goal 条件，需补：
  - goal-entry 事件决策点（进入 goal 半径时强制产生一次决策）；
  - goal distance/braking distance 观测；
  - 物体估计或公共 goal 的合法进度信号，不能读 env truth。
- 只靠“决策间隔 < 0.2m”不足以保证捕获：近切线弦长可能更短；且 STOP 有制动距离，可能进区域但停不住。建议 `s_trigger ≤ 0.1m` 或 goal-entry event。

## 8. 补充：option 预算从时间改为进度

把 option duration 从“持续 T 秒”改为“沿 option 有效进度的预算”：

| option | 进度定义 | 预算 |
|---|---|---|
| `PUSH` | 接触成立后，物体沿本次 push 法向的 signed 位移 | 0→contact，随后 .2m、.4m… |
| `REPOSITION` | 机器人沿 reposition 方向的累计 signed 位移（或路径） | 0→.2m、.4m… |
| `BRAKE` / `HOLD` | 无进度预算，固定短制动/保持 | 时间上限 |

关键边界：
- `PUSH` 的 0 点定义为**接触成立**。接触前是 acquisition 阶段，不由 RL 每步决策；acquisition 需要独立的 `contact_timeout` 和失败样本，不能把“没接触”伪装成 0.2m 进度。
- `PUSH` 的 .2/.4m 必须是**物体法向位移**，不是机器人自身位移。若当前观测没有合法 object position/progress 估计，不能直接用真值；需要先补估计器，或用接触点/q 的积分做明确近似并单独评估。
- `REPOSITION` 用机器人自身位移，不依赖物体估计。
- 到达预算 → 决策点；`Tmax` 仍按最慢有效速度计算；`distance_timeout` 单独标记。
- 接触丢失/安全/BRAKE/新意图可抢占；进度不倒退、不重复计同一段。
- 观测需包含：`since_decision_s`、`progress_since_decision_m`、`option_progress_m`、`actual_speed`、`contact`、`distance_timeout`、`goal_entry`。

## 9. 结论

- 方向值得做：RL 不再学习 PUSH approach 的逐 tick 行为，也不再按固定时间承受速度尺度漂移。
- 但必须先解决 PUSH 的合法 object progress 估计；没有它，进度预算只能退化为机器人位移代理，不能声称“物体法向 .2m”。
- 推荐顺序：先做 REPOSITION 的机器人进度预算（信号干净），再做 PUSH 的接触后物体进度；goal-entry 只作为可选事件，不改变基础 PUSH 语义。
