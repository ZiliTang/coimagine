# D009 v2 执行回报：无形状先验 EKF + 接触跟随（incomplete）

- 状态：**incomplete**。P0 修复后（同拍观测冻结、零时间差不更新、恢复力不再被 RECOVER 清零）40 条 task 开发评估 **30/40**；`reposition_0/2` 仍各 0/5。40 contract、184、估计/信息门槛未跑齐。22/40 是 P0 修复前的共享/Q 版，保留为历史。
- 新增物理运行只用于开发和 40 条 task 评估；**没有训练、没有替换默认控制器、没有发布 D010**。
- 任务书：`tasks/D009_v2_Autonomous_Implementation.md`；算法：`designs/D009_v2_Rigid_Motion_EKF.md`；配置：`tasks/D009_v2_config.json`。

## 一句结论

共享估计层、Q 量纲修正、去接触后追质心、全 FF 已接入；修掉同拍跨视角差分和 RECOVER 清零恢复力后，40 task 为 **30/40**：contact/approach/dual 各 5/5，reposition 仍 0/5。仍低于 32/40，**未达到最终验收，不能写成 engineering-accepted**。

## Post-P0 fixes（当前源码，30/40）

### P0-1 同拍跨视角差分

- 旧共享缓存被 A 本拍更新后，B 本拍与 A 本拍作差，库把时间差强行改为 `1e-6s`；这是空间差异，不是运动。
- 修复：每个 observer 保留自己的冻结线历史；本拍只与**同一 observer 的上一拍**比较；`dt<=0` 不生成任何 rate；同一 tick 重复 update 幂等；同一物理线的多视角 rate 按质量去重，选最高质量一份，不重复当独立证据。
- 注意：当前仍有 ±.16 m/s 的 FF 裁剪（不是无限幅）。

### P0-2 RECOVER 清零恢复力

- 旧 gate 在 `RECOVER/EXIT` 把双方力全部置零，包括 `PeerClearancePrefix` 的分离力。
- 修复：`RECOVER` 只阻止普通推进；若 clearance 正在执行或 joint recovery 活跃，则放行当前恢复力；`EXIT` 放行制动力；普通 PUSH 仍不能绕过安全门控。

### 4 条 dual 轨迹（P0 后）

| case | success | first contact | clearance | first gate-zero |
|---|---|---:|---|---:|
| T_dual_0_p0 | True | dec 11 | dec 2 started | dec 6 |
| T_dual_0_p4 | True | dec 7 | none | none |
| T_dual_2_p0 | True | dec 11 | dec 2 started | dec 6 |
| T_dual_2_p4 | True | dec 7 | none | none |

- 四条都成功；p0 进入过 clearance，但恢复力不再被总门控清零，随后仍完成推进；p4 没有 clearance。
- 这支持 P0-2 是真实条件缺陷，但还不能单独证明它是旧 8 条 dual 失败的唯一原因；本轮没有做“修复前/后单变量”以外的其它改动力度对照。
- 轨迹原始行：`outputs/agent_exchange/D009_v2_20261030_001/dual_trace_after_p0.json`。

### 40 task after P0

| family | success/total |
|---|---:|
| contact_0/2 | 5/5 each |
| approach_0/2 | 5/5 each |
| dual_0/2 | 5/5 each |
| reposition_0/2 | **0/5 each** |
| **total** | **30/40** |

- 仍低于 32/40，且 reposition 两根 0/5；不能写 engineering-accepted。
- `dev_task_eval_after_p0_fixes.json` 是当前结果；`dev_task_eval_before_p0_fixes.json` 保留 22/40；旧 30/40 为修正前历史。

## 1. 已实现

- `experiments/option_rl_v19/unknown_shape.py`
  - 固定世界参考点的刚体 twist EKF：`vx,vy,omega`；
  - 实际接触法向速度更新；接触只在真实 `contacting_object` 时启用；
  - 持续局部线的方向/法向位置更新；无尺寸、中心、矩形、端点、face_id 先验；
  - prior-free PUSH：观测 patch/line → 接近 → 真实接触 → 表面速度跟随；法向推进与有界侧向修正分开；
  - `UnknownShapeEnvMixin.install_prior_free()` 可替换 `OptionSystemEnv` 的 PUSH factory；REPOSITION/ROTATE 仍走旧 option（尚未完成全链路 prior removal）。
- 两个关键 bug 修复：
  1. 删除每 tick 先写 `reference.update((0,0))` 的双更新；
  2. 法向推进不再被切向 feedforward 的向量范数截断；切向单独限幅。

## 2. 开发证据

### 40 task（固定 acceptance 清单，非最终冻结）

| family | 修正前旧源码 | 当前共享/Q 修正版 |
|---|---:|---:|
| contact_0 | 5/5 | 5/5 |
| contact_2 | 5/5 | 5/5 |
| approach_0 | 5/5 | 5/5 |
| approach_2 | 5/5 | 5/5 |
| dual_0 | 5/5 | **1/5** |
| dual_2 | 5/5 | **1/5** |
| reposition_0 | 0/5 | **0/5** |
| reposition_2 | 0/5 | **0/5** |
| **总计** | **30/40** | **22/40** |

门槛 ≥32/40 且每根≥3/5；当前 22/40 失败。旧 30/40 不能绑定当前工作区：共享估计/去归中/全 FF 改变了 dual 与 contact 行为。

### 回归与失败机制

- 当前共享估计版 regression：dual_0/2 从 5/5 降到 1/5；两条机器人同时推时物体可能停在原地。共享线/接触融合或全 FF 的时序/符号还未校准。
- reposition 仍失败；旧/新在 REPOSITION 段一致，分叉发生在 PUSH_S 后；
- 新 PUSH 第一次接触在可见支持区间的偏心处（机器人 x≈1.91，物体 x≈2.50），接触扭矩使物体转到约 0.9–1.4 rad；
- 旧控制器用矩形/端点先验在面中心附近接触，因此成功；
- 尝试了可见支撑质心归中、切向 FF 关闭、偏心时先不施加全力法向推，均未修复；固定动作窗口内无法把可见支撑质心变成物体中心。
- 这不是 EKF 估计误差单独造成的；核心是 unknown-shape 的可观测性/探索问题。无形状先验时，可见支撑质心不是物体中心，不能靠加归中增益解决。

## 3. 未完成/未跑

- 40 条 contract 未评分（`C_*` 的 goal 固定反向，不能用 `env.success`；contract scorer 未实现）。
- 5 组 184 条最终比较（F_full 80、B_no_tangent 80、F_no_contact 8、F_no_geometry 8、F_no_adjust 8）未跑。
- 估计/信息/数值门槛：未生成 20Hz 真值/估计/置信度/约束来源日志，未跑先验哨兵/相同输入不变性/test_report。
- 未生成 6 张图、8 条代表轨迹、replay/index。
- 没有最终源码冻结和 source snapshot 完整性；当前源码仍在开发状态；30/40 与 22/40 的源码/参数未完整哈希绑定。
- ROTATE/FOLLOW/REPOSITION 的旧 option 仍保留旧 prior；全链路 prior removal 未完成。

## 4. 边界

- 没有修改物理、奖励、成功判据、动作频率、接触门限或安全投影。
- 没有训练、没有 checkpoint、没有 D010。
- 没有把旧矩形控制器作为维护默认；但也没有达到可替换的新默认。
- D002 B 仍冻结；D005/D008 历史缺口不借本轮补齐。

## 5. 产物

- `experiments/option_rl_v19/unknown_shape.py`
- `experiments/option_rl_v19/dev_d009_sample.py`、`eval_d009_tasks.py`
- `outputs/agent_exchange/D009_v2_20261030_001/preflight.json`、`progress.md`、`dev_task_eval.json`
- 本轮 task 评估：30/40；`reposition_0/2` 0/5。

## 6. 建议

先不要把这版接到正式 RL 或默认控制器。下一步应二选一：

1. 继续无先验路线：在接触前用可见支撑的**探索/重定位**获得更完整的局部线，或在偏心接触时拒绝全力 PUSH、先沿面滑动到更稳定接触点；需要新的接触策略，不是再调增益。
2. 走用户建议的回退：轮廓边界点云配准 + EKF，先验证能否在不使用尺寸/中心先验的情况下恢复稳定接触参考，再回到 40 task。

在达到 32/40 与 contract 门槛前，不写 engineering-accepted。

## Post-P0 20Hz 诊断（两 reposition 根）

冻结当前 30/40 版本后，从 REPOSITION→PUSH_S 交接前后逐控制 tick 记录（输出：`outputs/agent_exchange/D009_v2_20261030_002_diag/`）。

### T_reposition_0_p0

- 首次接触 tick 311（15.55 s）；动作是 PUSH_S `(0,-1)`。
- 首次接触选中面法向为 `(-1,0)`，与动作方向正交；这是“无方向匹配时回退最近 patch”的错误选面。
- 已只修这一项：没有方向匹配面时不再回退，改为 find/等待，不输出错误方向力。
- 修复后仍失败：物体终点 `(2.385,2.127, angle=-2.288)`，距离 .358；说明选面回退是真实 bug，但单独修复不足以完成该根。

### T_reposition_2_p0

- 首次接触 tick 311；选中面法向 `(0,-1)`，与 PUSH_S 一致。
- 接触点约 `(2.00,2.12)`；可见支持质心约 `(2.15,2.12)`；机器人 x≈1.90，接触明显偏离可见支撑中心。
- 接触前物体角 0；接触后角开始增大（tick 311→315，0→.043），是接触几何引起的转偏。
- 该段 `phase=IDLE`、`gate` 无清零；不是恢复门控或 FF 单独造成（FF=0 仍失败）。

### 当前结果

- no-fallback 选面修复后 40 task 仍为 **30/40**：contact/approach/dual 各 5/5；reposition_0/2 各 0/5。
- 结论：dual 已恢复；剩余瓶颈是 reposition 的 face-selection **加** contact-candidate/偏心接触，不是单一 EKF 数值问题。
- 下一步按复核只修 contact candidate：在可见支撑内选留足边距/稳定接触点，或在首次接触转偏时减速/撤力重定位；不应同时改 FF 或 EKF。
- 184/contract/估计门槛仍未跑；没有默认替换、训练、D010。
