# D003 执行回报：成功参考与失败窗口的接触建立/保持时序审计

- 状态：`complete`（8 个 case，16 次正式重放；witness 通过，6 个失败窗口全部复现且非干扰为 0）。
- 任务书：D003 v1（`tasks/D003_Contact_Timing_Audit.md`）；配置：`tasks/D003_config.json`。
- 开始/结束：2026-10-29 18:43–19:05（Asia/Shanghai）。
- 预算：正式物理重放 **16.27 s**（witness 5.75 s + 6 个失败窗口 10.52 s）；分析-only 修正 1.28 s；4 worker；总上限 300 s；聚焦测试 4 项 <120 s。未训练、未改控制器/奖励/观测/动作/掩码。
- 输入：D001 92/92、D002 A 12/12、v14 witness 两条参考、panel；25 个输入 SHA256 全部匹配。HEAD `22c1333...`，149 个未提交项；完整源码快照与哈希在输出目录。
- 实际命令与 supervisor：`run_d003.py ... --deadline 300 --grace 2`；`supervisor.json` 记录子进程组、10.52 s、未超时、status complete。外层组在超时测试中被 1 s/2 s 强杀验证过。
- 改动/新增：`experiments/option_rl_v17/contact_timing_audit.py`、`run_d003.py`、`postprocess_d003.py`、`tests/test_d003_contact_timing.py`；任务书顶部有执行侧补记（GPT 额度耗尽，用户临时授权 DeepSeek 补 config/执行；正文 8 样本、阈值、16 重放、300/120 s 未改）。

## 先回答

**正常成功过程是什么？** W_N/W_E 两条 v14 witness 参考在 129 tick/6.45 s 内成功：2.20 s 出现双方同时接触，2.25 s 出现双方 joint_active，随后连续 **64 tick（3.20 s）** 双方同时接触、59 tick 双方 joint_active；6.45 s 成功，最终距离与旧日志一致。baseline/instrumented 轨迹误差 0，旧日志误差 0。这就是控制器可执行的正常接近→接触→激活→推进序列；但它的制动原本含离线真值判断，只证明“控制器可解”，不是 RL 学会。

**失败最早在哪一步分叉？**

| 窗口 | 最早分叉 | D003 观察 |
|---|---|---|
| N_N_far | 从未双侧接触 | 4.8 s 内 0 tick 双方接触；持续 `ready_false`，并有 `local_failure`；gap/tangent/alignment 字段为 unknown；动作被持续保持 |
| N_N_direction | 从未双侧接触 | 4.8 s 内 0 tick 双方接触；持续 `ready_false`，`local_failure` + `peer_push_switching`；字段 unknown |
| N_E_one | 接触出现得太晚 | D002 3.2 s 内无双方接触；4.8 s 扩展后在 **6.45 s** 才出现 11 tick（0.55 s）双方接触，stable 10 tick，joint_active 14 tick；随后 1 tick 短接触丢失 |
| N_E_gap | 接触出现得太晚、且很短 | D002 3.2 s 内无双方接触；**17.60 s** 才出现 5 tick（0.25 s）双方接触，stable 3 tick，active 15 tick；三次 1/1/3 tick 短接触丢失 |
| L_E | 选择方向与目标相反 | 3.2 s 内双方接触仅 1 tick；但两侧 PUSH_NE 与目标方向的点积均为 **−0.686 ≤ −0.25**，持续全部 96 tick；接触丢失发生在 85.25 s |
| C_N | 原“末拍才接触”被延长窗口解决 | 原 D002 3.2 s 的最后一拍接触，在 4.8 s 中变成 **31 tick（1.55 s）** 双方接触、stable 30 tick、active 32 tick；D002 的右删失被解决，不再是“从未接触/末拍接触” |

**现在最该改策略选择还是执行过程？**

- **执行过程不是普遍缺陷**：witness 能连续 64 tick 保持双方接触，C_N 也能连续 30 tick 保持；没有证据支持“控制器整体不会保持接触”。
- **最早的共同分叉更像策略选择/准备条件**：N_N 两例整个 4.8 s 没接触且 `ready_false/local_failure`；N_E 两例接触要到 4.0–4.05 s（D002 3.2 s 之外）才出现；L_E 的推方向与目标相反 96 tick。动作在这些状态下被持续保持，但状态/方向并不适合立即同推。
- **接触保持是下游候选，不是首选**：失败窗口确实有 6 个 `contact_not_maintained` 短段（N_E_gap×3、N_E_one、L_E、C_N 各 1），但其中 C_N 只是 1 tick 瞬态后转为 30 tick 稳定，L_E 是 goal-opposed，N_E 的接触出现在 D002 窗口之后；不能据此把主因归为“接触保持逻辑坏了”。
- **证据足够选一个验证方向，但不足以指定代码级修复**：`approach_response_suspect` 与 `coordination_contract_violation_candidate` 都没有触发；gap/tangent/alignment 在 5 个窗口是 unknown，无法把准备条件拆到具体几何项；本审计只做相关时序，不做干预因果。

**本轮唯一最小后续验证候选：策略侧的“同推 readiness/selection gate”。** 具体说：当两侧准备/接触条件未满足，或所选推方向与目标方向相反时，不要继续提交/保持同推，先重选或制动；先用离线 gate 重放这 8 个固定窗口验证，不改控制器、不训练。若 gate 后仍有“已 ready 但接触 1–5 tick 丢失”，下一轮才把候选转为接触保持/恢复。这是候选，不是本轮已实施的修复。

## 8 个 case 的实测

| case | success | ticks | 双方接触 | stable | 双方 active | 关键事件/标签 |
|---|---|---:|---:|---:|---:|---|
| W_N | True | 129 | 64 (3.20 s) | 64 | 59 | 2.20 s 接触，2.25 s active，6.45 s 成功；old/baseline/instrumented 0 |
| W_E | True | 129 | 64 (3.20 s) | 64 | 59 | 同上；末尾 BRAKE_HOLD 不再被误标为 goal-opposed |
| N_N_far | False | 96 | 0 | 0 | 0 | 全窗 `ready_false` + `local_failure`；gap/tangent/alignment unknown |
| N_N_direction | False | 96 | 0 | 0 | 0 | 全窗 `ready_false` + `local_failure` + `peer_push_switching` |
| N_E_one | False | 96 | 11 (0.55 s) | 10 | 14 | 6.45 s 接触，6.50 s 丢失；D002 3.2 s 内无接触 |
| N_E_gap | False | 96 | 5 (0.25 s) | 3 | 15 | 17.60 s 接触，三次短接触丢失；D002 3.2 s 内无接触 |
| L_E | False | 96 | 1 (0.05 s) | 1 | 39 | 85.20 s 接触；PUSH_NE 与目标点积 −0.686 持续 96 tick |
| C_N | False | 96 | 31 (1.55 s) | 30 | 32 | 86.35 s 接触；D002 右删失被延长窗口解决 |

- 16 次正式重放：W_N/W_E 各 baseline+instrumented（4 条完整参考）；6 个失败窗口各 baseline+instrumented（12 条前缀+4.8 s 窗口）。
- 6 个失败窗口的 source 检查：raw_obs 误差 **0.0**、mask 误差 **0**、位姿误差 **6.2e-8–1.1e-7**、总 period 全部一致；前 64 tick 与 D002 hold 的位置/角度/力轨迹误差 **0.0**；baseline/instrumented 轨迹与奖励误差 **0.0**；全部 96 tick 采满，无异常/截断。
- 6 个失败窗口的 baseline/instrumented 都未成功；这是预期，因为审计的是原执行窗口，不是重选动作后的成功率。
- 新增 1.6 s 只用于观察；D002 的 3.2 s 门槛和负结果不变。N_E_one/N_E_gap 的接触出现在 D002 窗口之后，不能反过来重算 D002。

## 事件、blocker 与标签

- 事件/区间已改为使用**全局 control tick 与 t_start**，不是 list index。例如 N_E_gap 的双方接触 onset 是 tick 352/t=17.60 s，N_E_one 是 tick 129/t=6.45 s，C_N 是 tick 1727/t=86.35 s。
- `preparation_not_ready` 是失败窗口最稳定的标签：N_N 两例全 96 tick；N_E_one/N_E_gap 到 4.0–4.05 s；L_E 到 2.8 s；C_N 到 3.15 s。可确认的 blockers 是 `ready_false`、`local_failure`、`pending`、`pause:peer_push_switching`、`gap_large`；`gap_unknown/tangent_unknown/alignment_unknown` 明确标 unknown，没有补成猜测值。
- `goal_opposed_command` 只在 L_E 触发：两侧 PUSH_NE 与目标方向点积 −0.686；局部 mask 包含该动作，face normal 与动作方向匹配，因此不是“动作不可执行”，而是“执行方向与任务目标相反”。W_E 的末尾 BRAKE_HOLD 在修正后不再被误判为 push。
- `contact_not_maintained` 共 6 段：N_E_gap 3 段（1/1/3 tick）、N_E_one 1 段（1 tick）、L_E 1 段（1 tick）、C_N 1 段（1 tick）。这些段都没有新的命令/BRAKE/safety 变化；L_E 段同时 goal-opposed；C_N 段之后是 30 tick 稳定接触。
- `approach_response_suspect`：0；`coordination_contract_violation_candidate`：0。不能让这两个“没有触发”代替“已经排除”；gap/tangent/alignment 的 unknown 限制了接近控制假设的检验。
- `right_censored`（D003 窗口）：0；C_N 是 **D002 的右删失被 4.8 s 解决**，不是 D003 右删失。`left_censored`：0。
- 自动标签给出的最频繁候选是 `contact_maintenance_or_recovery`（6 段）；人工综合候选是**策略侧 readiness/selection gate**。两者差异和证据边界见“先回答”；回报不把自动计数当成因果结论。

## Witness 正对照与记录器

- W_N/W_E 两条参考也有 0–2.2 s 的初始准备段和 5.4–6.4 s 的末尾制动段，自动标成 `preparation_not_ready`；这是正对照的正常接近/停止段，不是失败，也不进入失败窗口的候选计数。
- W_N/W_E：旧 v14 witness 日志成功 6.45 s、129 行；baseline/instrumented 都是 129 tick、success；旧日志对比误差 **0.0**、baseline/instrumented 误差 **0.0**、奖励误差 **0.0**；双方接触 64 tick、joint_active 59 tick。
- 记录器 P0–P4 按真实调用顺序执行：P0 `Slot.prepare`、P1/P2 `_coordinate`、P3 `Slot.step`、P4 `env.step_control`，另记录 `clearance.step`；每个原方法恰好调用一次，baseline 不安装 wrapper；wrapper 在 finally 恢复。
- P0/P1/P2/P3/P4 每 tick 都带同一整数 control_tick、t_start/t_end；P4 后接触用 `backend.contacts()` 只读查询；没有重新调用 observe/_coordinate/skill.step/滤波器来补记录。
- `clearance` 接管时 P3 明确标 `skipped_by_clearance`，不是漏记录；本轮 8 个 case 的 P3 未被 clearance 接管，测试中已用合成 case 验证旁路路径。

## 测试、偏离与执行账

- 聚焦测试 `tests/test_d003_contact_timing.py` **4 项通过**：P0–P4 顺序/每原方法一次/clearance 旁路；激活不会被写成 never_activated；右/左删失与 pair/recovery 使旧 ready 失效；1 s deadline + 2 s grace 的进程组强杀（含孙进程）。
- 正式 16 次重放前有两次**无物理**的启动失败：supervisor 先写 `supervisor.json` 导致 audit 拒绝非空目录（1.0 s，0 重放）；failure job 读取 `a0` 字段而 config 用 `source_action`（3.4 s，0 重放）。修正后重新启动；两次失败均保留在命令/时间线说明中，未改样本、动作、阈值或预算。
- 正式重放后做了一次 **analysis-only** 修正：把 BRAKE_HOLD 从 push 判定中排除、把 events/blocker 时间改为全局 tick；raw_ticks、baseline 轨迹、witness 结果均未重跑或改写。`manifest.json` 标记 `analysis_only_retry=true`、`physical_replays=16`。
- 预算：正式 16.27 s + 分析-only 1.28 s < 300 s；聚焦测试 <120 s；无训练、无 optimizer、无 checkpoint、无 960/ledger/switch-cost，未解冻 D002 B，未并入 independent 停车，未发布 D004。

## 对任务假设的回答

- **“成功参考能正常建立接触并保持”**：支持。W_N/W_E 64 tick 双方接触、59 tick active、6.45 s 成功，旧日志/双执行误差均为 0。
- **“失败窗口存在一个共同的接触建立断点”**：部分支持。N_N 两例确实从未双侧接触；N_E 两例接触出现得晚且短；但 C_N 在 4.8 s 内能稳定 30 tick，说明不是所有窗口都“不会接触”。
- **“接触建立后很快丢失是主因”**：证据不足/部分反对。6 段短损失存在，但 C_N 之后稳定 30 tick、witness 稳定 64 tick；L_E 的短损失同时是 goal-opposed；N_E 的接触本来就在 D002 窗口之后。没有证据把主因归为控制器接触保持逻辑损坏。
- **“责任在策略选择/准备条件”**：支持但未到代码级。N_N 无接触 + `ready_false/local_failure`，N_E 晚接触，L_E 方向相反；这些是最早分叉。gap/tangent/alignment 多处 unknown，无法指定具体几何门控项，因此只能提出验证候选，不能宣称已定位 bug。
- **“是否需要改控制器”**：本轮没有支持“先修控制器”的证据；witness/C_N 的稳定保持不支持全局控制器缺陷。下一步应先验证策略侧 readiness/selection gate；若 gate 后仍有已 ready 的短接触丢失，再查接触保持/恢复。

## 限制

- 8 个固定窗口、6 个失败状态，不是训练集，不代表所有状态；4.8 s 只是短窗口，不是完整 90 s episode。
- witness 的制动含离线真值，是控制器可解性参考，不是 RL 学会，也不是训练示范。
- 标签是预先写定的候选标记，不是干预因果；`gap_unknown/tangent_unknown/alignment_unknown` 意味着部分准备条件无法判定。
- C_N 的 D002 右删失被 D003 延长窗口解决；N_E 的接触出现在 D002 3.2 s 之后；不能把这些新后缀用于重算 D002 门槛或启动 B。
- 本轮只读，不实施 gate/控制器修改；下一步由 GPT 复核、用户决定。

## 原始产物

- [summary.json](../../../outputs/agent_exchange/D003_20261029_1843/summary.json)
- [manifest.json](../../../outputs/agent_exchange/D003_20261029_1843/manifest.json)
- [case_metrics.csv](../../../outputs/agent_exchange/D003_20261029_1843/case_metrics.csv)
- [events.csv](../../../outputs/agent_exchange/D003_20261029_1843/events.csv)
- [blocker_intervals.csv](../../../outputs/agent_exchange/D003_20261029_1843/blocker_intervals.csv)
- [reproduction.json](../../../outputs/agent_exchange/D003_20261029_1843/reproduction.json)
- [raw_ticks.jsonl](../../../outputs/agent_exchange/D003_20261029_1843/raw_ticks.jsonl)（P0–P4；22 MB）
- [witness.json](../../../outputs/agent_exchange/D003_20261029_1843/witness.json)（含 4 条参考重放）
- [baseline/](../../../outputs/agent_exchange/D003_20261029_1843/baseline/)（8 个 case 的 baseline 20 Hz 轨迹）
- [timelines.html](../../../outputs/agent_exchange/D003_20261029_1843/timelines.html)
- [replay.html](../../../outputs/agent_exchange/D003_20261029_1843/replay.html)
- [source_hashes.json](../../../outputs/agent_exchange/D003_20261029_1843/source_hashes.json)
- [supervisor.json](../../../outputs/agent_exchange/D003_20261029_1843/supervisor.json)

## 下一步状态

D003 `complete`；`CURRENT.md` 更新为 `ready_for_review`，下一位 GPT。没有自动实施候选 gate，没有启动 D004，没有解冻 D002 B，没有并入 independent 停车。
