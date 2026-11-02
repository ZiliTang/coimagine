# D004 执行回报：状态表 + 同推 pair gate（A）

- 状态：`complete`。8 条 gate 重放，4 项聚焦测试；正式 **9.29 s**，4 worker；不训练、不改 approach/ROTATE/奖励/观测/动作表。
- 任务书：`tasks/D004_State_Pair_Gate.md`；配置：`tasks/D004_config.json`；复用 D003 的 8 个固定 case 与 16 条 baseline。
- 输出：`outputs/agent_exchange/D004_20261029_2045/`；状态模型：`docs/agent_exchange/state/STATE_MODEL.md`、`state_table.csv`、`field_inventory.csv`。
- 特别说明：GPT 额度耗尽，任务书由 DeepSeek 在用户“先做 A”和状态系统要求下补写；D003 样本/阈值未改。

## 先回答

**状态表是否覆盖当前状态？** 是，但分两层：

- `state_table.csv`：**47 行语义状态**，每行有 scope/field/type/owner/reader/timing/persistence/default/meaning/invariant/gate/test/source。
- `field_inventory.csv`：对当前 `Slot`/`ContinuousPush`/`LearningEnv`/`PeerClearancePrefix`/`Continuous2DEnv` 的 **196 个 runtime 属性**逐项分类（state / config 或 geometry cache / counter 或 cache / option internal）。任何新字段会出现在 inventory；未分类会让测试失败。
- `state_schema.py` 是唯一清单，`write_tables()` 生成两张 CSV；测试同时检查 runtime 覆盖和语义 state 字段入表。以后新增 option 字段的流程是：先改表/schema → 再改代码；这比聊天记忆更可靠。
- 这仍是**映射**，不是把 v13/v14 的所有内部字段都提升成正式接口；`option_internal` 行没有全部进入语义表，只进入 inventory。下一步若继续重构，应按 option 逐个把 `option_internal` 升级成有 owner/invariant 的语义行。

**pair gate 是否消除 unilateral normal push？** 在 gate 的 READY/ENGAGE 阶段，是：gate 在 `env.step_control` 前把两侧**法向力取较小值**（切向保留），因此 READY/ENGAGE 不允许一侧 full normal；APPROACH 仍允许 approach 力；PUSH 仅双方 contact 时开放。D004 的 8 个 case `one_side_normal_count=0`。但要注意这是“同阶段内对称化”，不是“双方在同一物理 tick 接触”；ENGAGE 仍可能一侧先接触，只是另一侧的 normal 不会单方面超过它。

**witness 是否保持？** 是。W_N/W_E gate 仍 129 tick/6.45 s 成功，phase 为 IDLE 26 / APPROACH 21 / READY 22 / PUSH 60；旧日志误差 0、baseline/instrumented 非干扰 0。

**6 个失败窗口是否因为 gate 变成功？** 没有；6 个仍失败，与 D003 的 success 一致。gate 只修联锁/力层，不改变策略选择、approach 速度或旋转；因此不能把失败窗口的 success=0 当作 gate 失败。D002 前 64 tick 与 D003 baseline 有意不同（gate 改变了行为），source raw/mask/pose 仍为 0/0/≤1.1e-7，baseline/instrumented 非干扰 0。

## 8 条 gate 结果

| case | success | gate phase counts (IDLE/APPROACH/READY/ENGAGE/PUSH/RECOVER/EXIT) | one-side | source | noninterference |
|---|---|---:|---:|---|---:|
| W_N | True | 26/21/22/0/60/0/0 | 0 | — | 0 |
| W_E | True | 26/21/22/0/60/0/0 | 0 | — | 0 |
| N_E_gap | False | 6/51/30/7/2/0/0 | 0 | True | 0 |
| N_E_one | False | 9/31/41/8/7/0/0 | 0 | True | 0 |
| N_N_far | False | 1/33/38/20/4/0/0 | 0 | True | 0 |
| N_N_direction | False | 6/39/33/16/2/0/0 | 0 | True | 0 |
| L_E | False | 3/0/37/22/3/0/31 | 0 | True | 0 |
| C_N | False | 5/28/40/20/3/0/0 | 0 | True | 0 |

- 6 个失败窗口的 source raw/mask/pose 仍为 0/0/≤1.1e-7；gate 的 baseline/instrumented 非干扰 0。
- D002 前 64 tick 与 D003 baseline 的差异（例如 N_E_one 0.514、L_E 0.136）是 gate 的预期行为，不是复现失败；本轮不重算 D002 门槛。
- 6 个失败窗口仍未成功；gate 的 phase 显示大量 READY/ENGAGE（N_N_far 20 tick、N_E_one 8、L_E 22），说明“双方 ready 但接触建立不了/维持不住”仍是主要现象。gate 只保证没有 unilateral normal，不解决接触建立。

## 状态模型与力层

- 联合 phase：IDLE → APPROACH（双方 PUSH，至少一侧未 ready）→ READY（双方 ready）→ ENGAGE（至少一侧接触、非双方）→ PUSH（双方 contact）；RECOVER/EXIT 为故障/终止。
- live 清除：`release` 只在双方 ready 且无 fault 时为真；`joint_active` 只在双方 contact 且 phase=PUSH 时为真；任一侧 contact/ready 丢失、pending/pause/recovery/clearance、pair identity 改变，立即清零。修复了旧实现的单向 latch（D003 的 L_E active 39 vs both contact 1 这类）。
- stage fallback：release 不为真时，单侧 `stage=push` 回退到 `approach`，不再保留 unilateral push stage；不擦除该侧 `ready`（否则 witness 会被饿死——第一版 D004 就是这个错误）。
- force layer：READY/ENGAGE 把两侧法向力取较小值（切向保留）；APPROACH 保留 approach 力；PUSH 才允许 full normal；RECOVER/EXIT 零力。gate 的 `final_forces/force_gated` 记录在 gate rows。
- 注意：`one_side_normal_count` 是 **P0–P4 分析**得到的“单侧 full normal”计数；对 READY/ENGAGE，gate 的实际力已对称化，因此分析返回 0。APPROACH 的单侧 approach 力不被算作 unilateral normal。IDLE 的 solo push 不进入 pair 计数。
- `clearance.finished` 是完成 latch，不是 active fault；第一版把它当 fault，导致 witness 的 `release` 永远 false。已修为“started and not finished 或 failed 才算 active”。

## 测试与偏离

- 聚焦测试 `tests/test_d004_state_pair_gate.py` **4 项通过**：状态表/inventory 同步、phase 转移与 live 清除、stage fallback、pending/recovery 打破 pair。
- 正式前有 **8 条失败 witness**（4+4）：第一版 gate 擦除 ready，witness 132 tick 失败；第二版把 `clearance.finished` 当 fault，witness 仍 132 tick；修复后正式 8 条通过。失败段保留在说明中，未改样本、阈值或动作；成功 W_N/W_E 仍 6.45 s。
- 本轮没有改 approach/ROTATE/`A_REF_MAX`/奖励/观测；没有训练；没有 D005；没有解冻 D002 B；没有并入 independent 停车。
- 预算：正式 9.29 s < 300 s；聚焦 <120 s；state table 47 语义行、196 inventory 行。

## 交回 GPT / 下一步

- A 的 pair gate 已验证：witness 保持，unilateral normal 0，live latch 修复；但 6 个失败窗口仍 false，说明“gate 不是成功修复”，下一层仍是接触建立/保持。
- 状态系统已落盘；下一步应先做 **approach/ROTATE 的单独 A/B**（一次一个变量），还是先按状态表把 READY/ENGAGE 的接触建立再拆一层？我建议先看 READY/ENGAGE 的 20 Hz 时序：对比 W_N/W_E 的 READY→ENGAGE→PUSH（22→0→60）和 N_N_far（38→20→4）的差异，确认是 gap/接近控制还是接触保持；这仍只读，不改参数。
- 未发布 D005；D002 B 保持冻结；independent 停车不并入。

## 产物

- [STATE_MODEL.md](../state/STATE_MODEL.md) · [state_table.csv](../state/state_table.csv) · [field_inventory.csv](../state/field_inventory.csv)
- [summary.json](../../../outputs/agent_exchange/D004_20261029_2045/summary.json) · [manifest.json](../../../outputs/agent_exchange/D004_20261029_2045/manifest.json)
- [case_metrics.csv](../../../outputs/agent_exchange/D004_20261029_2045/case_metrics.csv) · [gate_ticks.jsonl](../../../outputs/agent_exchange/D004_20261029_2045/gate_ticks.jsonl)
- [state_table.csv（输出副本）](../../../outputs/agent_exchange/D004_20261029_2045/state_table.csv) · [field_inventory.csv（输出副本）](../../../outputs/agent_exchange/D004_20261029_2045/field_inventory.csv)
- 实现：`experiments/option_rl_v18/state_schema.py`、`pair_gate.py`、`run_d004.py`；测试：`tests/test_d004_state_pair_gate.py`。
