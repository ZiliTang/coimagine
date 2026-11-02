# D004 状态模型：机器人 / 协调器 / 联合 option

2026-10-29。目的不是再记一份日志，而是给后续所有 option/controller 讨论一个**单一状态契约**：
每个字段属于谁、什么时候写、是否粘滞、哪些条件必须成立、由哪条测试守住。新增/修改 option 字段时，
先改这张表和 schema，再改代码；否则测试失败，避免“讨论完就忘”。

## 1. 分层

| 层 | 内容 | 写入者 | 时序 |
|---|---|---|---|
| 物理/观测 | robot x/y/angle/vx/vy/omega、object、contact、patch/normal/gap | 物理/传感器 | tick 开始观测；物理步后状态 |
| 每机器人 option 内部 | action、stage、ready、contact_ready、release、joint_active、joint_pair、done/exit、pause、pending、recovery、reference/force | Slot / ContinuousPush | 决策时提交；每 control tick 更新 |
| 协调器 | pair_identity、pair_phase、both_ready/both_contact/both_active、force_enable、pair start/engage/push tick、contact-loss latch、recovery/clearance/pause | ContinuousModeEnv / PairGateEnv | `_coordinate` 的 P1/P2 |
| 联合 option | IDLE/APPROACH/READY/ENGAGE/PUSH/RECOVER/EXIT | PairGateEnv | 每 control tick，P1 后、P3 前 |
| 动作/力 | 每侧 force、heading、stage fallback、force_gated | PairGateEnv + skill | P3/P4；物理前 |

关键：`ready` 是“接近完成”，`contact_ready` 是“已接触”，`joint_active` 是“双方都接触、允许 normal push”。
三者不是同义词；`release` 与 `joint_active` 必须是 **live**，不能像旧实现那样只置真不清零。

## 2. 联合状态机

| phase | 进入条件 | 允许 | 禁止/退出 |
|---|---|---|---|
| IDLE | 不是双方同族 PUSH，或 pending 未清 | 原 solo 行为 | 进入 pair 时 APPROACH |
| APPROACH | 双方 PUSH；至少一侧 ready=False | 各自 approach 到 staging | 不 release、不 joint_active、不 normal push；双方 ready→READY |
| READY | 双方 ready=True，均未接触 | 接触建立的 pair 阶段 | joint_active=False；任一侧 ready 丢失→APPROACH；一侧接触→ENGAGE |
| ENGAGE | 双方 ready，至少一侧接触，未双方接触 | 只允许 pair 接触建立（joint_engage 级别） | 不 normal push；双方 contact→PUSH；接触丢失留 ENGAGE；ready 丢失→APPROACH |
| PUSH | 双方 ready 且双方 contact | 双方 normal push | 任一侧 contact/ready 丢失→ENGAGE/APPROACH；recovery/pause→RECOVER |
| RECOVER | recovery/pause/clearance/失败 | 恢复流程 | 恢复后→READY/PUSH；退出→EXIT |
| EXIT | 任一侧 done/终止 | 保留分母 | 不产生力；新 pair→IDLE |

完整转移表在 `state_schema.PAIR_TRANSITIONS`；测试覆盖每条边。

## 3. 力层不变量

1. **PUSH ⇒ both_ready ∧ both_contact ∧ 双方 stage=push**。
2. **joint_active ⇒ both_contact ∧ both_ready**；任一侧 contact 丢失，同一协调 tick 清零，下一物理步之前生效。
3. **release live**：任一侧 ready 丢失、pending/pause/recovery/clearance、pair identity 改变，立即 false。
4. **no unilateral normal push**：一侧 stage=push 且 normal 力开启时，peer 必须 both ready/contact；ENGAGE 只允许 `joint_engage` 级别的接触建立，不算 normal push。
5. **stage fallback**：release=False 时不得保留单侧 `stage=push` 输出正常推；gate 把该侧回退到 ready/approach 并记录原因。
6. **退出/恢复/暂停**：任一 fault 立即清 release/joint_active，力归零；恢复不能沿用旧 contact/latch。
7. **所有力/阶段**必须能在 `state_table.csv` 找到字段；`force_gated`/`stage_fallback` 不允许静默。

## 4. 扁平表与字段清单

- [`state_table.csv`](state_table.csv)：语义状态（约 50 行），列包括 scope/field/type/owner/reader/timing/persistence/default/meaning/invariant/gate/test/source。
- [`field_inventory.csv`](field_inventory.csv)：对当前 runtime `Slot`/`ContinuousPush`/`LearningEnv`/`PeerClearancePrefix`/`Continuous2DEnv` 的**全部 196 个属性**做分类（state / config_or_geometry_cache / counter_or_cache / option_internal）。任何新字段都会出现在这里；未分类字段会让测试失败。
- [`state_schema.py`](../../../experiments/option_rl_v18/state_schema.py)：唯一字段清单；`write_tables()` 生成两张 CSV；`classify_runtime_fields()` 在测试中扫描 runtime 字段。

**新增字段流程**：先在 `FIELD_SPECS` 加语义行（或明确 `option_internal`）→ 运行 `python -m experiments.option_rl_v18.state_schema` → 两张 CSV 更新 → 测试通过 → 才改 controller。这样“讨论完就忘”变成“表里没有就不许进代码”。

## 5. 验证机制

- `tests/test_d004_state_pair_gate.py`：
  - 全 runtime 字段已分类；语义 state 字段在 `state_table.csv`；CSV 与 schema 同源。
  - 转移表：APPROACH→READY→ENGAGE→PUSH；contact loss 回 ENGAGE；ready loss 回 APPROACH；recovery/pause→RECOVER；EXIT。
  - 力：PUSH 才允许 normal；ENGAGE 允许 joint_engage；APPROACH/READY 不产生 unilateral normal；contact loss 清零。
  - stage fallback：一侧 release false 时不得留 push。
  - D004 的 8 个 case：W_N/W_E + 6 个 D003 窗口；witness 必须仍 6.45 s；6 个窗口 source/D002/baseline 误差同 D003；`one_side_normal_count=0`。
- 8 条 gate 重放 + D003 的 16 条 baseline；正式 ≤300 s；聚焦 ≤120 s；不训练、不改速度/旋转。
- 新增 option 字段未进表 = 测试失败；这是本轮的“系统级防遗忘”机制，不依赖聊天记忆。
