# D004：状态表 + 同推 pair gate（只做 A）

- 2026-10-29，DeepSeek 执行侧 v1（GPT 额度耗尽，用户临时授权；不改 GPT 的 D003 任务书/D002 结论）。
- 本轮只做两件事：
  1. 建立可审计的机器人/协调器/联合 option **扁平状态表**（设计、归属、时序、持久性、不变量、测试）。
  2. 在当前 v13/v14 控制循环之上加一个**只读可验证的 pair gate**，修“双方都选 PUSH 时一侧先推、另一侧还在接近”的联锁问题。
- **不改** approach 速度、`A_REF_MAX`、ROTATE 速度/旋转 profile、奖励、观测、动作表、PPO、训练；**不**解冻 D002 B；**不**并入 independent 停车；**不**发布 D005。

## 1. 固定问题与验收

固定复用 D003 的 8 个 case 和 16 条物理记录作 baseline；D004 再跑 **8 条 gate 重放**（W_N/W_E 各 1，6 个失败窗口各 1）。baseline 与 gate 使用同一源、同一前缀、同一 4.8 s/6.45 s 窗口、同一动作。总物理重放 8 条，worker ≤4，正式墙钟 ≤300 s；聚焦测试 ≤120 s。

必须满足：

- W_N/W_E gate 仍成功、129 tick、6.45 s，旧日志/基线轨迹一致；否则 `incomplete`，不做后续。
- 6 个失败窗口 gate 的 source raw/mask/pose 仍满足 D003 的 1e-5/0；D002 前 64 tick 只作 D003 baseline 参考，gate 允许有意不同（否则就是没有改行为），不得用它来判 gate 失败；baseline/instrumented 非干扰仍为 0。
- 任何 tick 不允许出现 `pair_phase=PUSH` 而 `both_ready`/`both_contact` 不成立；`joint_active` 只在 `both_contact` 时为真，接触丢失立即清零。
- 不允许 solo/one-side **normal push**：`PUSH` 阶段两侧 force 才可正常输出；`APPROACH/READY/ENGAGE` 只允许 pair 的接近/接触建立，不允许一侧进入正常推动。
- 若 W_N/W_E 失败，停止并保留 partial，不调 gate 阈值、不重跑正式；若只是记录/启动 bug，保留失败段并在回报中披露。

## 2. 状态模型（必须先落盘）

新增 `docs/agent_exchange/state/STATE_MODEL.md` 和 `state_table.csv`，以及 `experiments/option_rl_v18/state_schema.py` 作为唯一字段清单。表至少覆盖：

- **物理/观测层**：robot x/y/angle/vx/vy/omega、object、contact、patch/normal/gap、tick/time。
- **每机器人 option 内部**：requested/committed action、action kind/type/speed、skill stage、ready、contact_ready、release、joint_active、joint_pair、cooperative、done/exit、pause、pending、recovery、skill_time、face/contact target、reference/force。
- **协调器**：pair identity、pair phase、pair start/engage/push tick、both_ready/both_contact/both_active、force enable 0/1、recovery/clearance/pause、contact-loss latch、stage fallback reason。
- **联合 option 状态**：IDLE/APPROACH/READY/ENGAGE/PUSH/RECOVER/EXIT；进入/退出条件；live vs latched；强制不变量。
- 每行必须写：scope、field、owner/writer、reader、timing（pre/post）、persistence（live/latched/derived）、default、meaning、invariant、gate-use、test、source。
- `state_schema.py` 的字段表与 CSV 必须由同一清单生成/校验；测试必须检查 v13/v14 当前 `ContinuousPush`/`Slot`/`ContinuousModeEnv` 的字段都被登记，新增字段未登记时测试失败。

## 3. Pair gate 语义（只做 A）

实现 `experiments/option_rl_v18/pair_gate.py`，在 v14 `LearningEnv` 之上 subclass/wrap，**不修改 v13/v14 源码**。gate 固定：

- 只有两侧都选 PUSH（`ContinuousPush`、无 pending、同一 pair identity）才进入 pair 状态机；否则保持原 solo 行为，不伪装成 pair。
- `APPROACH`：至少一侧 `ready=False`；双方处于 approach/ready；**release=False、joint_active=False**；不输出正常 push。
- `READY`：双方 `ready=True`；进入接触建立的 pair 阶段；`release=True` 仅此阶段起；**joint_active=False** 直到双方 contact。
- `ENGAGE`：双方 ready，至少一侧已接触但不满足双方 contact；只允许 pair 的接触建立（`joint_engage_speed` 级别），不允许正常 push；任一侧接触丢失回退。
- `PUSH`：双方 `contact_ready=True` 且 `ready=True`；`joint_active=True`，双方 normal push 才可输出。
- **live 清除**：任一侧 `ready`/`contact_ready` 变假、pending/pause/recovery/clearance 接管、pair identity 改变，立即 `release=False`/`joint_active=False`；若阶段已进入 push，回退到 READY/ENGAGE 或 RECOVER，不允许粘滞。
- **stage fallback**：`release=False` 时不得保留 `stage='push'` 的单侧推动；把该侧回退到 ready/approach（只通过 gate 的状态字段和副作用，不改控制器公式）。
- **力层**：`PUSH` 才允许两侧 normal force；ENGAGE 只允许对称的接触建立；APPROACH/READY 的正常推为 0。若现有 `ContinuousPush` 在 joint_engage 分支仍产生单侧力，gate 必须在 `env.step_control` 前限制/清零，并记录 `force_gated` 原因。
- 本轮不新增“双方同时 contact”的物理保证；只保证没有 unilateral normal push，且所有阶段/力都有状态表记录。

## 4. 必须记录的配对指标

每条 gate 重放记录：pair phase 每 tick、both_ready/both_contact/both_active、per-side force、force_gated、stage fallback、pair start/engage/push tick、one-side normal push 计数（目标 0）、contact loss→force clear 延迟、witness success/ticks、6 个失败窗口的 source/D002/baseline 误差、state table hash。

## 5. 预算与交付

- 8 条 gate 重放 + 4 worker；正式 ≤300 s；聚焦测试 ≤120 s；不改 D003 原始目录。
- 产物：`outputs/agent_exchange/D004_<run>/`；`raw_ticks.jsonl`（gate + D003 baseline 对齐）、`case_metrics.csv`、`state_table.csv`/schema hash、`reproduction.json`、`manifest.json`、回放/时间线。
- 回报：`results/D004_State_Pair_Gate.md`；开头回答“状态表是否覆盖当前状态？pair gate 是否消除 unilateral normal push？witness 是否保持？还差什么？”。
- 更新 CURRENT 到 ready_for_review 后停止；不训练、不发布 D005、不解冻 D002 B。

## 6. 特别说明

GPT 额度耗尽，任务书由 DeepSeek 在用户“先做 A”和状态系统要求下补写；固定复用 D003 的 8 个样本和 baseline，不重选样本、不改阈值。若实现中发现 gate 需要改动超过本任务的范围（速度/旋转/奖励/训练），停止并交回，不自行扩权。
