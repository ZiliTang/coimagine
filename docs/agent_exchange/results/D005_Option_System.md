# D005 执行回报：option 注册表 + 中心协调器 + headless 测台会话

- 状态：`complete`。46 个 option 注册、三张自动表、中心 joint coordinator、OptionSystemEnv、headless session；正式 **7.84 s**，4 worker；6 项聚焦测试；不训练、不做 Web UI。
- 任务书：`tasks/D005_Option_System.md`；配置：`tasks/D005_config.json`。
- 输出：`outputs/agent_exchange/D005_20261029_2130/`；自动表：`docs/agent_exchange/state/option_table.csv`、`state_table.csv`、`param_table.csv`。
- 特别说明：GPT 额度耗尽，D005 任务与执行由 DeepSeek 在用户明确“开始编写 D005 并执行”的授权下完成；D001–D004 历史证据未改。

## 先回答

**option 是否已经在系统中注册？** 是。v13 的 46 个 action 全部进入 `SPECS`，每个有 family、参数、观测、状态、上报 stages、joint requirements 和 source；三张表由 `spec.write_tables()` 生成，不再手写。

**中心节点是否管理联合状态？** 是。`JointCoordinator` 拥有 IDLE/APPROACH/READY/ENGAGE/PUSH/RECOVER/EXIT、live release/joint_active、pair start/engage/push tick、stage fallback 和 READY/ENGAGE 的法向力对称化；option 只报告自身 stage/contact，不再自己写全局 joint 状态。

**是否已经有测台？** 有 **headless TestbedSession**：手动 5 Hz option 序列、同一 Pymunk/env 执行、129 帧、state strip、run JSON、参数覆盖、run 存档/列表/载入。Web UI、鼠标拖动、逐帧播放和多轮列表面板属于 D006，本轮没有假装完成。

**回归是否保持？** 是。W_N/W_E 129 tick/6.45 s 成功；6 个 D003 窗口 source raw/mask/pose≤1e-7、baseline/instrumented 非干扰 0；one-side normal 0；D002 前 64 tick 与 D003 的差异是 coordinator 的预期行为，不重算 D002。

## 架构落地

| 层 | 文件/对象 | 责任 |
|---|---|---|
| Option contract | `spec.OptionSpec` / `ParamSpec` | 参数、观测、状态、stages、joint requirements、来源 |
| Registry | `spec.SPECS` / `write_tables` | 46 个 action；三张 CSV 自动生成 |
| Skill factory | `spec.make_skill` | v13 实现 + per-family 参数覆盖 |
| Central coordinator | `system.JointCoordinator` | joint phase、live flags、stage fallback、力层 |
| Env | `system.OptionSystemEnv` / `OptionSlot` | 替换 Slot/协调器，不修改 v13/v14 |
| Session | `session.TestbedSession` | 手动序列、Pymunk 步进、帧/条带、run 存档、参数覆盖 |
| Regression | `run_d005.py` | 8 个 D003 case + 1 条 session |

关键约束：`OptionSlot._commit` 只换 skill factory；`OptionSystemEnv` 不重写 `joint_step`；`JointCoordinator` 不复制 v13 控制公式；session 不复制物理。这样 option 的实现、协调器、测台各自只有一层职责。

## 自动表

- `option_table.csv`：**46 行**；key/family/action_index/label/kind/effect_kind/speed/params/observations/states/stages/joint_requirements/source。
- `state_table.csv`：**52 行**；option 状态 + coordinator 状态；persistence/invariant。
- `param_table.csv`：**26 个 family 参数**；default/low/high/unit。
- 表由代码生成；改 `OptionSpec` 后重跑 `write_tables()` 即可。没有手写总结表参与运行。

## 回归与 session

| case | success | periods | source | noninterference |
|---|---:|---:|---|---:|
| W_N | True | 129 | — | 0 |
| W_E | True | 129 | — | 6e-15 |
| N_E_gap | False | 368 | True | 0 |
| N_E_one | False | 144 | True | 0 |
| N_N_far | False | 144 | True | 0 |
| N_N_direction | False | 1088 | True | 0 |
| L_E | False | 1744 | True | 0 |
| C_N | False | 1760 | True | 0 |

- 6 个失败窗口 source raw/mask/pose = 0/0/≤1.1e-7；D002 前 64 tick 的差异是 coordinator 的预期，不是复现失败。
- session：W_N 实例 + 33 个 witness 5 Hz 命令；129 帧/129 tick，success=True；`strip` 每帧包含 stage0/stage1、joint phase、contact、joint_active、forces；run JSON 可 save/load。
- 参数覆盖：`OptionSystemEnv(overrides={'PUSH': {'joint_engage_speed': 0.05}})` 经 `OptionSlot` factory 进入 `ContinuousPush.config`；测试验证。
- 多轮：`session.run()` 可重复，`list_runs()`/`save_runs()` 保存多次结果；D006 的列表只消费这些 run JSON。

> **D006 后续更新（2026-10-29）**：本报告中的 READY/ENGAGE normal equalization 已在 D006 A/B 后从 v19 代码删除；D005 输出是旧代码的历史证据，新 testbed/v19 不再使用 equalization。

## 测试与清理

- `tests/test_d005_option_system.py` **6 项通过**：registry/table、coordinator phases/live、force equalization、session 129/save-load、参数覆盖、多 run 存档。
- 删除了重复的 force-gate 实现（原先 coordinator 和 env 各一份）、未使用 import、session 的 pytest 收集警告；v19 不依赖 v18 pair gate，D004 保留为历史证据。
- 代码行数/模块：4 个实现文件（spec/system/session/run_d005），没有新的 UI/数据库/后端服务。
- 未跑全仓库 pytest；本轮只跑针对性测试。

## 限制 / 偏离

- 没有 Web UI、鼠标拖拽、逐帧播放、颜色条带页面、参数面板；这些是 D006。
- D005 的 manual sequence 是 agent 手填/命令行，不是用户画布操作；session 仍可直接复用这些数据。
- 参数覆盖目前 per-family，尚未做跨进程/多环境参数档案合并；run JSON 已带 overrides，D006 可导出。
- D002 B 仍冻结；independent 停车未并入；没有训练。

## 下一步

建议 D006 只做 UI/交互层：Canvas 拖拽、手动序列编辑器、播放/暂停/逐帧、颜色条带、run 列表/回放、参数面板与 override 导出。D006 不应再写一套 env/物理，直接消费 `TestbedSession` 的 run JSON 和 `OptionSystemEnv`。

## 产物

- [option_table.csv](../state/option_table.csv) · [state_table.csv](../state/state_table.csv) · [param_table.csv](../state/param_table.csv)
- [summary.json](../../../outputs/agent_exchange/D005_20261029_2130/summary.json) · [manifest.json](../../../outputs/agent_exchange/D005_20261029_2130/manifest.json)
- [case_metrics.csv](../../../outputs/agent_exchange/D005_20261029_2130/case_metrics.csv) · [session_run.json](../../../outputs/agent_exchange/D005_20261029_2130/session_run.json)
- 实现：`experiments/option_rl_v19/spec.py`、`system.py`、`session.py`、`run_d005.py`；测试：`tests/test_d005_option_system.py`。
