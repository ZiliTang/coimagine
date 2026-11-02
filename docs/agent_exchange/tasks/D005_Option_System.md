# D005：option 注册表 + 中心协调器 + headless 测台会话

- 2026-10-29，DeepSeek v1；GPT 额度耗尽，用户明确授权“开始编写 D005 并执行”。
- 范围：只做 option 系统核心和 headless 测台；**不做 Web UI（D006）**，不训练，不改 reward/物理/观测/动作表，不解冻 D002 B，不并入 independent，不发布 D006。
- 目标：把 D004 的“状态表 + PUSH pair gate”升级成可复用的 option 系统，并用同一 Pymunk env 验证 D003/D004 回归。

## 1. 必达

1. **Registry**：v13 的 46 个 action 全部注册；每个 spec 有 family、参数、观测、状态、向上报告 stages、joint requirements、来源。`option_table.csv`、`state_table.csv`、`param_table.csv` 由代码生成，不手写。
2. **中央协调器**：joint phase（IDLE/APPROACH/READY/ENGAGE/PUSH/RECOVER/EXIT）、live release/joint_active、stage fallback、READY/ENGAGE 法向力对称化、APPROACH/PUSH 力门控；option 不拥有全局 joint 状态。
3. **OptionSystemEnv**：在 v14 `LearningEnv` 上替换 `OptionSlot` 和 coordinator；使用同一 v13 skill 实现；支持 per-family 参数覆盖；不修改 v13/v14 源码。
4. **headless TestbedSession**：手动 5 Hz option 序列；同一 Pymunk/env 执行；保存 run JSON（scene/sequence/overrides/frames/state strip/metrics/table hash）；支持 reset、run 存档/载入、参数覆盖存档；暂不做鼠标 UI。
5. **回归**：W_N/W_E 129 tick/6.45 s 成功；6 个 D003 失败窗口 source raw/mask/pose≤1e-5、baseline/instrumented 非干扰 0；gate one-side normal 0；D002 前 64 tick 差异是 gate 的预期行为，不重算 D002。
6. **测试**：registry/table 覆盖；option lifecycle；coordinator 转移/力门控；session 手动序列/参数/run 往返；不跑全仓库 pytest。
7. **清理**：新代码不重复 D004 的 gate 逻辑；不保留未使用的 coordinator/env 方法；不删除 D001–D004 历史证据。

## 2. 预算

- 正式回归 8 条 + session 1 条；4 worker；墙钟 ≤300 s；聚焦测试 ≤120 s。
- 新代码位于 `experiments/option_rl_v19/`；D004 `experiments/option_rl_v18/` 保留为历史证据，但新工作不依赖它。

## 3. 交付

- `docs/agent_exchange/state/option_table.csv`、`state_table.csv`、`param_table.csv`（自动生成）。
- `outputs/agent_exchange/D005_<run>/`：回归 CSV/JSON、session run JSON、manifest、源码快照。
- `results/D005_Option_System.md`；CURRENT 到 ready_for_review。
