# 会话交接：Option v4.6（协同 staging + 短面软内缩）与 RL 训练

日期：2026-10-27  
交接对象：另一台 PC 上负责 RL 训练的 agent  
本轮目标：在**当前优化后的 option 版本**上做第一次有效 RL 训练；不重新设计 reward，不回到 v4.5 的旧 wrapper。

## 0. 先读这几条

1. 本仓库 `main` 的最新提交就是本交接提交；commit 消息里写了本文件名。
2. **不要**用旧的 run1/run2 结论判断当前训练器：旧 wrapper 有“24 技能早退不收费”的 bug，已经修掉。
3. 先跑 `pytest tests/test_push_v46.py tests/test_option_testbench.py tests/test_random_start.py tests/test_random_start_anchor.py tests/test_random_start_rl.py -q`（本机 40 passed），再做训练。
4. `outputs/` 不入库；本机的 run/页面数据在另一台 PC 上不存在，需要重跑。
5. 全量 `pytest -m "not slow"` 在本机有 1 个既有失败：`tests/test_observation_terminal_v1.py::test_full_chain_baseline_evaluation_is_not_degenerate` 需要 `outputs/observation_terminal_v1/terminal_observations_train.jsonl`。与本轮改动无关。

## 1. 本轮改动（相对 63082da / v4.5）

### 1.1 `PushSkill` v4.6：协同 staging（ready/release）

- 新增常量（`src/coimagine/random_start/spec.py`）：
  - `COOP_READY_GAP = 0.03` m：两侧各自到约 3 cm 边到边裕度；
  - `COOP_READY_BRAKE_GAP = 0.03` m：6 cm 内改用慢速径向伺服，避免惯性冲进接触；
  - `COOP_READY_ALIGN_DEG = 10°`：朝向与接触法向的对齐容差。
- `PushSkill` 新增 `ready` 阶段：协同双推时，到 3 cm 且朝向对齐后**零力等待**；`ready` 期间不触发 no-progress/contact-loss 计时。
- `RandomStartEnv.joint_step`：只有**两侧都 ready** 时，同一周期设置 `release=True`，两侧下一周期同时进入 `push`。
- 旧的 1.5 s 带推力 light-contact hold 已移除；`hold` 字段保留为 False 兼容旧日志。
- 日志 row 新增 `ready`、`release`、`cooperative` 字段；页面时间条蓝色段即 ready。

### 1.2 短面软内缩

- 旧硬 guard：可见线段 < `2*(R+0.02)=0.40 m` 直接 `insufficient_visible_segment`；0.25 m 短面永远不可推。
- 新 `_contact_inset`：宽面保持 `R+0.02` 内缩；窄面缩小内缩，最小允许 `0.1R=0.018 m` 的可见线段，目标点为可见片段中心。
- `insufficient_visible_segment` 只在可见线段短于 0.1R 时出现。
- 这是“接触优先”策略；窄面仍可能因接触点偏置产生力矩。

### 1.3 训练器 wrapper（上一轮已修，必须保留）

`experiments/option_rl_v45/train.py`：

- `MAX_SKILLS 24 → 120`，episode 跑到 `env.termination`；触顶补 `FAILURE_PENALTY`，不再允许“短 WAIT 早退省钱”；
- 每个 episode 最后一步标记 `terminal`；`returns_and_advantages` 在 terminal 处重置，修掉跨 episode return 泄漏；
- `observe_vector` 改为用环境的 `_last_progress`（合法 760 维里的 4 个进度特征不再恒 0）；
- `history.json` 超时也落盘；
- 新增 `experiments/option_rl_v45/fixed_policies.py`：固定策略 return 对照（wait / random push / legal goal heuristic / privileged diagnostic）。

## 2. 当前 option 行为证据

### 2.1 短面 `L1_train_0002`

- 旧：停在 gap≈0.125 m，`insufficient_visible_segment`，无接触；可见短面约 0.22 m。
- 新：
  - PUSH 0.39 m/s / 1.00 m：进入 push、接触，`signed=0.564 m`，物体位移 0.585 m，转角 −8.8°，exit `option_budget`；首次接触点离真实面中心约 0.002 m；
  - PUSH 0.24 m/s / 0.50 m：`push_complete`，`signed=0.508 m`，位移 0.518 m，转角 +4.0°。
- 1.0 m 快推仍会 option budget 用尽，是 timeout/行程问题，不是接触丢失。

### 2.2 协同双推（同侧长面，4.0×3.0 m 空场，无静态障碍）

9 个 `coop_*` 场景（对称/非对称/不等间隙/一侧远/近角点）：

- 两侧 `sync delay = 0.00 s`；staging gap 约 0.026–0.035 m；
- 从**首次接触之后**算，同侧 pair 的 after-engagement contact fraction = **100%**；无 `contact_lost`；
- 两侧接触前物体移动 < 0.02 m；最终转角≈0（角点 ±4–6°）；
- 一侧远距离（0.45 m gap）也等双方 ready，近侧不再提前推；
- `coop_adjacent`（下侧+右侧，短面/垂直法向）仍不稳定：`safety_abort` / `contact_lost`，右侧 after-engagement 只有 57–60%（个别 100% 但双方 safety_abort）。这是已知构型问题，不是 staging 的回归。

### 2.3 固定策略 return 对照（16 test roots）

> 这组数字是 **v4.5 控制器 + 修好的 wrapper** 下的对照；v4.6 只改变短面/协同 staging，
> WAIT/random 的结论不变，但具体数值可在新机器上重跑 `fixed_policies.py` 复核。

- 旧 wrapper 的 `WAIT[2s]×24`：return −2.86、48 s、无 failure penalty（**这个 loophole 已修**）；
- 修后 `WAIT`：−7.36（90 s，正常 failure）；
- random PUSH：−8.42；random legal：−8.13；合法目标启发式：−7.14；
- 特权目标推：−6.15 均值，1/16 成功；成功局 return **+7.43**（+10 success bonus，密度代价约 −3.6）。

结论：推有正收益路径，但成功稀疏；distance/time 权重没有压过成功奖励。**不要为了变正而改 reward 权重**。

## 3. RL 训练状态

- 训练器：`experiments/option_rl_v45/train.py`（760 维合法观测、28×28 因子化头、semi-Markov return、观测归一化、动作 mask、每评估 checkpoint、自动 dashboard）。
- 监控：`experiments/option_rl_v45/monitor.py` → `dashboard.html`。
- run1：无效（外部 timeout，未写 checkpoint，val_success 全程 0）。
- run2：旧 wrapper 下主动停止；贪心动作 WAIT/REPOSITION、PUSH 0；现已被 wrapper bug 解释，**不能作为当前训练器的结论**。
- 当前还没有在 v4.6 + 修好的 wrapper 上跑过有效训练。下一次算第一次有效训练。

## 4. 建议的训练顺序

### 4.1 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
# torch 若未装：pip install -e '.[deterministic-grid]' --extra-index-url https://download.pytorch.org/whl/cpu
```

### 4.2 先做一次短诊断（不是正式长训练）

```bash
.venv/bin/python experiments/option_rl_v45/train.py \
  --iterations 5 --episodes 8 --eval-roots 4 --hours 0.2 \
  --out outputs/option_rl_v45/run3_smoke
```

检查：

- 观测维度 760、动作 28；
- `explore_return` 量级应在 −6 ~ −8（修好的 wrapper 下随机策略），不是旧的 −2.8；
- 贪心动作里是否出现 PUSH（不一定成功，但不能长期为 0）；
- dashboard 能打开。

### 4.3 第一次有效训练

按上一轮预算：最多两轮、总限 12 小时；run1/run2 已消耗，下一次算第一次有效长训练。建议先 3–4 h：

```bash
.venv/bin/python experiments/option_rl_v45/train.py \
  --hours 4.0 --episodes 8 --eval-roots 16 \
  --out outputs/option_rl_v45/run3
.venv/bin/python experiments/option_rl_v45/monitor.py outputs/option_rl_v45/run3
```

重点看：

- `explore_return`、`explore_success`、`explore_progress`；
- `val_success`、`val mean_progress`；
- 贪心动作分布（尤其 PUSH 是否为 0）；
- reward 分量（progress / distance_cost / time_cost / success_bonus / failure_penalty）；
- entropy、policy/value loss。

如果 4 h 后 PUSH 仍为 0：优先改**探索**，不是 reward：

- `entropy_coef 0.01 → 0.05~0.1`；
- PUSH 优先采样 / 课程（先近距、低尺度、对称初始）；
- 从合法目标启发式 warm-start（如果实现，明确它是初始化，不是 oracle 输入）。

### 4.4 评估口径

- 固定 16/64 个 test roots；报告 success、mean progress、return、steps、cost、invalid pushes、contact after engagement；
- 不要用训练集 success 代替 test；
- 不要用隐藏真值裁剪候选或选动作；
- 任何 reward/评测条件改动都必须先和用户确认，不能为了正结果偷偷改。

## 5. 复现本轮证据（可选）

```bash
# 固定策略 return 对照（16 roots）
.venv/bin/python experiments/option_rl_v45/fixed_policies.py --roots 16

# 27 场景 / 93 分支可视化 + SUMMARY
.venv/bin/python experiments/option_testbench/continuity_eval.py
# 只重建 summary/HTML（已有 logs）
.venv/bin/python experiments/option_testbench/continuity_eval.py --skip-run
```

页面：`outputs/option_testbench_v1/round_035_pusheval/index.html`（静态 frame scripts，直接打开；93 分支首次加载稍等 1–2 秒）。`SUMMARY.md` 里有短面、角点、协同三张表。

## 6. 已知问题 / 未决

1. **相邻侧协同不稳定**：下侧长面 + 右侧 0.25 m 短面，两法向垂直；ready/push 能发生，但会出现 `safety_abort` / `contact_lost`。不要把它当成同侧协同的回归。
2. **1.0 m 快推/短面**：仍可能 option budget 用尽；0.5 m 可完成。是 timeout/速度问题，不是接触问题。
3. **`ENV_VERSION` 仍是 `random-start-oriented-v4.1`**：历史字符串没有随 v4.2–v4.6 bump；不要依靠版本字符串判断 branch 兼容性，旧 branch/checkpoint 不要跨本轮行为恢复。
4. **全量 pytest 的既有失败**：缺 `outputs/observation_terminal_v1/terminal_observations_train.jsonl`，与本轮无关。
5. 本轮没有提交 `outputs/`；所有 run 都需要在新机器上重跑。

## 7. 文件索引

- 控制器：`src/coimagine/random_start/skills.py`、`env.py`、`spec.py`
- 训练器：`experiments/option_rl_v45/train.py`、`monitor.py`、`fixed_policies.py`
- 可视化：`experiments/option_testbench/continuity_eval.py`、`continuity_template.html`
- 测试：`tests/test_push_v46.py`
- 本轮简版变更记录：`docs/handoff/20261027_PushSkill_v46_Ready_and_SoftShortFace.md`

Handoff file: `docs/handoff/20261027_Session_Handoff_Option_v46_RL.md`
