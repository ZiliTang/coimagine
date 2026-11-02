# 当前训练框架、模型结构与动作编码 Review

2026-10-29。本文只做代码核对与结构整理，不启动训练、不改 reward/controller/physics。所有数字来自当前代码或冻结 checkpoint；文件行号以本仓库当前版本为准。

## 0. 版本地图

| 版本 | 主要变化 | 当前是否使用 |
|---|---|---|
| v13 | 46 个持续闭环模式、5 Hz 决策、20 Hz 控制、共享 actor/critic、MC 回报、822 维观测 | 控制器/模式层仍被复用 |
| v14 | 944 维观测、分离 actor/critic、MC/GAE、0.8 s 动作保持、独立/自回归动作头、短历史未接入 | 观测/网络/PPO 基础 |
| v15 | 在独立动作头上加 14×14 联合类型残差；其余沿用 v14 | 当前联合头候选 |
| v16 | 精确 resume、每次评估保存完整权重、逐场景 retention | 续训/诊断入口，尚未正式跑 |

关键文件：

- 控制器/模式：`experiments/option_rl_v13/controller.py`、`mode_env.py`、`push_controller.py`
- 物理/奖励/成功：`src/coimagine/random_start/env.py`、`spec.py`、`skills.py`、`effect_push.py`
- v14 观测/网络：`experiments/option_rl_v14/interface.py`、`mode_env.py`
- v15 联合头：`experiments/option_rl_v15/interface.py`
- v16 PPO/评估/续训：`experiments/option_rl_v16/train.py`、`parallel.py`、`retention.py`、`continue_short.py`
- 固定开发面板：`experiments/option_rl_v14/learnability.py`、`outputs/option_rl_v14/learnability_002/panel.json`
- 全随机场景：`experiments/option_rl_v14/scenes.py`

## 1. 系统数据流

```
reset(instance)
  └─ 物理 120 Hz；控制 20 Hz；观测/记录 5 Hz；actor 决策 1.25 Hz
       actor 每 0.8 s 选一次联合模式 (a0,a1)
       0.8 s = 4 × joint_step；每个 joint_step = 4 × 0.05 s 控制 tick
       joint_step 内：Slot.select → 控制器 20 Hz 闭环 → 力 → 物理
       局部 skill 失败可提前结束 0.8 s 块；Step.periods 记录真实 tick 数
  └─ 观测 944 维 + 每机器人 46 维合法 mask
  └─ PPO：GAE(γ=1, λ=exp(-dt/10 s)) → 优势归一化 → clip → 4 epochs
  └─ 评估：每 N 轮，贪心解码；8 个扰动场景 + 8 个训练场景；v16 另存逐场景 retention
```

需要特别区分三个频率：

| 层 | 频率 | 代码 |
|---|---:|---|
| physics | 120 Hz | `continuous2d/config.py`，`spec.PHYSICS_DT` |
| control / controller | 20 Hz | `spec.CONTROL_DT=0.05` |
| observation / joint_step | 5 Hz | `decision_ticks=4`，每步 0.2 s |
| actor decision / hold | 1.25 Hz | `hold_seconds=.8`，4 个 joint_step |
| PPO sample | 每个 0.8 s 块一条 | `Step`；5 Hz 帧不是独立 actor 样本 |

`run_episode` 的 `repeats=round(hold_seconds/.2)`；每个 `joint_step` 跑 `decision_ticks=4` 个控制 tick。因此 `Step.periods=16`，`dense_dt=0.2 s`。actor 只在块首看到一次观测；块内的 5 Hz 观测只用于记录/世界模型，不参与 PPO 的 logp。`decision_flags` 标记真正的 actor draw。

## 2. 环境、任务、奖励与终止

### 2.1 任务

- 2D 平面，一个可推的矩形物体，两个机器人，一个目标点。
- 成功：物体中心距目标 ≤0.20 m，物体线速度 ≤0.02 m/s，角速度 ≤0.05 rad/s，连续保持 0.5 s。
- 时间预算：90 s。成功、异常或到 90 s 终止。
- 异常：数值非有限、穿透 >0.05 m、越界。异常 episode 记录但不进入 PPO；评估时仍计入分母并算失败。
- 动作不是“推 0.25 m”这种距离宏动作；是持续闭环模式（见第 4 节）。同一个模式可以跨多个 0.8 s 块保持。

### 2.2 奖励（`env.py:890-917`）

每个 0.2 s 控制段：

```
r = 1.0 * (d_before - d_after)
    - 0.02 * 0.05 * Σ d_after
    - 0.002 * periods
    + (success ? +10 : budget ? -2 : 0)
```

| 项 | 系数 | 量级 |
|---|---:|---|
| progress | 1.0 / m | 0.45 m 目标约 +0.45 |
| distance cost | -0.02 /(m·s) | 90 s、d≈0.45 约 -0.8 |
| time cost | -0.002 /control tick = -0.04 /s | 90 s 约 -3.6 |
| success bonus | +10 | 一次成功 |
| failure penalty | -2 | 仅 90 s 预算失败 |

含义：

- 成功奖励远大于时间/距离代价。6.45 s 成功回报约 +10.1；78 s 成功仍约 +6.2；90 s 失败约 -6。策略有强动机成功，但**没有强动机快**。
- progress 是原始米数差；报表里的 `progress` 是 `(initial-final)/initial` 再 clip 到 [-1,1]，两者不要混用。
- progress 可正可负，推近又推远会抵消；没有接触、探测、准备、协调的额外奖励。
- `failure_penalty` 只在时间预算失败时给；局部 skill 失败不直接罚，只通过时间/距离/进度间接体现。
- gamma=1，有限时域；成功/失败/预算终止都不 bootstrap。

### 2.3 固定开发面板 vs 全随机场景

`experiments/option_rl_v14/learnability.py:25-76`：

- 8 个固定场景：contact、approach、dual、reposition 各 2 个；方向 2（北）和 0（东）；初始物体-目标距离 0.45 m；没有初始在目标内的场景。
- contact：机器人距物体 0.008 m；approach/dual/reposition：0.10 m。
- 单机场景的 robot1 在远处 (4.4,0.5)；dual 的 robot1 在物体另一侧附近。
- reposition：目标在初始推方向的**反方向**；机器人需要绕到物体另一侧。
- 验证集是每个场景 robot0 沿切向扰动 1 cm 的 8 个场景；不是 64 个随机场景，也不是正式 test。
- 参考 witness 用真值距离决定制动，用真值判断“是否绕到目标反侧”；16/16 成功；这是离线可解性证据，不是策略输入、不是 PPO 示范。
- `experiments/option_rl_v14/scenes.py` 的另一套全随机场景：4×3 arena、目标在矩形内均匀采样、64 个固定 validation + 64 个 test、训练每 episode 新场景。v15 的 40 轮没有用这套 64 场景验证。

## 3. 观测：944 维到底装了什么

`observe(env)` 返回 `env.observation_vector(*boundary_observations())` 和两个 46 维 mask。当前 `context='extended'`，长度 944：

| 切片 | 维度 | 内容 |
|---|---:|---|
| `0:40` | 40 | 两机器人 own pose/velocity、相对位姿、patch 距离/法向、goal、接触标志、progress、时间 |
| `40:760` | 720 | 每机器人 72 条射线 × 5 通道：距离、hit、object、static、teammate |
| `760:822` | 62 | v13 控制器上下文：每机器人 31 = 14 当前/请求模式 + 11 phase + 6 标量 |
| `822:944` | 122 | v14 反馈：每机器人 61 = 14 执行 type + 14 phase + 13 failure + 3 其他 + 17 执行摘要 |

### 3.1 基础 40 维（`env.py:370-430`）

- 两机器人：`x,y,cos,sin,vx,vy`（12）
- `teammate_dx,dy,cos,sin`（4）
- 每机器人 patch：`valid,distance,normal_x,normal_y`（8）
- `goal_x,goal_y,goal_radius`（3）
- 每机器人接触：object/static/teammate（6）
- 每机器人旧 option progress/target（4，持续模式下置零）
- `elapsed_fraction,reserved_zero,remaining_fraction`（3）

### 3.2 v13 控制器上下文 31 维（`mode_env.py:105-112`）

- 14 维：`a = slot.pending or slot.action` 的 type one-hot（**请求模式优先**）
- 11 维 phase one-hot：`find, approach, ready, push, align, move, follow, brake, blocked, switching, peer_separate`
- 6 标量：`mode_speed/.5`、`mode_age/90`、`reference_vx/.5`、`reference_vy/.5`、`controller_failed`、`switching`

注意：`_logits` 的 persistence 读取 `start+29`，即 `controller_failed`，不是 `switching`；`failure_aware_persistence=True` 时当前模式 one-hot 会乘 `1-controller_failed`，即 skill 已失败时不再坚持该模式。这是有意的 failure-aware 设计，不是索引错误。

### 3.3 v14 反馈 61 维（`v14/mode_env.py:15-64`）

- 14 执行 type one-hot（`slot.action`，提交后的动作，不是 pending）
- 14 phase one-hot：增加 `recover, peer_wait, perception_wait`
- 13 failure one-hot + `failure_other, peer_wait, perception_wait`
- 17 摘要：`sensor_missing_fraction, guard_elapsed_fraction, guard_valid, geometry_age, geometry_valid, endpoint_observed_fraction, endpoint_valid, tangent_error, gap, gap_valid, ready, contact_ready, joint_active, recovery_requested, target_dx, target_dy, target_valid`

`target_dx/dy` 是“本机器人扫描得到的 contact target − 本机器人 own odometry”，不是物体真值；`gap` 是接触间隙；`ready/contact_ready/joint_active` 是双机接触握手状态；`recovery_requested` 是恢复请求。它们都由本机器人扫描/反馈或公开信息得到。

### 3.4 归一化

- `RunningMeanStd` 对 944 维都存 mean/var，但 `encoded=normalizer.normalise(obs); encoded[760:]=obs[760:]`：**只归一化前 760 维**；控制器上下文和反馈已经是 [-5,5] 或 one-hot，原样输入。
- warmup：micro panel 用 8 个 reset 观测；dynamic 用 64 个随机场景 reset 观测。
- 每轮 rollout 用冻结的 normalizer；PPO 后用本批 raw 观测更新一次；不是每帧在线更新。
- `raw_observations` 在并行路径只来自 valid episodes；串行路径在 anomaly 判断之前 append，因此 anomaly 的 raw 也会进入 normalizer 更新。这是一个实现差异，当前 v15/v16 用并行，影响较小，但值得统一。

### 3.5 信息边界

- 策略看到**两个机器人**的完整局部观测、控制器上下文和反馈；这是集中式全共享，不是 B-bit 去中心化策略。v13 metadata 明确写 `centralized full sharing of local observations; no B-bit deployment claim`。
- 没有把物体真值位姿/速度作为策略输入；`target_dx/dy` 来自扫描，`robot_states()` 的 own pose 是机器人自身 odometry。
- `summary['initial_geometry']`、`scenes.geometry_audit(instance)` 使用真值，但只写日志，不进策略。
- 观测里仍有旧 option progress 字段（持续模式下置零）；v14 反馈提供了局部执行进度，但两套字段并存，语义不完全一致。

## 4. 动作编码：46 个持续模式

`experiments/option_rl_v13/controller.py:16-30`：

| 索引 | 类型 | type_index | scale_index | 速度 |
|---:|---|---:|---:|---|
| 0-31 | PUSH E/NE/N/NW/W/SW/S/SE | 0-7 | 0-3 | 0.04, 0.08, 0.16, 0.32 m/s |
| 32-37 | ROTATE CCW/CW | 8-9 | 0-2 | 0.02, 0.04, 0.06 normal m/s |
| 38-43 | REPOSITION CCW/CW | 10-11 | 0-2 | 0.06, 0.12, 0.18 m/s |
| 44 | BRAKE_HOLD | 12 | -1 | — |
| 45 | SUPPORT_FOLLOW_PUSH | 13 | -1 | — |

`TYPE_COUNT=14`，`SCALE_COUNT=4`；动作表 46；codebook hash `7e8dd885...`。

### 4.1 每个模式实际做什么

- **PUSH_x**：`ContinuousPush`。控制器找可见面、接近、对齐、建立接触，然后以声明的法向速度做闭环力控。速度是“接触法向速度指令”，不是固定位移；持续到 actor 改模式或局部失败。`action_mask` 要求推方向与某个可见面法向夹角 ≤30°。
- **ROTATE_CCW/CW**：同一个 `ContinuousPush`，`effect_kind='rotate'`，用接触点产生旋转；`normal_v` 是法向速度尺度。它仍是 `kind='push'`，因此静态 joint mask 把它当 PUSH。
- **REPOSITION_CCW/CW**：`RepositionSkill`，沿物体边界外侧绕行，`rounded_scan`、`peer_detour='outer_arc'`、`progress_guard`；用于换面。`mode_key` 只看 kind，所以换向/换速度是同模式原地更新。
- **BRAKE_HOLD**：`HoldBrake`，零参考、制动；没有 persistence family。
- **SUPPORT_FOLLOW_PUSH**：`SupportFollowSkill`，保持相对可见面的位姿，不持续推；`completion_clock='peer_push'`，绑定 peer 的 push，`target_gap=.02`，peer guard；peer 不推/失败/切换时暂停。

### 4.2 mask

`controller.mask(obs)`：

- push：`effect_push.action_mask`，需要至少一个可见面，且推方向与某个面法向的 dot ≥ cos(30°)。所以一个状态下通常只有 **1 个方向 × 4 个速度**的 push + 6 rotate + 6 reposition + brake + follow = 18 个合法动作；没有可见 patch 时只剩 brake + follow。
- rotate：有可见面即可。
- reposition：`obs.patch is not None` 即可。
- brake/follow：总是合法。

mask 是**每个机器人自己的**，只依赖 own scan；不含 peer 状态。joint 层再加静态 `legal_pairs`：FOLLOW 只能和 `kind='push'` 的 peer 配对；其余 46×46 合法。`legal_pairs` 2101/2116。

注意：rotate 的 `kind` 也是 `'push'`，所以静态 mask 允许 FOLLOW+ROTATE；`mode_env.joint_step` 的检查也是 `kind!='push'`，同样允许。若语义上 FOLLOW 只应支持平移推，这是一个待确认的边界，不是当前已证明的 bug。

### 4.3 模式切换与 persistence

- `mode_key`：push 只看 `(kind,effect_kind,rotation_sign)`；translate push 的 8 个方向共享同一个 key；rotate 按 CCW/CW 分开；reposition 只看 `('reposition',)`；brake/follow 各自。
- 同 key：`Slot.select` 原地更新 skill 的 action/speed/direction；不制动、不重新握手。push 换方向、reposition 换向/换速都属于这一类。
- 不同 key：设置 `pending`，先 `HoldBrake` 到低速/零参考，再 `_commit`；`phase='switching'`。这保证模式切换不是瞬间跳变。
- 失败后同 key + 方向改变：`changed_intent_retry_requested`，重新制动检查；否则可能保持失败锁存。
- persistence：`mode_families` 把 8 个 push 方向归为同一族，rotate 各一族，reposition 两方向同族，follow 一族，brake 无；当前模式 one-hot × `1-controller_failed` × 2.5 加到 type logits。效果是“倾向继续 push/rotate/reposition/follow”，但 push 方向本身不被 persistence 锁定。
- actor 每 0.8 s 重选一次；同一个模式可以跨很多块保持。`mode_switch_*` 日志的分母是 5 Hz 执行记录（2 机器人），事件每次决策只出现一次，所以**每决策切换率约为 dashboard 值的 4 倍**。例如 dashboard 8.7% ≈ 每决策 34.7%。

### 4.4 双机联合动作

- 策略是集中式的：两个动作一起选，不是两个独立策略。
- 分布：`p(a0,a1) = p(type0,type1) × p(scale0|type0) × p(scale1|type1)`。
- 联合类型：`log p(type0) + log p(type1) + residual(type0,type1)`，再乘 `legal_pairs`。
- v15 的 `residual` 是 14×14，来自 actor hidden；只耦合 type，不耦合 speed。
- 采样：先 a0 边缘，再 a1 条件；logp 是两项之和；entropy 是 `H(a0)+H(a1|a0)`，即联合熵。
- 贪心：a0 边缘 argmax，再 a1 条件 argmax；不是联合 MAP。v15 诊断显示这两个双推场景的联合 MAP 和条件贪心恰好相同，所以不是解码 bug，而是分布本身把同推压掉了。

## 5. 模型结构

`experiments/option_rl_v14/interface.py:58-141`，v15 在其上增加联合头。

| 模块 | 形状 | 参数量 |
|---|---|---:|
| actor body | 944→256→256，Tanh | 307,712 |
| critic body | 944→256→256，Tanh，独立 | 307,712 |
| type heads | 2 × (256→14) | 2 × 3,598 |
| scale heads | 2 × (256→56) | 2 × 14,392 |
| value head | 256→1 | 257 |
| joint type residual | 256→196 | 50,372 |
| persistence bias | scalar | 1 |
| **合计** | | **702,034** |
| actor 参数 | | 394,065 |
| critic 参数 | | 307,969 |

- 独立头（无 joint）合计 651,662；v15 联合头多 50,372 参数（actor 的 12.8%）。这是一个**容量不等**的对照，4/8 vs 2/8 不能完全归因于“联合交互”。
- actor/critic 独立主干、独立 Adam、独立梯度裁剪（1.0）；v14 的共享/分离对照已做，当前默认分离。
- type/scale head 正交初始化 0.01；joint 头零初始化；value head 默认初始化；persistence 初始 2.5。
- 可选 `encoder='scan'`：雷达 2×72×5 经共享 Conv1d 编码成 64/侧，再与 40 维基础特征、184 维上下文/反馈拼接，body 输入 352；当前 v15 用 `flat`，未用 scan。
- 可选 `autoregressive`：动作 embedding + 条件 MLP，`p(a1|h,a0)` 真正条件化；当前关闭。
- 联合头不是完整自回归：它只加 14×14 type 残差；速度仍按各自 type 条件独立采样。双推需要两侧同速/同向，这个结构不能直接表达“speed 也要配对”。

## 6. PPO 训练管线

`experiments/option_rl_v16/train.py`：

### 6.1 rollout

- 每轮 8 个 episode，8 个 worker；每个 worker 用同一份冻结的 policy/normalizer 快照。
- micro panel 下 episode 轮流取 8 个固定场景；每次 attempt 的 ordinal 递增，anomaly 会消耗 ordinal 并补采，最多 3× episodes。
- 每个 episode 的动作随机采样；eval 是贪心。训练成功率和 eval 成功率不是同一个量。
- 每个 0.8 s 块一条 PPO transition；5 Hz 观测只用于记录/世界模型。
- 局部 skill 失败会提前结束当前 0.8 s 块，`Step.periods` 用真实 tick 数。

### 6.2 GAE / PPO

- gamma=1；lambda=exp(-step.periods×0.05/10)；0.8 s 块 lambda≈0.923。
- 终止步 next_value=0，不 bootstrap；本环境总是跑到 success/anomaly/budget，没有采样截断。
- batch 优势标准化；clip 0.2；4 epochs；minibatch 256；value coef 0.5；entropy coef 0.02；target KL 0.03，单个 minibatch KL >0.045 提前停该 epoch。
- 分离 actor/critic：两个 Adam 3e-4，各自 zero_grad/backward/clip 1.0/step。
- 更新前用当前策略重算 old logp，误差 >2e-5 直接报错。
- 更新后 normalizer 用本批 raw 观测更新一次。

### 6.3 数据量

v15 40 轮：320 个训练 episode，23,888 条 actor transition，89,257 条 5 Hz 密集记录；约 600 条 actor transition/轮，4 epochs，即每轮约 10 个 minibatch。这个量很小，8 个场景、1 个 seed、确定性 8 根评估，成功率的颗粒度是 12.5%。

### 6.4 评估

- micro panel：8 个 1 cm 扰动场景 + 8 个训练场景；都是贪心。
- 每 10 轮（v16 可 5 轮）；首轮也评估。
- `score=(validation success, mean_progress)`；`best.pt` 按验证集选；`policy.pt` 是最终权重；v16 每次评估另存 `snapshots/iteration_XXXXXX.pt`。
- `development_training` 的 8 个训练场景只用于 overfit/retention，不是泛化。
- 全随机 64 场景 validation/test 在 v15 没有使用；v13 的 3/64 全部是初始已在目标内的场景。

### 6.5 数据缓存

- `world_model_replay`：每 31 个全局 episode 存一个 5 Hz 序列；ring buffer 256；观测 944、动作 46、dt 0.2、reward、`actor_decision` flags；不是 PPO 数据。
- `rollouts/*.json`：每个 episode 的完整 action/execution 记录；用于审计和回放。

### 6.6 resume / 监控

- v16 保存 policy、optimizer、normalizer、history、ordinal、torch RNG、sampling RNG、manifest。
- resume 检查 architecture/rollout、episodes/seed/epochs/lr/entropy/advantage/trace/target_kl/panel、scene panel、parent source hashes；拷贝 parent evaluations/snapshots；恢复 RNG。
- 并行路径用 ordinal 派生 episode seed；v16 的 bitwise resume 测试只覆盖 workers=1；8 worker 的 smoke 只验证了可运行和结果合理，没有逐项 bitwise 对照。
- 监控：非有限、心跳 >300 s、连续 3 轮训练异常 ≥25%、连续 2 次验证异常 ≥10%、连续 3 轮 KL >0.5、持续低熵且无非平凡成功；触发写 STOP.json；supervisor 也检查 wall overrun。

### 6.7 实现层的小问题/待确认

这些不是当前结论的主要解释，但 review 时值得知道：

- `development_training` 的 `dev_eval` 没有传 `should_stop`；如果验证期间写 STOP，扰动验证会停，8 个训练场景的评估仍会跑完。8 根很快，影响小。
- checkpoint 的 `manifest['config']['seed']` 是 `None`，`manifest['seed']` 是 0；resume 用的是后者，功能上没问题，但读 config 时容易误解。
- normalizer 存 944 维 mean/var，实际只用前 760；760: 的统计被计算但随后被 `encoded[760:]=obs[760:]` 覆盖。
- 串行路径把 anomaly 的 raw 观测也放进 normalizer 更新；并行路径只放 valid。当前用并行，差异小，但两条路径不是严格同一协议。
- v16 的 bitwise resume 测试只覆盖 `workers=1`；并行路径靠 ordinal/`behaviour_seed` 保证 episode 级可复现，但没有逐项 bitwise 对照。8 worker smoke 只验证了可运行和结果合理。
- v15 的 `render_every=0`；`episodes.html` 是单独渲染的，不是训练中自动生成。
- `best.pt` 按 8 根验证集的 `(success, progress)` 选；成功率颗粒度 12.5%，best 选择本身有噪声。
- `mode_switch_*`、`changed_intent_retry` 的分母是 5 Hz 执行记录；每决策切换率约 dashboard 值的 4 倍。`failed_or_wait_rate` 则是 5 Hz 记录的时间占比，不宜再乘 4。

## 7. 当前经验状态（与框架 review 直接相关）

v15 40 轮，联合头：

- 固定训练 4/8，1 cm 扰动 4/8；逐场景是单机 4/4、双推 0/2、换面 0/2。
- 独立头固定训练 2/8、扰动 2/8；单机 2/4、双推/换面 0/4。
- 训练中随机策略：双推 41/80（联合）vs 50/80（独立）；换面 0/80（两者）。联合头在随机双推上并不更好。
- 冻结 checkpoint 的初始贪心：dual_2 = PUSH_N + REPOSITION_CCW（同推概率 0.5%）；dual_0 = PUSH_E + REPOSITION_CCW（0.8%）；独立头同推概率 16.1%/6.8%。
- 联合残差对 `PUSH_N,PUSH_N` −1.17、`PUSH_E,PUSH_E` −0.55、`PUSH_x,REPOSITION` +2.06；dual_2/dual_0 残差余弦 +1.000；双机 vs 单机 −0.34 到 −0.49。
- 换面场景贪心 90 s 一直朝远离目标方向 PUSH；参考 23.6 s（REPOSITION_CW 14.4 s + PUSH_S 8 s + 制动 1.2 s）。
- v14 容量探针（特权离线标签、独立 probe.pt）在训练/1 cm 扰动上闭环 8/8；它不是 PPO 结果，但说明当前观测/动作接口和 flat MLP 至少能表达参考行为。

解释边界：这些是 8 个固定开发场景、1 个 seed、确定性评估的结果；不是全随机泛化，也不能证明联合头原理上学不会协同。它说明当前训练数据/目标下，联合头学到的是一条“双机不要做同类型动作”的近似固定规则，正好压掉了双推需要的同推；换面则是探索问题（0/80）。

## 8. 需要 review 的问题

下面是我认为会直接影响下一步设计、需要你确认的点。先不改代码，等你逐条定。

1. **集中式全共享是否是目标基线？** 当前策略看两侧完整观测，metadata 明确不是 B-bit 去中心化。如果最终要验证通信/去中心化，这个基线的结论不能直接迁移。
2. **主指标用贪心还是随机策略？** 当前报表主指标是贪心；训练是随机。双推随机 50% 但贪心 0%，差距很大。是否同时报告两者，并以贪心为部署主指标？
3. **8 场景 overfit 门槛是否合适？** 现在只有 8 个训练场景、8 个 1 cm 扰动、只扰动 robot0；成功率颗粒度 12.5%。是否扩到 12–16 个场景/多方向扰动，再谈“学会”？
4. **联合头是否继续？** 它多 50k 参数、只耦合 type 不耦合 speed、4/8 全是单机、双推同推概率被压到 0.5%。是保留并加公开成对几何/门控，还是换回独立头 + 真正自回归动作头，或先做同容量对照？
5. **时间抽象是否合适？** actor 0.8 s 一次决策、控制器 20 Hz、5 Hz 观测；模式切换要制动握手。是否需要比较 0.4/0.8/1.2 s，或把“持续执行”与“模式重选”彻底分开？
6. **奖励尺度是否合适？** 成功 +10、时间 -0.04/s、失败 -2；78 s 成功仍 +6.2。是否要更强的时间压力，还是保持任务成本不变、只用课程/数据混合解决？
7. **数据混合是否要改？** 当前 8 个场景均匀采样，双推/换面各只占 1/4 的 episode。是否过采样双推/换面，或做“正确一侧起步”的换面课程？这比改 reward 更可控。
8. **换面任务是否先隔离训练？** 0/80 随机成功说明策略从未发现 REPOSITION 起手；先用只含换面的 panel 验证 PPO 能否在无单机干扰时学会，再谈组合。
9. **FOLLOW+ROTATE 是否应允许？** 静态 mask 和 mode_env 都用 `kind=='push'`，rotate 也算 push；如果 FOLLOW 只应支持平移推，需要收紧，但会改变动作空间。
10. **观测是否要加短历史/成对特征？** 当前是单帧 + 控制器摘要；双推的失败更像“多任务负迁移”而不是“记不住”。是先加 peer 距离/同面/接触的显式 pair 特征，还是上 GRU/scan encoder？另外，v13 上下文的“请求模式”和 v14 反馈的“执行模式”在切换时会不一致；旧 option progress 与新局部反馈并存；两套 phase 集合也不完全相同。这些是否需要在下一版统一？
11. **评估噪声是否要处理？** 单 seed、8 根、确定性；`best.pt` 按验证集选。是否要 2–3 seeds、重复随机评估，或更多扰动根，才接受“改善”？
12. **全随机训练何时恢复？** 我的建议是固定 8 场景稳定 ≥7/8、且双推/换面不再是 0/4，再跑 64 场景验证和有界全随机；不建议现在直接 12 h。

## 9. 一句话总结

当前框架是“集中式、单帧 944 维、46 个持续闭环模式、0.8 s 保持、分离 actor/critic、GAE、8 场景 overfit”的高层模式选择器。控制器承担了大量接近、接触、力控、换面、避让和双机握手；策略的杠杆主要是模式/方向/速度/时机。v15 的联合头增加了 14×14 type 耦合，但当前学到的是一条近似固定的“双机分工/差异化”先验：单机 4/4，双推/换面 0/4，双推同推概率被压到 0.5–0.8%。在决定下一步之前，最需要确认的是：主指标、8 场景门槛、联合头是否继续、数据混合/课程、以及是否接受集中式全共享基线。
