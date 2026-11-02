# D001 执行回报：双推成功经验为何没有形成稳定贪心行为

- 状态：`complete`。92/92 episode，4 个 8-episode 批次全部完整，0 异常。
- 任务书：D001 v1（`tasks/D001_Dual_Push_Credit_Diagnostic.md`）。
- 开始/结束：episode 2026-10-29 16:56:17–16:57:02；最终分析 17:07:51（Asia/Shanghai）。中间为首次分析 bug/调试/重复分析（实现时间）。
- 实际预算：episode 阶段 45.4 s；最终分析 6.5 s；合计 **52.0 s**（<900 s 硬上限）。另有实现期 0.4 s 探针和首次分析 bug 的调试空档，不计入仿真/分析预算，详见“偏离与限制”。
- 输出：`outputs/agent_exchange/D001_20261029_1644/`。

## 冻结输入与执行条件

| 输入 | SHA256 |
|---|---|
| panel | `a60bdcc120923a8091d8e8930eb4717406075aeeb55b5291e3f924dd48227126` |
| independent 第 40 轮 | `1e8a0c88ab65ef9a9a42b58a11b625094804a7b5127650ce8f4504ae05b19449` |
| joint_types 第 40 轮 | `dac637e6d21f64d6fa3bf98c658c176d12b1ac7409b66fe6ff84186e5121e6c5` |

- HEAD `22c1333027bbf2a39dea3246bee78638916b4171`；工作区 147 个未提交项。源码哈希与快照见 `source_hashes.json` / `source_snapshot/`；没有用 HEAD 代替实现。
- 两个 checkpoint 的 35/35 源文件哈希与当前源码匹配；第 40 轮、944 维、46 动作、0.8 s 保持、原 mask/控制器/物理/90 s/奖励均未改。
- 命令：`.venv/bin/python experiments/option_rl_v16/diagnose_dual_credit.py --out outputs/agent_exchange/D001_20261029_1644 --panel ... --independent ... --joint ... --workers 4 --deadline-seconds 900`
- 无 `optimizer.step`、无 normalizer 更新、无 checkpoint 写入；只对冻结权重求梯度。两个 checkpoint 在运行前后 SHA256 不变。
- 评估动作 seed：random 1000–1015；greedy 无随机动作。A 的 68 条 + B 的其他 24 条 = 92；B 的双推 8 条复用 A 的 joint 1000–1003。

## A. 冻结策略的随机与贪心差距

每模型/每根：贪心 1 条；随机 16 条（seed 1000–1015）。16 次是同一初始根上的动作重复，不是 16 个场景或 16 个训练 seed。

| 模型/根 | 贪心成功（最近距离） | 随机成功 | 随机回报均值 | 随机时间均值 | 进入目标未判稳 | 异常 |
|---|---:|---:|---:|---:|---:|---:|
| independent/dual_2 | 0/1（0.194 m） | 13/16 | +6.10 | 33.8 s | 1 | 0 |
| independent/dual_0 | 0/1（0.016 m） | 11/16 | +3.90 | 41.0 s | 2 | 0 |
| joint/dual_2 | 0/1（0.404 m） | 6/16 | −1.09 | 63.4 s | 4 | 0 |
| joint/dual_0 | 0/1（0.335 m） | 6/16 | −1.44 | 67.6 s | 3 | 0 |

- 冻结随机策略**仍然能成功**：joint 12/32，independent 24/32；贪心 0/4。随机-贪心差距真实存在，不能再用训练累计 41/80、50/80 解释末轮。
- 训练累计 41/80（joint）与 50/80（independent）是 40 轮训练期间的总和；末轮冻结随机是 6/16+6/16 与 13/16+11/16。两者不是同一个量。
- 两者的贪心失败模式不同：
  - independent 的贪心**已进入目标**（最近 0.194 m / 0.016 m）但未判稳，是到达后制动/判稳失败。
  - joint 的贪心**没有接近目标**（0.404 m / 0.335 m），是动作序列/执行在中途失败。

## A2. 同推概率与实际执行：joint 选了，但几乎没有两机同时推

对每个 actor 决策，完整 46×46 联合概率对全部合法对求和；“同推”定义为双方都选**同向平移 PUSH**，不是旋转或 FOLLOW。

| 模型/根（随机） | 同推概率均值 | 选中同推的决策 | 选中同推时长 | 实际双方 push 阶段 | 仅一方 push | wait | 预接触/对齐 | 切换/blocked/其他 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent/dual_2 | 18.56% | 109 | 82.2 s | **30.5 s** | 16.7 | 9.4 | 19.8 | 5.9 |
| independent/dual_0 | 11.29% | 109 | 85.1 s | **34.8 s** | 18.5 | 6.4 | 25.2 | 0.2 |
| joint/dual_2 | 3.08% | 39 | 30.8 s | **0.6 s** | 16.6 | 4.6 | 8.8 | 0.2 |
| joint/dual_0 | 3.02% | 41 | 32.8 s | **4.0 s** | 11.8 | 7.0 | 9.8 | 0.2 |

- joint 的随机策略确实选中了同推（约 31–33 s），但真正进入“两机都在 push 阶段”的时间只有 0.6 s / 4.0 s；大部分时间是一方 push、预接触/对齐或 peer wait。independent 在同样定义下有 30.5 s / 34.8 s 的实际双方 push。
- 这不是“同推概率低所以没试过”；joint 选中后也没有形成持续双方执行。该现象在 A 的 80 个 joint 同推决策上可复核；不是 top-k 截断。

### joint 的成功与失败

| 根 | 成功 episode | 成功中同推决策 | 成功中实际双方 push | 失败 episode | 失败中同推决策 | 失败中实际双方 push |
|---|---:|---:|---:|---:|---:|---:|
| dual_2 | 6 | 3 | 0.0 s | 10 | 36 | 0.6 s |
| dual_0 | 6 | 4 | 0.0 s | 10 | 37 | 4.0 s |

joint 的成功 episode **几乎不选同推**；失败 episode 反而集中了 36/37 个同推决策。成功主要来自 push+reposition / 其他动作，而不是双方同推。这个结论只覆盖这两个根和 16 个随机 seed，不外推到所有几何。

## A3. 换面：0 成功不等于没尝试

B 的 8 条 joint 换面随机轨迹（reposition_2 / reposition_0 各 4 条，seed 1000–1003）中：

- robot0 选择 `REPOSITION_*` 共 47 + 56 = **103 次**（占其决策 9.6% / 11.5%）；robot1 没有选 REPOSITION。
- 这 103 次都进入 `phase='move'`，平均执行 0.8 s；但仍然 0 成功。
- 决策序列里还包含 PUSH_N/PUSH_E、ROTATE、FOLLOW、BRAKE；换面不是“从未尝试”，而是尝试后没有完成目标侧换面/推进。
- 因此前一阶段“换面 0/80 成功”只能说明没有成功，不能说明没有 REPOSITION 动作。换面的下一步应看尝试后的执行/几何/方向，而不是把“没试过”当作既定原因。

## B. 四个 8-episode 批次的组成与权重

每批：joint 的 dual_2/dual_0 各 1 条（A 的 seed 1000–1003）+ 其他 6 根各 1 条（B 新增 24 条）；顺序沿用原 panel。4 批全部 8/8 valid，0 异常。

| 任务 | episode 数 | actor transition | 模拟秒数 |
|---|---:|---:|---:|
| dual | 8（25%） | 587（26.4%） | 466.1 s（27.6%） |
| single | 16（50%） | 659（29.6%） | 501.0 s（29.7%） |
| reposition | 8（25%） | 981（44.0%） | 720.0 s（42.7%） |
| 合计 | 32 | 2227 | 1687.1 s |

- 逐批 dual transition：55 / 133 / 226 / 173；reposition 241 / 247 / 245 / 248；single 300 / 163 / 111 / 85。
- 换面只占 1/4 episode，却占 44% transition 和 43% 模拟秒；两个长失败换面每批固定 180 s。single 占 1/2 episode，只占约 30% transition。**episode 的 1/4 不是梯度的 1/4**；这与任务书的提醒一致。
- 4 个批次的 dual 过渡数从 55 到 226，取决于双推成功/失败；长失败轨迹会放大 dual 的样本份额，不能把 dual 的 1/4 固定成 1/4 权重。

## B2. 同推优势：只在极少数成功样本上为正

joint 的 dual 任务、`same_push` 类别（raw / batch-normalized advantage 均值；n 为 transition 数）：

| 批次 | dual 结果 | 成功中的同推 | 失败中的同推 | 成功中的 push+reposition | 成功中的其他 |
|---|---|---:|---:|---:|---:|
| 1000 | 两条都成功 | **无同推决策** | 无 | 11（+4.61 / +2.77） | 44（+1.83 / +1.02） |
| 1001 | 一成功一失败 | 3（+5.70 / +4.03） | 8（−0.35 / −0.24） | 8（+3.57 / +2.52） | 9（+1.98 / +1.40） |
| 1002 | 两条都失败 | 无 | 7（−0.59 / −0.15） | 无 | 无 |
| 1003 | 一成功一失败 | 无（成功走 push+reposition） | 4（−0.95 / −0.56） | 15（+2.07 / +1.34） | 44（+0.52 / +0.36） |

- “成功轨迹中的同推得到正优势”只在 1001 的 **3 个同推 transition** 上成立；其余成功 dual episode 要么没有同推，要么走 push+reposition/其他。
- 失败中的同推优势为负（−0.24 / −0.15 / −0.56 normalized）。这会把同推与失败关联起来，而不是形成稳定的同推贪心。
- joint 的**成功经验主要是 push+reposition 和 other**，不是同推。任务标题里的“双推成功经验”在 final joint 策略上需要收窄：成功经验不是“双方同推”，而是“一推一换位/其他”。
- value bias（MC reward-to-go − value）很大：成功同推 batch 1001 平均 +7.21；失败同推 −8.94 / −3.76 / −8.03；成功 push+reposition 约 +5.7–7.1。value 对成功系统性偏低、对失败偏高；这会影响 GAE，但本轮没有把它当作独立无偏动作价值，也没有做 Adam/多 epoch 的因果追踪。

## B3. 梯度分解：没有看到“其他任务系统性压回同推”

在冻结权重上，对每批初始 PPO actor loss 分解为 dual / single / reposition / entropy；任务分量沿用完整批次的归一化优势和总 transition 分母，不分别标准化。critic 不混入。

| 批次 | dual 梯度范数（all） | other 梯度范数 | cos(dual, other) all | cos(dual, other) joint_types | L_dual | L_other |
|---|---:|---:|---:|---:|---:|---:|
| 1000 | 0.487 | 0.805 | +0.037 | −0.012 | +0.522 | −0.342 |
| 1001 | 0.608 | 0.342 | +0.001 | +0.001 | −0.367 | +0.236 |
| 1002 | 0.691 | 0.336 | −0.028 | +0.018 | −0.801 | +0.520 |
| 1003 | 0.723 | 0.295 | +0.005 | +0.002 | −0.308 | +0.252 |

- dual 与 other 的梯度余弦接近 0（all：−0.028 到 +0.037；body/base/joint 也接近 0），**没有系统性反向冲突**。dual 梯度在 3/4 批次比 other 大；没有“其他任务把双推梯度压没”的直接证据。
- `L_task` 的符号会随批次和优势- logp 相关性变化，不能只凭正负断言冲突。双推的 episode 份额、transition 份额和梯度份额不是同一个量；换面长失败占 44% transition，但梯度近似正交。
- 所有检查：old logp 重算最大误差 **1.43e-6**（<2e-5）；loss 重组误差最大 **8.94e-8**；分组梯度重组误差最大 **1.53e-8**；全部有限。4 批 `status=complete`，无异常。

### 同推的局部一阶方向（固定 20 个输入）

评测输入预先固定为：两个双推初始观测 + A 的 joint 贪心轨迹各前 10 个决策。原始 22 个状态中有 2 个是 reset 观测与首个决策的重复；按精确观测去重后为 **20** 个唯一输入。对 `log P(同向平移双推)` 求 actor 参数梯度，再与各 loss 梯度取负点积；正值表示普通梯度下降会一阶增加同推概率。

| 批次 | body | base_heads | joint_types | full |
|---|---:|---:|---:|---:|
| 1000 | +0.030 | +0.061 | −0.048 | +0.044 |
| 1001 | +0.226 | +0.149 | +0.204 | +0.579 |
| 1002 | +0.107 | −0.062 | +0.040 | +0.086 |
| 1003 | +0.008 | −0.461 | +0.080 | −0.373 |
| 均值 | +0.093 | −0.078 | +0.069 | +0.084 |

- `p_same_push` 均值 2.62%（几何均值 1.33%）；logP 均值 −4.32；梯度范数 body/base/joint/all = 2.72/6.38/5.67/8.96。
- 方向是**混合的**：full 3/4 正、1/4 负；base heads 2/4 负、均值为负；joint_types 3/4 正。没有一致、强到足以解释历史 40 轮更新方向的局部信号。
- 这是一阶局部量，不是 Adam、梯度裁剪、多 epoch、normalizer 更新后的真实变化，也不能追溯证明历史因果；没有明显方向时必须保留“不确定”。

## 对任务问题的回答

1. **冻结末轮策略是否仍能随机完成双推？** 是。joint 12/32，independent 24/32；贪心 0/4。末轮随机-贪心差距真实存在；训练累计 41/80、50/80 不能代表末轮。
2. **成功轨迹中的同推是否得到正优势？** 只在 joint 的一个成功 dual episode 的 3 个同推 transition 上为正（+4.03 normalized）；其余成功 dual episode 基本不选同推，成功经验是 push+reposition/其他。失败中的同推为负。**“双推成功经验”在 final joint 策略上不是同推经验**，不能据此期待稳定的同推贪心。
3. **双推与其他任务对同推概率的局部更新方向是否冲突？** 没有看到系统性冲突。dual/other 梯度余弦接近 0；同推局部一阶方向混合。换面占 44% transition，但梯度近似正交；不能把问题归因于“其他任务把同推压回去”。
4. **最强的证据指向哪一支？** joint 的**选择-执行不匹配**：同推被选中约 31–33 s，实际双方 push 只有 0.6 s / 4.0 s；大部分是一方 push、预接触/对齐或 peer wait。independent 在同样定义下有 30–35 s 实际双方 push，说明控制器本身可以在合适状态执行。joint 的同推选择落在未准备好的成对状态，或握手/接触没有形成。
   - 另一个独立问题是 independent 的贪心已进入目标（0.194 / 0.016 m）但未判稳；这是到达后停止/判稳/确定性解码问题，不是同推选择问题。

## 交回 GPT：建议的下一步

**优先做 joint 同推的固定动作/状态执行诊断**，不先做双推隔离训练、过采样或同容量对照：

- 取出 A 中 joint 选中同推的 80 个决策状态（或按根各若干），在同一状态固定执行同推动作，分别对 joint/independent 记录：`ready`、`contact_ready`、`joint_active`、`gap`、peer wait/switching、速度配对、实际双方 push 时间、接触/推进效果。
- 关键是区分：joint 选同推时对方是否已在同面/可接触；是 peer handshake 让一方等待，还是速度/接触几何不允许；以及同一状态用 independent 的动作是否可执行。D001 已经证明“选中但很少双方执行”，下一步是定位执行断点，而不是继续改信用分配。
- 同一状态固定动作只做离线执行诊断，不进入策略输入、不训练；如果固定动作能在这些状态执行，而 joint 的策略动作不能，再查动作概率/状态选择；如果固定动作也不能，查控制器握手/接触/速度。
- independent 的停止/判稳另开一轮（v15 stopping 诊断已有线索）；换面的 103 次 REPOSITION 尝试也另开一轮，先看尝试后为何失败。当前不建议 12 h 长训或直接全随机扩展。

## 偏离、预算与限制

- 首次 B 分析在 `gradient recombination` 的 tensor→scalar 转换处崩溃；92 条 episode 已完成且保留，修复后**只重跑分析，episode 重跑 0 条**。`manifest.analysis_retry` 记录。
- 实现期跑过 1 条 **0.4 s** 的走向探针（不是 90 s episode，不进入 92 条分析/预算）；正式 92 条随后运行。如实披露。
- `episode_seconds=45.4 s`，最终分析 `6.5 s`，合计 `52.0 s`；`wall_seconds_total≈694 s` 包含首次分析 bug、调试和重复分析空档（实现时间），不计入 15 min 仿真/分析预算。
- 全量 `pytest tests` 在 300 s 内未完成，已停止；使用 `tests/test_option_rl_v14.py`、`v15`、`v16` 共 **20 项聚焦测试**，全部通过。未跑完整测试套件，不用它冒充完整验证。
- 预检：两个模型的 4 个初始贪心动作与原 `greedy_coordination_001` 完全一致；GAE 终止隔离最大误差 0；每个决策联合概率和=1、非法对质量为 0、logp 重算 ≤1.43e-6；4 批梯度重组 ≤1.53e-8。
- `min_distance` / “进入目标未判稳”基于 5 Hz 记录；成功判据由环境 20 Hz 控制精确给出。两者分母不同。
- 同推概率/优势对全部合法 46×46 对求和；“同推”指双方同向平移 PUSH；同速是子集，不是预设成功必要条件。16 次重复只描述这两个根上的动作采样，不是 16 个场景/seed。
- 结果只覆盖 8 个固定开发场景、2 个双推根、4 个训练 seed（B）和 16 个动作 seed（A）；不外推到全随机、其他几何或“联合头原理上不会协同”。
- 初始状态残差相似只说明初始观测；D001 没有沿整条轨迹重复残差分析。80 个同推决策和 20 个梯度输入是采样状态，不能外推为整条轨迹不依赖几何。
- 没有修改 reward、成功判据、控制器、物理、mask、动作版本、normalizer 或 checkpoint；没有 `optimizer.step`。

## 原始产物

- [summary.json](../../../outputs/agent_exchange/D001_20261029_1644/summary.json)
- [manifest.json](../../../outputs/agent_exchange/D001_20261029_1644/manifest.json)
- [episode_metrics.csv](../../../outputs/agent_exchange/D001_20261029_1644/episode_metrics.csv)
- [decision_metrics.csv](../../../outputs/agent_exchange/D001_20261029_1644/decision_metrics.csv)
- [advantage_summary.csv](../../../outputs/agent_exchange/D001_20261029_1644/advantage_summary.csv)
- [same_push_gradient.json](../../../outputs/agent_exchange/D001_20261029_1644/same_push_gradient.json)
- [replay.html](../../../outputs/agent_exchange/D001_20261029_1644/replay.html)
- [source_hashes.json](../../../outputs/agent_exchange/D001_20261029_1644/source_hashes.json)
- [episodes/](../../../outputs/agent_exchange/D001_20261029_1644/episodes/)（92 个 NPZ + 92 个 JSON）

## 下一步状态

D001 执行完成；`CURRENT.md` 更新为 `ready_for_review`，下一位处理者 GPT。没有自动启动下一轮。
