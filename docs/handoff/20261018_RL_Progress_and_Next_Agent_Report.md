# CoImagine 近期 RL 工作与下一位 Agent 交接报告

日期：2026-10-18。仓库绝对路径：`/home/tzl/repo/coimagine`。
本报告基于实际代码、运行日志、检查点和本轮定向检查；不把研究假设写成实验结论。

## 0. 一页结论：接手时先知道什么

1. 主线仍是**固定预测能力与预算下，更好组织联合技能预测、反馈和决策**。RL是平台与行为数据工具，不是论文主贡献。
2. 当前平台已具备随机初态、合法局部雷达、PUSH/REPOSITION/WAIT、单世界同步执行。最新奖励/预算为 **v3**。
3. 已修复critic时间错位、未实现TBPTT、局部RNG失效、静默缩减训练子集等问题；也改掉了两种奖励/预算激励漏洞。
4. 8实例长训练曾达到3/8确定性成功，但不稳定；最佳保留在更新1550。
5. 256实例从该最好模型热启动，完成4648次更新、37184条训练轨迹，**最佳仍为更新0**；没有稳定超过热启动模型。
6. **新确认的严重问题：validation与train使用相同种子规则，64个validation实例与前64个train实例完全重复。** 所有此前“独立验证/泛化”表述应撤回；拆分未修。
7. 重复短技能退出使失败轨迹占据大量决策训练项；这是有证据的分布现象，但尚未证明其为退化主因。
8. 本报告生成时两个v3训练PID均已不存在，当前没有这两个训练进程在运行；manifest可能滞后，不能只看其status。
9. 不建议直接开启第三轮同配置24小时训练。先修评估拆分、回放轨迹、量化梯度贡献、做小型表达能力诊断。

## 1. 研究定位与文档优先阅读

- 近期研究定位：`docs/20261013_1937_CoImagine_研究主线与验证原则_v1 (1).md`。
- 最近完整环境任务书：`docs/tasks/20261015_CoImagine_RandomStart_Reposition_VisualReview_v1.md`。
- 用户随后明确授权：奖励重写、统一90秒预算、24小时长训练、256实例最佳模型热启动。
  因而当前v3并不逐项沿用旧任务书的gamma=0.997、30技能预算或60更新限制。
- `AGENTS.md`、原2D执行提示/v6研究方案保留信息边界、证据与研究背景要求；早期“不训练”的范围已获后续用户授权扩展。
- 根README和大量旧STATUS条目滞后；不要从中推断当前不存在学习模块。

核心研究问题：减少模型规划的物理乐观和信息乐观。在同模型、同信息、同预算下比较决策组织方式。
当前尚未证明新的预测/规划算法有效，也没有完成泛化、分布式融合或论文核心创新验证。

## 2. 当前平台与信息边界

- L0：3.2×2.2 m无内部障碍；L1有固定中央障碍，本轮RL只用L0。
- 物体矩形1.0×0.25 m；两个机器人从物体附近合法随机位姿出发。
- 每机器人15个离散技能动作，共225有序联合动作。
  PUSH为偏角×速度9种；REPOSITION顺/逆时针×0.25/0.50m四种；WAIT 0.5/1.0s两种。
- 共享世界，控制周期0.05s。技能边界不重置位置或速度；每周期双方先计算控制，然后世界只推进一次。
- 合法策略输入：机器人自身/队友状态、目标、公开预算、接触与360°72射线、1.2m测距及可观测类别、扫描拟合法向。
- 物体真实全局位姿/速度、奖励、成功标记、快照不得作为策略/critic/GRU/控制器输入。
- 成功：中心距目标≤0.20m，线速≤0.02m/s，角速≤0.05rad/s，持续0.50s。
  不要求物体完全进入目标圆；当前尺寸也不允许其被半径0.2m圆完全覆盖。
- 当前324维输入**缺少显式过去动作/技能阶段历史**，是已知未修缺口。原skills_fraction列在v3改为reserved_zero，维数不变但语义已变。

## 3. 修复和版本演变

### 3.1 v1语义排查

实现主要在 `experiments/random_start_reposition_v1/rl.py`：
- rollout用决策前V，update却先消费当前技能内部帧再算V；修为同一决策前状态。
- 声称按技能截断梯度但未detach；现每次读完actor/critic头后detach，下一决策仍可训练中间帧。
- Categorical.sample未使用传入generator；改为显式generator的multinomial。
- bandit旧检查不是完整PPO路径，却被过度解释为训练正确；已移除默认bandit及相关函数，以真实update测试替代。
- 源成功列表实际8例，旧scan=32静默丢掉0038/0056/0063；修为显式完整性检查。
- 修复S4门槛只检查末次的错误：需最近三个连续评价点≥75%，且比初始化提升≥20pp。

固定两条真实失败轨迹、60个决策的critic-only对照：修复路径解释方差−0.094→0.889，旧路径也能达到0.876。
因此实现缺陷确实存在，但没有证据单独证明它们是零成功主因。

报告：`docs/random_start_rl_audit_v1/report.md`。

### 3.2 v2稠密奖励

旧势函数为负中心距离，终止设Phi=0；完整MC的折扣塑形沿轨迹相消。
同初态、同时长、同终止事件的失败轨迹不能按最终距离区分。gamma=0.997还使延迟失败可能更划算。

用户提出并授权混合稠密奖励，现每周期：

```
r = (d_before - d_after) - 0.02 * d_after * 0.05 - 0.002 + event
event = +10(success), -2(budget failure), 0(other periods)
gamma = 1
```

真实终点距离保留，无吸收态补偿；事件只发一次，成功与预算同周期时成功优先。
关闭熵奖金（0.01→0）和AdamW隐含weight_decay（显式0），不再额外推动均匀分布或参数范数正则化。
保留PPO clip0.2、value权重0.5、梯度裁剪5和跨轨迹批次优势归一化。
这同时改变奖励与部分正则化，**不能把后续变化全部归因于奖励一项**。

报告：`docs/random_start_dense_reward_v2.md`。

### 3.3 v3统一物理预算（当前版本）

v2保留30技能上限：快速受阻可在远小于90秒时失败，少付时间/距离成本，形成另一种漏洞。
用户授权修正，v3删除30技能上限：普通失败只能在90秒真实物理期限发生。
普通技能退出不是episode退出；模拟异常单列并停止训练。

- 每个正常失败episode均1800周期，时间成本固定−3.6；成功更早完成才减少时间成本。
- 对照：1800次短技能与164次正常WAIT均跑90秒，总回报一致，独立日志复算差0。
- 未加剩余时间补偿、奖励bonus或隐式动作上限。
- 最多1800次决策是每技能至少推进一个周期的自然上限，非新增终止条件。
- 删除EnvConfig.skill_budget，旧参数调用报错；恢复检查点与分支有版本检查。

报告：`docs/random_start_time_budget_v3.md`。

## 4. 训练设置与实际运行

共同设置：MLP编码器＋GRU128＋条件联合动作头p(a0|h)p(a1|h,a0)＋价值头；actor/critic共享编码器和GRU。
完整MC回报减V，不用GAE；3个PPO epoch；AdamW lr3e-4；每批8条完整episode。
8独立采样进程，CPU单线程/进程，一个同步学习器；整批策略快照不变，按提交顺序合批。

### A. v1修复后有界重训

`outputs/random_start_rl_retrain_v1/`：45分钟到期，74/120更新、592条训练轨迹。
更新70恢复检查点；最新评价确定性/随机均0；未过S4。

### B. v2长训练（因预算漏洞人工停止）

`outputs/random_start_dense_reward_v2_long/`。
回报提高主要来自减少时间和距离成本，而非稳定成功；随后停止切换v3。
子项分析目录 `reward_breakdown/`：进展/时间/终局项由已有指标还原，距离成本是余额推算，
不是独立物理日志验证；不能混淆。

### C. v3固定8实例长训练（人工停止并转入256例）

`outputs/random_start_time_budget_v3_long/`。
- 实际日志/monitor截至更新4567，恢复检查点4565。
- 最佳模型更新1550：8例确定性3/8，最终距离中位数0.393m；随机执行5/24。
- 中期训练成功率约20.9%，后续明显退化、偶尔恢复。
- 用户授权停止，改用256实例；旧最佳权重未丢失。
- 当前manifest仍写running但PID414838不存在；这是状态落盘不一致，不是仍在训练。

### D. v3覆盖256实例（最新）

`outputs/random_start_time_budget_v3_coverage256/`。
- 从C的最佳1550权重热启动；best.pt没有优化器，因此明确新建AdamW。
- 每轮打乱256训练实例，每批8个不重复，32更新覆盖一轮；不再有放回抽8例。
- 原8例跟踪；固定32训练面板；标记为validation的64例；每50更新评价。
- 按64例“验证”确定性成功率、再按距离中位数选最佳，初始化也参与候选。
- 实际完成 **4648更新、37184训练episode**，monitor=time_budget_reached，恢复检查点4648。
- 训练PID691396不存在；manifest仍为running，没有可靠最终落盘，因此使用检查点/monitor/进程事实共同判断。
- **最佳仍是更新0**，即热启动模型；没有稳定超过原模型。

| 评价面板 | 初始化 | 最新完整评价4600 |
|---|---:|---:|
| 原8例确定性成功率 | 37.5% | 12.5% |
| 原8例随机成功率 | 20.8% | 16.7% |
| 固定32训练例确定性 | 0% | 3.1% |
| 标记validation64的确定性 | 17.2% | 7.8% |

最新200更新训练平均回报−6.60、采样成功7.06%；中间有退化和恢复，不能仅以末次指标概括全部曲线。

## 5. 新确认问题：场景拆分不独立（优先修）

文件：`src/coimagine/random_start/layout.py:build_split`。
所有split使用相同地图base_seed，再加index；split只改变名称/元数据。
本轮实际重新生成train256和validation64，比较除name/split外完整as_dict：**64/64相同**。
首个train/validation seed均20261015。

影响：
- 本轮验证集实为前64个训练实例，不能声称独立验证或泛化。
- 旧8例也大部分位于这些64例之中，原validation成功不能当泛化证据。
- 此问题不直接解释策略退化，但使评估解释和模型选择标签不成立。
- 代码对test也未分配独立种子空间；本轮没有读取/评价test，修复时应一并设计隔离。
- 不要静默改旧实例/结果。保留训练面板，生成新版本的独立validation/test并检测完整实例内容去重。
- 仅检测实例名称不同不足以检测此问题。

## 6. 证据支持的现象与尚未隔离的原因

### 已有证据

- 新训练没有稳定超过热启动模型，不能再只用“训练量不足”解释。
- 原有成功行为在后续更新中丢失：热启动8例37.5%，100更新降至0%。
- 最新200更新：113条成功、1487条失败；平均技能数成功约74、失败约293。
  成功episode占7.06%，但平均每批仅占2.63%的决策训练项；失败项占97.37%。
- 当前损失按全部决策平均，优势亦按决策统一归一化，所以重复短技能对训练分布影响大。
- 该占比是决策项数量，不是已测得的梯度贡献百分比。
- 更新1001–2000采样成功约1%，critic同批EV约0.90；高EV可以只是准确预测稳定失败。
- 探索分布曾收缩：更新501–1000记录的平均熵约0.51；熵日志为条件头的采样条件统计，不是额外奖励。

### 假设，不能直接当结论

- 重复失败决策可能稀释有效成功经验；需分解实际损失/梯度，不能仅凭数量推因果。
- actor/critic共享表示可能发生干扰；尚未独立测actor、critic梯度及动作分布变化。
- 完整MC、罕见成功、变时长技能与小批次造成高方差信用分配。
- 缺动作/阶段历史及技能级TBPTT可能限制多技能记忆。
- 低熵可能限制探索，但不能未经对照立即把熵奖励加回去。
- 热启动分布切换＋优化器重置也可能造成遗忘；当前实验不是单因素对照。

## 7. 推荐下一步：不要直接再跑同配置长训练

按信息量优先：

1. **先修split独立性**，版本化新评估面板，保留旧结果，添加内容去重测试。
2. **回放少量固定实例**：比较最佳1550、退化与恢复检查点；看短技能循环、换位、方向、目标保持失败。
3. **固定轨迹梯度审计**：分别量化成功/失败、actor/critic贡献；检查一次更新后已有成功动作概率如何变化。
   再决定是否做按轨迹聚合或分层采样决策的单因素实验，而不是先假定等权一定正确。
4. **小型表达能力诊断**：使用现有合法成功轨迹检查同网络能否拟合动作并闭环复现。
   这应标为诊断，不伪装成新增方法优势；不读取特权状态输入。
5. 有证据后只调整一项（输入历史、损失权重或优化稳定性），再做有界对照。

用户讨论过50option上限，但未授权实施；不要重新引入会让提前失败少付成本的终止出口。
当前没有新一轮长训练授权，也不要自动扩展S5采集/世界模型/规划器。

## 8. 代码入口与文件定位

所有以下路径均相对 `/home/tzl/repo/coimagine/`：

| 路径 | 用途 |
|---|---|
| `src/coimagine/random_start/spec.py` | v3常数、动作、版本 |
| `src/coimagine/random_start/env.py` | 同步执行、真实周期奖励、终止、白名单输入与分支恢复 |
| `src/coimagine/random_start/layout.py` | 场景采样，split种子问题在此 |
| `src/coimagine/random_start/skills.py` | 合法局部控制及退出 |
| `experiments/random_start_reposition_v1/rl.py` | GRU、rollout、MC target、PPO update |
| `experiments/random_start_reposition_v1/long_train.py` | 8进程采样、均衡256采样、评价、恢复、监督进程 |
| `experiments/random_start_reposition_v1/reward_breakdown.py` | 离线v2均值回报核算；不直接适用于256打乱采样规则 |
| `tests/test_random_start.py` | 环境、预算、传感和奖励 |
| `tests/test_random_start_rl.py` | critic时间、梯度、RNG、reward和门槛 |
| `tests/test_random_start_parallel.py` | 并行等价、恢复、打乱遍历覆盖 |

最新运行完整地址：

```
/home/tzl/repo/coimagine/outputs/random_start_time_budget_v3_coverage256/README.md
/home/tzl/repo/coimagine/outputs/random_start_time_budget_v3_coverage256/curves.png
/home/tzl/repo/coimagine/outputs/random_start_time_budget_v3_coverage256/updates.jsonl
/home/tzl/repo/coimagine/outputs/random_start_time_budget_v3_coverage256/monitor.json
/home/tzl/repo/coimagine/outputs/random_start_time_budget_v3_coverage256/initial.json
/home/tzl/repo/coimagine/outputs/random_start_time_budget_v3_coverage256/checkpoints/best.pt
/home/tzl/repo/coimagine/outputs/random_start_time_budget_v3_coverage256/checkpoints/resume.pt
```

原最佳模型：
`/home/tzl/repo/coimagine/outputs/random_start_time_budget_v3_long/checkpoints/update_01550.pt`。
各运行目录有 `code/` 源码快照、`manifest.json`源哈希和配置、`requirements-freeze.txt`。
best.pt/每点评价.pt为模型权重；resume.pt为含优化器的完整恢复状态，勿混用。

## 9. 测试、工作树与接手注意

- 最近启动256例前定向测试 **31 passed**；没有宣称全量测试通过。
- 旧平台阶段曾报全量496 passed/9 failed（历史产物缺失）；这是过去记录，不是当前复验结果。
- 当前工作树有大量本会话修改与未跟踪文件，尚未替用户提交；不要清理/覆盖这些文件。
- 快速检查：
  `.venv/bin/python -m pytest tests/test_random_start.py tests/test_random_start_rl.py tests/test_random_start_parallel.py`
- 本次交接只读取运行状态/检查点并写报告，不新增训练、模拟或改变训练参数。
- 报告数字是读取时快照；若核对新结果，以逐更新日志/检查点/真实进程为依据，不只看滞后的manifest。

**交接要点：环境基础已经可用，也曾学到部分成功，但优化不稳定；评估拆分又暴露实质缺陷。
下一步应增加诊断可信度，而非继续堆训练量或新增复杂模块。**
