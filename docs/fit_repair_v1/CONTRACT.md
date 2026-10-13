# CONTRACT — fit_repair_v1（训练协议修正与预测失效定位）

任务书 `docs/tasks/20261013_0157_CoImagine_训练协议修正与预测失效定位任务书.md`。
本文件在正式训练前冻结。优先修 `src/coimagine/fit_capacity/`、`experiments/fit_capacity/`；
不修改连续环境、数据采集器、Grid。新输出写 `outputs/fit_repair_v1/<run_id>/`，
旧 `fit_capacity_v1` / `data_upgrade_v1` 输出与模型全部保留。

本轮回答：修正训练实现后，加宽/充分训练的收益是否仍成立；不同检查点怎样影响候选选择；
终止分类退化和几何非法分别表现在哪里。**不采用强制晚停新规则，不新增数据/损失/模型类型/宽度，
不做闭环、多尺度、几何投影、分布式融合。** 真值替换只用于离线定位，绝不进入预测器或规划器。

## 0. 冻结项

- 数据：`data_upgrade_v1` 的原 train/val/test、matched Random/Branch、分支 val/test；
  整场景划分与样本顺序语义不变。两增强训练组各 3684 窗口（以数据实测核对）。
- 物理环境、场景版本、2 s 窗口、控制器、输入/输出语义、编码/解码、角度残差、几何阈值、
  终止优先级、评分 J、候选集合与真实收益参照全部冻结。
- 训练矩阵：Old+Random / Old+Branch × 2×256 / 2×512 ReLU MLP × seeds 0/1/2；
  Adam lr=0.001、batch=128、6000 updates、无 early stopping、无 scheduler/weight decay/重加权；
  四项原损失逐样本和，再对 batch 取均值。固定 CPU 4 线程、单训练进程。

## 1. 阶段 A：只修已确认的实现与统计问题

### A1 batch loss 归约

```python
per_sample = torch.stack(components, dim=0).sum(dim=0)
loss = per_sample.mean()
```
四项系数、state 内部维度平均、wrapped 角残差不变。训练日志的四项与 total 全部用逐样本均值，
并记录 batch 实际大小。必须与 `data_upgrade.training.per_sample_loss(...).mean()` 在相同输入上
数值一致。测试：相同参数/同批数据下标量 loss 与**参数梯度**与旧正确函数一致；
同批样本重复两遍，均值 loss 与梯度不变。不再用“手工均值 × 样本数”证明错误的求和实现正确。
说明：Adam 对整体梯度缩放有抵消，不能把旧 bug 解释为学习率放大 128 倍；末批大小不同也不能
认定两者完全等价。

### A2 在对应 update 保存中间模型

到达指定 update 时立即保存独立快照（或立即复制参数、稍后写盘），禁止引用后续会被修改的
tensor。两个 best 保留独立参数复制，final6000 单独保存。checkpoint 内记录实际 update、配置、
**参数内容 hash** 与数据内容 hash；文件 hash 与参数内容 hash 分开。
测试：缩小网络与步数、checkpoint 时刻可配置；重新加载早期 checkpoint 的参数/预测必须等于该
时刻即时捕获值，最终 checkpoint 等于最终值；早期 checkpoint 重测物理指标应与即时日志一致。

### A3 恢复功能必须真的能恢复

每个固定评估点**原子保存**恢复状态（临时文件 + rename），新增 resume 入口；保存模型、Adam、
数据 RNG、当前 permutation、cursor、updates/seen_samples、best 状态与分数、必要运行 RNG。
恢复前核对完整数据内容与配置 hash。测试：连续跑到 N 步，与跑到非 epoch 边界 M 步后恢复到 N 步，
比较最终权重、优化器步数、样本顺序与 best 选择；固定 CPU 无随机层下要求一致。

### A4 候选比较按共同样本配对

按 `(start_id, candidate_seed)` 配对，只在**两个模型都有可行候选**的行上计算收益差与损失差，
先按起点平均、再跨起点平均。不同 checkpoint 比较同一方法。输出完整样本的拒绝/无解比例与
共同子集上的收益差；uniform 与 candidate_best 也在同一子集重算，不把删去的难例藏起来。

### A5 最小报告更正（只用旧原始记录）

- 训练 loss 下降是 11/12 而非 12/12；
- 仅可行 chosen_true −0.252 与 −0.166 **高于** uniform −0.361；
- 较早模型不称“近初始化”；
- 原 test 无真实成功 → 召回 `null`，不是 0；
- “物理最优 checkpoint”只指实际比较范围与明确指标（旧脚本仅在某几个保存点上按两套 val
  物体位置误差之和比较），不据两个选点不同断言严重冲突，报告指标差值即可。

A 测试通过后才进入正式新训练。同一实现问题最多两次有依据的修复；仍不通过则交付 BLOCKED 与证据。

## 2. 阶段 B：正确协议下重新运行 12 次训练

新目录 `outputs/fit_repair_v1/<run_id>/`；全部从相同种子的新初始化开始，不接续旧批求和训练；
复用已有数据，不重新采样。记录所有原始数据文件的**完整 SHA256**。
同宽同 seed 两数据组初始参数相同；同数据同 seed 两宽度使用相同 minibatch 顺序。
独立初始化 RNG 与数据 RNG；对整个训练样本索引流滚动计 hash（不只看前 64 批）。

评估时间：update 0、100，以及每 250 次至 6000（去重，即含 250、500、750）；每次在完整 train、
原 val、分支 val 上评估统一指标。训练 batch loss 与 checkpoint 全集 eval loss 分开存档。

保留 checkpoint：`fixed_1000`、`fixed_3000`、`fixed_6000`；`best_mixed_val`
（最小 `0.5×原val总loss + 0.5×分支val总loss`）；`best_original_val`（最小原 val 总 loss，次级规则）。
best 排除 update=0，并列保留较早者；**禁止**增加“至少训练多少步才允许选”的限制。
全部训练完成后冻结模型清单，才运行 test 与候选评估。

统一指标沿用上轮：四项 loss、各体位置误差与物体角度/速度误差、原始预测非法率与违规类型、
终止混淆矩阵与成功 precision/recall（空分母 `null`）、persistence 对照；列窗口微平均、
场景等权平均与样本数。输出同预算/同种子的进度表、256/512 配对表、三种选择规则表。
旧有偏差结果只作历史附件，不与新结果混算种子均值。

## 3. 阶段 C：两项限定诊断

### C1 终止类别为什么恶化（观察证据，不改训练）

对 12 run × 三个 checkpoint（mixed/original/final6000），在 train、原 val、分支 val 上增加：
1. 真实各类别数量、平均 CE、该类真实标签概率均值、各类对总 CE 的贡献（按全数据分母计算，
   相加恢复总体 CE）；
2. 成功 precision/recall、预测成功数、真实成功数；无真实物理失败标“未验证”；
3. 对真实成功窗口预测 `window_complete` 的比例与真实 success 概率；对真实普通窗口预测 success 的比例；
4. 按 S0/S1 分组；不新增阈值、ECE 分箱或损失权重搜索。

只能称观测归因，不宣称终止头造成位置错误或多头梯度竞争已被证实。分支 val 的 CE 恶化不称
mixed 规则“失效”，说明它在状态回归与终止判断间的取舍；注明原 val 无成功样本是原 val-only
选择的实际盲点。

### C2 几何非法主要与哪部分误差相关（限定离线替换）

固定两个 run：**Old+Random/256/seed0** 与 **Old+Branch/256/seed0**（看结果前确定，不挑种子），
各取 mixed/original/final6000，在自身 train 与分支 val 上诊断。

先重用原预测：统计各违规 pair 的发生比例、最大穿透分布，以及对应物体/机器人位置与角度误差
（一个样本可有多种违规，比例不必相加为 1）。

再做三种独立反事实，只调用冻结几何检查，速度与终止类别保持原值：
仅替换物体 pose、仅替换两机器人 pose、三刚体 pose 全部替换（检查参照）。
分别报告原非法→合法、合法→非法、总非法率。全 pose 真值参照若仍明显非法，先检查时刻/场景/解码，
停止该项归因。这些混合状态是离线敏感性实验，不称可部署提升，不送入 rollout，不据 test 调阈值；
若替换相互依赖，报告“联合误差”，不指定唯一责任头。

## 4. 阶段 D：补齐检查点的候选选择对照

- **D1**：12 runs × 三种 checkpoint（mixed/original/final6000）在冻结原 test 与分支 test 复测统一指标；
  相同参数内容可缓存但保留三个逻辑标签。既有 test 只称固定留出回归集。
- **D2**：同上 36 个逻辑 checkpoint，全部沿用上一轮 12 起点、candidate seeds 0/1/2、H=3、K=128
  的既有候选与真实参照；核对初态/候选序列/评分 hash，复用精确收益，不新增 H/K/起点。
  逐行保存 `(group,width,model_seed,selection,checkpoint_hash,start_id,candidate_seed)` 与实际 K。
  两套协议：未筛选（chosen_true、uniform、candidate_best、selected_bias、selection_excess、regret）
  与执行筛选（拒绝率、无可行候选比例、可行选择真实收益、误拒损失、集合内排序损失及和）。
  无可行行的选择收益/损失填 `null`；无解率同时列“pool 行数/全部 pool 行数”与
  “三个候选种子全无解的起点数/12”。同 run 的 mixed→original、mixed→final 用共同样本配对；
  256→512 只在同数据组/模型种子/选择规则下配对；列 Δ真实收益、Δ拒绝率、Δ两类损失及分母；
  有解↔无解的变化单独计数。精确对照 bias/regret=0；逐行检查误拒+排序=执行 regret。

本轮不据 D 的收益重选 checkpoint，不把事后最优选点当正式算法，不运行闭环。

## 5. 资源、失败处理与交付

新训练累计预算 60 分钟、候选评估累计 60 分钟；按完成批次实际计时外推（禁止把整个 run 耗时
当 500 updates 耗时）。日志与结果按 run/checkpoint 增量保存，可跳过已完成且 hash 一致的项。
预算耗尽记 `partial` 与恢复命令，不以“全部完成”概括。

测试范围：A 的必要测试、指标分母/null、配对共同子集、两类 regret 恒等式、既有非 slow 测试。
禁止改测试期望掩盖错误，不扩跑 slow 环境实验。

交付：`docs/fit_repair_v1/{CONTRACT,STATUS,report,errata}.md` + manifest；
12 run 的配置/训练轨迹/checkpoint 索引与新数据 hash 保护记录；终止分组表、几何替换表、
留出指标表、逐候选池选择结果及配对汇总；最多三张图（训练/验证位置与非法率；
状态 loss 与终止 CE；三种选模的真实候选收益与拒绝率，不混分母）；
`fit_repair_v1_review.zip`（修改源码、测试、配置、表格、完整评估日志、36 个逻辑选点对应模型，
重复权重只存一份并映射；大原始数据在 `MISSING.md` 列路径/hash/恢复命令）。
正文 ≤1500 中文字。
