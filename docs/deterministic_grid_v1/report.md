# 确定性 Grid 控制实验报告（B）

run_id `grid-20261012-194632`。契约见 `docs/deterministic_grid_v1/CONTRACT.md`；实现独立于旧
Grid，不复用旧检查点、编码器、信念或评分。正文 ≤1200 字，指标明细在运行目录。

## 阶段完成度

B0 契约/动力学/评分、B1 精确链路验证、B2 数据与点预测模型、B3 固定候选诊断、B4 解释规则
均完成，无 BLOCKED。A（连续平台收尾）另见 `docs/closeout_v1/`。

**B1 精确链路（全通过）**：两图 witness（empty 4 步、obstacle 9 步，只作环境验收）；状态
导出/恢复与候选顺序不改变真实结果；用“完美模型”从完整 rollout→终态→评分入口得到
J_hat=J_ref，regret 恰为 0；成功吸收（进入目标后只计进入那一步）；初态成功 J=10、0 步；
手算候选集覆盖“立即推”优于“等待”与“先移动换位再推”。

## 数据与单步模型

30,000 条去重 (map,state,joint) 转移；按状态键哈希分 train/val/test = 23791/3119/3090 转移、
11970/1564/1571 状态（80/10/10）；物体移动率 9.6%、机器人移动率 58%、整步回滚 1.1%、
含无效动作 55%。20 个评估起点取自 test 状态组，**0 个出现在训练状态组**。

MLP 2×256 ReLU，三头（物体 25 锚点、机器人 A/B 各 36 格），交叉熵，3 个种子。
test 单步：物体 0.967、机器人 0.936/0.939、完整状态 0.860、非法状态率 0.05。
**分组**：物体确实移动的样本完整状态准确率 0.53，no-op 样本 0.89。

## 三个研究问题

**1) 模型是否准确，尤其在推动/换位后？** 总体较准（完整状态 0.86），但关键转移明显更差：
物体移动样本 0.53 vs no-op 0.89。故“已学准”只对多数 no-op 成立，推动后的转移仍是弱项。

**2) 被选中方案是否比普通候选更偏乐观？** 是。所有 H/K 下 selected_bias 为正
（均值 +0.36…+1.25），且 selection_excess = selected_bias − candidate_mean_bias 为正并随 H
从 +1.03 升到 +2.70：高估集中在被选中的候选上，不是全体候选同等高估。精确模型对照行
bias/regret 恒为 0，说明该指标链路本身无偏。

**3) 固定 H 下 K 增大是否让实际选中收益下降？** 否。chosen_true_return 随 K 平或略升
（H=4 均值 −5.34→−5.21），候选集最佳收益升得更多（−5.15→−4.55），而 candidate_regret 随 K
上升（0.19→0.66）。即 K 增大提供了更好的候选，模型的**选择**没有跟上，而不是搜索越宽越差。

| H | K | model | selected_bias | selection_excess | candidate_regret | chosen_true_return | candidate_best | 选中成功率 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 16 | mean | +0.361 | +1.029 | 0.047 | −2.472 | −2.425 | 0.000 |
| 2 | 64 | mean | +0.567 | +1.538 | 0.392 | −3.417 | −3.025 | 0.006 |
| 4 | 64 | mean | +1.117 | +2.543 | 0.661 | −5.319 | −4.658 | 0.011 |
| 4 | 256 | mean | +1.250 | +2.700 | 0.664 | −5.214 | −4.550 | 0.017 |
| 任意 | 任意 | exact | 0.000 | 0.000 | 0.000 | =uniform | =chosen | 0.000 |

## 递归对照与证据界限

teacher-forced vs recursive（同一批 16 条固定序列，H=4）：递归完整状态错误从 0.19 升到 0.34，
teacher-forced 维持 0.12–0.19（三种子均值）；递归使误差增加约 0.15，但**不据此断言递归是唯一
原因**。

候选成功率上限很低：连精确候选的最佳成功率在 H≤4 也只有 0–5%，J 差异主要由步数与距离构成。
因此本表对“成功”的结论应标注**候选覆盖不足**，不把低成功率解释为模型好坏。20 个起点 × 多
候选顺序不是独立大样本，只作描述性机制证据，不宣称普遍规律或论文新颖性。本轮不要求低于
随机、不要求多尺度、不要求双机器人不可替代。

## 文件与复现

```bash
python experiments/deterministic_grid/run_env_checks.py --run-id <id>
python experiments/deterministic_grid/run_dataset.py --run-id <id>
python experiments/deterministic_grid/run_training.py --run-id <id>
python experiments/deterministic_grid/run_diagnostic.py --run-id <id>
```

源码 `src/coimagine/deterministic_grid/`；命令与全部指标见运行目录
`outputs/deterministic_grid_v1/grid-20261012-194632/`；两张图 `figures/fig_k_selection_bias.png`、
`figures/fig_k_true_return.png`；审查包 `deterministic_grid_review.zip`。

## 下一步（仅一项）

在同一冻结候选与评分下，把模型改成按物体是否移动分头的校准调整（或在 no-op 与推动样本上
重加权），复核 selection_excess 是否下降；若不变，则本简单确定系统不足以暴露该机制，回到连续平台。
