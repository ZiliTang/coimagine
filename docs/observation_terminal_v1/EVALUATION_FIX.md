# 评价修正（评审后，未重训）

评审指出本轮评价里有两个具体错误。**两个都成立**，本文记录复现、修复与影响。
修复后**复用三个既有检查点重算**（`reevaluate.py`，不重训）。
新旧对比表：`terminal_report_train.json`（旧，错误）→ `terminal_report_reeval.json`（新，正确）。

## 错误 1：P1 的连续量单位错了；P0 的缺失观测回退也受影响

`_standardise` 把 `y` 换成标准化数组、原文留在 `y_raw`。`baselines.py` 却从 `y` 计算：

- P1 的**每个**命令组统计（几何、位移、速度、时长中位数与法向均值）；
- P0 在**起始片段不可见**时的逐列中位数回退。

结果是把标准化后的数字（≈0）当成米、秒交给评价器，与原始单位的 `y_raw` 比较。
复现（评审给的最小例子）：train 时长 3 s 与 5 s，正确中位数 4 s，旧函数返回标准化的 0，
再被当作 0 s 评价。

**修复**：`baselines.raw_block()` 统一从 `y_raw` 取标签，所有组内统计量都在原始单位计算。
新增手算测试 `test_command_grouped_baseline_statistics_are_in_original_units`
（3 s/5 s → 必须得 4 s）与 `test_persistence_fallback_median_is_in_original_units`。

**影响**（test，中位误差）：

| P1 | 旧（错） | 新（正确） |
|---|---:|---:|
| 片段中点二维 (m) | 0.6323 | **0.0960** |
| 法向角 (°) | 21.49 | **3.359** |
| 位移 dx (m) | 1.0327 | **0.0647** |
| 时长 (s) | 3.8472 | **0.3000** |

P0 主要几何比较未变（机器人 0 起始可见），只有回退项受影响：
P0 的 `r1_mid_2d` p90 **0.5968 → 0.5859**、max 1.010 → 0.878（中位不变 0.1468）。

## 错误 2：二值基线被当成 logits

网络二值头用 `binary_cross_entropy_with_logits` → 输出 **logits**；P0/P1 输出 **0/1**。
评价器对两者都做 sigmoid 再 `≥0.5` 判正 → `sigmoid(0)=0.5` 也算正，**0 和 1 全被判成正类**。

P0 的 `r0_contacting` 旧数字：accuracy 0.215、confusion `[[0,402],[0,110]]`、recall 1.0
——即"全部报正"。修复后：**predicted_positives = 0、recall 0.0、accuracy 0.785**
（P0 复制起始 `contacting`，而技能开始时尚未接触，所以它正确地从不报接触）。

**修复**：`evaluate_terminal.coerce_binary()` 要求显式声明类型
（`BINARY_PROBABILITY` 或 `BINARY_LOGIT`），并对声明为概率的输入**校验取值范围**，
越界直接报错而不是静默转换。类别目标另行区分：1 维 = 类别索引，2 维 = 逐类分数。
基线调用点传 `probability`，网络调用点传 `logit`。
修复后的二值对比（test）：

| r0_contacting | 预测为正 | 真值正例 | 召回 |
|---|---:|---:|---:|
| P0 | 0 | 110 | 0.00 |
| P1 | 46 | 110 | 0.14 |
| **模型（三种子）** | 96–108 | 110 | **0.84–0.92** |

## 为什么原有测试没拦住

原来的"单位一致性"测试只比较**真值减真值**（必然为 0），没有经过
"标准化 → 基线生成 → 评价"整链。已补三项：
`test_full_chain_baseline_evaluation_is_not_degenerate`（真数据集上跑完整链，
断言时长基线必须是秒、二值判正数必须等于基线自己 0/1 里的 1 的个数）、
`test_binary_predictions_must_declare_probability_or_logit`、以及上面两个手算样例。

## 复现

```bash
.venv/bin/python -m pytest tests/test_observation_terminal_v1.py -q   # 12 项
.venv/bin/python experiments/observation_terminal_v1/reevaluate.py     # 复用检查点重算
```
