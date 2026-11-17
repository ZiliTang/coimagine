# CAP检索与程序路由基线 v1：实现及实验接口

日期：2026-11-17。对应[研究计划v5](CAP_Integration_Mechanisms_Research_Plan_v5_20261117.md)。当前为并行基础设施建设与工程验收，不改变正在采样的GDA扩数据实验，也不产生正式架构优劣结论。

## 已实现的两种选择方式

| 配置 | 索引证据 | 选择频率 | 默认行为 |
|---|---|---|---|
| R0 / initial | 开发初态与各CAP完整执行成败表 | 初态一次 | 最近初态中的成功CAP，之后持续闭环运行 |
| Rt / boundary | 各CAP成功轨迹的中间决策边界 | 每个option边界 | 最近成功前缀所属CAP输出当前动作 |

默认k=1、固定库序破平。R0可扩为k近邻加权已观测成功率，未测标签保持未知；附近无成功证据时回退到全索引开发成功率最高的CAP，不在线试执行。Rt的k>1使用成功前缀加权投票，**不是**将未尝试CAP标为失败后的成功率估计。失败记录仍在原始数据和R0标签表中。

每条成功轨迹均匀取最多16个边界，包含初始和最后一个动作前边界，不包含完成后的终态。这样控制单轨迹的索引规模，但不声称已消除成功轨迹数量、状态密度或CAP覆盖不均造成的全部偏差。

所有原始结果来自显式指定、已封印的开发采集目录；构建器拒绝验证和测试split，输出保存输入路径/哈希、选定初态、边界和来源。索引只读，修改数据需另建目录。完整实验应匹配GDA经验池，不能把索引多用了成功轨迹或成败标签的收益归因于路由架构。

## 表示与信息权限

复用G的公开实体/扫描几何编码，不调用CAP来构建检索特征。实体排列、目标相对坐标与缺失值规则沿用现有编码。

- `snapshot`：当前几何向量；包括最后看见的实体缓存、上一实际动作和初始目标相对坐标系。因此是“当前紧凑状态估计”，不是严格无记忆图像。
- `history`：与G相同的初始向量＋最近8边界表示。

距离为训练索引内标准化后均方欧氏距离；缺失值用均值填充并添加有效性掩码。均值和尺度只由索引拟合，尺度下限0.1，查询不更新统计。精确近邻分块计算，固定顺序破平。距离与k是显式基线配置，不宣称最优，应在开发/验证集选定后再测独立留出。

运行时只接收合法观测—动作历史；场景ID与真实仿真状态不输入距离函数。训练场景ID仅保存在来源追踪和日志中。

## 接管和执行接口

新模块：

- `experiments/planar_tasks/semantic_rl/program_retrieval.py`：索引、几何编码、路由器契约与执行适配器。
- `experiments/planar_tasks/authoring_v2/program_retrieval_baselines.py`：索引构建与有界执行CLI。

`RoutedFeatures`接入既有`rollout(features=..., model=None)`，复用控制器、合法性、动作尺度和任务预算。没有修改现有runner或GDA运行模块。

每个CAP在每个实际边界都更新记忆，即使没有被选择；更新接收真实已执行动作。选中CAP的第一个建议若非法，按单CAP相同规则记录失败，不偷偷尝试其他CAP。支持真实前缀重放，跳过历史边界会显式报错。

每次选择记录：近邻及距离、CAP得分/投票、选中程序、动作、切换标志、公共历史哈希，以及记忆/表示准备、检索与建议输出耗时。回放结构保持既有格式，并增加`routing_trace`、`routing_summary`；旧UI是否展示新增字段需单独接入，不能将保存日志等同于已完成可视化。

当前为公平记忆维护运行全部CAP的状态更新，不以减少专家计算作为本版收益。R0只执行选中CAP的轻量优化、延迟唤醒等应在同值验收后另行处理。

## 学习式路由接口

`Router.select(vector)`返回CAP名称及证据；同时提供固定库顺序、表示版本、CAP源码/动作协议封印。检索与未来学习式路由共享这一契约，执行器不变。

**LR的接口已预留，训练目标、训练器和正式LR实验尚未实现。**后续训练必须区分初态成功标签、自己成功轨迹前缀、跨CAP真实接管标签，以及未知结果；不把每条成功轨迹的作者身份强制视为唯一正确专家。

F的建议并集约束消融也尚未实施。当前基础设施完成不代表v5全部对照已建成。

## 命令示例

```bash
# 示例：先用旧开发集的8个分层初态验收，不读取正在写入的采样目录。
PYTHONPATH=. .venv/bin/python -B -m experiments.planar_tasks.authoring_v2.program_retrieval_baselines build \
  --rows outputs/semantic_rl/random_room_coverage_20261117_02/collection/result.json \
  --out outputs/research/program_retrieval_20261117_01/index --max-cases 8

PYTHONPATH=. .venv/bin/python -B -m experiments.planar_tasks.authoring_v2.program_retrieval_baselines run \
  --index outputs/research/program_retrieval_20261117_01/index/initial --mode initial \
  --cases outputs/semantic_rl/random_room_coverage_20261117_02/parent/development.json \
  --case-id newdev_2_002 --out outputs/research/program_retrieval_example_r0
```

Rt将index子目录和mode分别改为`boundary`；历史匹配版构建时指定`--representation history`。正式数据规模、距离/k、教师经验池与模型预算需另行冻结；本文件不自动调参或发起全场景比较。

## 本次验收范围

先运行针对性测试：未知标签、无成功近邻回退、成功前缀投票、缺失表示、文件封印、拒绝测试数据、边界采样、R0锁定及Rt切换时实际历史更新。再用既有开发数据构建小索引，做少量完整任务与同历史建议同值验收。

工程运行最多4个完整执行，每例1800ticks/300决策/180秒墙钟，串行；新物理预算至多7200ticks，不计入GDA扩数据的7800正式执行。索引构建、历史重算不执行仿真，不调用LLM。报告包含正负结果与不完整状态，不因失败自动扩大任务或调参。

后续正式比较先在同经验来源下测R0/Rt/F，再加LR与表示匹配消融。当前GDA扩数据结果仍按原封印独立报告。
