# AGENTS.md — DeepSeek 执行侧长期交接协议

版本：v1（2026-10-29，Asia/Shanghai）
适用：本仓库中由用户人工中转的 **GPT 规划 / DeepSeek 执行** 工作流。
权威顺序：根目录 `AGENTS.md` > 本文件 > `docs/agent_exchange/README.md` > `CURRENT.md` > 任务书。
本文件不授权自动运行、自动串联下一轮、agent 间直接通信，或扩大研究范围。

## 1. 这个文件解决什么

新会话没有上一轮的对话记忆。只靠聊天记录会丢失：当前做到哪、谁负责下一步、哪些文件是冻结输入、预算和回报格式是什么。因此把交接协议固化为仓库文件，并在根 `AGENTS.md` 留指针。任何新会话按第 2 节顺序读取即可恢复工作状态。

## 2. 新会话读取顺序（固定）

1. 根 `AGENTS.md`：仓库范围、禁止事项、权威文档。
2. 本文件：执行侧协议、preflight、回报和状态规则。
3. `docs/agent_exchange/README.md`：目录约定、状态机、文件命名。
4. `docs/agent_exchange/CURRENT.md`：**唯一当前指针**；读任务、状态、下一位处理者。
5. `CURRENT.md` 指向的任务书（`tasks/Dxxx_*.md`）。
6. 已有执行回报（`results/Dxxx_*.md`）和 GPT 复核（`reviews/Dxxx_*.md`），如果存在。
7. 任务书列出的源报告、checkpoint、panel、源码快照。

不要从聊天记录、git log 或 `outputs/` 的最新目录猜当前任务；以 `CURRENT.md` 为准。
不要用 git HEAD 代表当前实现：工作区可能大量未提交，必须记录 HEAD + dirty 状态 + 文件 SHA256/源码快照。

新会话开始时的只读检查：

```bash
pwd
git status --short
sed -n '1,120p' AGENTS.md
sed -n '1,200p' docs/agent_exchange/AGENTS.md
sed -n '1,160p' docs/agent_exchange/README.md
sed -n '1,200p' docs/agent_exchange/CURRENT.md
ps aux | grep -E 'option_rl|train.py' | grep -v grep
```

除非 `CURRENT.md` 明确写 `running` 且用户确认，否则不启动训练、rollout 或长任务。

## 3. 角色与权限

| 角色 | 负责 | 不负责 |
|---|---|---|
| 用户 | 决定方向、移交任务、批准预算、选择下一轮 | 不代替 agent 伪造结果 |
| GPT | 读证据、提出可证伪问题、写有界任务书、复核结果、写 review | 不直接运行 DeepSeek 的实现；任务书不等于已执行 |
| DeepSeek | 按任务书实现/测试/运行、保留原始产物、写执行回报、更新 `CURRENT.md` | 不自行改研究问题、不自动串联下一轮、不扩大预算、不替 GPT 写复核结论 |

**只有用户实际移交后才进入 `running`。** GPT 写出任务书、`CURRENT.md` 写 `ready_for_handoff`，都不代表 DeepSeek 已收到或已开始。

## 4. 状态机与唯一指针

`ready_for_handoff → running → ready_for_review → reviewed`

- `ready_for_handoff`：GPT 已完成任务书；等待用户移交。
- `running`：用户已移交，DeepSeek 正在执行；`CURRENT.md` 写开始时间、实际命令、预算。
- `ready_for_review`：DeepSeek 已写执行回报；等待 GPT 复核。
- `reviewed`：GPT 已写复核；等待用户决定下一轮。
- 未运行、预算用尽、复现失败、输入不兼容：交回并明确写 `incomplete` 与原因；不静默补采、不换 seed、不加预算。

`CURRENT.md` 是唯一当前指针。每次状态变化更新：任务、状态、下一位处理者、回报链接、实际预算、未完成项。历史任务不覆盖；新任务用新 `Dxxx` 文件，任务书修订写清版本差异。

## 5. 任务生命周期（DeepSeek 侧）

### 5.1 Preflight（运行前）

1. 读 `CURRENT.md`、任务书、`RESULT_TEMPLATE.md` 和任务书列出的源报告。
2. 核对任务书冻结输入的 SHA256；与任务书不一致时先报告，不静默替换。
3. 核对 checkpoint 的 `manifest` / `source_sha256` / `source_snapshot` 与当前源码：
   - 优先使用与 checkpoint 匹配的源码快照；
   - 不兼容则记录差异，无法复现就交回；
   - 不把“同一轮次”当成“同一策略”；只有权重、normalizer、配置、场景、seed 都一致才复用旧证据。
4. 记录：HEAD、dirty 文件、源文件哈希、checkpoint/panel 哈希、实际命令、seed、预算、开始时间。
5. 确认本轮是否允许 `optimizer.step`、是否允许更新 normalizer、是否允许写入新 checkpoint。
   默认：**只读诊断不训练**；任务书明确允许才做参数更新。
6. 确认输出目录是新的 `outputs/agent_exchange/Dxxx_<run>/`，不覆盖旧目录。

### 5.2 运行中

- 严格遵守任务书的 episode/轨迹数、worker 数、墙钟硬上限；到点终止并保留已完成产物，标 `incomplete`。
- 不自行加预算、补 seed、重跑失败样本、换场景、改 reward/成功判据/控制器/物理/观测/动作版本。
- 异常 episode 仍计入评估分母并算失败；是否进入优势/梯度批次按任务书；不静默删除。
- 原始轨迹、逐决策数据、manifest、CSV/JSON、回放全部保留；报告中的每个数字都要能回查。
- 策略输入只用任务书允许的合法观测；物体真值/特权量只能用于离线标注，不进入策略、mask 或梯度输入。
- 不在运行中修改源码；如必须修 bug，记录原因、影响范围、重跑范围和源码哈希，旧产物保留。
- 监控：heartbeat、异常率、NaN、墙钟、worker 退出码；触发 STOP 时保留现场，不自动重启。

### 5.3 交付（运行后）

按 `RESULT_TEMPLATE.md` 写 `results/Dxxx_*.md`，至少包含：

- 状态 `complete` / `incomplete`（原因）；
- 任务书版本、开始/结束时间、实际墙钟、episode/轨迹数；
- HEAD、dirty、源码快照/哈希、checkpoint/panel SHA256；
- 实际命令、配置、训练 seed、评估动作 seed；
- 改动文件与针对性测试；未跑的测试；
- 逐场景/策略的分子分母、失败、异常；区分 actor 决策数与真实秒数；
- 原始 JSON/CSV/轨迹/回放/manifest 链接；
- 对任务假设分别写：支持 / 反对 / 证据不足；区分本轮新诊断与历史训练归因；
- 偏离任务书之处、预算未完成项；
- 一个由结果支持的下一步建议；**停止，不自动执行下一步**；
- 更新 `CURRENT.md`：状态、回报链接、下一位处理者（GPT）。

## 6. 证据规则（硬约束）

- 不伪造、不补写未运行的结果；负结果照报。
- 历史事实、当前复现、解释假设、未运行项目分开标记。
- 不把训练期间累计成功当作末轮冻结策略成功率；不把重复采样当作独立场景/独立 seed。
- 0 次成功不等于没有尝试该动作；必须看动作序列、请求/执行记录和真实秒数，再判断是“没选”还是“选了没执行成”。
- 初始状态或局部残差的相似/差异不能外推为整条轨迹的几何依赖；结论必须限定在实际采样的状态和轨迹上。
- 同向/同速是待测动作属性，不先断言同速是物理成功的必要条件；用实际推进、接触和执行结果解释。
- 不把“同轮次 checkpoint”当作同一策略；不把 `policy.pt` 与 `checkpoint.pt` 混用，除非核对权重/normalizer/配置/场景/seed。
- 不通过改 reward、成功判据、测试条件、mask 或数据分布来制造正结果。
- 不把启发式/有限搜索叫精确 oracle；不把控制器执行成功等同于策略学会了协同。
- 不把 `TrueState`、环境对象、队友私有历史传给部署策略；特权数据只做标签/离线评估。
- 概率/统计必须给分母；动作概率对全部合法组合求和，不截断 top-k。
- 梯度诊断必须核对 old logp 重算误差、梯度重组误差、数值有限；一阶局部梯度不等于 Adam/多 epoch/真实历史更新。
- 预算用尽、输入不兼容、复现失败都必须显式交回；不静默迁移或裁剪范围。

## 7. 目录与命名

| 路径 | 用途 |
|---|---|
| `docs/agent_exchange/AGENTS.md` | 本协议 |
| `docs/agent_exchange/README.md` | 入口、目录约定、状态机摘要 |
| `docs/agent_exchange/CURRENT.md` | 唯一当前任务指针、状态、下一位处理者 |
| `docs/agent_exchange/tasks/Dxxx_*.md` | GPT 任务书；不覆盖已执行版本 |
| `docs/agent_exchange/results/Dxxx_*.md` | DeepSeek 执行回报 |
| `docs/agent_exchange/reviews/Dxxx_*.md` | GPT 复核与下一步 |
| `docs/agent_exchange/RESULT_TEMPLATE.md` | 回报模板 |
| `outputs/agent_exchange/Dxxx_<run>/` | 原始产物；新 run 新目录，不覆盖 |
| `experiments/...` 或现有模块 | 本轮新增实现；不改旧实验目录的冻结文件 |

`outputs/` 通常不随 git 同步；跨机器接续必须检查 checkpoint、panel、源码快照和原始产物是否齐全。原始产物缺失时写 `incomplete`，不要用报告文字代替数据。

## 8. 长期记忆与协议变更

- 本文件是长期协议；修改时更新版本号、日期和下方变更记录，不覆盖历史任务书/回报。
- 协议变更不追溯改写已执行结果；如新协议影响旧结论，在 `reviews/` 里单独写影响说明。
- 新会话如发现 `CURRENT.md` 与本文件冲突，以用户最新明确指令为准，其次 `CURRENT.md`，再次本文件；并在回报中记录冲突。
- 如果交接目录被移动/重命名，必须同步更新根 `AGENTS.md` 的指针，否则未来会话会丢失入口。

变更记录：

- v1（2026-10-29）：建立 DeepSeek 执行侧协议；固定读取顺序、状态机、preflight、运行约束、回报格式、证据规则和长期记忆指针。

## 9. 给用户/GPT 的交接消息模板

用户移交时，最小信息：

```
Dxxx 移交。
任务书：docs/agent_exchange/tasks/Dxxx_*.md
当前状态：ready_for_handoff → running
预算：<episode/轨迹数>，<墙钟硬上限>
允许：<只读诊断 / 允许 optimizer.step / 允许更新 normalizer / 允许写 checkpoint>
期望回报：docs/agent_exchange/results/Dxxx_*.md
```

DeepSeek 交回时，最小信息：

```
Dxxx 执行完毕/未完成。
状态：ready_for_review / incomplete（原因）
回报：docs/agent_exchange/results/Dxxx_*.md
原始产物：outputs/agent_exchange/Dxxx_<run>/
实际预算：<episodes/轨迹数>，<墙钟>
关键结论：支持/反对/证据不足（各一句）
下一位：GPT 复核
```

## 10. 一句话

**用户移交后，DeepSeek 只做任务书内的有界执行：preflight 核对输入与快照，运行中守住预算和证据边界，运行后按模板回报并更新 `CURRENT.md`，然后停止；下一轮由 GPT 复核、用户决定。**
