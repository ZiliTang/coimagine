# GPT / DeepSeek 固定交接入口

建立于 2026-10-29（Asia/Shanghai）。用户负责两个 agent 之间的 handoff。

- GPT：读证据、提出可证伪问题、限定预算、审查结果，写任务书与 review。
- DeepSeek：按用户移交的任务实现、测试、运行、保留原始结果，写执行回报。
- 用户：决定任务移交与下一轮方向；本目录不自动触发任何 agent 或训练。

2026-11-03 用户明确更新当前 testbed 工作方式：GPT 已直接接管 8765，后续由用户与 GPT 在测台交互调优；这一范围内无需先交 DS。继续先围绕一个实例查看中间结果、逐步修改，再决定回归范围；不授权自动训练或全场景搜索。实时进程/版本查 `/api/status`，运行与重启记录见 `CURRENT.md`、`outputs/testbed/server.json`。

同日用户明确评价重点：option自身行为合理性与计算开销，task complete仅作附加标签；不要为了让所有task成功而改变option优化目标。性能诊断要分开控制/感知/物理、记录/序列化、历史列表与前端等待，不能把页面等待统称为option计算时间。

同日用户决定先移除轮廓SupportMemory，让初始q只依赖对应机器人当前可见支持；已做小范围行为对照并上线8765，等待人工反馈。运动估计时间历史与锁定q状态保留，不自动恢复轮廓memory。版本以CURRENT与实时status为准。

每次会话先读 [AGENTS.md](AGENTS.md)（执行侧长期协议），再读 [CURRENT.md](CURRENT.md)，再读其中指向的任务书、执行回报和源报告。
根目录 AGENTS.md 已登记这个入口；不需要每轮重新寻找最新交接路径。

目录约定：

| 路径 | 用途 |
|---|---|
| `CURRENT.md` | 当前任务、状态、输入证据、下一位处理者；维护唯一当前指针 |
| `tasks/Dxxx_*.md` | GPT 编写的有界任务书；修订写清差异，不覆盖已经执行的版本 |
| `results/Dxxx_*.md` | DeepSeek 执行回报；按模板写，链接原始产物 |
| `reviews/Dxxx_*.md` | GPT 复核后的结论与下一步；与执行回报分开 |
| `RESULT_TEMPLATE.md` | 执行回报模板 |
| `briefs/Dxxx_*.md` | 给用户阅读的任务动机、目标和结果分流说明 |

用户沟通约定（2026-10-29 补充）：每次 GPT 准备任务书时，同时给用户一份简明说明，
在聊天中直接解释，并保存到 `briefs/`。说明必须回答：当前卡在哪里、为什么先做这项、
具体检查什么、不同结果会怎样改变下一步、对 RL 改善有什么帮助，以及本轮不能证明什么。
用具体行为解释术语，不只列指标、函数名和预算；任务书负责执行细节，简明说明负责对齐目标。
执行回报复核时，也用同样的语言说明原先猜测哪些成立、哪些被否定。

当前任务的简明说明：[先做通一个换面失败实例](briefs/D011_Why_One_Case.md)。
2026-10-30 用户更新迭代方式：固定一个 failure case，先完整可视化，再由用户介入逐步优化同一实例；实例做通并确认后再评估其他 case。不要每想到一项就改算法并跑全体矩阵。多个串联子问题必须在同一行为链中展示；本轮先交可视化和诊断，不自主开始新算法搜索。
历史说明：[m3b八次独立验证](briefs/D010_M3b_Why_Probe.md)。
历史说明：[D010 接触获取长周期修复](briefs/D010_Why_Contact_Acquisition.md)。
历史说明：[D009 v2 这一夜的目标](briefs/D009_v2_Overnight_Objective.md)。
D009 v1 尚未执行，现已被 v2 取代；旧文件保留供追溯。
历史说明：[D008 的动机与目标](briefs/D008_Why_This_Audit.md)。
历史说明：[D007 的动机与目标](briefs/D007_Why_This_Calibration.md)。
历史说明：[D003 的动机与目标](briefs/D003_Why_This_Audit.md)。
历史说明：[D002 的动机与目标](briefs/D002_Why_This_Experiment.md)。
历史说明：[D001 的动机与目标](briefs/D001_Why_This_Diagnostic.md)。

状态使用：`ready_for_handoff → running → ready_for_review → reviewed`。
未运行、预算用尽和复现失败都可以交回，明确写 `incomplete` 与原因。
只有用户实际移交后才进入 running；GPT 写出任务书不代表 DeepSeek 已收到。

源码放 `experiments/` 或现有模块，原始产物放 `outputs/agent_exchange/Dxxx_<run>/`，
本目录只放任务、报告与索引。原有 `docs/handoff/` 保留，不搬移、不重写历史证据。
`outputs/` 通常不随 git 同步：跨机器接续必须检查 checkpoint、panel、源码快照是否齐全。

不要以 git HEAD 代替源码快照：当前工作区包含大量未提交实现。
每次执行记录文件 SHA256、配置、动作与观测版本、种子、命令和实际预算。
报告中的历史事实、当前复现、解释假设与未运行项目分别标记。
