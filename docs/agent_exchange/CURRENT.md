# 当前工作状态

研究主线：[计划v6](../research/CAP_Program_Trace_Selection_Plan_v6_20261117.md)：先在CAP完整建议并集中选择动作，以自动提取的程序执行特征比较几何输入，保留真实接管修复与监督整合。旧v5及冻结实验保留，不将新候选限制追溯应用于旧结果。

## 下一阶段：第二台PC实施，尚未启动

用户要求将方案落盘、提交代码并增量移交，后续大计算在另一台PC执行。[移交说明](../development/Program_Trace_Transfer_20261117.md)：依赖原`priority_teacher_v1`，新增`cap_program_trace_v1`携带本轮证据、模型、回放及相对索引，不复制约10GB派生数组。自动插桩、稳定特征schema、完整CAP建议池与新候选级评分器尚未实现。先做无行为改变及示范候选覆盖验收，再冻结下一轮训练预算与新测试集；当前没有新训练、PPO或LLM运行。

增量包已完成，约1.41GB，3770文件读回校验通过；相对索引、真实边界特征、12模型及历史donor跨目录验证通过。[验收记录](reviews/Program_Trace_Transfer_20261117.md)。两台PC根路径无需一致，不需重传已有priority包。

## 已完成

- 冻结10个随机房间CAP；270示范整合与覆盖诊断：[结果](reviews/Random_Room_Coverage_Repair_20261117.md)。
- 同270示范/28000更新/seed17，旧100已查看诊断面板G/GA/GD/GDA分别1/48/5/50成功；最佳CAP003为46，CAP并集79。R0/Rt为36/41。GD已完成100评估、封印和报告收尾。[16方法交互报告](../research/reports/CAP_Coverage_Comparison_20261117/index.html)及[分层读数](../research/reports/CAP_Coverage_Comparison_20261117/analysis.md)。
- 单训练种子、固定已见诊断，不作为新盲测；覆盖相似不等于行为相同。

## GDA扩数据已完成

[结果报告](reviews/GDA_Coverage_Scaling_20261117.md)：原270与扩570示范各3seed、42000更新，全部7800次采集/评估完成，宿主退出0。新200例seed17/29/43：原270为86/80/85，扩570为74/77/90；教师并集153/200。平均收益不明确，不能只看单seed推断扩数据必然有益。

权威产物：`outputs/semantic_rl_authors/random_room_gda_scaling_20261117_03/RESULT.json`、`study.exit.json`。当前没有该任务的剩余评估。

## 本机已完成：D/A 联合表示

用户已明确授权自主实现和实验。[本轮报告](reviews/DA_Joint_Representation_20261117.md)、[协议](tasks/DA_Joint_Representation_20261117.md)、[实现与解释边界](../development/DA_Joint_Representation_20261117.md)。监督器退出0；输出 `outputs/semantic_rl/da_joint_20261117_01`，权威结果 `outputs/semantic_rl_authors/da_joint_20261117_01/RESULT.json`。正式宿主约3小时20分钟，无新CAP/PPO/API调用。

48 donor、112分叉、12fit（各28000更新）、1536模型评估全部完成。24身份、44首动作配对和288模型接管前缀审计通过；270示范旧通道精确一致。新80初态两seed：原数据current 50/44、enriched 46/46、joint 47/47成功；补10后缀后分别40/41、49/50、46/45。联合乘积没有显示额外优势；enriched补数据自身提升4.375个百分点，但区间跨零。所有修复模型在10个示范接管点均10/10成功，从同10场景初态只有5～8成功；20次失败均在原第8接管边界前偏离旧donor前缀。局部修复可学习成立，稳定泛化与能力保持尚未成立。完整逐场景比较、排序诊断、图表和局限见报告；没有自动启动下一轮。

## 远端优先级对照

计划v5已登记：最快成功教师 vs 开发成功率优先教师的等场景/等更新预算监督对照。第二台PC专用迁移包和运行工具已准备并通过不同根目录/冷缓存验收（[记录](reviews/Priority_Transfer_20261117.md)），见[迁移说明](../development/Priority_Transfer_20261117.md)。本机没有启动或改动该优先级实验。

用户最新提供远端执行回报：20:01结束、约3小时7分钟，6模型各42000更新、1200评估无错误。共享200初态上seed17/29/43的fastest为89/82/75成功，priority为87/82/83，均值41%→42%，未显示稳定优势。每seed约34%～36%场景改变成败；两组示范拟合均超过99%。priority含24114边界，fastest含20007，故不是纯标签交换。该信息来自用户粘贴的远端报告，本机尚未读取原始CSV或独立核验；不与源机历史fastest跨机器拼成因果比较。远端报告位置：`/home/tzl/repo/coimagine/docs/agent_exchange/results/Priority_Teacher_20261117.md`，数据位置：`outputs/semantic_rl/priority_teacher_20261117_01/per_case_results.csv`。

## 清理与接续

[清理记录](reviews/Workspace_Cleanup_20261117.md)记录永久删除范围、账本和保留目录。保留当前CAP库、示范、冻结模型、对照评估、关系迁移实验、运行表格、基础控制器/模型和研究汇总。旧CAP/RL原始输出及旧方案已删除，不再声称能完整复现旧阶段；仓库外旧agentic暂存也已永久删除。

扩展宿主JavaScript堆OOM已被日志确认；outputs等已加入VS Code监听/搜索/Python分析排除。清理不等于已确定或修复崩溃根因。

## 2026-11-18：完整CAP trace开发数据移交

[验收记录](reviews/CAP_Trace_Value_Transfer_20261118.md)与[使用说明](../development/CAP_Trace_Value_Dev_Transfer_20261118.md)。`outputs/transfers/cap_trace_value_dev_v1.tar.gz`独立包含所选7CAP×760开发初态的5320条完整原始gzip（1759成功、3561失败），零缺失/异常；原数据完整审计、包内历史重验及6150文件归档回读均通过。附760场景、10CAP的7600摘要、程序/依赖、相对路径索引、源哈希和无评估标签的场景去重清单。没有训练、仿真或补采。接收方先验SHA256再执行包内说明的verify；内部程序trace仍须重算并验收等价性。
