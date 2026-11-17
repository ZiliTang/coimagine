# 当前工作状态

研究主线：[计划v5](../research/CAP_Integration_Mechanisms_Research_Plan_v5_20261117.md)：多CAP覆盖、扁平语义与几何动作评分、监督整合及检索机制对照。此页已于2026-11-17清理历史流水账。

## 已完成

- 冻结10个随机房间CAP；270示范整合与覆盖诊断：[结果](reviews/Random_Room_Coverage_Repair_20261117.md)。
- 同270示范/28000更新/seed17，旧100已查看诊断面板G/GA/GD/GDA分别1/48/5/50成功；最佳CAP003为46，CAP并集79。R0/Rt为36/41。GD已完成100评估、封印和报告收尾。[16方法交互报告](../research/reports/CAP_Coverage_Comparison_20261117/index.html)及[分层读数](../research/reports/CAP_Coverage_Comparison_20261117/analysis.md)。
- 单训练种子、固定已见诊断，不作为新盲测；覆盖相似不等于行为相同。

## 正在运行

GDA扩大数据：[任务书](tasks/GDA_Coverage_Scaling_20261117.md)。原270与扩570示范各3seed、42000更新，6模型已训练完；正在3800次比较评估（6模型×300场景＋10CAP×200新场景）。

实时权威状态：`outputs/semantic_rl_authors/random_room_gda_scaling_20261117_03/progress.json`、`study.exit.json`、`RESULT.json`。截至本次清理检查约1116/3800，数值会继续变化；禁止仅按本页旧数字判断停止。源码与输入未修改。

## 待办

计划v5已登记：最快成功教师 vs 开发成功率优先教师的等数据/等预算监督对照。尚未启动，不改当前评估。当前不新增CAP、RL或模型调用。

## 清理与接续

[清理记录](reviews/Workspace_Cleanup_20261117.md)记录永久删除范围、账本和保留目录。保留当前CAP库、示范、冻结模型、对照评估、关系迁移实验、运行表格、基础控制器/模型和研究汇总。旧CAP/RL原始输出及旧方案已删除，不再声称能完整复现旧阶段；仓库外旧agentic暂存也已永久删除。

扩展宿主JavaScript堆OOM已被日志确认；outputs等已加入VS Code监听/搜索/Python分析排除。清理不等于已确定或修复崩溃根因。
