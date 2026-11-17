# D/A 联合表示实验实现说明

协议：`docs/agent_exchange/tasks/DA_Joint_Representation_20261117.md`。这是候选参数/条件绑定的第一轮机制实验，不是完整新谓词发现算法。

## 固定输入与表示

保留十个 CAP 的原始全部 G/D/A。每个 CAP 新增 78 个候选通道：8 个条件数值、26 个原始建议操作数、12 个候选兼容性值、32 个绑定乘积。所有通道只读合法观测历史和该程序根据真实历史产生的结果。

8 个条件为 target_seen、target_estimate_age、target_contacts、blocked_robots、object_progress_5s、stagnation_seconds、nav0_distance、nav1_distance；分别除以 1、5、2、2、0.5、5、2、2，再限幅到 [-2,2]。缺失仍为 NaN。它们保留冻结程序原定义，不把字段名当物理真值，也不改动程序的估计器。

26 个操作数：建议合法标志、空目标标志、任务目标标志、ticks/40、两个机器人各 11 维动作编码（8类、速度、目标参考系方向余弦/正弦）。

12 个兼容值：目标匹配；每机器人类型匹配、方向点积、速度差、动作ID精确匹配；联合动作匹配、时长差、已选前缀匹配。尚未选择的坐标为 NaN，显式 None 目标与未知目标区分。

32 个乘积：8 条件分别乘以目标匹配、目标×机器人0动作匹配、目标×机器人1动作匹配、目标×联合动作匹配。每个乘积保持同一 CAP 的绑定，没有将甲的条件与乙的建议混接。不同 CAP 原始估计不强制去重。

三臂使用同样维度和参数总数：current 的新增 78×10 槽全零；enriched 提供前46×10操作数，乘积槽全零；joint提供全部。enriched/joint原始信息相同，后者将关系显式计算出来。活跃通道数不同，所以不能声称有效容量完全相同。原始CAP已经有部分匹配语义，本轮是补齐参数并显式绑定的增量比较。

current是在本轮共同网络结构与28000更新预算下重训的原信息对照，不是直接复用历史最佳GDA；诊断donor则是历史42000更新模型。表示效应应在本轮同预算对照间判断，不将历史donor与本轮模型的差值直接归因于表示。

该适配器仅用于本轮三种 GDA 表示比较：新增槽在三臂中均允许，不是新的 G/GA/GD/GDA 消融入口。8个条件还包括程序意图相关量（例如规划导航距离），不能将联合表示的收益单独归因于纯描述性D。若以后做分组消融，必须重新明确混合通道的归属及掩码。

## 数据和评价

270 条原示范所有旧通道逐数组精确核对；新增输入只提取一次，各臂从同一缓存投影。修复训练在16个开发接管状态中选择最快成功的连贯后缀，与原示范合并，三臂一致；没有使用失败动作负标签，也没有引入多正例排序训练损失。

完整动作覆盖审计（`analysis/demonstration_action_coverage.json`）：270示范/9292边界中，直接操作非任务目标物体为0；target-free为3144、包含MOVE为2792、包含旋转为121、双PUSH为718（后几项并非互斥分类）。因此本轮不能支持“主动搬开其他物体的清障子技能已被整合”这一主张；目标物体运动引起的被动接触带动不计作主动选择障碍。所有24个争议状态均在预定候选中的第8边界被选中，本轮尚不代表全部晚期接管分布。

独立 rank_probe 对已执行候选的完整联合动作 log-probability 排序，允许多个成功答案。其标签只适用于冻结 donor 的后续执行，不是 Q*，也不是新模型闭环成功的替代指标。8个接管留出来自已有开发池，仅称诊断留出；真正未使用的最终面板是新80初态。开发池历史结果参与了CAP候选优先级统计，必须与最终盲测区分。

## 前缀修复与审计

旧 LibraryFeatures 在 decision 中更新 CAP 记忆，在 panel 中更新几何历史。强制重放前缀不评分，因此首次 GDA 身份对照失败。新增 HandoffLibraryFeatures 在每个真实边界增量更新几何；正常评分再调用是幂等的。旧模块和冻结结果不改。

初次分叉目录 branches_attempt01_failed 保留。16个已启动worker按每次1800 tick上界计账，12份已完成summary保留；缺失原始结果明确列为基础设施失败，不能作科学样本。修复后24个身份对照与原轨迹全程一致，112次正式分叉无工具错误。

## 运行与产物

- 诊断：`python -m experiments.planar_tasks.authoring_v2.da_joint_study diagnose`
- 表示训练/评估监督器：`python -m experiments.planar_tasks.authoring_v2.da_joint_run`
- 后验排序探针：`python -m experiments.planar_tasks.authoring_v2.da_joint_rank_probe`
- 完成后的审计、分层配对统计与图表：`python -m experiments.planar_tasks.authoring_v2.analyze_da_joint_results`（只读分析输入，输出独立分析产物和报告附录；不参与训练或模型选择）
- 数据：`outputs/semantic_rl/da_joint_20261117_01/`
- 日志与阶段状态：`outputs/semantic_rl_authors/da_joint_20261117_01/`

源码封印后不在运行中修改。监督器为各阶段保留独立日志和退出码。旧数据与远端优先级实验不变。
