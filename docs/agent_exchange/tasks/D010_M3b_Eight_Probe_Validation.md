# D010 m3b小物理验证：独立8次预算方案

2026-10-30，GPT。**准备就绪，等待用户人工移交；未移交不运行。**
用户实际移交本文件，才授权这笔明确的**新增8次attempt、20min总墙钟**；不是旧D010剩余额度，不覆盖/伪造历史unknown，也不自动开始D011。
这是验证冻结版本的短任务，不是新一轮算法探索。原D010整体仍incomplete。

必读：[m3b复核](../reviews/D010_M3b_Offline_Review.md)、[配置/冻结输入](D010_M3b_Probe_config.json)、[用户说明](../briefs/D010_M3b_Why_Probe.md)。

## 1. 固定8个attempt

| 顺序 | 版本 | case |
|---:|---|---|
| 1 | B | T_reposition_0_p0 |
| 2 | F | T_reposition_0_p0 |
| 3 | B | T_reposition_2_p0 |
| 4 | F | T_reposition_2_p0 |
| 5 | B | T_dual_0_p0 |
| 6 | F | T_dual_0_p0 |
| 7 | B | T_dual_2_p0 |
| 8 | F | T_dual_2_p0 |

B来自D010_planning_20261030/baseline_source.zip冻结闭包；F来自本任务config指向的m3b冻结闭包。
两个版本均在隔离进程使用各自快照，不混当前文件；断言unknown_shape、system、mode_env和coimagine包实际路径/哈希。
case初态/command逐项来自D009_v2_acceptance.json，root0的PUSH为W、root2为S。保持原脚本时长与BRAKE，不自适应延长。

## 2. 预算与禁止项

- 独立新预算：最多8次attempt（启动错误、失败、异常、取消均计）；无额外physical smoke、单测或补seed。不执行第9次。
- workers≤2，每个job墙钟≤90s；总20min从本任务preflight起计算，含记录器实现、纯测试、运行、分析和报告；最后3min报告。
- 首个物理前完成快照导入/日志schema的纯测试，不实例化backend。统一budget controller先在append-only ledger预留attempt_id再启动job；进程失败也消耗该id。
- 每个job新进程组，超时TERM后≤5s KILL/reap。到总17min终止仍运行的计算并汇总；不足8条就incomplete，不缩水分母。
- 算法、参数、EKF/FF、物理/成功/奖励/动作/初态全部冻结。允许纯记录/分析修复；需要改算法就保留原件、停止本小任务，交回缺口，不边看结果边改版本。
- 不训练、不替换服务/default、不跑40/new/contract/260、不自动继续下一批。旧账unknown原样保留，新账精确从0到最多8。

## 3. 先确保20Hz记录正确

复用实际执行时的观测包；可包装observe缓存，不能为记录额外scan/update估计器。
每tick保存：

- pre_obs：tick/time、raw contact/原始接触点、own xy/angle/v、当前扫描patch支持/法向；
- controller：requested/executing action、candidate/id/q/边距/误差/原因、phase/定时器、raw/ref/FF/力、clearance、gate_pre/post；B没有新candidate字段时写not_applicable，不能推测填充；
- truth_pre/truth_post：物体位姿/速度/角速度、机器人状态，仅scorer消费；分别标t_k与t_k+.05，不把post状态标成pre时间。

原始接触从pre_obs.contacting_object取，不能用controller_ready替代。cp_world从原始contact_point_robot和同刻own pose变换。
完成无物理schema golden检查；关键字段缺失则显式错误。记录不反馈控制；不把真值传给controller/mask。

## 4. 固定分析问题

每条分别报告：

1. 是否success/termination、最终距离、指定推向净推进、相对初角峰值；不把几毫米变化称成功。
2. 首次真实接触前后是否对齐/等待/强行touch；精确pre tick和时间。
3. F的候选是否持久、何时重选；首次cp相对可见支持的切向坐标/边距、接触法向与选面夹角。
4. 若有SHIFT：记录计划目标、实际own里程计路径/投影位移、shift后candidate与重新接触是否改变。不能用speed×timer当真实位移。
5. 首次接触后.5s的相对角变化与接触保持；不足窗口标right_censored。

B/F首触点可额外在各自同刻物体坐标系表示，用于离线几何比较；真中心/真角只在scorer，不输出给F。
两条dual F应保持2/2；若失败，先指出首次阶段/clearance/gate断点。两条reposition F是否成功与是否改善接触几何分别报告。
只有4个工程case，不报总体成功率改善/统计显著性，不宣称D010 accepted。不要求结果为正。

## 5. 交付后停止

新目录`outputs/agent_exchange/D010_m3b_probe_<run>/`：8个attempt原件、逐tick日志、源码/配置/输入哈希、准确budget/ledger/命令、逐case B/F表。
给四张B/F对照图（每case一张，可多子图）：机器人/物体路径、candidate/首触点、角偏、阶段/gate。缺失字段或失败如实标注，不补造。
回报`results/D010_M3b_Eight_Probe_Report.md`：首先回答“候选是否改变了首触、角偏/推进是否改善、dual是否保住”，再列方法与限制。
全部8条与数据齐全写本小任务complete（仍可not-supported）；缺记录/未跑齐写incomplete。更新CURRENT为ready_for_review；不自动扩大。
