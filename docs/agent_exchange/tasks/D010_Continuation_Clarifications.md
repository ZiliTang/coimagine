# D010继续执行说明：无新增研究范围、无新增预算

2026-10-30，GPT。补充原[D010任务](D010_Contact_Recovery_Autonomous.md)，不覆盖历史任务书/报告，不创建D011。
阅读[执行复核](../reviews/D010_Execution_Review.md)和[用户说明](../briefs/D010_Continuation_Why.md)。用户人工移交后处理，不直接向另一agent发消息。

## M0：先补预算与仪表

1. 从真实执行记录重建原D010的开始时间、deadline、实际wall/物理attempt、缺失/失败/重跑。不能从7张表直接假定只用了280次；缺失项写unknown或可证明的保守上界，不编造精确数。
2. 本说明**不重新起算12h、不重给1600次**。只有可核实的原预算剩余才可继续物理运行；若已到限或无法核算，先停止新增物理、补离线交付，明确写budget_exhausted或budget_unknown。
3. 保存当前30/40版本与规划B为两个不同源码快照。逐attempt新文件，记录配置/源码/命令；旧结果不覆盖。
4. 完成pre-control/control/post-physics schema；action直接存ID/方向；角偏相对初角；日志包括candidate/支撑、phase转移原因、真实contact、FF/ref/force owner与gate、估计x/P及拒绝因子。真值单独scorer。
5. 离线/合成状态测试优先；有预算才用两条reposition和两条dual做逐tick记录非干扰检查。记录本身不能改变控制输出。

## M1：先把状态机做成真正可执行、可中断的状态机

显式按phase调度，RETREAT、SHIFT等不得只存在于contact=True分支。contact丢失是转移条件，不能自动覆盖phase。
至少通过这些事件测试：

- 退让过程中contact丢失，退让按安全/位移/定时条件继续结束，之后进入SHIFT或明确的等待，不自动向物体重推。
- 无接触SHIFT仍继续执行；达到目标或deadline才结束；不能靠试验预设.18×.45声称实际移动了.081m。
- 已在RETREAT期间，omega仍大不应每tick重设deadline或增加retry_count；只在状态进入时记一次attempt。
- 候选跨tick锁存；支持不足不假装候选可靠；有界视角更新要有触发和次数账本。
- align达标并稳定后才touch，超时转等待/换候选/失败，不能超时就强行触碰。
- BRAKE/新意图和安全退出可抢占所有准备/撤退/shift；P0恢复力仍可执行。
- 主动相对侧移≤.08m/s；转动响应的调整方向必须可解释/校准，不能直接照搬−sign(omega)。

可重构模块与状态机，不要求继续叠加if补丁；冻结EKF/Q/R/FF和物理限制不动。此次状态反例是控制逻辑bug修复，不是更改研究目标。

## M2：用局部真实轨迹证明动作链，而不是立刻再跑40条

在可核实剩余预算内，先跑两个reposition p0；输出每阶段持续时长/真实路径、是否失联、是否真的shift、首触点变化和首触后.5s转角。
出现dual回归时只取两个dual p0定位首个断点；达到局部阶段预期再扩到10条reposition/10条dual，最后40回归。
局部轮廓累积、实际有代价探视、更合理候选评分与必要局部配准，均是原D010已授权的后备路径；按真实缺口继续探索，普通负结果不用再次求相同方法许可。
所有探视在真实扫描/安全路径中执行，不读隐藏尺寸/中心，不免费重置，固定动作窗口不延长。

## 最终交回

仍按原D010的验证规则/最多260比较执行，不沿用D009的184作为D010数量。若未跑齐，固定分母和缺口照报。
更新独立补充回报（保留phase1旧报告）：`results/D010_Continuation_Report.md`，给准确停止原因，状态ready_for_review；不启动D011/RL。
