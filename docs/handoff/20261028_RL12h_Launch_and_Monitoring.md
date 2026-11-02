# 新 Option RL：12 小时启动与夜间监控

北京时间 2026-10-28 02:43，按用户明确授权启动 12 小时预算；预计在 14:43 左右到达训练
时间上限，可能因正在结束的迭代/验证略有延迟。没有同时启动多个长训练。

- tmux：`coimagine_rl10_20261028`
- 监督进程 PID：279051；训练进程 PID：279053。当前状态以 processes.json / 实时进程为准。
- [实时 dashboard](../../outputs/option_rl_v10/overnight_20261028/run10_001/dashboard.html)
- [每分钟摘要](../../outputs/option_rl_v10/overnight_20261028/run10_001/monitor_summary.md)
- [训练日志](../../outputs/option_rl_v10/overnight_20261028/training.log)
- [冻结监控计划](../../outputs/option_rl_v10/overnight_20261028/supervision_plan.json)

## 本轮配置

动态合法训练场景；64 根固定验证、另留 64 根测试；68 动作；条件 type/scale；新版 PUSH、
RP 外侧绕行/静态通道停止；8 episode/iteration；seed 0；entropy 0.02；gamma=1；
无 BC、无课程、无奖励调参。固定验证首轮及每 10 轮执行；首轮及每 20 轮生成行为回放。
源码及动作/控制器配置在 manifest 冻结，关键源码另存 source_snapshot。

## 实时性验收

1. dashboard 在每轮训练更新后、长验证开始前即显示当轮指标；完整验证完成后补入同一轮
   评测值，没有重复迭代点。验证期间展示 `eval_done / 64`。
2. 18 张曲线覆盖回报、成功率、progress、policy/value loss、熵、奖励项、异常率、
   actor/critic 梯度、KL/clip fraction、采样/PPO 概率差、动作占比、阻塞与短动作、距离分层。
3. 训练曲线每轮更新，固定验证每 10 轮实际测量。页面每 10 秒刷新；独立监控每 15 秒重绘。
   HTML 与 history/status 采用原子替换。浏览器已实测自动刷新，无脚本错误，390 px 无横向溢出。
4. 18 项针对性测试通过。用真实短任务验证 2 个请求 episode 恰好采到 2 个；也验证了
   STOP 请求可在当前回合结束后停训并保存结果。

在增加实时状态的开发验收中，live_check_001 曾因采样循环缺少达到目标数后的 break，
采满了最大重试次数（请求 2、实际 6）。已在正式任务启动前修复并加入回归测试；
live_check_002 实际请求/采样均为 2。两轮原始记录保留，不混入正式训练曲线。

## 持续监测

本地 supervisor 不依赖聊天网络连接。每分钟保存 observations.jsonl 和 monitor_summary.md；
每小时保存阶段快照，较好验证权重另存 best.pt，每轮保存 checkpoint.pt。

明确故障的停止规则：

- 非有限数值，或状态心跳超过 300 秒；
- 连续三轮训练异常率 ≥25%，或连续两次验证异常率 ≥10%；
- 连续三轮 PPO KL >0.5；
- 连续十轮熵 <0.1，且最近两次验证无初始目标外的成功、progress ≤0；
- 运行中关键源码指纹改变。

不因为早期验证为零或短期波动自动停训。触发时写 STOP.json，请求回合边界停止并保留结果。
程序不会自动编造修复或调整奖励；需要修复时根据证据另开新目录，保留原轮次。
代理的实际复核另写 agent_observations.md，与自动摘要区分。


## 02:58：8 worker 接管

用户指出采样效率偏低。确认机器有 24 逻辑核、无 cgroup CPU 配额限制，串行训练只占约 1 核。
原轮已正常停在 8 个完整训练迭代，约 443 s；首轮含验证约 239 s，普通迭代约 22–31 s。
停止原因是性能优化，不是发现模型训练失效；所有原始数据和 checkpoint 保留。

新活动页：[8 worker dashboard](../../outputs/option_rl_v10/overnight_parallel_20261028/run10_001/dashboard.html)。
北京时间 02:57:52 启动新的 12 小时任务，预计 14:58 左右结束。
从原 policy.pt 恢复已训练 8 轮的权重、优化器、归一化统计和下一场景序号 68；新目录独立记轮次。
没有迁移 run9 的旧 28 动作权重。

8 个持久 worker 同时负责训练/验证，每 worker 1 线程；固定每更新 8 个 episode。
主进程统一固定每批归一化与权重快照，PPO 后合并统计，避免 worker 独立改输入变换。
从头训练的并行入口有一轮串行归一化预热，当前续训直接使用已保存的有效统计。
第一版冷启动冻结零统计时曾出现很多短安全退出（parallel_live_check_001），原记录保留；
当前采用已验证统计恢复路径，parallel_resume_check_001 为 62 个 transition / 8 episode，
短 option 约 13%，无物理异常。

[实际吞吐基准](../../outputs/option_rl_v10/parallel_benchmark_001/benchmark.json)：8 个同快照完整回合，
串行 29.49 s，8 worker 冷池 6.73 s、热池 6.00 s，热池约 4.92 倍。
串并行结果/动作完全一致，log-prob 和 value 最大差为 0。19 项框架/并行/实时监控测试通过。
这是固定快照吞吐对照；训练输入归一化的更新时序也变化了，不用它声称学习效率已提高。

新任务原监控、每分钟摘要、每小时阶段报告和故障停止规则继续生效。
