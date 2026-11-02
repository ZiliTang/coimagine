# 进度事件决策与集中式PPO接入

2026-11-03，用户要求把进度预算和约束组合接入RL并检查。已新增`experiments/option_rl_v20/`，保留当前v19测台及v13–v16历史训练代码。版本`event-progress-v1`；不兼容旧动作/观测语义的checkpoint。

## 执行语义

- 合法有序组合：PUSH＋PUSH、PUSH＋WAIT、REPOSITION＋REPOSITION、REPOSITION＋WAIT、WAIT＋WAIT；左右角色可互换。排除ROTATE、FOLLOW、PUSH＋REPOSITION及旧绕行挡位别名。
- v19保留46个历史ID，当前第一版RL的有效单机动作是32个PUSH、2个REPOSITION、1个WAIT；联合静态合法矩阵1097个有序组合。各机器人仍有自己的几何可行mask，联合矩阵在概率归一化前共同生效，环境入口也拒绝非法组合。
- 控制tick为0.05s。单机PUSH的0点为该机真实接触连续0.15s；双机PUSH则双方同时真实接触连续0.15s。接触事件产生后进度归零。切换参与PUSH的机器人会重新判定；同一个组合继续时不重复产生接触事件。
- 每个事件区间内，PUSH使用接触成立前后两拍的自身里程计增量，沿请求的世界方向做有符号投影并累加；它是机器人进度，不是物体位移。REPOSITION读取该skill已有的有符号沿边界进度增量，不使用总路程。任一参与者达到0.20m即产生联合决策。
- 接触丢失持续0.25s、控制器新失败、任务终止、未接触12s、接触后或绕行8s无进展、双方WAIT 1s，也产生事件。接触获取和推进超时各有独立reason。超时按真实控制tick计时，从事件动作开始即计时。
- 每次新决策区间重新计两台的进度增量；选择相同动作时维持原skill、q及运动状态。决策点不强制刹车。接触丢失事件只在一次丢失中报告一次，恢复稳定接触后重新允许报告。
- observation沿用944维基础及控制反馈，追加14维：上次区间实际时长、两台进度、两台当前接触和9类event one-hot，共958维；仍只归一化前760维。奖励逐tick累加，gamma=1；沿用v16按实际periods计算GAE衰减。策略集中式，未宣称去中心化。

这些数值是首版接口设置，已通过功能检查，未做参数寻优或学习效果结论。各动作的法向/绕行进度可能受接触、滑动、控制与观测影响；达到进度不保证物体推动同样距离。

## 固定动作检查

| 实例 | 决策事件 | 结束tick | 该区间进度m |
|---|---|---|---|
| 接近东推＋WAIT | contact | 18 | 0 |
| 同上继续PUSH | progress | 38 | 0.210 |
| 同上继续PUSH | progress | 57 | 0.204 |
| 同面双北推 | contact | 18 | 0 / 0 |
| 同上继续双推 | progress | 36 | 0.205 / 0.205 |
| 同上继续双推 | progress | 54 | 0.207 / 0.207 |
| 绕行＋WAIT | progress | 26 / 43 / 60 | 0.202 / 0.212 / 0.212 |

持续同一PUSH时skill与candidate ID保持。额外测试在建立接触后人为把机器人移开，正常传感路径在0.25s后报告`contact_lost`；这是受控扰动检查。另验证获取超时、推进超时、WAIT与非法组合。

固定动作40控制tick，对比20Hz单tick推进与原0.2s批量推进：物体、机器人状态和每段奖励一致到1e-12；事件奖励之和与环境累计奖励一致，时间成本按真实秒数累计。

## PPO试运行

最终源码的`smoke_003`：单进程、seed 20261103、hidden 64、2个完整训练episode、1次PPO epoch/更新，另在同一1cm扰动场景各执行1次贪心和随机评估。总墙钟约10.68s（含四个episode、更新及保存），仅用于接线，不代表长期训练吞吐。

- 训练：97个事件transition，两个episode均按原90s时间预算结束，0物理异常；47次相关聚焦测试通过。
- 新PPO更新：采样和重算log probability最大误差**0.0**；actor/critic梯度和损失有限；checkpoint保存、重载与版本校验通过。
- 贪心0/1、随机0/1；该样本太小，且扰动场景源自训练根，不能评估学习或泛化。未把无成功写成接入失败，也未声称改善RL。
- 另外有两轮较早的2-episode/1-update接线调试`smoke_001`/`smoke_002`，它们早于timeout分类和最终源码；共6个训练episode、3次更新及2个最终评估episode，均为短验证，不是正式训练。

## 文件与使用边界

- [事件环境](../../../experiments/option_rl_v20/event_env.py)、[策略和联合mask](../../../experiments/option_rl_v20/interface.py)、[单进程PPO runner](../../../experiments/option_rl_v20/train.py)、[事件测试](../../../tests/test_option_rl_v20_events.py)。
- `outputs/agent_exchange/event_rl_20261103/verification.json`及三个`*_ticks.json`保存固定动作事件和逐tick诊断；tick真值仅供离线核验，未进入事件、mask或policy输入。
- `outputs/agent_exchange/event_rl_20261103/smoke_003/`保存训练/评估记录、checkpoint、normalizer、optimizer、panel hash及29项源码hash/快照；`tests_final.txt`为聚焦测试结果。历史smoke保留。
- runner提供按需贪心/随机评估、checkpoint保存及加载校验；目前是单进程，未实现旧v16的多worker、精确resume、retention、正式训练监控及独立泛化评测。应先设计有界训练和相应验证，再考虑长训。
- 本RL事件接口独立计时；旧testbed `TestbedSession`的预算安装时机和时间换算问题本轮没有改，不能拿旧testbed的时间预算实现冒充新RL语义。8765测台没有重启或改行为。
