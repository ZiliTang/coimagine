# Run12：持续闭环模式、5 Hz高层决策

独立新实验，不替换run11，不续用旧codebook权重。当前是控制模式/PPO试验，尚未训练历史编码器或世界模型。

## 接口

- 物理120 Hz、低层20 Hz；高层每0.2 s选择联合动作；测台可对照10 Hz。
- 46项：8方向PUSH × 4个法向速度（0.04/0.08/0.16/0.32 m/s）、双向ROTATE × 3个接触法向速度（0.02/0.04/0.06 m/s）、双向RP × 3个切向速度（0.06/0.12/0.18 m/s）、BRAKE、FOLLOW。
- ROTATE强度单位是接触法向速度，不是期望角速度。PUSH/RP没有名义行程或名义完成时间。
- 重复命令保留控制器；同模式族内更新方向/速度也保留跟踪/参考速度。PUSH与ROTATE、旋转正反向、RP/FOLLOW/BRAKE等模式族切换先真实制动，再提交新控制器。RP正反向可连续更新并保留加速度限制。
- 保留接触建立、端点跟踪、可见接触点分配、绕角/同伴外侧绕行、静态障碍停止、联合接触恢复；失败状态锁存至不同模式族请求。
- FOLLOW仅与PUSH配对；mask按动作kind处理，BRAKE始终可用。安全恢复的参考速度与活动控制器连续交接。
- 不再按两个请求行程比例调速；双PUSH各自速度由策略指定，保留接触就绪/联合恢复。
- 原760维观测加62维合法控制器上下文（模式、阶段、请求速度、年龄、参考速度、失败/切换状态），共822维。仍是集中共享本地观测，不作B-bit部署声明。

## 策略与数据

MLP仍为2×256，type/scale因子化＋联合合法配对归一化；采样/PPO条件概率一致。原760维使用运行统计归一化，控制器上下文本身已缩放并直接输入。

为避免初始随机策略每0.2秒换族、无法建立接触，actor包含可学习的模式族保持偏置，初值2.5；BRAKE不加保持偏置。它不强制保持、不添加奖励；不同族仍可随时请求并按物理制动执行。必须把该先验作为新方案差异报告，不能将变化全部归因于更密集样本。

任务仍90秒，reward按真实20 Hz物理控制周期累计：gamma=1，progress/距离时间积分/时间/成功/失败系数保留。不能每次高层决策额外收取原整段option成本。

`world_model_replay/`为可选数据准备：每第32个训练root保存完整5 Hz边界观测序列；NPZ原子写入，保留最近256个被选episode的环形缓存。包含合法原始观测T+1、联合动作T、真实dt、reward和scene ordinal/采集迭代元信息；世界模型未训练。完整reset/action/reward/termination摘要仍在规范rollouts中，旧缓存可按冻结控制器重放恢复。

另有10 Hz内部观测/控制命令收集接口（`collect_dense=True`），当前训练未开启。不能把采集数据宣称为学成的belief。

## 验证和限制

最终短测台：`outputs/option_testbench_v1/continuous_modes_v1/round_008/index.html`。
10个指令时间表×5/10 Hz共20分支；相同物理指令时刻下状态/力逐周期一致，无安全触发，穿透≤2mm，参考速度每周期变化≤0.025m/s。真实位姿独立重算reward一致。

9项聚焦测试通过；1000合法初始化及48首物理步无异常。最终smoke_006完成2轮×8episode、8 worker、每轮8根验证及回放；logp重算误差≤1.91e−6，无物理异常。没有运行完整仓库测试，也不据此宣称RL已学会任务或新方案更优。

持久控制器的完整branch快照/恢复尚未实现，API明确拒绝；支持从初态完整重放。不可直接套用旧的“只恢复物理世界”分支快照做规划。

## 运行

```bash
.venv/bin/python -m experiments.option_rl_v12.testbench \
 --out outputs/option_testbench_v1/continuous_modes_v1/round_NEW
.venv/bin/python experiments/option_rl_v12/supervise.py \
 --job outputs/option_rl_v12/JOB_NEW --hours 2 --workers 8 \
 --episodes 8 --eval-roots 64 --eval-every 10 --render-every 20
```

保持动态训练、64固定validation、64固定test，seed0、无BC/课程/奖励调整。从头训练，先做2小时试轮。普通迭代的样本/计算量与run11不同，比较时同时报告episode数、模拟秒、transition和墙钟时间。

原本地故障监督、checkpoint/best权重、source snapshot及每轮dashboard保留。另增模式族切换请求/提交、切换制动状态比例；“失败状态比例”按决策统计，不当作独立失败事件次数。

## 已排队的夜间接续

`handoff.py`等待run11训练器及supervisor正常结束，归档最终摘要，复核已验收新源码指纹后启动一次2小时新试轮。
任何异常退出、源码变化或目标目录已存在均暂停接续。具体方案和状态在
`outputs/option_rl_v12/handoff_20261029/`；新任务目标为
`outputs/option_rl_v12/continuous_5hz_pilot_20261029/run12_001/`。
程序不在未来自动生成/修复代码、不改奖励，也不续训旧参数。
