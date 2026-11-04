# V2 十二小时事件RL训练

日期：2026-11-04（Asia/Shanghai）。用户同意先进行长程V2训练，询问开训前是否存在阻塞。本轮沿用上一轮有界规模：**12h、16个spawn worker、32个有效episode/轮**。V2从头初始化；V1冻结保留。状态与启动时刻见CURRENT和运行目录的plan/processes。

**已启动：2026-11-04 12:03:41，预计11月5日约00:04结束。** supervisor2462130、trainer2462169；源码提交`4266042`。首轮正式32episode/1513transition、0异常、初始logp误差4.77e-7。数据只说明运行正常，尚无学习结论。[启动命令与版本](../assets/v2_launch_20261104/launch.json)。

## 目标与边界

本轮检验新增YIELD、PUSH方向闭环和无尺寸先验平移mask后，策略能否学到可用行为，以及是否减少BRAKE挡路和持续低运动对抗。保持原奖励、成功判据、物理、动作速度和场景生成器；**施力成本系数为0，没有方向一致惩罚**。施力与对抗只作离线诊断。

本轮不是单因素消融，也不是V2优于V1的预先结论：动作空间、观测与控制契约均有变化。两个版本均为集中共享、单训练seed、固定矩形分布，不声称有限通信或未知形状泛化。新增动作不保证一定被RL有效使用。

## 开训前完成的工程项

- 独立V21长训器、16worker、supervisor和dashboard；不把973维/47动作权重送入V20 runner。
- worker冻结同一policy/normalizer快照，校验动作/事件metadata和内容hash；主进程统一更新PPO，之后更新normalizer。记录每个worker的快照ID与行为seed。
- 修正旧STOP路径不一致：监督器同时写外层和`run/STOP.json`，训练器读取后按边界退出。
- 预算及心跳使用单调时钟，避免墙钟跳动影响；优先正常保存，停止后120s仍不退出则SIGTERM进程组，30s后仍不退则SIGKILL；异常不自动重启。
- manifest保护实际导入的82项项目依赖（预检查实测），覆盖底层参考速度/力控制，不只冻结顶层入口；保留源码快照、环境/物理配置和面板。
- 中间文件有保留上限：每250轮周期快照，最多8份；逐根原始验证保留最近8次、第一轮、每500轮及当前best专门记录；完整聚合history、best/final和配置保留。

没有改动测台控制器；8765可以继续查看/运行实例。训练期间不要修改manifest保护的控制、观测、环境、PPO、事件或采样源码；会触发source-drift停止。文档和静态测台UI可独立更新。

## 固定训练配置

| 项目 | 设置 |
|---|---|
| 时限/并行 | 12h采样/更新预算；16常驻worker，每轮32个有效episode |
| 初始化 | V2随机初始化，不resume V1或预检查模型 |
| 模型 | 973维、双256Tanh、分离actor/critic、15类×4scale因子、联合合法组合归一化 |
| 动作 | PUSH32＋REPOSITION2＋BRAKE＋YIELD；1168静态合法有序组合 |
| PPO | lr3e-4；最多4epochs；target KL0.03；entropy0.02；gamma1；GAE时间尺度10s |
| 场景 | 沿用simple-legal-v2；4×3m空场、随机初态/目标、固定矩形；90s物理预算 |
| seed/验证 | 训练seed20261103；与V1相同32个validation根；第1轮及每10轮greedy评估 |
| test | 另用seed20261104生成32个新test根；内容hash隔离，训练期间不执行 |
| 保留 | 周期250轮/最近8快照；最近8次原始验证＋里程碑＋best；best/final/history完整 |

首次验证0/32或大量YIELD/BRAKE可发生在随机初始化阶段，不据此宣称失败或修改reward。监控学习曲线和真实行为，异常规则按数值/心跳/异常率/源码漂移/预算触发。

## 新增只读行为指标

使用协调门控后实际电机力，记录平均力、力平方归一化积分；它们不是物体接触合力或真实电机能耗。

`opposing_contact_s`：双方有物理接触、各自电机力>1N、力方向余弦<-0.5。`stalled_opposing_contact_s`另要求物体线速<0.02m/s、角速绝对值<0.05rad/s；该真值仅用于离线标签，不进入观测、mask或reward。该指标是低运动对抗的候选量，不能把制动瞬态或所有反向力都宣布为错误行为。

另记录YIELD实际执行/active/blocked机器人秒，以及BRAKE/YIELD机器人接触物体的机器人秒。两个机器人秒可以超过episode秒数；动作比例仍按两侧决策计数，不能混同物理时间占比。

本轮保留聚合和关键验证轨迹，不保存所有训练轨迹作经验库；后续可用冻结策略单独采集经验，避免训练日志膨胀。

## 预检查证据

[preflight.json](../assets/v2_launch_20261104/preflight.json)：73项聚焦测试通过；正式256隐藏层、16worker两轮，共64训练episode＋8验证episode，0异常。PPO概率重算最大误差9.54e-7；两轮KL约0.00431/0.00459，YIELD决策占比约0.421/0.419，仅为接线证据。

真实停止探针正常保存退出，外层STOP传到训练器，约5.9s完成。首次探针因包装器提前在要求空目录中创建log而被监督器拒绝，0物理；保留该失败，不计入64条正式采样。

7项D010/D005历史失败已经在修改前V1复现，见V2控制器审查报告；本次未宣称全测试套件通过。预检查目录与正式训练分开，不将其episode混入正式学习统计。

## 运行中与结束后的入口

正式目录：`outputs/option_rl_v21/event_12h_20261104/`。主进程写`run/history.json`、`checkpoint.pt`、`best.pt`；结束写`policy.pt/result.json`。监督器写根目录`plan.json/processes.json/monitor_summary.md/supervisor_result.json`。

Dashboard计划入口：**http://127.0.0.1:8772/event_12h_20261104/dashboard.html**。本地监督器继续执行，不依赖聊天会话在线；它不会自动续训或自动完成未来人工复核。

先保留best与final，结束后按明确选择规则做test和逐实例行为检查。V1已有test@2580结果不能绑定V2；新test不用于挑checkpoint。若YIELD仍未减少内耗，再用有界对照检验施力成本，不在这次运行中途加惩罚。
