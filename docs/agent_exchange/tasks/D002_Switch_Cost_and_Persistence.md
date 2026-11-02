# D002：持续执行验证与衰减切换成本短对照

2026-10-29，GPT规划v1；ready_for_handoff。用户已要求确定下一轮调整与测试。
主要方案、数值和分流条件已定，执行侧不自行选择超参数或扩展方案。
先读[用户说明](../briefs/D002_Why_This_Experiment.md)和[固定配置](D002_config.json)。

## 1. 问题与冻结项

检验“减少刚开始执行就换模式，能否让现有RL更稳定完成任务”。
先用同状态物理分支检验持续执行是否有价值；只有通过预定门槛，才做切换成本训练对照。
门槛未通过也是有效结论；停止，不换课程、网络或参数。提高同推比例本身不算任务改善。

冻结actor周期0.8 s、观测/记录5 Hz、控制20 Hz、物理120 Hz、任务90 s、原成功判据、
46动作与掩码、底层控制器、flat 256×256分离actor/critic、联合类型头、原persistence_bias。
只用joint第40轮作为共同父模型，independent保留作历史参照，不增加第二套架构训练。
0.5 s不符合当前0.2 s整数倍检查，本轮不改调度/频率。
用户本轮讨论允许将切换成本作为受控训练候选；它改变训练目标，不改变原环境任务reward。
原任务回报与新增成本分开记录，评估排名只用原任务指标。

## 2. 实现与输入

- 新实现放 `experiments/option_rl_v17/`，版本 `v17-age-decay-switch-cost-v1`。
  复用v16 PPO/控制/场景，runner和worker适配放v17，禁止改写冻结v13–v16行为。
- 父模型：`outputs/option_rl_v15/joint_ablation_002/joint_types/joint_types/checkpoint.pt`，
  SHA256 `dac637e6d21f64d6fa3bf98c658c176d12b1ac7409b66fe6ff84186e5121e6c5`。
- panel：`outputs/option_rl_v14/learnability_002/panel.json`，
  SHA256 `a60bdcc120923a8091d8e8930eb4717406075aeeb55b5291e3f924dd48227126`。
- D001输入：`outputs/agent_exchange/D001_20261029_1644/`，不得修改。
- 产物放新目录 `outputs/agent_exchange/D002_<run>/`，存task/config、输入hash与完整源码快照。
- 这是版本化warm-start对照，不是v16精确resume；不能绕过原resume校验冒充原协议续训。

## 3. A阶段：原后续与坚持3.2秒，12对固定状态

12个源边界写死在D002_config.json，每根6个、每episode最多1个。
选样未按最终成败/分支效果筛选：候选是长度1的same_push请求段，下一决策不含BRAKE且改变
至少一侧非制动模式，本段结束两侧exit均为continuing；按seed、decision_index排序，各episode
取首个名义切换成本≥0.01的候选，各根取前6个。名义成本仅按动作历史/时钟推算，未应用失败豁免。
这是对“成本会作用的短暂同推”的定向诊断，不是全部状态的无偏样本。

每个边界两条分支各自从reset重放D001的5 Hz命令前缀，不能只设位置/速度。
保留控制器、pending、握手、扫描历史与内部计时。到source decision的t_start检查：
raw_obs最大绝对误差≤1e-5、mask完全一致、记录中的物体/机器人位置角度误差≤1e-5
（角度取包裹差）、总period一致。12对必须全部通过；不匹配就incomplete，不换样本。

从该次同推开始，两分支各执行3.2 s（64 control tick）：

- `recorded`：逐0.2 s重放原execution_samples，到窗口终点或原任务终止。
- `hold`：整个窗口维持source的(a0,a1)，方向/速度均不变，每0.2 s调用原joint_step。
  原安全、失败、恢复全部生效，不重置失败锁存，不加真值制动。局部失败按原控制器处理、
  仍计时；任务success/anomaly/budget立即终止。

共24次前缀重放和24条短窗口，不算24条新90 s任务。
窗口截断记diagnostic truncation，不能作为廉价任务失败/完整成功进入任务回报比较。
按20 Hz只读记录：请求/提交动作、stage、ready、contact_ready、joint_active、pending、gap/有效性、
局部失败/等待、真实接触、施加力、物体位置速度/距目标。真值仅离线评估；保留5 Hz回放。
不得改变控制顺序；recorded分支与旧日志重叠段检查轨迹一致。

每对计算：

- progress=d_start−d_end；delta_progress=progress_hold−progress_recorded（米）。
- engaged_seconds：两侧提交动作均为同向平移PUSH，且tick开始时双方contacting_object、
  contact_ready、joint_active均为真，按0.05秒累加。此指标表示接触/握手条件，仍结合位移解释。
- push阶段时长另列；切换/等待、局部失败、异常、任务成功、最小距离也单列。

**A通过门槛，必须全部满足：**

1. 12对完整且可复现，hold无新增物理异常；
2. ≥6/12的delta_progress≥0.01 m，且每根≥2/6；
3. delta_progress中位数≥0.01 m；
4. delta_progress≤−0.02 m的严重退化不超过2/12；
5. ≥4/12的engaged_seconds增加≥0.40 s。

这是预定工程筛选规则，不是统计显著性声明。未通过就交付A并停止，B不运行，不改阈值或样本。

## 4. 切换成本定义：每侧最大0.08，2.4秒归零

仅A通过才实施B。成本在actor决策边界计算一次，不能20 Hz重复扣分。
每机器人维护动作/自身反馈可重建的ledger。非BRAKE模式族固定5种：
TRANSLATE（8方向合并）、ROTATE_CCW、ROTATE_CW、REPOSITION（CW/CCW合并）、FOLLOW。
与原mode_key的模式重建边界一致；速度、平移方向、RP换向不收费，不按动作ID差异收费。

ledger字段：last_family（5种或NONE）、last_change_time、failed_since_change。
时间由整数control period换算；reset为NONE、failed=false。
决策前读取已生成的自身反馈，再算cost、更新ledger，最后执行动作：

1. requested=BRAKE：cost=0，last_family/last_change_time/失败位均保留。
2. last_family=NONE且requested非BRAKE：cost=0，设置新family、当前时间、failed=false。
3. requested等于last_family：cost=0，不重置时钟；同族速度/方向改变也不重置。
4. requested为另一非BRAKE族：age=now−last_change_time；
   `cost_i = 0 if failed_since_change else c_max * max(0, 1-age/2.4)`；
   无论cost是否为0，更新为新family、当前时间、failed=false。

失败位只由**已生成**的own controller_failed反馈触发，且已提交执行模式族与last_family相同。
触发后保持到下次不同非BRAKE族请求；pending期间旧模式失败不得污染新族。
peer_wait、感知等待、准备耗时不直接豁免；不读物体真值或未来反馈。
监听在原5 Hz边界/真实局部失败提前交还处进行，不窥视未生成的信息。

age是“距最近一次非制动模式改变的物理时间”，包括准备、切换制动和中间BRAKE时间，
不是累计有效推力时间。BRAKE不会立即清成本，但满2.4秒自然归零。
首次工作免费；A→BRAKE→A免费；A→BRAKE→B仍按A建立时刻计算。
自动恢复/等待/安全制动不改ledger。actor改变pending到另一族算一次请求变更，
重复相同pending不重复扣分。时钟从请求开始，避免握手时长令成本无限不衰减。

每决策总cost=cost_0+cost_1≤0.16，不加其他bonus、cost封顶或动态系数。
`r_train = r_task - cost_0 - cost_1`，只进入当前actor Step的训练reward与critic目标。
环境reward、原components、success、dense物理reward标签保持原值；单列训练reward/cost。

| age | 单侧最大cost |
|---|---:|
| 0 s | 0.080000 |
| 0.8 s | 0.053333 |
| 1.6 s | 0.026667 |
| ≥2.4 s | 0 |

2.4秒=3个actor间隔；0.08≈原任务2秒时间成本，双侧最坏0.16≈4秒时间成本，远小于+10成功奖励。
这是固定的保守首个候选，非已验证最优值。不做参数扫描，负结果保留。

## 5. 两组共同追加16维可观察ledger

BRAKE前的历史不能暗藏在reward里。每机器人追加8维：
one-hot `[NONE,TRANSLATE,ROTATE_CCW,ROTATE_CW,REPOSITION,FOLLOW]`（6维），
min(age/2.4,1)（NONE时1），failed_since_change。robot0在前，robot1在后。
观测944→960，均来自自己的命令/反馈历史和公共时钟，不加隐藏物理或几何pair特征。

A0/A1同样使用960维、ledger与日志，只差c_max。
父模型actor/critic首层权重右侧补16列零，其余权重/偏置原样拷贝；normalizer原944维和count继承，
新mean=0、var=1。只归一化前760维，后续原样输入；两组都在每批PPO后更新normalizer一次。
任意新ledger值不应改变初始化时的父策略概率/value，验证误差≤2e-6。
persistence_bias继承父模型实际值并正常学习，不归零、不调强、不替换手工规则。
新模型声明960维/ledger-v1/训练reward版本；原v16加载器不得静默接收。
评估仍更新ledger，但不按cost手工修正动作、不强制保持，只看策略自身选择。

## 6. 四次配对短训：两组×两个后续随机流

| 组 | c_max | 其余 |
|---|---:|---|
| A0 | 0 | ledger、960维、初始化、PPO同A1 |
| A1 | 0.08/机器人 | 同A0 |

共同父模型joint第40轮；fork_seed=200/201，各20轮，每轮原8根各1个episode。
总4×20×8=640训练episode。这是两条后续随机流，不是两次独立从头训练。
顺序A0/200、A1/200、A0/201、A1/201；最多4个rollout worker，不并行启动多套4worker。
两组均重建actor/critic Adam，lr=3e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，
避免输入扩展后的旧Adam状态迁移混淆。继承权重/normalizer，不继承旧优化器状态；不重置critic权重。
torch PPO shuffle RNG=fork_seed；第k轮第j根动作seed=`behaviour_seed(fork_seed,8*(k-1)+j)`，
j按panel原顺序0..7。复制初始化不消耗后续随机流；同seed首批A0/A1物理动作轨迹应完全一致。

固定gamma=1、GAE trace_seconds=10（真实dt）、4epochs、minibatch256、clip0.2、entropy0.02、
value_coef0.5、actor/critic各自grad_clip1.0、target_kl0.03（沿用1.5倍早停阈值）。
优势按完整batch标准化，不改任务权重/采样/课程；不对cost单独标准化。
任何训练异常、NaN、old logp重算误差>2e-5都停止该run，保留现场，不补采。

## 7. 评估及判定

- 更新前16根贪心（8训练+8既有1cm扰动）只跑一次；4run初始化相同，复用同一基线。
- 每run第5/10/15/20轮评估16根贪心，保存完整快照；训练/扰动分别报。
- 第20轮每run两个dual训练根各16条随机，seed1000..1015，共128条；这是复用的开发种子。
- 完整episode预算：训练640+贪心272+随机128=1040；A的24短分支/前缀另列。
- 不用best.pt挑结果，主比较固定20轮；同时检查10/15/20轮保留性。

逐根/每fork报告success、原回报、进入目标未判稳、末距/最小距、任务秒数、非BRAKE族切换数及
每模拟分钟次数、连续请求段、engaged_seconds、等待/失败时长、task/training reward、总cost与
收费次数/BRAKE豁免/失败豁免。失败仍按原90秒规则计费，不能靠少切换而隐藏停滞。
固定回放每run末轮两根dual贪心及每根seed1000随机，共16条，不挑最好样本。

“值得下一轮复验”的工程门槛（全部满足，不是显著性检验）：

1. 4run均完成20轮及指定评估；
2. 每fork的最后3次dual贪心成功数A1≥A0，训练/扰动分别判；
3. 两fork合计，训练dual最后3次成功数至少多2/12，扰动dual也至少多2/12；
4. 单机4根最后3次成功数不下降（每fork、训练/扰动分别判），换面成功单列不得隐去退化；
5. 最终随机dual成功数每fork≥A0−2/32，合计原任务平均回报≥A0；
6. 无新增物理异常。

达标仅记promising，后续仍需独立初始化/更多几何验证，不自动长训。
不达标记no_clear_gain或regression，注明未满足项，不调整c_max/T重试。

## 8. 验证与硬预算

只跑针对性测试，不跑全仓库pytest：

- ledger数值、真实不等dt、总cost≤0.16；首次免费、同族不重置、不收费；
  BRAKE不收费且不清历史，A→BRAKE→B按A时刻计、A→BRAKE→A免费。
- 匹配族已生成失败才豁免、等待不豁免；重复pending不重复费、pending改族一次费；两侧独立、reset无残留。
- 944→960保留初始概率/value≤2e-6、前944输入不变；相同合法历史下改变隐藏真值不能改变ledger/cost。
- 同seed首批A0/A1物理动作/logp一致、只有training reward差cost；利用正式首批，不另跑episode。
- 原task reward从物理日志独立重算；training差额恰为cost，dense标签/物理reward未污染。
- PPO old logp≤2e-5、GAE终止隔离、checkpoint保存/加载保留新维度与ledger协议。
- STOP/deadline覆盖训练、训练根/扰动/随机评估；停止后清理子进程。

数值测试总上限180秒。正式仿真/PPO/分析/评估执行总上限1200秒，最多4worker：
A≤120秒；每run含周期评估≤210秒（4个≤840）；最终随机/汇总≤240秒。
记录全部执行段（含失败尝试）、总elapsed和人工实现时间，不能只计最后成功一次。
A通过后可以结束A进程，再实现B；B启动从manifest读取A已耗执行秒数，执行总和≤1200秒。
正式各阶段内用外层墙钟deadline兜底，不修改运行中源码；发生bug停止并交回incomplete，
不自动修复重跑正式实验或换seed。到期保留完整已完成快照；不得为凑20轮减episode/物理精度。

## 9. 交付与停止

manifest、A逐对CSV/JSON/回放与门槛布尔项、聚焦测试结果；进入B则补4run全部rollout/快照、
周期及随机评估、分开的任务/训练回报表与回放。
回报写 `results/D002_Switch_Cost_and_Persistence.md`，先用自然语言回答：
“多坚持是否有用？惩罚是否转化为稳定成功？有没有妨碍停止或退出失败动作？”
更新CURRENT为ready_for_review并停止，不自动选新参数或D003。
