# 当前工作状态

**迁移暂停（2026-11-25）**：本机频繁重启，用户要求立即上传GitHub并换PC接续。长期goal已暂停；优先阅读[迁移交接](FOUR_STAGE_MIGRATION_20261125.md)。后文“运行中”是暂停前记录，非当前进程状态。

<!-- CAP_FOUR_STAGE_ACTIVE -->
**2026-11-25：四段式 CAP 自主闭环实验运行中。** 用户授权持续goal。64项针对性测试通过。C++前端把一条440决策/1800tick在线轨迹降到7.59秒（前端1.85秒），仍未达到一秒一局。首轮8开发房成功0/8、推进0。首批32物理房完成5627条分叉；第二轮仍停滞，独立审计发现两个早停探针各2/3次经验放行运动越界，不能以零接触宣称安全。正在补采切换状态，修补预测支持域、包络校准及停止依据。最终v0/v1/v2与确认未完成。[开发报告](../research/key_findings/2026-11-25_four_stage_loop/README.md) · [研究面板](http://localhost:8775/research-journal#four-stage-loop)。提交273b45b已由用户推送，之后改动另计。

<!-- CAP_FOUR_STAGE_DESIGN -->
最新执行语义：预测器可读当前原始观测和合法历史，只预测一次动作后的语义与过程摘要；检查中复用第三段但不递归调用未来预测器。运行时按分数选首个通过当前检查的候选，可扩展准备动作，不要求正分或单步直接收益；实际下一边界用新观测重算。一步限定深度，同一边界仍可预测多个候选。以下为设计状态，不是已运行结果。

**2026-11-25：四段式接口与分阶段闭环设计已对齐，尚待实现。** 准备公开证据、注册语义、纯函数生成候选与评分、宿主预测检查；真实决策与预测复用同一份第三段代码。同输入、记忆与受支持运算下，语义注册应保持规则策略表达能力；预测摘要是否充分另行核验。首轮开发暂缓强制预测接续检查，同步采集；框架补充分叉拟合模型，先只报告不拦截，允许第二轮针对反例修订，再冻结策略与第0版开展模型迭代。保留技能限制与已有可靠安全检查，未验证项不记通过。后继可启动不等于下一边界实际放行。房间划分与各项数值预算尚待冻结；此次仅更新设计，无新增实验结果。[接口与验收](../research/future_directions/2026-11-25_cap_contract_responsibilities.md) · [作者材料](../research/future_directions/2026-11-25_cap_author_packet.md)。

<!-- CAP_SYMBOLIC_MAINLINE_SCOPE -->
**2026-11-25：主线优先符号决策与拟合预测，残差价值学习留作拓展。** 用户明确要求减少同时变化的因素。先整合可抽取符号评分、启用条件、同边界备选与准备动作，以及预测后继中的同一选择语义；再接入联合效果模型和在线rollout。作者开发完成后冻结程序、评分参数、候选、门槛及试算触发和预算，只更新效果模型开展配对比较。主线不增加价值网络或优势学习；安全停止与有效推进分别报告。[范围与实现顺序](../research/future_directions/2026-11-25_cap_contract_responsibilities.md) · [作者材料](../research/future_directions/2026-11-25_cap_author_packet.md)。本次更新为范围确认，整合实现尚待完成。

<!-- CAP_CONTRACT_RESPONSIBILITIES -->
**2026-11-25：CAP 作者与宿主的责任边界已整理。** [接口设计](../research/future_directions/2026-11-25_cap_contract_responsibilities.md) · [作者初始材料模板](../research/future_directions/2026-11-25_cap_author_packet.md)。明确预先给定的规格和动作语义、作者可发现的派生条件与策略、模型的联合效果职责及检查器保证范围。下一步整合须区分启动、过程安全、完成与终止，并把新版unknown证据接入作者CAP和预测链。本轮是文档整理，新增字段尚非现行API；未启动新实验。下方保留已有冻结证据。

<!-- UNKNOWN_SAFETY -->
**2026-11-25：unknown、安全观察与完整停止的小场景验收已完成。** 旧14个首次箱体静态接触全部复现：8个障碍此前在策略边界扫描中未见，6个已见且仍在记忆中。新12组冻结三臂复测：空场景观察后有效推进并完整停止12/12，遮挡障碍观察后拒绝12/12，不观察推进0/12；36局无接触、无范围越界，完整停止36/36。结论依赖静态传感器、最小障碍面积、初始分离、CAD及运动范围假设；尚未重测旧64房间或建立整局安全。13项针对性测试通过，旧914个冻结源文件未变。[报告](../research/key_findings/2026-11-25_unknown_safety/README.md) · [任务书](tasks/CAP_Unknown_Safety_20261125.md) · [回放](http://localhost:8775/journal-file/unknown-safety/replay.html)。下方保留上一轮冻结结果。

<!-- SYMBOLIC_GENERAL -->
**2026-11-25 用户授权的 CAP 可抽取符号策略框架与三版预测器闭环：实现及评估已完成。** [完整报告](../research/key_findings/2026-11-25_symbolic_general/README.md) · [执行记录](tasks/CAP_Symbolic_General_20261125.md) · [研究面板](http://localhost:8775/research-journal#symbolic-general)。64间确认房，每版1024候选，v0→v2 的 C 为48.44%→60.45%、F为9.12%→8.18%、K为155.08→91.80，预定局部主假设得到房间配对区间支持。整局v2仅1/64，同一CAP无预测门33/64，v2仍有14间箱体静态接触：局部收益没有转化为较高整局完成率，安全保证未成立。原始证据 `outputs/semantic_rl/cap_symbolic_general_v1/` 已冻结；下方保留历史主线。


无门控 A/B 路由的默认更新采用分叉监督和全局软保持，成功轨迹回放仅在显式对照中启用。[单轮消融与适用范围](../research/key_findings/2026-11-23_ungated_no_replay/README.md)

[未来研究方向讨论](../research/future_directions/README.md) · [论文关键发现](../research/key_findings/README.md)

[区域策略的邻域与决策时刻诊断](../research/key_findings/2026-11-20_regional_advantage/README.md)已完成。96 个父房间，首次分歧时读出 206/353，固定动作 actor 211/353。[面板同步回放](http://localhost:8775/advantage-replays)

<!-- ACTOR_REPLAY -->当前 actor 加历史回放已完成 24 轮和 200 次最终闭环。回放 105、仅当轮 109、监督起点 117；旧标签保留改善。96 个同前缀分叉的 192 个首步中仅 actor 成功 19 个、仅提案专家成功 9 个。[主报告](../research/key_findings/2026-11-20_regional_actor_replay/README.md) · [后缀诊断](../research/key_findings/2026-11-20_regional_actor_replay/suffix.md)

<!-- ACTOR_ITERATION -->十维 actor 的分叉迭代已完成 24 轮。最终 200 房间中监督起点 117 次成功、迭代 109 次、冻结采样集中拟合 112 次。[报告](../research/key_findings/2026-11-20_regional_actor_iteration/README.md)

单次改选完成 200 个开发房间的分叉采集及 200 对新房间验收。728 个分叉能救回 45/89 个原失败房间；冻结候选在验收中救回 21 个、丢失 15 个，单侧下界 −4.61 个百分点，保留原监督 actor。[单次改选报告](../research/key_findings/2026-11-20_regional_once/README.md) · [监测面板](http://localhost:8775/#single-override)

十维编号 actor 完成 2,000 局 PPO 微调及 505 局固定模型闭环。自然留出成功 113→120/200，最佳单 CAP 为 117/200，困难族 30→30/90。十个 C++ 策略通过配对验证，16 进程完整采样吞吐提高到 1.92 倍。[结果、训练曲线与 C++ 对照](../research/key_findings/2026-11-20_regional_id_ppo/README.md)

十维 CAP 编号 actor 硬标签对照完成 1,010 次闭环和 215 条留出教师轨迹重放。固定预算下，编号和合并动作 actor 分别在自然留出成功 113/200、110/200，在困难族扰动成功 30/90、23/90。[评测与切换诊断](../research/key_findings/2026-11-20_regional_id_actor/README.md)

同数据、同预算的时间软标签对照已完成 1,010 次闭环。6,150 个训练边界中 272 处有多个实测成功动作，固定 24,000 步的自然留出成功数从硬标签 110/200 到软标签 106/200，困难族扰动 23/90 到 26/90。[配对结果与训练分歧诊断](../research/key_findings/2026-11-20_regional_soft_actor/README.md)

最快成功教师 actor 在十成员库的 215 个训练房间上完成闭环。原 directQ 保留专家并集 108/161，单教师 actor 24,000 步保留 160/161，低学习率续训的训练选定检查点保留 161/161。两个冻结 actor 随后完成 580 次留出执行，训练选定模型在自然留出成功 118/200、困难族扰动成功 24/90。[留出配对结果](../research/key_findings/2026-11-20_regional_fastest_actor/held.md)。[逐场景成败与首次偏离](../research/key_findings/2026-11-20_regional_fastest_actor/README.md)

十成员区域库独胜收益对照已完成 505 次闭环。成功轨迹的 V 目标改为 1/N 后，18 个留出独胜房间的赢家排名中位数从 5 提高到 2.5，directQ 在同一子集成功 2→3 例，全部留出房间成功 112→88/200。[逐场景对照](../research/key_findings/2026-11-20_regional_inverse_v/README.md)

区域作者第 6–10 轮及新旧 directQ 配对评测已完成。200 个新房间上五成员→十成员并集为 130→160，开发选定单程序 107→117，原 Q→新 Q 为 110→112。新增五轮修复原失败父房间 19/45，其中 8 族由同一新增成员同时解决父房间及两个未见扰动。[报告与训练预算](../research/key_findings/2026-11-19_regional_extension/README.md) · [评测面板](http://localhost:8775/#ten-cap-directq)

五轮区域精化、常规增量、同预算通用 CAP 已完成 3,960 次留出闭环。200 个新房间上区域最佳单程序 119 例成功，同库开局路由 113 例，五成员并集 139 例。[作者对照](../research/key_findings/2026-11-19_regional_cap_authoring/README.md) · [区域成员逐轮地图与 directQ 进度](http://localhost:8775/)

区域库无 trace directQ 已完成 seed17 的 520 次闭环。相对区域最佳单程序，开发池 90 对 92，新房间 122 对 119，开发邻居 95 对 93；新房间配对区间跨零。[逐场景配对](../research/key_findings/2026-11-19_regional_directq/README.md)

最高剂量 GLOBAL-attention＋trace 已完成全部 600 个闭环。固定 seed17，attention 成功 314 例，无 trace GA-Q 成功 323 例，配对差 −1.50 个百分点，95% 区间 [−5.50, +2.50]。[闭环对照](../research/key_findings/2026-11-19_global_attention/README.md) · [程序特征与样本选择假设](../research/future_directions/2026-11-19_policy_switch_representation.md)

最高剂量专家 V 切换与动作 Q 配对已完成。600 个新初态上 V 切换 260 例成功，GA-Q 323 例，最佳单 CAP 267 例。V 开局全部选择 CAP-009，37,129 个实际动作中只有 3 个偏离其提案。[配对结果与行为诊断](../research/key_findings/2026-11-19_cap_policy_switch/README.md) · [研究定位](../research/future_directions/2026-11-19_policy_switch_representation.md)

[按原场景成功数反比采样的千条规模曲线已完成](../research/key_findings/2026-11-19_inverse_scale/README.md)。200 个可解开发族，新增成功序列剂量目标 0／250／500／750／1001，固定 seed17，600 个新初态评估。[进度页](http://127.0.0.1:8767/) · [协议](../../outputs/semantic_rl/cap_inverse_scale_20261119_01/protocol.json)

场景族定向增广实验已完成。每组 5,376 条训练轨迹，普通组新增 364 条、定向组新增 226 条不同的成功动作序列。新面板 seed17 原始／普通／定向 Q 为 96／102／112 成功；独有场景 10／11／16（共 26），定向对普通的主比较区间跨零。[关键数据与启示](../research/key_findings/2026-11-19_family_augmentation/README.md) · [结果页](http://127.0.0.1:8766/report.html)

无 trace 的冻结离线 GA-Q 与 24k 单教师 GA 已在同一诊断面板配对。两种子平均成功率 53.00% 对 43.75%，差 +9.25 个百分点，场景配对 95% 区间 [+3.50, +14.75]。七 CAP 并集保留率差 +10.27 个百分点，28 个单 CAP 独有场景的差值区间跨零。[结果与配对回放](../../outputs/semantic_rl/cap_offline_q_closedloop_20261119_01/REPORT.md) · [论文关键发现索引](../research/key_findings/README.md)

无 trace 的 64×64 单教师 GA 完成 400 次闭环执行。同一 200 场景上，6,000 到 24,000 步的平均总成功率从 37.00% 升至 43.75%，七 CAP 并集保留率提高 9.25 个百分点，场景配对 95% 区间 [+1.42, +17.22]。独有场景和并集外场景没有观察到净提升。[闭环结果与配对回放](../../outputs/semantic_rl/cap_actor_closedloop_20261119_01/REPORT.md)

64×64 单教师 actor 的 GLOBAL trace 对照已完成。固定 24,000 步，GLOBAL 相对 GA 的留出教师一致率差 +0.05 个百分点，场景配对 95% 区间 [−1.82, +1.96]。两个种子的训练一致率均提高，留出交叉熵均升高。本轮只测离线拟合。[结果与接入方式](../../outputs/semantic_rl/cap_actor_trace_20261119_01/REPORT.md) · [训练面板](../../outputs/semantic_rl/cap_actor_trace_20261119_01/live/index.html)

单教师 GA actor 的容量与预算对照已完成。64×64／6,000 到 128×128／24,000，训练教师一致率从 72.86% 到 99.47%，留出从 71.28% 到 75.07%，留出交叉熵从 0.72 升至 3.30。四模型为纯监督拟合诊断。[报告](../../outputs/semantic_rl/cap_actor_capacity_20261119_01/REPORT.md) · [逐种子面板](../../outputs/semantic_rl/cap_actor_capacity_20261119_01/live/index.html)

GA actor 单教师与成功动作集合对照已完成，四模型共 800 次闭环。在同一批 200 个已查看场景上，seed17/29 的单教师组成功 69/79 例，多解组成功 72/85 例；多解组相对单教师组的七 CAP 并集保留率差 +2.05 个百分点，场景配对 95% 区间 [−6.55, +10.78]。两组均直接监督训练至固定末轮。[拟合、保留与配对回放](../../outputs/semantic_rl/cap_actor_distillation_20261119_01/REPORT.md)

新 CAP 拟由 LLM 直接用 C++ 编写完整策略，模板默认预留可关闭采集的 trace 接口。当前为设计提案，后续先验证一个新 CAP，再决定是否切换作者默认语言。[原生 C++ CAP 方案](../development/Native_CPP_CAP_20261119.md)

七 CAP 可信实验入口默认启用共用的 `clearances` 和 `navigate` C 内核。固定 24 初态、12 进程，完整回放每批从 17.77 秒缩短到 10.78 秒，192 次计时执行均与归档一致。[对照报告与复现命令](../../outputs/semantic_rl/cap_shared_kernel_20261119_01/REPORT.md)。

可信实验入口已默认启用冻结 CAP009 的位姿规划 C 内核。七 CAP 路由的 24 场景测速中，共用的 `clearances` 和 `navigate` 分别占累计 worker 时间的 16.7% 和 16.3%。[默认配置](../../experiments/planar_tasks/semantic_rl/program_trace_value_v1/pose_kernel/README.md) · [逐 CAP 耗时及行为核对](../../outputs/semantic_rl/cap_shared_timing_20261119_01/REPORT.md)。

**监督学习与能力保留对照已完成。** 6,000次新面板执行；七CAP并集146/200。自由重选相对开局选一次在六个条件中均有净救回。半批成败配对采样的主比较为 −3.77 个百分点，95%区间 [−10.74,+3.15]。[结论与回放](reviews/CAP_Retention_Supervised_20261119.md)。输出 `outputs/semantic_rl/cap_retention_supervised_20261119_01/`。

**seed17 接续 2000 episode 已完成。** 新200场景上，GA 94→105，GLOBAL 103→97；差距变化 −8.5 个百分点，配对95%区间 [−17.0, 0.0]。[报告](assets/cap_trace_ppo_continue_20261118_01/REPORT.md)。输出 `outputs/semantic_rl/cap_trace_ppo_continue_20261118_01/`。

**GA / GLOBAL 三种子 PPO 确认实验** · <!-- TRACE_PPO_STATUS -->已完成。新确认集上，GLOBAL＋PPO 为 56.83%，GA＋PPO 为 54.00%，配对差 +2.83 个百分点，95% 区间 [−1.33, +7.00]。两组微调收益之差 +8.00 个百分点 [2.50, 13.33]。[完整报告](assets/cap_trace_ppo_confirm_20261118_01/REPORT.md)。输出 `outputs/semantic_rl/cap_trace_ppo_confirm_20261118_01/`。

<!-- PPO_SEED17 -->
**seed17 PPO对照已完成。** PPO 117/200，同顺序DQN 93/200，共同起点112/200。2888次新增执行完成。[报告](assets/cap_trace_value_ppo_20261118/REPORT.md)。
<!-- PPO_SEED17_END -->

<!-- LONG_ONLINE_START -->
**分层洗牌＋500更新筛查已完成。** 新组93/200，旧组108/200，差-7.50pp。
[报告](reviews/CAP_Trace_Value_Shuffle500_20261118.md)；[曲线](assets/cap_trace_value_shuffle500_20261118/dashboard.html)。单seed、已查看诊断，两因素组合不能独立归因。2888新增执行完成，新确认集0次；停止，不自动扩预算。
<!-- LONG_ONLINE_END -->

**CAP trace＋价值学习完整实验已完成，状态`complete`。** [完整评估与验收报告](reviews/CAP_Trace_Value_Full_20261118.md)。新冻结200测试上，GLOBAL＋RL三个seed成功112/113/107，平均55.33%；原10CAP最强CAP003为86/200（43%）。多CAP价值选择有明确净收益，但trace相对无trace基线仅+2.17 pp、GLOBAL在线相对离线+3.17 pp，两项配对区间均跨零，尚不能独立确证trace或在线RL的稳定额外收益。GA离线已达53.50%，GA在线53.17%，LOCAL在线51.50%。

正式9份模型各完成16000次V预训练、8000次Q初始化、200在线任务及6000次Q更新；8264正式物理执行全部完成，主实验与报告均退出0。18模型版本×200＋10单CAP×200＝5600最终执行，共享200个不同初态。所有模型先冻结后测试；没有挑seed、checkpoint或按测试追加训练。新在线200根与新测试200根及已知历史内容不重合，仍为同一场景生成规则。

输出：`outputs/semantic_rl/cap_trace_value_full_20261118_02/`；宿主：`outputs/semantic_rl_hosts/cap_trace_value_full_20261118_02/`。验收从[报告](reviews/CAP_Trace_Value_Full_20261118.md)、[逐场景CSV](assets/cap_trace_value_full_20261118/per_case_results.csv)和[配对回放](assets/cap_trace_value_full_20261118/replays.html)进入。任务书：[完整实验](tasks/CAP_Trace_Value_Full_20261118.md)；此前[v7计划](tasks/CAP_Trace_Value_RL_20261118.md)及历史实验保留。本轮没有剩余执行项；后续若据此改方法，应另冻结确认测试集。

## 本轮开发数据与执行记录（已完成）

7CAP为009/003/001/010/006/007/002，在760开发根上保留560/570已知成功覆盖；完整无损覆盖仍需10CAP。[覆盖证据](assets/cap_trace_value_plan_20261118/development_cover.json)。2026-11-18新`cap_trace_value_dev_v1`已在本机全量数据验收通过：5320完整运行，1759成功、3561未成功，缺失/异常0；369726次执行决策、375046个含初态观测边界。此前4510条轨迹缺口已补齐，旧[数据缺口](assets/cap_trace_value_plan_20261118/data_requirements.json)保留为计划时的历史盘点。[本机验收报告](reviews/CAP_Trace_Value_Dev_Acceptance_20261118.md)。64条decision_budget保持独立截断标记；该次数据验收只检查已有数据，未执行CAP、插桩、仿真或训练；后续完整实验见上方。

目标机旧增量包严格特征逐值验收失败：72个D/A方向点积差异，最大4.27e-17；12模型在该参考边界评分相同，仍不等价于strict通过。[历史检查记录](assets/cap_trace_value_plan_20261118/transfer_check.json)。用户已授权直接开展新方法，不再把旧通道差异作为新训练门槛。新trace实现与原CAP已通过短前缀上下文/记忆同值，pilot全部22693边界的行为CAP建议与记录一致。

已完成336轨迹pilot与三组各2000次V更新、48次policy-switching闭环验证。依据用户新增明确授权，完整三seed离线/在线/新测试均已完成。新目标为成功1/其余0、gamma=1；旧reward不改写。软件集成检查另有12次短优化更新与1次开发根执行，单独保留，不进入研究结果；早期dtype修复前的特征提取尝试保留于pilot `_01`，未训练或执行物理任务。

## 既有v6增量移交

以下记录2026-11-17移交时的历史状态，不代表上方已完成实验的当前实现状态。

用户要求将方案落盘、提交代码并增量移交，后续大计算在另一台PC执行。[移交说明](../development/Program_Trace_Transfer_20261117.md)：依赖原`priority_teacher_v1`，新增`cap_program_trace_v1`携带本轮证据、模型、回放及相对索引，不复制约10GB派生数组。自动插桩、稳定特征schema、完整CAP建议池与新候选级评分器尚未实现。先做无行为改变及示范候选覆盖验收，再冻结下一轮训练预算与新测试集；当前没有新训练、PPO或LLM运行。

增量包已完成，约1.41GB，3770文件读回校验通过；相对索引、真实边界特征、12模型及历史donor跨目录验证通过。[验收记录](reviews/Program_Trace_Transfer_20261117.md)。两台PC根路径无需一致，不需重传已有priority包。

## 已完成

- 冻结10个随机房间CAP；270示范整合与覆盖诊断：[结果](reviews/Random_Room_Coverage_Repair_20261117.md)。
- 同270示范/28000更新/seed17，旧100已查看诊断面板G/GA/GD/GDA分别1/48/5/50成功；最佳CAP003为46，CAP并集79。R0/Rt为36/41。GD已完成100评估、封印和报告收尾。[16方法交互报告](../research/reports/CAP_Coverage_Comparison_20261117/index.html)及[分层读数](../research/reports/CAP_Coverage_Comparison_20261117/analysis.md)。
- 单训练种子、固定已见诊断，不作为新盲测；覆盖相似不等于行为相同。

## GDA扩数据已完成

[结果报告](reviews/GDA_Coverage_Scaling_20261117.md)：原270与扩570示范各3seed、42000更新，全部7800次采集/评估完成，宿主退出0。新200例seed17/29/43：原270为86/80/85，扩570为74/77/90；教师并集153/200。平均收益不明确，不能只看单seed推断扩数据必然有益。

权威产物：`outputs/semantic_rl_authors/random_room_gda_scaling_20261117_03/RESULT.json`、`study.exit.json`。当前没有该任务的剩余评估。

## 源PC已完成：D/A 联合表示

以下为源PC移交记录；对应完整outputs未随Git迁入当前目标PC。用户已明确授权自主实现和实验。[本轮报告](reviews/DA_Joint_Representation_20261117.md)、[协议](tasks/DA_Joint_Representation_20261117.md)、[实现与解释边界](../development/DA_Joint_Representation_20261117.md)。监督器退出0；输出 `outputs/semantic_rl/da_joint_20261117_01`，权威结果 `outputs/semantic_rl_authors/da_joint_20261117_01/RESULT.json`。正式宿主约3小时20分钟，无新CAP/PPO/API调用。

48 donor、112分叉、12fit（各28000更新）、1536模型评估全部完成。24身份、44首动作配对和288模型接管前缀审计通过；270示范旧通道精确一致。新80初态两seed：原数据current 50/44、enriched 46/46、joint 47/47成功；补10后缀后分别40/41、49/50、46/45。联合乘积没有显示额外优势；enriched补数据自身提升4.375个百分点，但区间跨零。所有修复模型在10个示范接管点均10/10成功，从同10场景初态只有5～8成功；20次失败均在原第8接管边界前偏离旧donor前缀。局部修复可学习成立，稳定泛化与能力保持尚未成立。完整逐场景比较、排序诊断、图表和局限见报告；没有自动启动下一轮。

## 教师优先级对照已完成

最快成功教师 vs 开发成功数优先教师对照已在目标PC完成（[执行回报](results/Priority_Teacher_20261117.md)）。fastest三seed成功89/82/75，priority为87/82/83，各组分母200；平均41% vs 42%，未显示跨seed稳定收益。六份42000更新及1200评估全部完成、错误记录0、退出0。迁移准备的历史验收见[记录](reviews/Priority_Transfer_20261117.md)及[说明](../development/Priority_Transfer_20261117.md)。当前不新增CAP、RL或模型调用。

### 目标PC执行（2026-11-17）

用户已明确移交并授权本机完成固定优先级对照。执行状态：`complete`，交接状态：`ready_for_review`；等待用户决定后续，不自动扩展实验。
目标路径为 `/home/tzl/repo/coimagine`；未修改冻结数据或运行源码。压缩包、1244负载文件、运行源码哈希及真实示范四级特征/目标/历史逐值验收通过，指定四组测试为11 passed。

- 持久会话`priority_teacher_20261117`已随监督器正常结束；2026-11-17 16:54:39至20:01:38（Asia/Shanghai），总墙钟约3小时7分钟。
- 实际入口：`outputs/transfers/venv_py311/bin/python -u -m tools.priority_transfer run --bundle outputs/transfers/priority_teacher_v1 --out outputs/semantic_rl/priority_teacher_20261117_01 --workers 8`；`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- 环境：保留原 `.venv`，独立安装Python 3.11.18（源机3.11.4）；交接指定六个包版本均匹配。系统bubblewrap与开发头文件已补齐，编译隔离保留。
- 预算：两组各570示范、seed17/29/43、各42000更新/7200秒训练上限；4提取进程、6模型顺序拟合、8评估进程；1200执行共享200已查看诊断初态，每例1800ticks/300决策/1800秒。
- 验收和环境记录：`outputs/transfers/priority_teacher_preflight_20261117/preflight.json`；日志：`outputs/transfers/priority_study.log`。
- 产物：`outputs/semantic_rl/priority_teacher_20261117_01/`；`FINISHED.json`及`study.exit.json`均complete。模型哈希/更新数/评估分母核实通过，已生成`AUDITED_SUMMARY.json`、`per_case_results.csv`和`REPORT.md`。
- 无剩余必需执行项。fastest/priority边界数20007/24114；拟合率及逐场景得失见回报。评估使用NumPy雷达回退；本机fastest不同于历史源机结果，跨机器完整复现未验证，不能混用两机结果形成教师对照。

## 清理与接续

[清理记录](reviews/Workspace_Cleanup_20261117.md)记录永久删除范围、账本和保留目录。保留当前CAP库、示范、冻结模型、对照评估、关系迁移实验、运行表格、基础控制器/模型和研究汇总。旧CAP/RL原始输出及旧方案已删除，不再声称能完整复现旧阶段；仓库外旧agentic暂存也已永久删除。

扩展宿主JavaScript堆OOM已被日志确认；outputs等已加入VS Code监听/搜索/Python分析排除。清理不等于已确定或修复崩溃根因。

## 2026-11-18：完整CAP trace开发数据移交

[验收记录](reviews/CAP_Trace_Value_Transfer_20261118.md)与[使用说明](../development/CAP_Trace_Value_Dev_Transfer_20261118.md)。`outputs/transfers/cap_trace_value_dev_v1.tar.gz`独立包含所选7CAP×760开发初态的5320条完整原始gzip（1759成功、3561失败），零缺失/异常；原数据完整审计、包内历史重验及6150文件归档回读均通过。附760场景、10CAP的7600摘要、程序/依赖、相对路径索引、源哈希和无评估标签的场景去重清单。没有训练、仿真或补采。接收方先验SHA256再执行包内说明的verify；内部程序trace仍须重算并验收等价性。
