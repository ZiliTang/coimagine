# 当前交接

- **跨PC打包入口（2026-11-05）**：[单压缩包、内容范围与恢复步骤](Cross_PC_Handoff_20261105.md)。最新交付位于`outputs/handoff/20261105_completed_rl/`，含完整Git bundle、V1冻结产物、V2训练结束产物、审查回放和中文说明；大文件不随Git push上传，精确hash见包旁PACKAGE_INFO.json。
- **V2正式训练已结束并复核（2026-11-05）**：[本轮总结、V1对照与行为边界](reviews/V2_Training_Final_20261105.md)。12h单调时钟到点正常保存退出，4170轮/133440 episode/5657193事件transition，训练/完整验证0异常，82项受保护源码无漂移。
- best@3900验证29/32；最近完整验证@4160为28/32；最终权重@4170尚未额外验证，test仍未运行。最近200轮训练成功均值83.8%、验证79.1%，不以best峰值代替稳定性。
- 三个重点失败根2/24/27在best和最近完整验证均成功；YIELD实际有使用，但BRAKE接触/反向施力仍存在，尚不能宣布减少内耗或优于V1。没有续训、调参或自动test；下一步由用户决定冻结test与单实例可视化。
- 原产物`outputs/option_rl_v21/event_12h_20261104/run/`的best.pt/policy.pt/source_snapshot保持；权重SHA与统计见总结附带summary.json。下一位：用户审阅。以下为开训时历史快照，running/PID不再代表当前状态。

## V2开训历史快照（已结束）

- **V2正式12h训练运行中（2026-11-04 12:03:41，Asia/Shanghai）**：[开训检查、配置和停止规则](reviews/V2_Event_RL_12h_Launch_20261104.md)。目录`outputs/option_rl_v21/event_12h_20261104/`；supervisor2462130/trainer2462169，16 worker，每轮32episode，随机初始化；计划11月5日约00:04结束，预算按单调时钟计。
- Dashboard：**http://127.0.0.1:8772/event_12h_20261104/dashboard.html**。原奖励，施力成本为0；日志新增门控后施力、低运动对抗、YIELD实际执行/blocked及非推动方接触。V1仍冻结。验证沿用V1同32根，新test用seed20261104且训练中不执行。
- 开训前73项聚焦测试通过，16worker两轮64episode＋8验证0异常，logp误差最大9.54e-7；STOP实测约5.9s正常保存退出。首轮正式采样32episode/1513transition、0异常，初始logp误差4.77e-7；仅确认接线与运行，不作学习结论。
- 82项实际导入项目源码已快照/hash保护；训练中不要改控制器、事件、环境、PPO、采样与监测Python源码，源码漂移会按STOP规则停训。8765继续可看回放/运行手工实例；V1 dashboard8771保留。自动监督不自动重启或续训，下一位：用户查看曲线/到点后复核。

## 以下为V2开训前实现与历史交接

- **Option V2 已上线，等待用户审查（2026-11-03晚）**：[改动、8个实例、检查与限制](reviews/Option_V2_Yield_20261103.md)。8765已重启，PID2260541，`option-v2-yield-signed-push`；PUSH方向贯穿选面/分配/跟踪/执行，新增YIELD(ID46)和P+Y等组合；973维/1168合法对。无新训练、无惩罚、未禁止真实对向PUSH。
- 先看[同起点BRAKE](http://127.0.0.1:8765/?run=20261103_215538_49f501)与[YIELD](http://127.0.0.1:8765/?run=20261103_215539_8fe6f2)，再看[近墙边界](http://127.0.0.1:8765/?run=20261103_215539_2baf31)。开阔例接触7.55→0s、角偏0.432→约0；近墙仍有0.5s接触/1s blocked，不能宣称完全解决。
- **上一轮RL明确冻结为第一版V1基线**：[登记](../rl_baselines/README.md)。训练已按预算结束，5331轮/170592 episode；best@4640验证32/32，末次@5330为29/32；旧best@2580 test26/32不是新best结果。best/final/完整源码686项已保留，禁止用当前改过的控制器冒充V1复现。
- 清理周期权重、冗余逐轮原始评估/worker缓存、被替代pilot，以及已压缩迁移包的解包暂存，释放约2.30GB；未删用户presets、既有RL回放和v14–v16历史诊断。最终V1包`outputs/rl_baselines/V1_event_progress_20261103.tar.gz`，23.6MB，本机未上传。此前12:50包为历史中期版本。
- 62项聚焦检查通过；7个D010/D005历史失败已在修改前源码独立复现，未隐去/伪装通过。V2仍是手工option回放验证；原16worker长训器绑定v20，不能直接当V2 runner。下一位：用户在测台审查，不自动训练。
- 跨PC：V1最终权重包见基线登记；V2现成回放包`outputs/handoff/20261103_option_v2/testbed_review_v2.tar.gz`；代码增量bundle为同目录`coimagine_option_v2.bundle`。GitHub认证仍缺失，不能把本机提交视为已push。部署/读取校验见V2报告及verification.json。

## 以下为12:50及以前的历史交接快照

- **跨PC接续（2026-11-03 12:50冻结）**：[工作状态、权重/回放迁移、启动与下一步](Cross_PC_Handoff_20261103.md)。3937轮、125,984训练episode；best@3500验证32/32，最近@3930为30/32；旧best@2580中期test26/32，不能绑定新best。原训练仍运行，本次没有新训练/评测或修改受保护源码。
- **新研究路线已持久化，尚未实施**：[物理交互经验如何帮助冻结LLM选择/组合option](../research/LLM_Option_Experience_Research_Direction_20261103.md)。先做单一阻挡实例与经验表示对照，再决定benchmark扩展；不把神经世界模型作为前置条件。
- Git同步源码/文档/图表/紧凑证据；权重与回放仍被ignore，另有原PC迁移包`outputs/handoff/20261103_cross_pc/coimagine_state_20261103_1250.tar.gz`，**未上传GitHub**；具体SHA/复制方式见跨PC文档。不要在新机器自动重启训练。
- **本次push因GitHub认证缺失而失败，远端尚未更新。** 本地文档提交已保留；配置GitHub登录后`git push origin main`。另生成`outputs/handoff/20261103_cross_pc/coimagine_docs_20261103.bundle`，可与模型迁移包一起拷至新PC，按交接文档恢复。

- **导师汇报已复核重写**：[问题—动机—改动—效果—研究目标](../option_optimization_and_event_rl_report_20261103.md)。曲线冻结到11:18的3124轮；最近完整验证3120轮25/32、best@2580为30/32；另纳入DS的best中期test26/32。结果支持当前矩形分布可学，尚不能单独归因事件决策。
- 本次复核订正：v20事件读取仿真接触布尔反馈；RL几何mask仍经`rectangle_size=(1.0,.25)`面估计，尚未完成端到端去形状先验。未改训练源码或新增评测，训练继续。快照/图表/原稿备份在`outputs/agent_exchange/mentor_report_review_20261103/`，最终数字在`refresh_1118/`。
- **12h 事件进度并行 RL 运行中，学习已明显生效**：[启动记录](reviews/Event_RL_12h_Launch_20261103.md)、[早晨状态与测台回放](reviews/Event_RL_Morning_20261103.md)。目录 `outputs/option_rl_v20/event_12h_20261103_v2/`；04:13:48 启动，预计 16:13 结束。
- 11:12 状态：**3081 轮 / 7.51 h**；16 workers；最后 10 轮约 6.2–8.3 s；无 STOP/NaN/源码漂移/训练异常；PPO KL 0.0143、EV 0.83、entropy 1.16、logp 误差≤1.1e-5；内存约 6 GB。
- 学习：训练最后 100 轮 return **+5.79**、success **82.9%**、progress **0.726**；验证第 1 轮 1/32 → 第 2580 轮 **30/32（progress 0.790，best.pt）** → 第 3080 轮 **28/32（0.731）**，最近 5 次 0.75–0.9375。
- Held-out test（仅 `best@2580` 只读一次，未参与训练/验证/调参）：**26/32（81.25%）**、progress 0.711、绝对距离中位 **+0.880 m**、94% roots 改善、0 anomaly；6 个失败多为接近目标超时。**不是最终 test**，结束后应在最终 `policy.pt`/`best.pt` 上重做并写正式报告。
- 测台回放：test panel 的 6 条 `best@2580` 轨迹已加入 `outputs/testbed/runs/`，Runs 列表可见；20Hz 精确逐 tick 回放（periods/success/最终距离与 RL eval 一致），sequence 为 0.2s 量化仅供编辑。示例 http://127.0.0.1:8765/?run=20261103_111051_rl06 。RL 场景未注册进测台 `SCENES`，按 Run 会报 unknown scene；本轮未改测台源码/未重启。
- dashboard：http://127.0.0.1:8771/event_12h_20261103_v2/dashboard.html （文件 `outputs/option_rl_v20/event_12h_20261103_v2/dashboard.html`）；[monitor_summary.md](../../outputs/option_rl_v20/event_12h_20261103_v2/monitor_summary.md)、[plan.json](../../outputs/option_rl_v20/event_12h_20261103_v2/plan.json)、[manifest](../../outputs/option_rl_v20/event_12h_20261103_v2/run/manifest.json)。
- 运行结束前不要修改 `experiments/option_rl_v20/long_run.py` 的 `SOURCES` 内文件；supervisor 会按源码漂移写 STOP.json。结束时 trainer 自己写 `run/result.json` 与 `policy.pt`，supervisor 写最终 dashboard/summary；之后用 `best.pt` 与最终 `policy.pt` 各做正式 test，并与固定时间决策基线做同场景对照。
- v1（04:11 启动）因补齐 `parallel.py` 源码哈希被主动停止，12 轮记录保留在 `outputs/option_rl_v20/event_12h_20261103/`；pilot（单进程 20 轮、418 s）保留在 `outputs/option_rl_v20/pilot_20261103/`，仅作接线/吞吐证据。

- **进度事件已接入集中式PPO并完成短检查**：[实现、数据和边界](reviews/Event_RL_Integration_20261103.md)。新目录`experiments/option_rl_v20/`；0.20m、20Hz检查、接触0点、真实时间超时与WAIT，支持P+P/P+W/R+R/R+W/W+W，联合mask1097对。已有v19测台未改。
- 固定动作：单机接触tick18，随后进度tick38/57（0.210/0.204m）；双机接触tick18，随后tick36/54；绕行progress tick26/43/60。相同动作保持skill/q；47项聚焦测试通过，采样/更新logp误差0。
- 最终`smoke_003`：2训练episode/97决策/1次PPO更新，另贪心及随机各1次1cm扰动评估，均0/1、无异常；是接线检查，无学习或泛化结论。单进程，不支持精确resume/retention；没有正式长训。历史`smoke_001/002`为早期接线版本。
- 证据`outputs/agent_exchange/event_rl_20261103/`；最终checkpoint/代码快照在`smoke_003/`。旧testbed预算时钟问题未改，新RL事件接口使用独立的正确tick时钟。
- 最新动作速度已上线：[方案、实测与兼容方式](reviews/Option_Speeds_20261103.md)。REPOSITION每方向仅0.30m/s；PUSH平移0.10/0.20/0.30/0.40m/s，修复candidate接触分支原固定0.30绕过挡位的问题；保留原接触反馈/q′。旧绕行ID仅读取映射，不作额外选项；v13等历史动作表未改。
- 同一绕行后南推preset：MOVE 19.35→8.15s，参考0.12→0.30m/s、实际约0.10→0.25m/s，完整仿真26.9→16.15s。四挡PUSH实测实际速度约0.069/0.140/0.214/0.286m/s。54项聚焦测试通过。
- 0.30挡的接近/N+E/双北推/八边形四例完整frames/strip与原无memory版精确一致。试过恒法向候选导致N+E偏转增加，已弃用，产物保留；未扩全矩阵或训练。
- 8765当前PID **1370453**，controller **a0576536…**，runtime **41e51b0c…**，动作版本`testbed-speeds-v1`。可选42动作、存储保留46历史ID；当前策略mask应使用v19 spec。刷新页面后可查看run `20261103_033050_835410`（单速0.30绕行）。以下旧PID/runtime为历史。
- 本轮证据 `outputs/testbed/experiments/20261103_option_speeds/`。进度/接触事件RL决策仍待单独实现；本轮未改预算时间语义、没有训练。
- 最新观测提速已上线：[实现与实测](reviews/Observation_Kernel_Speed_20261103.md)。合并雷达交点归约、复用精确射线方向/拓扑索引、相邻射线连续性只算一次；不存历史轮廓、不改采样或拟合标准。
- 同实例5次中位数：无记录执行108.26→95.14ms（−12.1%）；RL式完整环境采样113.84→103.66ms（−8.9%，无网络/PPO/IPC）。离线几何负载扫描＋拟合耗时减少16.4%；不外推所有场景或训练吞吐。
- 288组几何scan、576次拟合、新旧输出精确一致；同例状态/力/q/观测/mask完全一致，另3个典型preset完整frames/strip一致；47项聚焦测试通过。无全矩阵、无训练。
- 8765当前PID **1365280**，controller仍**41ec39f4…**，runtime **11d519ee…**，restart_required=false、memory=false。以下旧PID/runtime均为历史。上线run `20261103_031212_3603c0`。
- 证据 `outputs/testbed/profiles/20261103_observation_kernel/`、`outputs/testbed/experiments/20261103_observation_check/`。GPT继续代管，原测台入口可用。
- 日期：2026-11-03（Asia/Shanghai）
- **当前测台已切换为无轮廓memory版**（用户明确要求）：[改动与对照](reviews/No_Contour_Memory_20261103.md)。SupportMemory/每tick维护/历史支持注入接口已移除，q初选只取对应机器人当前可见面；保留EKF运动历史、q锁定状态和同刻感知缓存。
- 8765当前PID **1360366**，controller **41ec39f4…**，runtime **88ce6431…**。header显示“无轮廓memory”，Run中新frame不含memory；旧run仅兼容历史显示。以下旧进程/版本记录均为历史。
- 单例缺陷fixture与35项测试通过；6个典型preset前后对照均无明显option行为异常，5例状态轨迹逐tick完全一致，双机北推仅约7e-15数值差。task complete仅作标签，不作为本轮门槛。没有全矩阵或训练。
- Runs已发布6条“无轮廓memory · …”；绕行后PUSH可看 http://127.0.0.1:8765/?run=20261103_024743_0193b4&tick=393 。HTTP在线验证另1条，保持检查过的候选轨迹。
- 产物与日志 `outputs/testbed/experiments/20261103_no_contour_memory/`。下一步用户人工查看/反馈，GPT继续代管，不自动添加memory补偿或拓展实验。
- 最新只读核对：[Support_Memory_Usage_Audit_20261103.md](reviews/Support_Memory_Usage_Audit_20261103.md)。SupportMemory实际参与PUSH提交时初选q/锚点，但持续tracker只用自身当前visible_faces；无接触calibrate每拍重建会丢双方均不可见旧面。纯观测探针证实共享支持选出的q可能超出自身tracker可见范围。未改算法、未新增物理；先明确memory语义，再优化这部分。
- 最新核心提速：[Core_Simulation_Speed_20261103.md](reviews/Core_Simulation_Speed_20261103.md)。同实例无记录.119→.113s；模拟v16 actor/dense观测消费的无学习基准.150→.123s（5次中位数，约18%耗时减少），不包含网络/PPO/通信。
- 同时刻感知几何缓存、nearest延后坐标计算、跳过无效单点分组已实施；RL式实际扫描/拟合240→182，91物理tick/20Hz/72射线/5m保持。状态/力/q/观测/mask/reward逐项完全一致。39项聚焦测试通过。
- 8765当前PID **1352102**，runtime **a3e0ead1…**，q控制器仍f30f7f72…。页面显示q/sim双版本，关键仿真文件修改也会触发重启提示。以下旧PID与耗时记录为历史。
- 产物 `outputs/testbed/profiles/20261103_core_speed/`，上线验证run `20261103_022125_5e8280`；无正式策略rollout或训练。当前数据约741控制tick/s，仅本实例环境基准，不外推PPO或多worker吞吐。
- 当前重点：**option行为合理性和计算开销；task complete仅为附带标签**（用户最新明确要求）。不为task标签调控制或把task失败当作option失败。
- 最新耗时诊断：[Testbed_Runtime_Profile_20261103.md](reviews/Testbed_Runtime_Profile_20261103.md)。固定“进度预算 · 接近阶段东推”91tick/4.55s，主要长耗时是每次读取解析约2.37GB历史来刷新摘要；已改为持久化metadata索引。
- 当前测台PID **1346288**（替代下方旧PID），8765已重启；q控制器仍f30f7f72…。三次API约.24–.30s，历史列表5–7ms；全部frame与优化前完全一致。5项索引测试通过，无观测/控制/物理/算法改动。
- 页面新增仿真/记录、回放整理、接口、同步绘制与历史列表耗时，刷新页面生效。浏览器实际绘制/呈现耗时未独立采样，UI同步JS计时不当作完整呈现延迟。
- 产物/日志：`outputs/testbed/profiles/20261103_approach_east/`；当前进程以`outputs/testbed/server.json`和`/api/status`为准。仅同一实例6次完整测量，另profile包装启动失败1次（0tick），无训练/广泛矩阵。
- 当前工作方式：**用户已明确授权 GPT 接管 testbed，后续用户与 GPT 直接在测台进行交互调优**；本范围不再要求先交 DS。坚持单实例逐步诊断，不自动展开训练/广泛矩阵。
- 测台状态：**http://127.0.0.1:8765/ 在线，GPT 已重启接管**，PID `1341738`，已加载 q 修复版 f30f7f72…；旧进程1336911已停止。
- 启动/日志：`outputs/testbed/server.json`；`outputs/testbed/operations/20261103_gpt_takeover/server.log`。以 `/api/status` 返回的实时 PID/加载SHA/restart_required 为准。
- 页面会显示加载SHA、q边段/来源；磁盘 q 控制器变更而未重启时提示并拒绝新run；run记录使用进程加载SHA，不再误用结束时磁盘SHA。
- 接管验证run：`20261103_014527_d08ce0`，同一octagon 800tick、q跳变0、task仍false。可直接打开 http://127.0.0.1:8765/?run=20261103_014527_d08ce0&tick=100 。
- 历史514个preset/run/override JSON逐文件hash核对未改；本次接管仅新增1条同实例API验证，无训练。操作记录在 `outputs/testbed/operations/20261103_gpt_takeover/`。

## q 修复的诊断依据（以下运行结果仍有效，8765旧进程说明已由上述接管记录取代）

- 任务：**D013 八边形 q 跳变代码诊断与修复（用户直接要求 GPT 实施）**
- 状态：`reviewed`；唯一 octagon 实例复现/修复验证完成，q 跳变消除；polygon task 仍 false、其他 case 未回归，不冒充整个控制器验收。
- 最新交接：[D013_handoff_20261103.md](D013_handoff_20261103.md)；[本轮诊断/修复/验证](reviews/D013_Q_Jump_Fix_20261103.md)。
- 根因：锁定分支后无条件 simple selector 覆盖 q，绕过后置迟滞；控制器又独立选最近 patch，导致 q 与 n 不同段。已改为每 skill 一次提交、有限段关联、direct 随动/法向校正、同段控制与失配等待。
- 同实例800tick：前130tick跳变39→0（>5cm），全程125→0，最大q步长36cm→2cm；q/控制线一致，候选ID保持1。原/修复 task均false，修复后最终距离反而更大（斜面持续推向东北），方向/任务问题保留。
- 逐帧对照：**http://127.0.0.1:8767/**；原8765进程未重启，仍旧逻辑，不能当作修复已加载。
- 产物：`outputs/agent_exchange/D013_q_jump_20261103_gpt/`；源码hash f30f7f72…；16项当前相关测试通过。3次同实例运行，独立账本；无其他物理case/训练。
- 下一位：用户查看同实例可视化，决定后续介入方向；不自动扩全体矩阵或开始训练。以下 D011/D010 条目是历史记录。

# D011 与后续人工迭代历史指针（不代表本轮验证）

- 下一轮规划提案（未授权实现）：[D012_Distance_Gated_Decision_Proposal.md](designs/D012_Distance_Gated_Decision_Proposal.md) —— 距离触发 option 决策点，0.25m 或按最慢速度算的 Tmax；只作提案。
- 测台可视化：主 Scene 增加可选 `est. face/velocity` 图层；每侧绘制该机器人**已观测拟合面**和估计表面速度箭头（q/cp 处，仅评估显示，不作控制输入）；m3b debug 文本改为 Scene 右侧浮层，压缩纵向空白。
- 新增剩余失败 preset：`D011 failure contact_0_p1` id `20261102_160011_80c498`；`D011 failure dual_0_p1` id `20261102_160011_94ca84`。
- 双机同面 union/切向分配已接入；final→touch 阈值从 .015 放到 .035、final 最小速度 .10 后，dual_0/2 p0 均 success(129，tick32 接触、无 1s 停顿)。inside-band 直接 final 后，dual 10/10；当前 40 task **34/40**，剩余 contact_0/2 p1/p2（4）和 reposition_0_p2、reposition_2_p1（2）。
- 新增典型失败 preset：`D011 regression dual_0_p0` id `20261102_152827_40a2af`（true fail，双推未形成）；`D011 regression reposition_0_p2` id `20261102_152827_d67bc2`（true fail，接触1.5s推进≈0）；`D011 regression reposition_2_p1` id `20261102_152827_790897`（true fail，接触1.5s但推进方向/效果错误）。
- 新增回归 testbed preset：`D011 regression contact_2_p0` id `20261102_145936_d0f2b9`；`D011 regression dual_2_p0` id `20261102_145936_133ad1`；8765 可直接加载。
- option 级标准与首批分类：[D011_Option_Level_Evaluation.md](results/D011_Option_Level_Evaluation.md)（contact_2_p0=option ok/task incomplete；dual_2_p0=true fail）。
- D011 当前 40 task 回归为 **22/40**：previous 10 个 reposition 失败中仅 `T_reposition_0_p2`、`T_reposition_2_p1` 仍失败；但新方案导致 contact_0/2 p0-p2（6）和 dual_0/2 全部（10）回归。单实例成功不等于全量；当前新方案不可接受。
- 日期：2026-10-30（Asia/Shanghai）
- 任务：**D011 一个换面失败实例的完整可视化与人在环诊断**
- 状态：`ready_for_review`（D011 Stage A 完成，等待用户查看/介入）；页面 http://127.0.0.1:8766/；未改算法，无 B/C 自动运行。
- 用户最新方向：先固定一个 failure case，看清全部中间结果并逐步做通；用户介入后续修改，做通后才评估其他 case。
- 唯一实例：`T_reposition_2_p0`，固定 seed12006 / 25.6s 原脚本。B/F 是同实例的冻结版本；不跑 dual/镜像/40/new/contract/260。
- 任务书：[D011_Single_Case_Visual_Diagnosis.md](tasks/D011_Single_Case_Visual_Diagnosis.md)；[配置/冻结输入](tasks/D011_config.json)；[用户说明](briefs/D011_Why_One_Case.md)。
- 本次仅执行 A：复用既有 B/F 原始日志，必要时只补记录；交付逐帧交互诊断页、事件/故障链，随后交用户查看。暂不自主改算法或开始 B/C。
- 拟定独立预算：用户实际移交后 A≤4h、最多4次新增物理 attempt、worker1、每次≤90s，含启动/失败/取消；旧 D010 unknown 保持。写任务书不等于已运行。
- 下一位：用户查看页面并介入；DS 不自动开始优化或跨场景回归。
- 当前证据：[D010八次复核](reviews/D010_M3b_Eight_Probe_Review.md)：F 首触候选误差约34cm，实际SHIFT约6cm后仍差约21cm；不沿用shift=0错误。D010仍incomplete，m3b不推广。

# D010八次验证历史交接（已复核，不是当前待执行任务）

- 日期：2026-10-30（Asia/Shanghai）
- 任务：**D010 m3b 独立8次小物理验证方案**
- 状态：`reviewed`（D010总体 **incomplete**；8 条固定运行完成、m3b not-supported）。F dual 0/2 回归、reposition 0/2；DS 的 SHIFT=0 归因已被原始轨迹复算否定，分析交付需订正。
- 下一位：用户阅读本轮复核并决定方向；可人工交 DS 订正已有分析，不授权第9次或其他新物理运行、不自动发布下一轮。
- 任务书：[D010_M3b_Eight_Probe_Validation.md](tasks/D010_M3b_Eight_Probe_Validation.md)
- 冻结配置/8个attempt：[D010_M3b_Probe_config.json](tasks/D010_M3b_Probe_config.json)
- 给用户的说明：[D010_M3b_Why_Probe.md](briefs/D010_M3b_Why_Probe.md)
- 最新复核：[D010_M3b_Eight_Probe_Review.md](reviews/D010_M3b_Eight_Probe_Review.md)：逐 tick 复算确认任务负结果；两条 SHIFT 实际路径约 6.3/6.2 cm，非零；首触候选误差仍约34 cm。dual 首触处于 peer_separate，candidate=None 不等于候选丢失。附阶段进展/缺口梳理。
- 历史离线复核：[D010_M3b_Offline_Review.md](reviews/D010_M3b_Offline_Review.md)；21项测试重跑通过、factory/step/Slot合成fixture成立，只代表局部分支；不保证真实运动先对齐再接触。
- 版本：B=原D010规划基线归档；F=m3b闭包归档`outputs/agent_exchange/D010_m3b_probe_planning_20261030/candidate_source.zip`，控制器SHA c3f515b6…。
- 方案预算：**明确新增最多8次attempt、总20min、workers≤2**，含启动错误/失败，不得第9次；新账从0登记。仅用户实际移交该方案后生效。
- 旧D010 attempt未知保持unknown；不计算旧剩余、不宣称原1600总量仍满足、不重置或延长旧12h。新方案独立记账，不再以恢复旧账为运行前置。
- 矩阵：reposition_0/2 p0＋dual_0/2 p0，各B/F一次=8；冻结算法/参数/物理/脚本，只记录与分析，不调参/扩40/new/contract/训练。
- 8-probe 交付：[D010_M3b_Eight_Probe_Report.md](results/D010_M3b_Eight_Probe_Report.md)（complete/not-supported）；原始日志/图在 `outputs/agent_exchange/D010_m3b_probe_20261030/`。
- GPT新增物理0、运行算法未修改；旧30/40不绑定m3b。完整contact-acquisition/未知形态/RL验收仍未通过。

# D010历史交接（原12h任务，不代表新8次已授权）

- 日期：2026-10-30（Asia/Shanghai）
- 任务：**D010 接触获取长周期自主诊断、修复与验证**
- 状态：`reviewed`（D010仍 **incomplete**）；M3四项集成修复已用真实step复核，16测试重跑通过；但ALIGN外层误差>.015/内层<=.015矛盾使稳定累计不可达，误差小会绕过速度/稳定时间直进TOUCH；源码快照与manifest不匹配。M3物理仍未运行。
- 下一位：用户移交当前M3集成复核 → DS在原许可范围内完成ALIGN真值表与版本冻结；不新增M4/重试策略。旧attempt上界unknown不伪造，物理继续暂停；若未来单列新验证预算需独立记账，不当旧剩余。
- 最新复核：[D010_M3_Integration_Review.md](reviews/D010_M3_Integration_Review.md)：四项修复、BRAKE抢占已确认；ALIGN明确反例仍待修，快照hash不一致。
- 历史M3复核：[D010_M3_Offline_Review.md](reviews/D010_M3_Offline_Review.md)，四项缺口描述属于上一版，已按本轮证据确认修复。
- 历史continuation复核：[D010_Continuation_Review.md](reviews/D010_Continuation_Review.md)，其中candidate_err_t一直None的结论对应修复前源码。
- phase1复核保留：[D010_Execution_Review.md](reviews/D010_Execution_Review.md)，其中原RETREAT/SHIFT依赖contact的缺陷描述属于修复前版本。
- 最新执行合同：[D010_M3_Candidate_Contract.md](tasks/D010_M3_Candidate_Contract.md)；[用户说明](briefs/D010_M3_Why.md)。原[继续说明](tasks/D010_Continuation_Clarifications.md)保留历史；均不增加预算/修改验收。
- M3 执行回报：[D010_M3_Report.md](results/D010_M3_Report.md)（offline implementation; physical verification not run）。
- 最新复核原件：`outputs/agent_exchange/D010_20261030_m3/gpt_review_integration/`（真实step/Slot fixture、源码/测试/manifest快照、checks.json）；16测试通过，新增物理0、运行算法未修改。
- 任务书：[D010_Contact_Recovery_Autonomous.md](tasks/D010_Contact_Recovery_Autonomous.md)
- 参考方案：[D010_Contact_Acquisition.md](designs/D010_Contact_Acquisition.md)
- 配置/哈希：[D010_config.json](tasks/D010_config.json)
- 固定新40-case验证：[D010_validation.json](tasks/D010_validation.json)
- 用户说明：[D010_Why_Contact_Acquisition.md](briefs/D010_Why_Contact_Acquisition.md)
- 输入订正：[D009_v2_Handoff_Diagnostic_Review.md](reviews/D009_v2_Handoff_Diagnostic_Review.md)：root0为PUSH_W不是PUSH_S；两条first-contact法向都匹配请求；原diag是pre/post混合快照，必须先修仪表/相对角语义。
- 范围：冻结当前共享EKF/Q/R/FF与P0修复，集中修候选、先安全间隙/切向对齐/后法向触碰、必要探视或响应重定位。失败后可在本任务内继续多轮，不必逐项等用户确认；每内部版本只改一个主要机制。
- 新预算：从DS开始preflight起12h；workers≤4；物理attempt≤1600（开发1280＋比较最多260＋登记预留60）；9h30m最晚转最终验证，11h30m停计算，最后30min报告。
- 冻结基线：`outputs/agent_exchange/D010_planning_20261030/baseline_source.zip`及manifest；当前40 task为30/40，待B复现；新动作/接触语义版unknown-shape-contact-v1。
- 验收：已知任务≥36/40且换面每根≥3/5、原六根≥29/30、dual≥9/10；新任务≥32/40每根≥3/5；contract≥32/40每族≥7/10；同时交信息边界/20Hz日志/scorer/replay。全部固定case/脚本不为过关调整。
- 结果：[D010_Contact_Recovery_Autonomous.md](results/D010_Contact_Recovery_Autonomous.md)（incomplete）；补充：[D010_Continuation_Report.md](results/D010_Continuation_Report.md)（incomplete）。
- 本轮不训练、不替默认服务、不改物理/奖励/成功判据、不自动D011。D009全系统先验剥离与估计验收缺口继续保留；D010局部通过不等于RL就绪。
- GPT已做纯几何验证40个新初态、基线源码归档和输入哈希；新增物理运行0。

# D009历史交接（以下不是当前任务）

- 日期：2026-10-30（Asia/Shanghai）
- 任务：**D009 v2 无形状先验的刚体运动 EKF＋接触跟随，自主实现与校准**
- 状态：`reviewed`（执行结果仍为 **incomplete**）；P0修复后的当前结果 **30/40**：contact/approach/dual六根各5/5、reposition两根各0/5；22/40为共享版P0前历史，旧原型30/40为另一历史版本。contract/184/估计门槛未跑。
- 下一位：用户决定后续任务；本轮不接正式RL、不发布D010、不自动追加运行。
- 最新GPT复核：[D009_v2_Post_P0_Review.md](reviews/D009_v2_Post_P0_Review.md)：同拍差分修复通过隔离探针；4条dual决策采样轨迹均成功，p0的恢复期间非零力通过gate。当前只支持功能恢复，不等于完整安全/估计验收；下一候选是reposition交接链20Hz定位。
- 历史共享回归复核：[D009_v2_Shared_Regression_Review.md](reviews/D009_v2_Shared_Regression_Review.md)，对应22/40源码，保留历史。
- 首次源码复核（对应历史原型）：[D009_v2_Design_Review.md](reviews/D009_v2_Design_Review.md)；其“未共享/Q旧/追质心”等描述属于修正前版本，不当作当前源码结论。
- 最新复核产物：`outputs/agent_exchange/D009_v2_20261030_001/gpt_review_post_p0/`（修复后源码/数据快照、哈希和离线复核）；历史`gpt_review_shared/`、`gpt_review/`保留。新增物理运行0，运行算法未修改。
- 最新授权：用户允许睡眠期间长时间自主探索；本任务可改方法、调参、多轮对照，不必遇到一次负结果就交回。信息边界、物理/奖励/成功判据、最终矩阵和总预算不可改。此授权仅适用于D009 v2。
- 任务书：[D009_v2_Autonomous_Implementation.md](tasks/D009_v2_Autonomous_Implementation.md)
- 算法：[D009_v2_Rigid_Motion_EKF.md](designs/D009_v2_Rigid_Motion_EKF.md)
- 配置/输入哈希：[D009_v2_config.json](tasks/D009_v2_config.json)
- 固定验收80-case与184次比较：[D009_v2_acceptance.json](tasks/D009_v2_acceptance.json)
- 给用户的说明：[D009_v2_Overnight_Objective.md](briefs/D009_v2_Overnight_Objective.md)
- 核心：实际接触法向速度＋持续局部线的方向/法向位置校准共同刚体运动；未接触机器人也可贡献几何；不需要端点/物体中心/全轮廓ICP。PUSH直接跟随表面速度并可加一项有界纠偏。
- 硬改造：维护路径移除矩形尺寸/中心/面数先验，不回退。新版本`unknown-shape-motion-v2`；旧checkpoint不兼容；新版本性能未验收时不用于正式RL。
- 预算：从DS开始preflight起≤8h（含开发）；最后20min汇总；workers≤4；所有物理尝试≤1200=开发1000＋最终184＋纯I/O修复预留16，失败/探针/物理单测/重跑全入账。
- 最终验收：任务≥32/40且每根≥3/5；控制契约≥32/40且每族≥7/10；同时通过信息/数值与估计误差/覆盖门槛。固定最终源码后一次性比较，不看最终集继续调参。
- GPT已做：80初态纯几何合法性检查；100组随机状态的观测Jacobian有限差分/参考点不变性检查。**没有物理运行、没有实现EKF、没有训练、没有重启服务。**
- 执行回报：[D009_v2_Rigid_Motion_Implementation.md](results/D009_v2_Rigid_Motion_Implementation.md)（incomplete）；产物 `outputs/agent_exchange/D009_v2_20261030_001/`。
- D009 v1的设计/任务/config/brief保留为已被取代的未执行版本，不执行其16条/40min门槛，也不沿用其端点锚定PUSH设计。历史文件：[v1任务](tasks/D009_Unknown_Shape_Edge_Implementation.md)。
- D008 incomplete及其错误评分边界保留；不以旧R0逐位复现为本轮前置。D002 B仍冻结；D005历史验收缺口不借本轮冒充补齐。无D010/训练授权。

## D008 历史交接记录（以下不是当前任务状态）

- **用户新增设计约束（2026-10-30，优先于此前已知尺寸建议）**：未来控制/策略不预设物体尺寸、完整形态或中心；跟踪局部边及运动，端点/长度只由合法观测约束。相同可见面可对应不同背面，策略按目标需要决定是否绕行获取信息。详见 [未知形态边状态设计](reviews/Unknown_Shape_Edge_State_Design.md)。此为下一份方案的约束；D008 仍按历史任务记录待复核，尚未发布新实施任务。

- 日期：2026-10-30（Asia/Shanghai）
- 任务：**D008 端点、anchor 与参考点的离线审计**
- 状态：`ready_for_review`；2026-10-30 DS 已完成离线审计，但 **R0 重放覆盖 38%<99%，本轮 incomplete**；R1/R2 未运行。
- 下一位处理者：GPT 复核；先处理 R0 重放缺口，不修改控制器/估计器。
- 任务书：[D008_Endpoint_Reference_Audit.md](tasks/D008_Endpoint_Reference_Audit.md)
- 固定配置/66 个输入哈希：[D008_config.json](tasks/D008_config.json)
- 给用户的说明：[D008_Why_This_Audit.md](briefs/D008_Why_This_Audit.md)
- 源依据：[D007 订正版](results/D007_Lateral_Transfer_Calibration_Corrected.md)、[D007 GPT 复核](reviews/D007_Lateral_Transfer_Review.md)
- 主证据：第二轮 D007 的 34 条完整记录，3,934 tick、7,868 side-tick，其中 7,568 条有 endpoint；两条稀疏 nonint 仅来源审计。首轮不扩进数值证据。
- 范围：全量测量区间/法向/anchor/参考误差账本、16 对 probe 的恒等分解、历史 estimator 严格重放，再做 R1 去速度预测与 R2 当前区间中点的固定离线对照。
- 预算：新增物理运行/传感器样本/训练/生产控制器改动均为 0；workers≤2；全量分析≤1200 s，纯离线测试≤180 s，从首个测试至报告图表总计≤1800 s；最多一次纯分析 bug 修复重跑，计入总预算。
- 执行回报：[D008_Endpoint_Reference_Audit.md](results/D008_Endpoint_Reference_Audit.md)（incomplete）；产物 `outputs/agent_exchange/D008_20261030_003/`。
- 停止点：解释参考点误差与偏移抵消的来源，输出唯一下一步建议；不修改估计器、不发布 D009、不启动训练。A 保留，B/C 不推广，D002 B 冻结，D005 完整验收缺口保留。

## D007 历史交接记录（离线订正已完成，以下不是当前任务状态）

- 日期：2026-10-30（Asia/Shanghai）
- 任务：**D007 侧向控制减法与传递校准**
- 状态：`reviewed`；2026-10-30 GPT 已检查源码和原始数据并独立离线复算。候选 C 不推广；D007 存在分析错误与未满足条款，非全部验收通过。
- 下一位处理者：用户决定移交 D007 离线订正；无新增物理实验授权、无 D008。
- GPT 复核：[D007_Lateral_Transfer_Review.md](reviews/D007_Lateral_Transfer_Review.md)；订正数据：`outputs/agent_exchange/D007_20261030_002/gpt_review/`。
- 任务书：[D007_Lateral_Transfer_Calibration.md](tasks/D007_Lateral_Transfer_Calibration.md)
- 固定配置/输入哈希：[D007_config.json](tasks/D007_config.json)
- GPT 审查：[D006_Control_Audit.md](reviews/D006_Control_Audit.md)
- 给用户的说明：[D007_Why_This_Calibration.md](briefs/D007_Why_This_Calibration.md)
- 执行回报：[D007_Lateral_Transfer_Calibration.md](results/D007_Lateral_Transfer_Calibration.md)（原版，保留）；**[D007_Lateral_Transfer_Calibration_Corrected.md](results/D007_Lateral_Transfer_Calibration_Corrected.md)（复核后订正版）**
- 预算：正式物理轨迹最多 36 条、workers≤4、正式阶段≤600 s、聚焦测试≤180 s；从首个测试/运行到分析回放完成总计≤900 s，进程组硬截止。
- 方案：A 原版；B 关闭四项 PUSH target 修正；C 在 B 上去掉额外接触切向阻尼。第二轮 36 条是最终证据；加首轮至少 72 条尝试、正式运行合计约 42.614 s，超过原数量上限。修正后共同有效 6/8、G 增益改善 +.04665（要求+.10）、RMS 比 1.11251（要求≤.90）；baseline 角偏/推进退化，C 仍未过门槛。
- 机制订正：旧分析读取缺失的 effect.n 后把两机器人都投影到世界 Y，robot1 support=0 和镜像增益抵消结论撤回。偏移已正常进入 target，扰动窗可见性/reservation 裁剪均为 0；跟踪参考点变化会抵消净目标变化，先订正离线分析与图表，不按旧结论修偏移入口。
- 范围：只实施可选择候选与观测记录，不训练、不换默认、不改物理/奖励/动作频率/接触门限/安全投影；D002 B 冻结。完整 D005 8-case 当前版本验收仍未补齐，本任务不冒充该验收。
- 当前服务：沿用用户已有服务，本任务未重启；D007 输出 `outputs/agent_exchange/D007_20261030_002/`；旧 `_001` 为首轮记录缺口尝试。

## D005/D006 历史交接记录（以下不是当前任务状态）

- 日期：2026-10-29（Asia/Shanghai）
- 任务：D005 收尾 + D006 Web testbed；**额外控制器改动待 GPT 复核**
- 状态：`ready_for_review`；2026-10-29 21:20 用户授权，DeepSeek 执行；D005/D006 功能与 testbed 已交付；GPT 反馈已收到，事实性修正已落，尚未正式验收
- 下一位处理者：GPT 复核；用户可继续手动使用 testbed
- 任务书：[D005_Option_System.md](tasks/D005_Option_System.md)、[D006_Testbed_UI.md](tasks/D006_Testbed_UI.md)
- 固定配置：[D005_config.json](tasks/D005_config.json)、[D006_config.json](tasks/D006_config.json)
- 执行回报：[D005_Option_System.md](results/D005_Option_System.md)、[D006_Testbed_UI.md](results/D006_Testbed_UI.md)、**[D005_D006_Consolidated_Report.md](results/D005_D006_Consolidated_Report.md)（本次汇总，先读）**、**[D006_Control_Equations.md](results/D006_Control_Equations.md)（控制公式/诊断）**、**[D006_Lateral_Offset_Report.md](results/D006_Lateral_Offset_Report.md)（侧向偏移简报，给规划侧）**
- 最新更新：侧向偏移简报已写，等待规划侧做符号/数学审计并另发任务书；控制公式/诊断文档已写；PUSH.rotation_feedback_gain 的 signed sweep 显示正 gain 放大 CW、负 gain 才阻尼（−0.15~−0.18 附近接近零偏转）；诊断 run 已进测台列表；Goal 改为 amber 虚线圈并可拖动，run/preset 会保存移动后的 goal；Presets 去重复时间戳、右侧 520px、Timeline 压扁到 260px；contact-ready 容差 `0.05 m`，recovery 丢接触阈值实际为 `0.07 m`；同侧双推 preset 位移 1.465/1.538 m，但 `success=False`；Timeline 与 force/speed 控制曲线合并为同一 canvas/时间轴；可选 lidar/target/force/contact 图层；equalization 已从 v19 删除；所有测试场景同质 preset（8 个 seed preset，无 Scene selector/built-in/独立起点场景）；description 自动由 option 序列 + created 时间戳生成。
- 当前输出：`outputs/agent_exchange/D005_20261029_2130/`；`outputs/testbed/`（runs/presets/overrides）；Parameters 卡已可列出/加载已存 override profile
- 本地 URL：`http://127.0.0.1:8765/`（server job `bash-419`，已 curl 验证）
- 范围：D005/D006 功能完成（行为验证未闭环）；D006 期间修改了 v13 `push_controller.py`（contact-ready/recovery），**超出 D006“只做 UI”任务书**，且 D005 的 READY/ENGAGE 对称化已删除；这些额外变更需 GPT 复核。不训练、不修改 reward/物理/成功判据、不发布 D007。

## 前轮证据

- [D004 执行回报](results/D004_State_Pair_Gate.md) · [D004 原始产物](../../outputs/agent_exchange/D004_20261029_2045/)；D004 完成 pair gate + 47 行手动状态表，D005 已用自动 registry/coordinator 取代其新工作入口。
- [D003 执行回报](results/D003_Contact_Timing_Audit.md) · 原始产物 `outputs/agent_exchange/D003_20261029_1843/`；D003 已完成但 GPT 复核未出，D004 在其固定 8 个 case 上继续。
- [D002 执行回报](results/D002_Switch_Cost_and_Persistence.md) · [D002 GPT 复核](reviews/D002_Switch_Cost_and_Persistence.md) · 原始产物 `outputs/agent_exchange/D002_20261029_1757/A/`
- D002 A：12/12 配对完成；3/12 改善、median +2.46 mm、3/12 严重退化、engaged 改善 0/12；B 未运行，切换惩罚保持未验证候选。
- D002 复核修正：engaged=0 不等于握手从未激活；双方接触稀少、未与激活持续重合；不能直接宣判握手故障。
- [D001 执行回报](results/D001_Dual_Push_Credit_Diagnostic.md) · [D001 GPT 复核](reviews/D001_Dual_Push_Credit_Diagnostic.md) · `outputs/agent_exchange/D001_20261029_1644/`
- D001：92/92 episode，0 异常；末轮随机仍能成功（joint 12/32、independent 24/32），贪心 0/4；joint 成功主要走 push+reposition/其他。

本轮 D005/D006 已实施 option registry + 自动三表 + 中心 coordinator + headless session + Web testbed；没有训练、没有改 reward/成功判据/物理；没有解冻 D002 B；没有并入 independent 停车；没有发布 D007。D006 期间额外删除了 v19 equalization、调整了 v13 contact-ready/recovery，并在 testbed 中完成 preset/可视化/时间线合并；这些变更的复核边界见汇总报告第 5 节。

- 复核 remaining reposition_2_p1 / reposition_0_p2：均为 option_effect_ok/task_incomplete，距离 .212 vs goal 半径 .2（差 1.2cm），不是 option true fail；testbed 的 no success 仍是 env task 判据。

- 按新 option 标准复核当前 34/40 的 6 个 task 失败：全部 contact 正常、正推进、角度≤.074，均为 task-incomplete；当前没有 true-fail。按用户要求 testbed 已删除这 6 个 preset，仅保留 seed 基础场景。

- 全量 40 option 审计：task success 36/40；option pass 38/40；true fail=2（T_dual_0_p2、T_dual_2_p2，最大相对角 .7212，task success）。testbed 现只保留 `truefail dual_0_p2` / `truefail dual_2_p2` 两个 preset；contact p1/p2 四个为 task-incomplete，未放。heading 在 standoff/final/touch 已改为面向 support 法向、线速度独立。

- peer gate + push label 修正后 40 审计：task success 36/40，true-fail 0；dual p2 角度 .0902（原 .7212），testbed 已删除 truefail preset。final 慢主要是法向接近速度/等 peer，不是切向。

- 测台持久预设（按实际内容命名，供复核）：单机东推·基础接触 `20261102_172225_a5b70f`；单机北推·基础接触 `20261102_172225_aac5fb`；单机东推·近目标余量 `20261102_172225_91208b`；单机北推·近目标余量 `20261102_172225_623c86`；双机东推·同侧接触 `20261102_172225_413b24`；双机北推·同侧接触 `20261102_172225_29b914`；双机北推·旋转压力 `20261102_172225_c8c699`；绕行后西推 `20261102_172225_43dfd1`；绕行后南推 `20261102_172225_ce15a2`；接近阶段北推 `20261102_172225_351d3c`。

- N+E 双侧协推：修正 peer gate 为先自身 heading 对齐、双方 contact+aligned 才 push；tick29 R1 orient、R0 wait，tick50 双方才 push。该场景后续仍 task fail、角度 .449，需继续查；40 case 尚未在该 gate 后重跑。

- N+E：q 在 contact/wait/push 后按共享 EKF 的 v+ω×r 积分，R1 目标 q 随物体右移；tick55 双方 push，最大角 .285（原 .449）。task 仍 false，属 task-incomplete，非 q 冻结。

- N+E 切向：q 随共享 twist 已正常；u=1.5×err 当前 maxangle .285/task false。尝试 4×err/.20 与 paired surface FF 均更差（.667/.676、contact 丢失），已回退。下一步是速度级切向跟踪，不是简单加大 gain。

- 2D 位置 P（q+outward*(R-.05)）在 N+E maxangle .649，比 best .285 更差；已回退。下一阶段需 pair-level 力分配/接触速度协调。

- 按用户要求切到探索版：q' 切向 offset + ref_C 的 2D P，不再坚持 best。testbed 当前典型失败 preset：单机东推·接触短推进 `20261102_175825_b3e03d`；绕行后南推·弱推进 `20261102_175825_aaa34c`；双机北推·低推进 `20261102_175825_17fdbc`；接近阶段东推·弱推进 `20261102_175825_f64481`；N+E 双侧协推 `20261102_172943_1d04c5`。该版 40 audit task success 0、true-fail 0；作为探索版保留。

- P 探索版参数：q' tangential offset rate .02；contact 2D P 改为 K=6、速度上限 .30（原归一化固定速度）。弱推进 contact_0_p2：contact 18 tick、final x2.568、角度 .0117；N+E：contact 128 tick、maxangle .0732（原 .155）。两者 task 仍 false；40 未在该 K 下重跑。

- 接触抖动修复：push 后 raw contacting_object 丢失 <0.15s 不退回 touch/approach；弱推进 approach_0_p0 阶段不再连续 push/touch 回跳（push41/touch33，先后段），推进 x≈2.625，角度 .0007；N+E maxangle .1026。两者 task 仍 false；recover 本来已被 P 分支绕过。

- contact 定义：真实 Pymunk 接触（gap 阈值=0）；coordinator contact-ready gap=.05。contact hold 从 .15→.20。尝试 soft contact gap≤.02 作为接触代理会直接让弱推进/N+E 不动，已回退。hold=.20 结果：弱推进 contact23/final2.614/角度.0005；N+E contact91/maxangle .0835；task 仍 false。

- 按用户要求彻底去掉真实 contact：控制中 contact = patch_gap≤.02；不再使用 contacting_object/contact_point_robot；estimator 不再吃真实 contact，cp 用 candidate q/None。结果（K6+hold.20）：弱推进 est_contact=0、物体停在2.502；N+E est_contact=0、物体2.501；合法接触版本当前无法触发接触/推进，是明确失败，待修 approach 到 .02。

- approach 两段化（|e_t|>.05 去安全 band，否则直接 q+outward*R；contact 仅 gap≤.02）已实现但未触发接触：弱推进 run `20261102_181604_60fc9a` est_contact=0、物体2.547；N+E `20261102_181604_6360b4` est_contact=0、物体2.547。task 失败，需继续查 q/参考与实际几何。

- approach 两段化验证：weak tick15 gap=.0039、contact_ready=True；tick20 gap=-.0004；已进入 0.02 阈值。testbed contact 显示改用 estimated contact_ready（修掉 stage=push/contact=false 的显示错误）。弱推进 task 仍 false，问题转到 P 接触力/推进，不是 approach 进不了阈值。

- 链路定位：P 的 K=6 不是瓶颈；ref_C 的 .05 侵入因“中心到 q≈R>.05”判断永远不生效，desired 只有 .023。改为 surface gap=‖q′-own‖−R 判断后，weak desired≈.299、contact 74、final x3.368（overshoot goal2.95）、maxangle0；N+E maxangle .271。下一步是 terminal stop/brake，不是加大 gain。

- D012 提案已补充：基础 PUSH goal-free、不加强制 terminal；可选 PushToGoal/goal-entry 事件；option 预算从时间改进度（PUSH 接触后物体法向 .2/.4m，REPOSITION 机器人 .2/.4m）。关键未决：PUSH 物体法向进度当前没有合法 object position 估计，需先补估计器或明确代理。

- 双预算记录已进 testbed frames（progress_m/budget_m/budget_reached/time_budget_s）。当前追踪：single PUSH progress 恒 0（stage push 但 progress 条件未累计），只在 10s time budget 触发；reposition 路径 progress 到 1.97m < 2.4m 预算，30s time budget 触发；budget 在部分 tick 为 None，说明 skill 重建后未稳定重挂。需要修 progress 累积和预算重挂，不是简单调预算值。

- 预算状态已从 skill 移到持久 OptionBudget（env 层）；PUSH 用机器人 Δ 投影 observed normal，REPOSITION 用自身路径，双预算 time/progress。testbed 只设一次并读 tracker。结果：单机东推进度 .45m 成功 112 tick（progress .495）；reposition 1.9m 预算成功 512 tick；dual/approach 早前成功；N+E .4m 仍 false。

- 评价口径：task success=false 不再表示 option 失败，只表示任务层没进 goal；option 验收看 stage/contact/progress/stability。testbed 标签改为 TASK SUCCESS/TASK INCOMPLETE。N+E 进度预算 .4m 已到达，属 option 正常；场景 goal/初始姿态本身不构成任务。

- UI 预算丢失修复：app.js loadPreset/loadRun 之前只保留 decisions，导致 UI 跑 N+E 864 tick；现已保留/显示 budget_m、time_budget_s、progress。Playwright 验证：N+E preset 在 UI 中 sequence 带 0.4m/10s，Run 132 tick（task incomplete，progress budget 已触发）。

- 进度预算语义：当前每个 side 一个 OptionBudget；PUSH 用各自机器人 Δ 投影 observed normal，REPOSITION 用各自路径。testbed 用 **any side reached** 停止，因此实际是“进度最大/最早到预算的机器人”触发，不是两侧都到。新 goal-fixed N+E preset 也沿用该语义；若要求双机都完成，应改成 all/min 或共享物体进度。

- q/q' 排查：屏幕 q 是 base candidate q；q' 现已接入 debug/UI（蓝色叉）。reposition 后南推 frame463：q=(2.7446,1.9172)，q'=(2.7389,1.9175)，q' 在 q 的左侧（offset=-.0057）；frame480 q' 左移 .0335。但机器人 x=2.63/2.83，q' 仍在机器人右侧，所以 desired 仍向右；q' 相对 q 的方向正确，绝对世界左移量不足（base q 随 support extension 右移约 .25m）。不是符号反了。

- 核实 q 选点：当前 `choose_candidate_q(lo,hi)` 只取可见 support 区间中心（margin clamp），support extension 后替换为新区间中心；**没有**按目标推力/力矩匹配选 q。reposition 后南推 q 右移来自 support 区间中心，不是转动补偿；q' 只是小 offset。下一步需要 torque-aware q 选点，而不是继续调 P/q' 增益。

- q 近角点原因核实：support interval 在 frame440–480 基本恒定 [2.1004,2.9699]（中心≈2.535），但 q 从 2.502→2.989，已到/超过右端；不是 choose_candidate_q 取中点，而是 `_advance_candidate_with_motion` 每 tick 用 v_at(q) 积分 q，未重新投影/clamp 到当前 support。因此 q 从中心漂到观测端点，q' 仅小 offset。需在 motion 更新后把 q 投影/裁剪回 support，或改为 torque-aware 重选。

- 新增 D013 规划提案：[D013_Visible_Contact_Pair_Selector.md](designs/D013_Visible_Contact_Pair_Selector.md)。核心：可见 support 上的力/力矩+分离双接触点选择；可靠 EKF 时 q 随动并 clamp，不可靠时停止积分、用当前观测重投影；当前 LIDAR_RANGE=1.20m 建议单变量试 1.8–2.0m。未实现。

- D013 step1 已实现：LIDAR_RANGE 1.20→1.80；q 运动积分增加 support 投影/clamp；当前在速度不可靠时默认关闭 q 的 v/ω 积分（fallback）。reposition 后南推 run `20261102_192440_aafe08`：q 不再漂到 2.99，frame460 q≈2.496，task success 547 tick；q' 目前只在 P contact 更新，部分帧会滞后，torque-aware pair selector 未实现。

- D013 实现进度：LIDAR_RANGE 1.8；q motion support clamp/不可靠关闭；select_contact_pair 纯函数（最近点单初始+坐标下降，不交叉）测试3通过；paired fallback 接入。reposition task success 547；N+E 172 task false，q reason 仍 support_extension，selector 实际选中需核实。

- D013 selector 已改为 paired 优先：N+E run `20261102_192951_e06fe7` q reason=pair_alloc（selector 结果），q0/q1 分离约 .40m；max angle .183（之前 .271），172 tick（progress budget 内），task false 但 option 进度正常。selector 单测 3 通过。

- selector 0.1s + 1 tick v/ω 预测实测：N+E run `20261102_194224_09c0be` maxangle .434（比每 tick .183 差），tick40 R1 q 跳到 (0.955,1.523) 后 wait；selector 无时间连续性/visibility gate，不能保留。reposition task success 547。下一步：previous-q 热启动/travel cost、可见性门控，再恢复 0.1s。

- local selector（approach 全量、接触后 prev q±.1m）已实现；0.1s 重算 + q 短预测。N+E run `20261102_194612_3e54f3` maxangle .084（好于 .183/.514），但 q 仍有异常跳变（如 side1 q≈0.93、side0 y≈1.01），说明 q 预测/selector 在接触后仍有未门控跳变；task false 但 angle 改善。下一步需 gate q prediction 或隔离 selector。

- 隔离测试：N+E selector-only（q prediction off）run `20261102_194910_01fb39`：q 连续无 0.9 跳变，maxangle .273，156 tick；q 预测开启时 angle .084/.183 但 q 跳 0.93/1.01。确认跳变来自 q 短预测，不是 selector；默认暂关 D013_Q_PREDICT，待加协方差/最大步长门控。

- 按用户决定删除 q 短预测（无 q_motion_enabled / D013_Q_PREDICT / v_at 积分方程）；只保留 selector（0.1s 重算，中间 hold q，selector 内部 clamp）。N+E 最新 run `20261102_195134_b288a5`：q 连续、maxangle .273、156 tick；reposition task success 547。

- 全可见面 selector 已实现（不再按 action 过滤 face；approach 枚举 face-pair；online local 面/±.1m）；单测4通过。N+E goal fixed 仍选 q0 左侧、q1 下方（run `20261102_200241_24644f`，angle .273），R0 顶面未被选中：需要先确认 top face 是否在 R0 当前可见 patches 中；若不可见，selector 不能凭空选，必须加观察/迁移，而不是改代价。

- 语义修正：N+E 里“R0 上半侧”指**左侧面的上半段**，不是顶面。当前 selector 的 q0≈(2.39,1.85) 在左侧下半段，仍然不对；目标是左侧 y>2 以产生 CW 力矩抵消 R1。下一步不再加 top-face 探索，而是在 R0 当前左侧 support 内调 q0/torque cost；若上半段不可见则需观察/迁移。

- selector torque reference 修正：o_ref 从“两 support 中心的中点”改为“两侧可见 support 点的 bbox 中心”；N+E goal fixed run `20261102_200538_e6efeb`：task success 130 tick，angle .21，q0 从 y1.95→2.21（左侧上半段），M≈±.006。根因是旧 o_ref 偏下 y≈1.72，导致下半段 q0 力矩更小；修正后 q0 移到左上。

- 单机 candidate retarget 也加 EMA（α=.30，单步≤.05）+ paired selector EMA。接近阶段东推 run `20261102_202201_23bd18`：success 188 tick，angle 0；frame40 q 仍由 support_invalid 从2.375→2.452（较原来2.465减小），stage final 仍出现。q 跳变来自单机 support_invalid，不是 selector。

- 单机 PUSH 也接 selector：`select_contact_point` + candidate EMA 最大步长 .02；approach_0_p0 run `20261102_202408_3ab153` q 连续、stage push、task success 91 tick、angle0；q reason 改为 single_selector。

- selector 双锚点已实现：world q + support-relative anchor(u/n/rho/support)；local 时先映射到当前同面，面/跨度不可靠回退 world q；修掉 `_last_anchor` 覆盖 env 原字段的 bug（改名 `_selector_anchor`）。N+E goal fixed run `20261102_203650_f948d2` success 140 tick、angle .121，q 连续；reposition `20261102_203651_977369` success 535、angle .008。selector 测试5通过。

- 估计窗口拆分：独立 object-centred 放大窗口（左侧，135px/m），主 Scene 移除 est 层；机器人淡化，support fit/memory 加粗。新增 SupportMemory（同线区间 union、hits/last、世界系），env 每 tick 更新并写入 frames.memory；暂未接入 selector。frame0 无记忆，播放后可见（direct session frame100 有27段）。

- 修复世界系记忆残影：SupportMemory 每 tick 用共享 EKF (v,ω) 对历史线做刚体预测，再做同线/interval union；N+E frame100 memory 降到 3 条（此前 30+ 条），估计窗口可辨认。窗口机器人已淡化，memory 白粗线/当前 fit 彩色。memory 仍未用端点可信度，也不直接进 selector。

- memory 记忆策略：非接触（REPOSITION/APPROACH）motion 强制 0，只做同线 union；检测到 contact(gap≤.02) 即 freeze，不再扩 interval。绕行后南推 run `20261102_220045_75a786` success 535，frame440 memory 10 条且左侧边不再被 EKF 侧向速度拉长。反向 scan-to-shape 求速度未实现。

- EKF 不可观测子空间先验已实现：每个 accepted face 线加入 H=[n,0]/[0,0,1]；SVD 求零空间，将零空间方向 x 投影为 0，并把该方向协方差设为 σ=.05 先验；两面非平行满秩时先验自动消失。测试 `tests/test_ekf_nullspace.py` 2 通过。N+E run `20261102_220828_d3234d` success140/angle.121；南推 run `20261102_220834_21d938` success535，approach 末 v_est≈0。

- 直接接触速度解算诊断（scripts `experiments/option_rl_v19/velocity_direct_audit.py`）：用两侧 contact 的 q/n/own_v 解 `n·(v+ω×q)=n·v_robot`，λ=.02 正则。N+E goal 直接解 mean .016 / p90 .025，EKF .272/.560；南推 direct .176 vs EKF .016，approach .163 vs .007，单机东推 .133 vs .061，双机北推 .058 vs .006。结论：两面接触时直接解能解决 N+E，但不能全局替代 EKF；下一步做 rank/残差门控的 hybrid。

- Hybrid 直接解已接入 `SharedMotionEstimator._try_direct`：接触行 H=[n,n·(Jq)]，rows≥2/rank≥2/cond≤20/res≤.12 才接受；通过则覆盖 EKF v，否则保留 EKF。N+E goal run `20261102_221923_ac24ab`：平移误差 mean .0396（EKF .2459，直接诊断 .016），p90 .0698；南推/approach/单机/双机北推保持原 EKF，未误触发。N+E 角速度误差变差到 .071（原 .017），因 direct 同时覆盖 omega；下一步应只覆盖平移或保留 EKF omega。

- Hybrid 最终版：接触行 3D 直接解（rank≥2、cond≤20、res≤.12、λ=.02）+ 平移 EMA(.30)；rank<3 时 ω 保留 EKF 线角速度；否则 direct ω。N+E goal run `20261102_222503_dbd8af`：success140，平移 mean .0417 / p90 .0723（EKF .2459/.5595），contact .0465，角速度 .0174（与 EKF 相同）。南推/approach/单机/双机北推保持原 EKF，无回归。

- 联合 ω 行试验（负）：将各线 dphi 中位数作为低权重 [0,0,1] 行加入 3D direct；N+E run `20261102_223227_f529d1` 平移回到 .2459（角 .0174），说明 N+E 的 dphi 与接触行不一致/被门控，联合解退化回 EKF。已回滚；恢复 contact-only direct + EKF ω 版本 `20261102_223308_c51b7f`：N+E .0417 / ang .0174。联合 ω 行需要先解决 dphi 异常/关联，不能直接加。

- q=own±nR 测试：两种符号结果完全相同（n·J(±nR)=0，只有切向中心进入 ω 项）；不是 q 符号问题。用真实物体/机器人速度代入接触行仍有 .2–.5 m/s 残差。发现参考中心不一致：`v_est` 实际贴近物体中心速度，而接触行用绝对 q（世界原点 twist），两者混用；联合解因此冲突。dphi 本身与真实 ω 吻合（−.194 vs −.207 等）。下一步先统一参考中心：选 c，行用 q−c，真值同样变换；c 可由 support memory/bbox 估计。

- 接触点参考中心验证：取 q_i=own_i+n_iR，q1 为参考；H1=[n1,0]，H2=[n2,n2·J(q2-q1)]；加 dphi 行解 v1,ω，再得 v2=v1+ωJ(q2-q1)。N+E 离线采样误差均值 .0099 m/s（v1/v2 对真实接触点速度 .001–.015），ω 也吻合。确认之前联合解失败是错用绝对 q/中心；接触点参考模型成立，待接入控制器。

- 接触点参考解已接入 `_try_direct`：q_i=own_i+n_iR；q1 为参考，H1=[n1,0]，H2=[n2,n2·J(q2−q1)]，加 dphi ω 行；输出 contact_solution(q_ref, v_ref, omega, 每侧 q/v)，并把 twist 转回世界原点写 `ekf.x`。N+E run `20261102_224445_b7b01f`：success140，接触点速度误差 mean .0103 / p90 .0181，ω .0195；双机北推 success119，direct err .0069；南推/approach/单机单接触回退 EKF，均 success。

- 冻结轮廓记忆已接入 selector：SupportMemory.faces() 转 face；approach/push 均传 extra_faces；先试“全记忆面”（含当前不可见）N+E 退化到284 tick，改为只允许记忆**延长当前可见同线 interval**、不引入不可见面后恢复：N+E success140，南推543，approach91，单机90，双机北推119，全部成功。

- 南推 tick514 诊断：EKF 切向分量实际为 0（n·Jv≈0），零空间先验生效；错的是法向：EKF .617 vs 真实 .139。单接触 direct（法向取机器人接触速度）给出 .205，仍不匹配；说明接触点/机器人法向速度与物体接触点速度不一致，或接触模型不成立，不是单纯 q。memory 被错误法向速度拖漂移。下一步需用雷达射线对 memory/接触行做可见性残差约束，并核对单接触 contact 状态。单接触 direct 已开启，南推 success542。

- 旧南推 kick514 截图来自旧 run（225515，one-contact direct 之前）。当前 run `20261102_230202_917668` success542：旧“切向划出”现象未复现；量化 memory vs 当前可见 face：法向偏差最大 .256、切向端点偏差最大 .18（approach 阶段法向 ~.25、切向 ~.03；push 后切向增长）。当前窗口仍绘制所有历史 memory 线，其中有些是旧面/不可见，视觉上会误以为在漂移；selector 只延长当前可见同线。下一步应做 memory 可见性/射线剪枝，再决定是否需要修正。

- q/q' 核实：q candidate 由 selector 每 0.1s 更新、两次之间 hold；q' 每 control tick 用 yaw_err 更新 offset 并用于 P。尝试在 selector 间隔内用 EKF velocity_at(q) 预测 q（support clamp）后，N+E/南推/approach/单机/双机全部失败（284/1252/284/284/284），已回滚；回滚后 N+E run `20261102_230814_b035b9` success145。当前不做 q prediction。

- 需澄清：`_try_direct` 通过后会写 `self.ekf.x`，因此 q prediction 和 memory 的 `motion=ekf.x` 实际读的是 direct 结果（direct 被拒时才是 EKF）；但 memory update 在 motion_estimator.update 之前调用，实际用的是上一 tick 的 direct。q prediction 在 direct 下仍全部失败并已回滚。后续若要再试，应显式保存 last_direct/contact_solution，memory 用当前 tick direct，q 用 q 所在接触点的 direct 速度，而不是再读 ekf.x。

- q prediction 参考修正：用 direct contact solution 的 v_contact 外推 `v_q=v_contact+omega×J(q−q_contact)`（不再 `velocity_at` origin twist）后，仍全场景失败（284/1252 tick）。已回滚 q prediction；回滚后 N+E run `20261102_231331_cc20d7` success145。`velocity_at_contact` 公式保留待用；q 仍只在 0.1s selector 更新。

- q prediction 崩溃根因找到：预测后用 `sk.candidate.support` clamp，但 allocated candidate 的 support 被硬编码为 `(-1.0,1.0)`；frame30 q 被投影到 y≈0.986，完全离开物体，因此所有场景失败。不是 velocity_at/参考中心本身。修复需把 selector 的真实 support 存到 allocated q 并用于预测 clamp；当前 q prediction 仍关闭。

- q prediction 修复：真实 selector support 写入 allocated_q（不再用 allocated candidate 的 (-1,1) 占位）；q_pred 用 direct 接触点速度 `v_contact+ω×J(q−q_contact)` 并 clamp 到真实 support。五场景全部 success：N+E 145、南推 542、approach 91、单机 90、双机北推 119。q prediction 重新启用。

- q vs q' A/B（q prediction 已启用、真实 support）：默认 q' vs D013_USE_QPRIME=0 q-direct。q-direct：N+E success140、南推542、approach91、单机95、双机北推119；q'：N+E145、南推542、approach91、单机90、双机119。全部成功；q-direct 在 N+E 略快，单机东推略慢。当前 8765 进程以 q-direct 启动；代码默认仍 q'，需环境变量切换。

- N+E q-direct selector 力矩检查：tick64 真实中心 (2.726,2.154)；q0 (2.553,2.112) tau≈+.078（CCW），q1 (2.522,1.651) tau≈−.092（CW），净 CW；但 selector 的 o_ref=(2.608,2.120) 下 q0≈+.020、q1≈+.016，二者同号，selector 认为无 CW，导致 q1 选在下侧左边。根因：o_ref 是当前可见 support bbox 中心，偏左 .12m，把 q1 的力矩符号算反。任务虽 success，但 selector 力矩目标错误；下一步应用更可靠的中心估计（memory 全轮廓/线交点）而非两 face bbox。

- 尝试 push 阶段冻结 selector + q 递推 + q' 积分 offset：N+E 失败284 tick，南推640、approach195、单机127、双机122（变慢），已回滚。q' 保持默认、selector 仍每0.1s；回滚后 N+E `20261102_233819_b87e8f` success145。结论：当前 q' 积分不能替代 selector 的力矩修正；应先修 o_ref/质心，再考虑冻结。

- o_ref 凸包实验（负）：用 memory+当前可见 face 端点凸包面积质心替代 bbox；N+E 失败192 tick，tick64 o_ref y≈1.996 vs 真实2.143，仍偏低（memory 含旧面/不完整线段）；q1 虽移到右侧但任务失败。已回退 o_ref_est=None，N+E run `20261102_234104_25c67f` success145。contour_center 代码保留但禁用；需要更可靠的闭合轮廓/边质心（只用一致线段/角点）再试。

- 简化 selector 试验成功：单机取 support 中心；双机同面取 1/3/2/3（保证 2R+margin 间距），异面各取 face 中心，不再用 o_ref/力矩代价；仍启用 q prediction + q'。五场景全 success：N+E 122、南推538、approach91、单机95、双机北推119；N+E 明显好于复杂 selector 145。当前 8765 以 D013_SIMPLE_SELECTOR=1 启动；代码默认仍复杂 selector。

- 分层简化 selector 已实现：两侧枚举可见/记忆 face-pair；每对 face 固定取点（异面中点、同面1/3+2/3并保距），用该 q 的合力/力矩（bbox o_ref 近似）选 face pair；选定后不再连续下降 q。五场景全 success：N+E 122、南推538、approach91、单机95、双机北推119。当前 8765 以 D013_SIMPLE_SELECTOR=1 启动；代码默认复杂 selector。曲边按多段 face 枚举。

- 简单分层 selector 已设为默认（D013_SIMPLE_SELECTOR 默认1）；复杂 selector 仍可用 env=0 回退。当前 testbed 物体几何仍硬编码矩形（OBJECT_WIDTH/HEIGHT、box polygon），T-shape/三角形/八边形需要先给 random_start 加多边形物体（物理/lidar/render），不能直接用现有 scene 预设测试。

- 合成多边形 selector 测试已加：`experiments/option_rl_v19/synthetic_shapes.py`（T/triangle/octagon 边 patch），`tests/test_simple_selector_shapes.py` 验证 face-pair 选择、q 落在多边形边上、双机间距；测试通过。完整 polygon 物理/lidar/render 环境尚未改，当前 testbed 仍是矩形。

- 完整 polygon 物理闭环已加：SceneSpec/InstanceSpec.object_polygon；Pymunk moment_for_poly/Poly body；lidar 扫描改用 polygon；env true-edge-gap/testbed scene meta/app 渲染均使用 polygon；新增 poly_T/poly_triangle/poly_octagon seed presets。smoke：三场景 20 decisions 无异常；R0 东推 200 decisions 均 task false（800 tick），说明物理/观测通路工作但任务序列/控制器尚未调通。合成 selector 形状测试通过；当前 polygon 仍无 task success。

- polygon testbed presets 已保存：`poly_T · 东推`(`20261103_000354_9b9758`)、`poly_triangle · 东推`(`...e10dbd`)、`poly_octagon · 东推`(`...7c7e29`)；序列= R0 PUSH E / R1 BRAKE_HOLD 200 decisions(40s)。UI 可加载并显示 polygon 与序列；实际运行 task false 800 tick（序列/控制器未调通）。

- Pymunk 非凸修复：SceneSpec.object_polygon 经 ear-clipping 三角分解，多个 triangle shape 挂同一 object body（body 先加入 space）；contacts 遍历所有 object_shapes；object_corners 用原 polygon；lidar/render 仍用原非凸轮廓。poly_T/triangle/octagon 均能跑；R0 东推仍 task false 800 tick（任务序列未调通，不是凸包虚空碰撞）。

- LIDAR_RANGE 5.00m；SupportMemory 更新不依赖 paired/PUSH，每 tick 用两侧 observations 的 patches 更新（静止阶段 motion=0，双方都贡献）。5m 下 N+E goal success122、南推 success538，memory 在南推 R1 brake 时仍增长到10条。当前是共享 memory union，不是每机器人独立 memory。

- Memory 校准/fit：非接触静止阶段用两侧当前 patches + 已有端点聚类、PCA 拟合，每个聚类只存一条拟合线；接触阶段冻结。LIDAR 5m 下 N+E success122（memory 2 条）、南推 success538（memory 在 300/500 tick 稳定为7条，不再持续增长）。

- Memory 校准修订：静止阶段 calibrate 只用当前两侧 patches（span≥.04，≥3点）重建拟合线，不再把旧 memory 端点重新喂回聚类；每个线聚类只保留一条 PCA 拟合线。octagon run `20261103_002053_4d2392` memory 在100/300/500/700 都稳定为3条（之前9→18）；N+E success122、南推530。polygon task 仍 false（序列/控制）。

- Approach 首见 q 锁定：非接触时首次 selector 得到 q 后锁定，不再每0.1s重选；进入 contact/push 后解除锁定，q prediction/selector 按原逻辑。五场景无回归：N+E122、南推530、approach91、单机95、双机119；octagon run `20261103_002618_9a0623` 仍 task false800（序列/控制），但 q 不再 approach 期间反复切换。

- “selector 每 PUSH option 只选一次”试验（q prediction 保持开启）仍失败：N+E/双机北推 284 tick；说明接触后需要 selector 修正，q prediction 不能替代。已回滚为 approach 首见锁定、contact 后解锁。恢复后 N+E run `20261103_003751_24d7aa` success121，双机北推 `...a41860` success119。q prediction 继续启用。

- q 的 support-anchor 校正：第一次 selector 保存 face n/rho/u；非接触锁定期间每 tick 用当前同线 support 重投影 q（不积分、不重选）。contact 后仍解锁 selector；q prediction 开启。结果：N+E success105、南推530、approach91、单机95、双机北推90（改善）。完整 option 只选一次仍失败284；contact selector 仍是必要修正。

- 完整 option 锁定 + env-level commit + support-anchor 重投影（q prediction 关闭）仍失败：N+E/双机北推 284；q 后续 reason 回 simple_support，锁定/校正未能替代 contact 后的 selector。已回滚 contact 解锁；恢复后 N+E run `20261103_004918_a64206` success105。当前保留 approach 首见 + anchor 校正，contact 解锁。

- 进一步核对“contact 同一中点”假设：完整锁定 + anchor + q prediction 仍 N+E/双机 284；contact 后 selector 实际把 q0 从2.477改到2.586（tick50），不是同一中点微差。恢复 contact 解锁；当前 N+E run `20261103_005321_0a00bb` success122。approach anchor 校正保留。

- 全锁定根因修复：`_match_anchor_face` 原来要求绝对 rho 差≤.05，物体平移后线 rho 改变导致匹配失败、q correction 跳过，q 卡在旧世界点、法向距离从 .18 缩到 .075。改为同法向匹配并选离 anchor rho 最近的平行面。全锁定 + anchor 重投影 + q' offset 五场景成功：N+E102、南推530、approach91、单机95、双机北推87；q 法向距离稳定 .18。

- octagon 全锁定 q 跳变诊断：q reason 全是 support_anchor_correct，不是 selector；跳变对应 support 从 [0.7,1.0] 切到 [2.5,2.63]/[0.78,1.0] 等不同段；同法向候选 + support overlap 仍不足以固定段，q 在 standoff/final 可见面切换时跳到另一平行/分段 support。全锁定 N+E 105/102，但 octagon 仍 task false800；下一步需要 face segment ID/滞回，而不是继续调 q 投影。

- octagon q 跳变核查：跳的两段法向同为 [0.924,0.383]，是同一法向的不同 support 段；tick17 support [1.53,1.62]、tick33 [2.50,2.63] 与主段 [0.70,1.04] 交替。尝试 support-overlap 过滤后仍 28 次跳变（run `20261103_011029_3d4c70`），说明 anchor/committed support 或匹配路径仍有问题；当前全锁定 testbed 保留。

- tick100 核查（用户指出的区间）：octagon q 在 support [0.74,1.06] 与 [2.62,2.96] 间跳，q≈(2.07,1.83)↔(2.07,2.17)；对应 n 在 [0.924,-0.383] 与 [0.924,0.383] 交替，确实是相邻面（45°），不是同线同法向。当前 `_match_anchor_face` 阈值 .98 理论上不应匹配，但 q reason 仍 support_anchor_correct，说明另有路径/陈旧 q 在切换；全锁定仍未解决。

- 按“不重投影，直接枚举当前可见 face 的中点/1÷3，选离当前 q 最近”实现后，octagon q 仍跳 39 次（run `20261103_011816_b719ad`，task false800）。说明候选集本身在可见面变化时切换；需要 face identity/滞回或只在候选属于同一 committed face 时更新，单纯最近点不够。

- 最近候选 + 0.10m 跳变滞回后，octagon q 仍跳 39 次（run `20261103_012344_f85fd4`），说明跳变不经过 `_nearest_visible_q` 的返回路径，或 q 在调用前已被单机 committed_q/selector 覆写。当前全锁定 testbed 保留；下一步必须打印 q 每次变化的来源（nearest/single/committed/prediction）而不是继续调阈值。
