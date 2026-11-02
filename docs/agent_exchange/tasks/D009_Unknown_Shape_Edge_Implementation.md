# D009：实现未知形态边滤波，并剥离矩形先验

2026-10-30，GPT v1。用户明确授权 DS 编写滤波代码和剥离矩形先验；GPT 本轮提供方案，不代跑实现。状态 ready_for_handoff。

**目标：让新控制路径只凭实际扫描持续估计局部边，物体尺寸/中心/完整形态不进入估计器、目标和 mask。先实施、离线验证，门控通过后做固定16条闭环检查。**

- 数学与接口规格：[D009_Unknown_Shape_Edge_Filter.md](../designs/D009_Unknown_Shape_Edge_Filter.md)，必须逐项实现，不自行改滤波器类型、状态、参数或裁剪规则。
- 数值/输入/矩阵：[D009_config.json](D009_config.json)。用户说明：[D009_Why_This_Filter.md](../briefs/D009_Why_This_Filter.md)。
- D008 保留 incomplete 记录；新方案不以原R0达到1e-7为实施前置条件。先纠正其评分错误：用记录 face_midpoint 对真实面中点直接评分，不用两倍 anchor_distance 反推边长，不引用24cm作为真实参考误差。7,568行的直接评分应复现 median≈.00845465、P95≈.04719911、max≈.14050104 m，允许不同分位数插值造成1e-4 m差异。
- 默认旧服务、历史训练/checkpoint、D002 B、奖励/物理/成功判据、动作ID/档位、力/加速度/ready/recovery阈值不变。新几何版本显式选择，无旧先验fallback；不训练，不开发真实非矩形物理后端，不发布D010。

## A. 实现与先验剥离

建议新增 `src/coimagine/random_start/edge_state.py`（纯估计、无env/backend依赖）、`edge_geometry.py`（无先验观测/目标支持）；新控制集成放 `experiments/option_rl_v20/`。复用同一物理环境、传感器、力控制和持久动作协议，可以提取共享hook，不能复制整份step/环境。旧路径作为明确历史对照保留，不改为隐藏fallback。

EdgeBank 属于每机器人环境感知状态，跨动作切换保留；mask、prepare、skill.step及绘图读同tick不可变快照。同一扫描不可被多次update。建立独立 `UnknownShapeGeometryConfig`，不能带 rectangle_size/face_length_priors/object_depth，不给旧FaceEstimate伪造length/centre。

交付逐项 `prior_removal.csv`：源码位置、旧依赖、替代、测试。至少覆盖 estimate_face/available_faces/所有mask、EndpointTracker length融合、旧AnchorTracker隐式调用、face.centre/depth、历史面关联、PUSH目标/守护、ROTATE偏置、SameFaceReservation、v14反馈及v19 session/render。传感器/环境reset/奖励/离线scorer可使用真值，但必须在调用图中和部署路径区分。原base760与controller62槽位保留，反馈122槽位可保持布局并用新有效位/边反馈填充；所有语义变更写入版本manifest，旧checkpoint拒绝隐式加载，不新增特权特征。

新路径的estimated_centre必须None，新增edge_reference/edge_normal/covariance/endpoint bounds等明确字段；目标可视化消费新字段。生产server不重启，提供headless选择新版本与只读回放入口即可。

必须用 actual factory+观测+mask+一次控制计算测试，而非只检查config或重写公式。对新分支的旧rectangle/中心补全函数设raise哨兵；同一可见输入下改变隐藏尺寸/背面metadata，输出、mask、目标与命令力完全一致。允许scorer读取这些metadata，不允许候选路径读取。

## B. 固定离线验证（不新增物理轨迹）

完整读取D007第二轮34条tick记录，3,934 tick/7,868 side-tick；其他两条稀疏记录只记来源。重建记录扫描时不传raw_labels、不使用D007融合后端点或truth face ID来初始化候选；每tick用扫描独立拟合patch并更新EdgeBank，不能只用历史动作选中的patch。物体真值只用于之后评分。

比较三组：H=历史矩形先验日志（只作上下文，非公平未知形态基线）；U=无记忆、无先验的raw patch/独立物理端点区间中点基线；F=新滤波器。U没有端点时输出missing，不补长度/中心。H与F的精度和可用率不得直接称同信息公平优劣；主比较F/U，按共同可见面绑定后的端点评价，并单独报告全体分母与失配。

沿用D008离线material face绑定规则仅供评分；逐轨迹/机器人/面/端点kind给法向角误差、面法向距离误差、端点位置误差median/P95/max、有效率、连续丢失、估计长度的误差与未知比例。对持续边参考，评价它相对初始已绑定物理面坐标的漂移，不把a当真实边中点；评价1–3s与全程，初始化前不得用未来信息对齐。

不确定度：端点2D位置用完整Jacobian传播P；报告真端点落在名义95%椭圆内的分子分母、sigma误差关系和无端点静线的切向协方差增长。滤波cov正定不等于实际校准通过。所有missing/conflict/ambiguous都计入可用率，不靠拒绝难例改善P95。

新增8条**解析几何传感器fixture流**，每条101帧（t=0:.05:5），不用Pymunk、不算物理轨迹；使用普通多边形光线求交生成合法72ray/1.2m扫描，真值仅在生成器和scorer。名称/几何/观测路径见config：S0静止+自运动、S1平移、S2旋转、S3遮挡、S4非90度角、S5/S6相同正面不同背面、S7镜像S2。不得用矩形专用scan_lidar生成S4等非矩形输入。S5/S6前2s扫描必须一致，后续相同绕行观测路径能产生不同扫描；估计器只能在观测分歧后改变。该fixture不等于已完成背面主动探索实验。

离线测试必须包括解析Jacobian有限差分、区间条件矩与一维数值积分（含尾部/窄/宽区间）、P半正定、初始化augmentation交叉协方差、未知端点不补全、区间重复更新去相关、时序幂等、镜像/坐标变换、单线切向不可观测、非90度角、真实schema、state跨动作保持、工厂/所有46动作的先验raise哨兵和同观测不同metadata不变性。

**闭环门控，全部满足才进入C：** 全流无NaN/非PSD；信息边界/时序测试通过；候选fresh line有效率≥U的.90；真实面法向角P95≤.5deg、法向位置误差P95≤.01m；共同可用端点F的P95≤U+.005m，候选端点输出覆盖≥U的.90；fixture中角点/端点/rigid transform关联正确且未把遮挡当端点。任一失败则保存代码和全量结果，报告complete-negative（若实现/评分本身坏则incomplete），C skipped，不扫参、不补物理探针。

“滤波有效”的更强报告条件单列：F端点P95≤max(.005m,.8*U的P95)，尾部max不比U增加>.01m，漂移/可用率/延迟一起可接受；门控通过不自动等于滤波有效。

## C. 固定16条闭环集成检查

沿用原矩形物理后端，仅控制候选不知道尺寸。formal轨迹和任何env/Pymunk测试共用**16条总上限**；不额外启动smoke或物理单测。先跑W_N候选这一条作为正式集成首条，后续bug不得自动整批重跑。失败也是一条；修后重跑需另交回，不能用“非正式”排除预算。

1. D007四个preset场景×旧A/新F：8条，原5/5/8/8s，原动作/override；镜像严格按冻结变换生成。
2. W_N/W_E参考instance+commands×旧A/新F：4条（先执行的W_N/F计在这里）。正常success可提前结束，不人为裁掉失败尾段。
3. W_N实例，新F：`[32,32]`20decision（CCW normal_v=.02，4s）、`[35,35]`20decision（CW，4s），2条。仅检查实现/几何/实际角运动，不假定未知形态下局部偏置必然给对力矩方向。
4. W_N实例，新F两条6s混合序列：M0=`[42,44]×10,[11,45]×20`；M1=`[11,45]×10,[44,44]×5,[38,11]×10,[44,44]×5`，2条。验证RP/FOLLOW/BRAKE切换与滤波历史、未知形态mask均实际走新路径。

baseline和候选从reset执行同一序列。不得用新mask偷偷替换冻结动作；若请求不可用，使用公开的blocked+brake协议并计失败，逐项报告mask和原因。物理传感器/碰撞/termination沿用原版；不为新版本改成功阈值。

报告原始全0.05s tick：P0观测、edge快照、P3请求/投影/反馈/命令力、P4推进后真值与真实contact；记录schema测试在纯fixture先通过，防止再跑一批才发现effect被覆盖。给每次instance创建/physics advance计数，并验证总16以内。

## D. 预算、证据与停止

workers≤2；纯测试≤300s；离线全量/fixtures≤1200s；闭环阶段≤300s；从首个测试到图表/报告完成总计≤2400s，进程组TERM后2s KILL。代码开发时间单列，运行阶段最多一次纯分析bug修复重跑，计总预算；不因此重跑物理。每条完成即落盘，不能整批结束才保存。

preflight先复制改动前完整相关源码，之后复制候选源码并记录独立hash；不覆盖原数据。每阶段记命令、起止时间、实际输入/输出/fixture/物理次数；未知版本若仍调用旧几何先验，立即停并交回。

交付新目录 `outputs/agent_exchange/D009_<run>/`：manifest、baseline/candidate源码快照、prior_removal.csv、离线全量CSV/score、fixture生成器与结果、gate.json、最多16条tick日志、12张固定诊断图（D007三类case各两侧共6，S0/S1/S2/S3/S4/S5-S6各1共6）、回放index。图表明确a≠物体中心，H有尺寸先验。

回报 `results/D009_Unknown_Shape_Edge_Implementation.md`：是否彻底剥离先验；新滤波是否改善尾部且诚实表达未知；控制还能完成哪些动作/哪些失效；唯一建议改动。失败结果保留，门控未通过不推广。CURRENT→ready_for_review，GPT复核，停止；不训练、不切换默认、不启动D010。
