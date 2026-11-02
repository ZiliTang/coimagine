# D009 算法规格：未知尺寸的刚性边状态滤波

2026-10-30，GPT，v1。本文与任务书/config 共同构成 DS 的实现规格。所有数值是首版工程设置，不是已验证最优值；不得在执行中扫参救分。

## 1. 契约

输入只有本机器人扫描、里程计、时间及已公开的机器人尺寸/动作；扫描的原始隐藏 labels 清空。无物体尺寸、中心、真实运动、质量/惯量、后台面 ID、完整地图或队友私有历史。

以一条局部直边为实体，不拟合整物体。每条边共享平移和角速度；物理端点未知时不补全，不要求相邻边垂直。记录中的矩形真值只在独立 scorer 中使用。新版本名 `unknown-shape-edge-v1`，同一 Pymunk/传感器契约，不新增物理模型。先作为显式选择的新版本对照，用户现有 server/default 不替换，无回退到矩形模型。

## 2. 状态、坐标与输出

基状态 6 维：

```
x = [a_x, a_y, phi, v_x, v_y, omega]
n(phi)=(cos(phi),sin(phi)); t(phi)=(-sin(phi),cos(phi))
```

a 是初始化时选取的边上参考点，不是物体/边的中心。初始化取第一次 patch 的点均值在拟合线上的投影，此后**不重置为当前可见点均值**。它可位于边所在直线的延长线上；参考原点的选择本身不证明材料点已被观测。

某侧物理端点首次得到有限区间后，才追加一个静态坐标 s_i 到联合状态（最多8维），端点 `p_i=a+s_i*t(phi)`。低/高端沿持续 +t 编号。未知端点没有伪造位置、长度或有限“默认不确定度”，用 absent/None。`L=s_high-s_low` 仅在两端均建立状态且顺序一致时作为**估计值**输出，不用于补全未见端点。

输出 `EdgeEstimate`：track_id、time、a/n/t、v/omega、完整 P、端点 mean/variance/状态、估计长度或 None、当前可见支持区间、当前安全目标区间、观测时间、关联分数、line/endpoint innovation、validity。无 `object_centre/depth/rectangle_size`；不把 visible_span 命名为物理 length。

## 3. 共同刚体预测

```
a_minus = a + v*dt
phi_minus = phi + omega*dt
v_minus=v; omega_minus=omega; s_i_minus=s_i
p_i_minus = a_minus + s_i_minus*t(phi_minus)
```

两个端点因此共享同一刚体变换。参考点速度采用短时常速度模型；真实旋转带来的加速度由过程噪声表达，不假定参考点就是转动中心。

F 为上述映射的解析 Jacobian：F[0,3]=F[1,4]=F[2,5]=dt，其余单位阵；P_minus=F P F^T+Q。位置/速度的两组连续白加速度块为 `q_v*[[dt^3/3,dt^2/2],[dt^2/2,dt]]`，q_v=.6²；phi/omega 同式 q_omega=1.5²。已知 s_i 加 `(.001 m)^2*dt` 的微小形状过程噪声（防止近似更新锁死，不代表真实伸缩）。

初始化 v=0、omega=0；P 基状态对角为 `[.01²,.01²,(3deg)²,.20²,.20²,.40²]`。首帧线观测初始化后不再重复作普通校正；但可以建立端点。dt 使用真实扫描间隔，跨过>.25 s 缺测不继续作有效控制输出；分步预测最大步长 .05 s，时间倒退直接报错。相同 robot/tick/sample_time 的更新幂等，无重复证据更新。

## 4. 一次拟合线观测：校正方向和法向位置

复用现有 scan_patches 的实测点，不插值增加信息量。匹配 patch 的点均值 mu、拟合法向角 phi_obs 与 RMS 来自合法扫描。

以 `z=[phi_obs,0]`、`h(x)=[phi,n(phi)·(mu-a)]` 作一次2维 EKF 更新。角残差 wrap；H 第一行 phi 列=1，第二行 a 列=-n，phi 列=`t·(mu-a)`，其他=0。

R=diag(sigma_phi²,sigma_plane²)，固定：

```
sigma_plane=max(.003, rms/sqrt(N)) m
sigma_phi=max(.7deg, atan2(max(rms,.003), sqrt(sum(q_j-q_mean)^2)))
q_j = point_j · t_obs
```

至少3点、可见 span≥.04 m、sigma_phi≤10deg，否则该 patch 不作可用线观测。使用点均值处的残差以减少角度/截距耦合；这是近似噪声模型，不宣称射线独立或精确置信度。线的 Mahalanobis innovation 阈值9.21，更新用 Joseph 形式，单次线性化，不重复用同一批点迭代收紧 P。

## 5. 端点观测：未知长度、区间似然

观测生成保留 observe_endpoints 的“相邻射线夹逼/遮挡删失”语义，但**不做已知 length 的两端区间融合**。也不得调用旧 EndpointTracker 中的长度补全。

角点细化允许任意两条夹角非退化的边：`abs(cross(n1,n2))>=sin(20deg)`，替代旧近90°限制；交点必须在本端射线区间±.005 m 内，且接近另一边实测片段端部（沿用 max(.025,2*当前射线夹逼距离) 容差）。不根据矩形朝向/尺寸识别角点。

不假定物体凸。邻接射线若先命中同类object的另一表面，且距离比当前拟合线交点近>.02 m，只有上述两边交线验证通过才可认作角点；否则记`self_shadow_censored`，不把自遮挡边界当物理端点。static/teammate遮挡沿用删失处理。补一个纯几何自遮挡测试，不以后台面ID作判定。

每个有限端点区间，以当前观测切线 t_obs 得到 l/u。端点的测量函数与 Jacobian：

```
h_i(x)=t_obs·(a+s_i*t(phi))
H_a=t_obs; H_phi=-s_i*(t_obs·n(phi)); H_s_i=t_obs·t(phi)
```

端点首次出现：`p_mid=(lower+upper)/2`，s_i=`t(phi)·(p_mid-a)`，这是**宽区间下的初始化中心**，不是精确测量。追加协方差：J_a=-t(phi)，J_phi=`-n(phi)·(p_mid-a)`，J 其余0；P_xs=P J^T，P_ss=J P J^T + width²/12 + .005²。该端本 tick 不再更新一次。width 按区间切向宽度计算，无 size/length prior。

后续有限区间使用以下高斯条件矩公式，按低端→高端各一次，R_e=.005²：

```
mu=h_i(x); S=H P H^T+R_e; C=P H^T
alpha=(l-mu)/sqrt(S); beta=(u-mu)/sqrt(S)
Z=Phi(beta)-Phi(alpha)
d=(pdf(alpha)-pdf(beta))/Z
w=1+(alpha*pdf(alpha)-beta*pdf(beta))/Z-d*d
x_plus=x+C/sqrt(S)*d
P_plus=P-(1-w)/S*C*C^T
```

这是对线性化测量的区间条件矩做高斯近似，不是“测量值=区间中点”的普通 KF。宽区间且预测远离边界时，校正自然很弱；区间会约束概率分布，而不要求预测值冲向中点。对 shadow/range censored 不追加端点、不给虚拟区间：该端只有预测，置信度不因“仍见到一段直线”自动变高。

用 math.erfc 分尾部计算 Z，避免两个接近1的 CDF 相减。Z<1e-9 时跳过该端更新并记 endpoint_conflict；连续3帧 conflict 时将该端标 conflict，不重置到真值/中点，保留线状态。w 数值仅在 [-1e-10,1+1e-10] 内允许裁到[0,1]，超出直接报数值错误。极窄区间 width<1e-5 m 用标量点更新，观测方差仍至少 .005²，不能除零。

为控制量化/同一扫描的重复计数：端点最近更新<.2 s，且把上次区间按预测刚体变换带到现在后，两界各变化≤.002 m、当前区间未缩窄≥.005 m，则跳过重复端点因子。若预测均值已在当前区间外>.01 m，仍允许立即更新。记录 skip/reason；此规则只是首版保守去相关，不宣称得到精确贝叶斯后验。

两端顺序冲突时不交换物理 ID、不拿已知尺寸纠正，标 extent_conflict，长度输出 None；允许继续使用可靠局部线及当前可见支持。

## 6. 数值、不确定度和可观测性

每步 P 对称化，特征值<-1e-10 视为失败；只允许加1e-12 I 的数值 jitter并计数。每次校正后通过添加半正定项设置基状态位置 std≥.002 m、角 std≥.3deg；s_i std≥.003 m。它们是工程下限，不是覆盖保证。内部phi保持unwrapped，仅角innovation wrap。非有限位置、速度模长>1 m/s或|omega|>4 rad/s时标track_invalid，不静默裁值后仍输出很小P；不按物体尺寸/中心限幅a或端点。

单条无端点的静止无限直线不观测切向平移；测试必须确认切向速度/位置不确定度不会因重复线更新而塌缩。只有端点/角点提供相应信息。不要用“line observed”替代“tangent observable”。

标志分别为 line_fresh、tangent_localized、endpoint_low/high_observed/predicted/conflict。line_fresh 要求距最后线观测≤.10 s、sigma_phi≤5deg、在机器人法向投影处的面位置 sigma≤.02 m。切向参考 std≤.04 m 才可用于位置纠偏；超过则 lateral 修正置0并明确 tangent_unlocalized，不能编零误差。该状态下只允许≤.04 m/s 的温和法向接近/维持，不自动加速推进。>.25 s 缺测进入 perception_wait/制动，延用现有安全退出机制。

## 7. 边关联与存活

每机器人独立 EdgeBank，最多8条；属于环境感知状态，跨动作/skill 切换保留，每 tick 最多更新一次。技能和 mask 只读取不可变快照，不各自再滤一次。

候选必须法向夹角≤20deg、观测片段与预测线法向距离≤.10 m，投影支持区间重叠或间隔≤.15 m；不按预设长短边、物体中心或90°关系关联。代价=线 innovation 的 NIS+(支持间隔/.15)²；按代价、track_id、patch_index 固定排序作一对一匹配。最优与次优代价差<2 时标 ambiguous，不强配；未匹配 patch 可建 provisional track，连续3帧匹配后可供控制。法向反号不自动翻成同一边。

同一条边有遮挡碎片时，本版只选匹配最好的一个片段更新，不把多段当多份独立线证据，也不通过无限直线外推把不连通共线片段强行合成一个实体。丢失>.5 s 的 inactive track 可删除；满额优先删最旧 inactive，否则拒绝新建并记录容量限制。

PUSH在动作意图/track失效时，从fresh且有安全区间的track中选`dot(n,d)`最大、法向间隙最小、track_id最小者（按此优先级）；选中后保持身份，不能每帧追当前“最好面”。ROTATE选法向间隙最小，再按安全区跨度最大、track_id最小。没有唯一可靠候选时报告unavailable/ambiguous，不能用真值绑定替选。

## 8. 先验剥离后的控制/接口替代

保持46个动作 ID、速度档、0.8 s RL hold（本轮不训练）、testbed .2 s decision、.05 s control、力/加速度限制、接触 ready=.05/recovery=.07、奖励与任务判据。几何版本变化必须阻止旧 checkpoint 无提示加载；可保持944个向量槽位，但新增语义版本与有效位，不宣称旧权重可兼容。

| 旧依赖 | 新规则 |
|---|---|
| rectangle_size / 长短边判别 | 从有效 patch 建 EdgeBank；不知道另一端也可以有有效线 |
| face.centre/depth 与物体中心 | 不生成；内部 effect_reference 存边参考a，日志旧 estimated_centre 置None并新增edge_reference，不能把a叫物体中心 |
| anchor±L/2 和旧中点目标 | 动作开始在所选边上固定目标坐标 q0：机器人中心在边上的投影，裁入当前安全区；后续 target=a+t*(q0+delta) |
| 依赖 L 的反馈限幅 | 在当前安全支持区内裁剪，记录 requested/projected；没有有限安全区则 unavailable |
| 原 heading/rate/lateral 外环 | heading .20、rate .10、lateral .35、point_rate .16、direct rot gain0、contact tangent damping .7 保留；lateral_error只用a相对动作初始参考的变化，并须 tangent_localized，否则该项invalid/0；同一边的 omega 来自滤波状态，不再对raw法向求差分另滤一次 |
| 连续 PUSH 进度/守护 | 继续用合法接触下机器人 normal_travel；不把参考点变化冒充物体位移/中心进度，q处normal surface speed=`n·v(a)-omega*q` |
| ROTATE 的 rotation_offset_fraction*L | 目标接触偏置 `q0-rotation_sign*.08 m`，经相同安全裁剪；它是旋转请求的局部偏置，不保证未知形态下一开始力矩符号；availability要求该方向至少.04m可用余量。报告真实旋转方向，不用真值纠偏；本轮不加新的力矩控制器 |
| face.length 相等/centre接近的面关联 | 使用 EdgeBank track_id 与上述纯观测关联 |
| SameFaceReservation 以面长分半 | 使用下述可见支持/peer separator，不以隐藏总长分半；机器人顺序锁存与碰撞避免保留 |
| 旧 AnchorTracker face_length_priors | 新版本不构造/调用此先验分支，_last_anchor保持None或明确unknown结果；观察生成仅保留纯扫描/patch路径 |
| 动作 mask 调用矩形 estimate_face | 读同 tick EdgeBank有效线/安全支持与请求方向≤30deg；FOLLOW组合约束沿用当前版本；不得读真实shape/尺寸验证可行性 |

当前可见区 S 用相对a的坐标 `q=dot(point-a,t)` 取该 tick 实测 patch 点的 min/max，内缩 .01 m；q0、peer/own坐标与separator也全部相对同一a。已知物理端点状态若收紧范围，低端用 s_mean+2std+.01、高端用 s_mean−2std−.01；未知端不额外补边。最终 S 必须与当前可见区相交，不凭预测穿过遮挡。q0 只在动作意图/选定track改变时重建，速度档变化不重置；没有端点时也不每帧重取可见中点。

同面保留：peer 的观测中心距本线外侧深度在 [R−.03,R+.35]，且切向投影落在 S 扩张 R 的范围内，才启用保留。沿用侧序、separator、separation=2R+.04+.01+.3*peer_age 及动态避障余量。将S与peer侧约束相交；无区间报lane_unavailable。不得以减小机器人间距补偿可见范围不足。Robot R 是机器人公开尺寸，允许使用。

REPOSITION/FOLLOW/BRAKE 的纯局部扫描实现继续复用；审计所有回调，禁止从新路径调用 legacy estimate_face/EndpointTracker/AnchorTracker。可以提取共享动力/生命周期函数，不能复制整份环境或整份 step 形成第二套物理/协议。新几何配置类型不得包含 rectangle_size、face_length_priors、object_depth；旧版只作显式历史对照，不能在未知版本中静默fallback。

## 9. 必须承认的局限与参考

该设计只跟踪可见局部边，不恢复未见背面/物体中心；光滑、共线、遮挡会留下真实不可观测性。位置尾部误差是否下降须实测，不能由KF名称保证；协方差须独立检查覆盖，不能只看是否正定。未知形态版本不能承诺所有ROTATE意图都能实现，失败与可用率都入账。

EKF基本递推参见 [Särkkä, Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf)，第5.2节。删失观测条件矩思想参见 [Loumponias & Tsaklidis, Kalman Filtering With Censored Measurements](https://arxiv.org/abs/2002.08597)。上述局部边模型、双端区间处理及去相关规则是本项目首版设计，不是照搬论文即可获得的性能保证。
