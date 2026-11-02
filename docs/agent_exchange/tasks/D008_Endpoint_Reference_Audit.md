# D008：端点、anchor 与参考点的离线审计

2026-10-30，GPT v1，`ready_for_handoff`。用户已同意下一轮；由用户移交给 DS 后执行。本轮做彻底的数据与算法审计，**新增物理轨迹 0、控制器改动 0、训练 0**。

- 目的：解释参考点为什么在物体坐标系中移动；区分测量区间、法向、状态预测/校正和 anchor 选择，输出唯一下一步建议。
- 说明：[D008_Why_This_Audit.md](../briefs/D008_Why_This_Audit.md)。固定输入、运行清单、哈希与数值：[D008_config.json](D008_config.json)。
- 先读 D007 GPT 复核及订正版，不用旧世界 Y 分析。B/C 不推广；D002 B 保持冻结。
- 允许新增 `experiments/option_rl_v19/audit_d008.py`、离线辅助模块、仅离线测试及新产物。禁止修改生产 controller/tracker/lidar/physics/server、旧报告/旧原始产物、默认参数；不得启动 env、Pymunk、scan_lidar、训练或补采。`scan_patches/observe_endpoints/EndpointTracker` 对**记录扫描**的纯几何重算允许，须与物理运行分开命名。

## 1. 冻结范围、分母与坐标

主证据只用 `D007_20261030_002/runs` 的 34 条完整记录：12 baseline、16 probe、6 witness，共 **3,934 tick、7,868 side-tick**；其中有 endpoint 结果的 side-tick **7,568**。不能把无 endpoint 的 300 条静默删除，也不能把 mirrored 当成新的独立场景。两条 nonint 只做索引/来源审计，不进入时序统计。`_001` 仅引用 manifest 记录既有预算，不并入新数值证据。

核对全部冻结 SHA256，复制本轮分析源码和输入索引，记录 HEAD+dirty、依赖版本与实际命令。config 区分历史候选快照和当前分析脚本，禁止用冻结基线哈希冒充候选哈希。旧快照缺口仍是缺口，本轮不补造。

规范化每条记录：`run_id,tick,t_start,side,skill_uid,profile,action,stage,pause/clearance,scan,face_normal/points,endpoint,target,offset,reference,P4.before/after`。P3 对应 P4.before，P4.after 仅用于推进后评价。字段缺失、非有限值、长度不齐、重复时间均显式报错或 missing；**禁止备用坐标轴/零值补造、按最短长度截齐**。正常缺 endpoint 与损坏 schema 分开。

使用 `x_B=R(-theta_object)*(x_world-c_object)`。t 的正向始终是 `J*n`；镜像在自己的局部轴上定义，不手动翻增益结果。角度先 unwrap；不得 world-angle 重采样 body-fixed lidar。

真实几何使用冻结 rectangle_size=(1,.25) 与当前物体位姿，只用于离线评价。给每段同一 skill 的 face 标注物体四个物理面的 material ID；用拟合法向与真实面 inward normal 最近角，角差≤15°且与第二候选间隔≥10°，再检查原始 patch 点到该真实面的距离中位数≤.01 m。否则标 ambiguous/mixed-face，不硬配最近角点。段中换面必须分段；端点低/高顺序沿该面一致的 +t 对应，不能逐 tick 选择误差最小的任意角点。

## 2. 全量误差账本：先事实，后归因

全 7,868 side-tick 出账，至少含：

1. **测量层**：每个 endpoint 的 kind、observed、测量区间宽度、真端点在当前拟合 t 上的区间覆盖、离区间距离、区间中点误差；法向角误差、平面法向位置误差、长度类别。`silhouette_interval` 的中点不是精确测量，区间覆盖不能由中点误差代替；shadow_censored 无点不计为“测量错误”，另报覆盖率。
2. **跟踪层**：历史 anchor/两个 tracked endpoint 相对固定真实端点的切向/法向误差、是否落在当前测量区间、fresh/predicted、age、anchor 切换、重捕获/重置、观测间隔；没有内部 velocity/innovation 记录的字段先 missing，待第 4 节重放补齐，不能猜。
3. **参考层**：`M_hat=anchor+s*(L_hat/2)*t_hat`，s=+1 对应低端，否则 −1；先与 `face_midpoint` 和 `target_before_visibility-t_hat*(offset+role+q_rot)` 分别核对，残差≤1e-7 m。preparation hold 覆盖另列，不能混为 estimator 漂移。
4. **下游层**：visibility/reservation 修正、direct offset、e_t、v_ref、提交力和相对机器人位置；只解释链路，不用命令力代算真实接触力矩。

在物体坐标系向量上做严格恒等分解：

```
M_hat - M_true = E_anchor + E_normal + E_length
E_anchor = A_hat - A_true
E_normal = s*L_hat/2*(t_hat - t_true)
E_length = s*(L_hat-L_true)/2*t_true
```

这里 A_true 是已绑定的同一个真实端点，M_true 是同一物理面的真实中点。逐 tick 向量闭合残差≤1e-7 m；同步给切向与法向分量。anchor 误差还需拆成区间中点误差 + 跟踪相对中点偏移；这一代数分解**不是传感器误差的因果百分比**，相关分量须保留符号和抵消。

每侧/场景/profile 分别报 median、P95、max、区间覆盖分子分母、|切向参考误差|>.02 m 的时长及连续≥.25 s 的事件。补报>.01/.04 m 敏感性；不把工程阈值称为泛化保证。按两端均观测/单端观测、kind、面长、anchor_id、机器人/面分组；必须给所有 rejected/unavailable 的数目。

## 3. 围绕 D007 抵消做成对分解

全部 16 probe 与同 case/profile 零偏移基线，复核共同前缀和阶段对齐；固定 1.95 s 基线 body tangent。报告 2–3 s 全曲线及 2.8–3 s 均值：

```
Delta target_B = Delta true_face_midpoint_B
               + Delta reference_error_B
               + Delta direct_offset_B
               + Delta visibility_B + Delta reservation_B
```

面匹配稳定时第一项严格为 0；不稳定则该段不强行比较。把 Delta reference_error 按第 2 节进一步拆开。逐项可还原净目标差，闭合≤1e-7 m。e_t 的重算还原≤1e-7 m；同时输出 raw world 位置只作图示，不参与增益判定。

明确回答：robot1 后段约 4–6 cm 的反向参考变化主要发生在测量区间变化、预测漂移、校正不足、法向变化，还是 anchor 选择？不要从一个相关性曲线宣布因果。检查 ±扰动/镜像；另用 A、同侧和 witness 看是否也有相同误差，不能只看 B/C。

## 4. 历史估计器重放与两个固定离线对照

先从记录扫描构造 LidarScan（origin/heading/angles/distances/valid/category，robot_id/tick/time 来自本行，raw_labels 为空），只重建当时可见 patches，匹配日志选中的 patch。`face_points/face_normal` 匹配容差 1e-7；多匹配或不匹配显式标明。长度从冻结 public prior 与历史 estimate_face 合法几何逻辑恢复，不允许用真值选长度/面。可用记录中点-anchor 反推 L 做一致性检查，但不能用此反推冒充独立恢复。

按 `(run,side,skill_uid)` 重建 EndpointTracker，保持历史调用时刻、暂停、reset、normal-jump、重复 observation 去重。参数固定 tau=.10、max_age=.25、max_speed=.30、residual_limit=.15，并用快照核对来源。历史 R0 必须先复现记录：anchor/endpoints/midpoint≤1e-7 m，status/anchor_side/observed_count/anchor_switched 完全一致。

匹配覆盖需达记录 endpoint 行的≥99%，其余全部逐项解释。若未达门槛，第 2/3 节仍可交付，**停止 R1/R2 比较**，把无法复现作为最先修复的数据/分析问题；不放宽容差、不从真值补状态、不新跑 env。原始 measurement 与融合后区间须分列；记录的 measurements 已经过两端长度约束融合，不能误称 raw measurement。

R0 通过后，在相同录制扫描序列上重算以下两组，只改变离线 estimator 实例，不写生产代码：

- **R1 去速度预测**：初始化、区间融合、gain、normal-plane 重投影、anchor 选择与超时不变；每次 predicted=state.point，更新后 velocity=0；去掉 velocity*elapsed 及速度状态累积，其余不调。
- **R2 无历史区间中点基线**：使用当前融合后端点区间中点，anchor 端选择沿用同 tick R0 的 anchor_side，仅在该端 observed 时输出；不滤波、不预测、不填补无观测。它用于辨别“历史跟踪是否优于当前测量区间”，不是部署方案；与 R0/R1 在三者共同可用行上比较并另报各自覆盖。

记录 R0 的 predicted_q、interval bounds、selected measured_q、innovation、gain、velocity pre/post、clamp/reset reason，解释 interval 内“保持预测值”的行为。`observed` 不等于点定位准确，velocity 从区间投影残差累积也不等于真实切向速度。

报告各组 reference/anchor 的误差、P95、逐步抖动，以及误差和 uncertainty 的对应；既不能单凭更平滑判更准，也不能仅凭更准中点判闭环更稳。对照只说明**相同扫描条件下的估计差异**，不回算“新成功率/新物体轨迹”，不据此推广 C。

## 5. 检查、交付、预算与结果分流

- 新离线测试必须覆盖真实 D007 schema、缺 face_normal 直接失败；已知刚体平移/旋转不改 body 坐标误差；mirror 局部语义；material face 绑定和低/高端交换；区间内真值可有较大中点误差；区间外覆盖失败；预测暂停/重复观测/anchor 切换；两条误差恒等式与实际记录重放。不得只把实现公式再抄一遍作测试。测试可用解析几何造小数组，**不能生成新传感器/物理样本**。
- 运行入口/测试均禁止实例化环境、调用 backend 或 scan_lidar；测试用失败哨兵验证。允许读取既有真值标签进行离线评分，估计器输入只能是原合法扫描/公共先验。
- 单次全量分析，workers≤2、≤1200 s；纯离线测试≤180 s；从首个测试到最终报告图表总计≤1800 s，进程组硬截止 TERM 后 2 s KILL。允许最多一次纯分析 bug 修复重跑，仍计总预算并保留失败产物/脚本哈希；不扩大矩阵/扫参数。无物理预算可调用。
- 输出新目录 `outputs/agent_exchange/D008_<run>/`：manifest、输入哈希与完整分析源码、schema/coverage/replay 校验、全量 side/endpoint CSV、summary.json、16 对 paired decomposition、R0/R1/R2 比较、12 张图与可检索 index.html。图数固定：adjacent/镜像的 B/C×robot0/1 8 张（各含 ±probe、零基线）；A 的 adjacent 两侧各 1 张；同侧/witness 各 1 张，选择 reference 切向误差 P95 最大的记录并写明规则。图必须分开真值、区间、估计、预测/校正、参考和净目标。
- 重点事件表固定最多 12 段，按 |参考切向误差|>.02 m 的连续≥.25 s 事件中峰值排序，同一 run/side/UID 的间隔<.1 s 合并；相同镜像对应只占一个主位，附对应曲线。若不足12，不补造事件。全量统计不受事件筛选影响。
- 回报 `results/D008_Endpoint_Reference_Audit.md`，首段用人话回答：误差在测量还是跟踪；为什么会抵消偏移；能否只删一项。给闭合残差、复现覆盖和所有分母，列出真正未确定的部分。
- 结果分流：区间覆盖<95% 且区间外误差>.01 m 的事件连续≥.25 s，优先提出测量/关联校准；区间覆盖良好而跟踪误差大，检查预测/校正。若 R1 对全体共同可用行参考误差 RMS≤R0*.80、P95≤R0+.005 m、覆盖不降，且 adjacent 与至少一种同侧/witness 上分别成立，可推荐下一轮唯一候选“去速度预测”；不自动修改。R2 仅作定位。若误差主要落在宽区间内，优先报告可观测性限制，而非强行归因算法 bug。分量相关时写 mixed/unknown，不硬贴唯一原因。
- 完成更新 CURRENT 为 ready_for_review，下一位 GPT。无 D009、无估计器修复、无正式验收/训练扩展；完整 D005 回归缺口保留。
