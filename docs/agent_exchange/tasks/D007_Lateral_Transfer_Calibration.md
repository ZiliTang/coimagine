# D007：侧向控制减法与传递校准

2026-10-30，GPT v1；状态 `ready_for_handoff`。用户移交后由 DS 实现和执行，完成即停止。

**目标：测清接触目标能否传到实际侧移，比较去掉额外阻尼是否更好；不根据单个终点角度翻转旋转反馈符号。**

- 原理与审查：[D006_Control_Audit.md](../reviews/D006_Control_Audit.md)。用户说明：[D007_Why_This_Calibration.md](../briefs/D007_Why_This_Calibration.md)。
- 冻结输入/哈希/数值：[D007_config.json](D007_config.json)。旧 DS 两份 D006 公式报告是待修正证据，不覆盖本任务的公式。
- 允许：新增可选 PUSH profile、诊断偏移入口、纯记录字段、对照脚本及针对性测试。复用 v19 session 和 v13 控制器，避免复制整份 step；诊断入口默认零且不进入策略输入。
- 禁止：训练、默认 profile 替换、改 reward/物理/成功条件/动作频率/感知、改 ROTATE/FOLLOW/REPOSITION 行为、改接触 .05/.07、力/速度/加速度限幅、同面 reservation 或 endpoint 算法。D002 B 仍冻结。

## 1. 实现三组，不自行选参数

| profile | PUSH 目标修正 | PUSH 接触阶段侧向控制 |
|---|---|---|
| A `legacy` | 冻结配置；direct `rotation_feedback_gain=0` | 原两级截断，c=.7 |
| B `outer_off` | `feedback=False`；heading_gain/angular_rate_gain/lateral_gain/rotation_feedback_gain 均 0 | 与 A 相同，c=.7 |
| C `minimal` | 与 B 相同 | c=0，即 `sat(k_t*e_t, ±v_t_max)` |

每个 preset 的其余 override 原样保留；adjacent 的 k_t/vmax 是 8.8/.5，同侧原版是 4.8/.213333333333。不得同时“统一参数”。profile 最后应用并序列化**解析后的**完整 config；unknown profile/非法数值报错，不静默忽略。

只在 translate 且未 follow_started 的目标 desired 中加入 `diagnostic_tangent_offset`（m，默认 0），位置在 point_offset 的 clamp/rate-limit **之前**。保留 role_offset 和后续所有安全投影。诊断脚本根据仿真时钟写入，不读取真值作决策；普通 run 一律 0。此入口不开放成 RL 动作，不在正式默认中设负 gain。

补记录并纠正含义：`n,t,heading,lateral_error,omega_est,anchor/side/status,point_offset,role_offset,q_rot,target_before_visibility,target_before_reservation,target_final,e_t,v_t,k_t*e_t,first_clip,c*v_t,lateral,radial,raw_before/after_avoid,raw_before/after_norm_cap,v_ref,F_pre/post_norm_cap,F_pre/post_gate`。记录 reference 限速、投影裁剪和 gate 的实际操作，不用 phase 名猜 force 被禁用。保留旧字段用于兼容，新增明确字段并修文档说明即可。

每行对齐：P0 tick-start 观测/真值（真值仅离线）；P1 coordinator 读取旧 flags；P3 本 tick 控制结果；P4 物理推进后状态与几何 contact points。`CONTROL_DT=.05`、decision_dt=.2；不要用当前 UI 混合帧直接重算 e_t。拟合角速度使用原有实际观测间隔，不改滤波系数。

## 2. 固定矩阵：最多 36 条正式物理轨迹

1. **12 条基线**：config 中四个场景 × A/B/C。adjacent 5 s，其 x 镜像 5 s，同侧 8 s，同侧 closer 8 s；从 reset，完整冻结序列。
2. **16 条小扰动**：仅 adjacent/其镜像 × B/C × robot0/robot1 × ±.04 m。每条仍 5 s，只在 `2.0≤t<3.0 s` 给指定机器人偏移，另一侧为 0，其他时刻 0。零偏移对照复用第 1 项，共同前缀 0–2 s 的位姿/控制误差应 ≤1e-9。必须从 reset 重放，不能只 restore 位姿；此约束仅同 profile 扰动与零对照间成立。
3. **6 条参考序列回归**：D005 的 W_N/W_E × A/B/C。使用冻结日志中的 instance/commands；无 preset override，仅统一应用三组 profile。报告本次结果，不要求冒充旧版本 129 tick 的逐行复现。
4. **2 条记录非干扰复跑**：adjacent 的 A/C 各一次关闭新增记录，与第 1 项比较所有 post-tick 位姿及提交力，误差 ≤1e-9。仅测试记录开关，不关闭 profile。

镜像严格使用 `x'=5-x,y'=y,angle'=wrap(pi-angle)`，镜像物体、机器人、goal、场景静态几何与向量；action E→W（3→19）、N→N（11），robot id 不换。不从 UI 拖动重造；保留变换前后 JSON、哈希及动作映射。若几何不合法停止该项，不挪位置修到可运行。

输入哈希必须先核对。先跑纯数学测试与两条非干扰所依赖的基线；失败立即交回。其余顺序按 1→2→3；不补 seed、不重跑负结果、不加 probe。缺失/异常/提前终止均入账；2 s 前未进入可用状态则该扰动标 unavailable，不改触发时间。

## 3. 指标与判定（不事后改门槛）

- 按轨迹报告实际秒数、success/termination、异常、沿两动作方向向量和的单位方向推进、goal 距离、最大 |theta-theta0|、末角、积分 |omega_object|、真实双方 contact 时长、near-ready 时长、每层裁剪/限速比例。角度先 unwrap；接触时长不能由 ready 代算。
- 对扰动侧：记录 projection 后 target 相对本 profile 零扰动的切向差、机器人和几何 contact point 的物体坐标系切向差，以及 `Delta omega_object`。离线以 2 s 前选定面的 body-frame tangent 固定符号；反射后按本地 +t 重新标定。无接触则实际接触点为 missing，不补零。
- 响应取 2.8–3.0 s 均值；`G_robot=sign(delta)*Delta q_robot/.04`，`G_target` 同理，同时报告 2–3 s RMS e_t 和全曲线。目标向 +t 偏移与机器人向 +t 响应必须分列，不以 object angle 代替位置响应。
- 有效跟踪样本：2–3 s 内请求方向 target 差≥.02 m 累计≥.5 s，扰动侧真实 contact≥.5 s，且无跳面/anchor 切换、无异常。其余全部保留并标 projection-blocked / no-contact / geometry-change / anomaly，可多标签；不给它们编一个“零增益”。
- C 的删除阻尼候选获得支持须同时满足：B/C 共同有效扰动≥6/8；这些配对中 C 的 G_robot 中位数比 B 至少高 .10 且 RMS e_t 中位数≤B 的 .90；四条无扰动场景相对 A 的推进损失各≤max(.02 m, .05*|A推进|)、最大角偏增加≤.05 rad、双方真实接触时长≥.9*A-.10 s；W_N/W_E 保持 A 的成功，耗时≤A+max(.5 s,.1*A耗时)；无新增异常或失败退出。A 自己失败的 witness 不算候选通过该验收。
- 上述是工程筛选门槛，不是统计泛化结论。有效样本不足写 inconclusive；任何退化照报，不调参数救分。B 对 A 的结果单独报告，不能把外环删除与阻尼删除合并归因。
- 不用本轮 angle_end 决定 q_rot 默认符号。若拟合 omega 与真值明显异号/跳变，列出窗口；若裁剪为主，列最早阻断层；接触力矩未直接观测则写 unknown。

## 4. 校验、预算与交付

- 纯数学/配置测试：四个轴向法向的 `cross(t,n)=-1`、ROTATE lever 方向；同几何切换高/低 anchor 的 target 等价；B/C 外环确实为零、非 PUSH 未改；偏移经过 rate/projection；c=0 等于一次位置截断；上文两级截断数值 .36 与 .5 不混淆；有限值、mirror involution/动作映射；日志 gate 与实际提交力一致。
- 聚焦运行：新 D007 测试 + `test_d003_contact_timing.py`、`test_d004_state_pair_gate.py`、`test_d005_option_system.py`、`test_joint_contact_recovery.py`、`test_option_rl_v13_recovery.py`、`test_same_face_allocation.py`。测试中运行的物理实例单列，不混进 36；不跑全量 pytest，不另做临时物理探针。测试的 rollout 计入测试墙钟。
- workers≤4；正式轨迹阶段≤600 s，聚焦测试≤180 s；从首个测试/物理运行到分析回放写完总计≤900 s。外部 supervisor 硬截止整个子进程组（TERM 后 2 s KILL），包含正在运行的 worker；不能只 cancel future。到点保留产物，incomplete，禁止自动重启。
- 运行前复制基线源码快照并记录 HEAD+dirty；修改后保存候选快照及全部输入哈希，固定 Python/依赖版本。运行期间不得改源码。允许纯分析 bug 修复后只重跑分析，记录新旧脚本哈希并仍守总预算；不因此补物理轨迹。
- 交付 `outputs/agent_exchange/D007_<run>/`：manifest、冻结输入与源码、36 条以内原始 tick JSON、配对 CSV、六张链路图（至少覆盖镜像/正负）、可选 profile 的本地回放。保留命令、每阶段墙钟、未完成项和测试创建实例数。
- `results/D007_Lateral_Transfer_Calibration.md` 首段用人话回答：去掉哪些项后，机器人更能跟随目标了吗；是否牺牲推进/稳定性；下一处断点在哪里。若未找到主因，明确说未定位。不自动发布 D008，不重启/替换用户正在使用的默认 server。
- 更新 CURRENT 为 ready_for_review，下一位 GPT。通过门槛也只交回候选，不自动推广为默认；完整 D005 8-case 和 RL 验证仍未完成。
