# D013 提案：可见边界上的双接触点选择与 q 跟踪

- 状态：`proposal_for_planning`；未授权实现/训练。参考 `mode opt.txt` 的思路，不复制其已知形状/真值依赖。
- 目标：让 q 不再从 support 中心无约束漂到端点；接触点选择显式考虑法向合力/力矩与参考推动方向，且双机接触点保持分离。

## 1. 已核实的问题

- 当前 `choose_candidate_q` 只在初始化/support extension 时取可见 support 区间中心；没有力/力矩优化。
- `_advance_candidate_with_motion` 每 tick 用 `v_at(q)` 积分 q，但**不把 q 投影/裁剪回当前 support**；q 会从中心漂到观测端点/角点。
- 实测 reposition 后南推：support≈[2.10,2.97]（中心 2.535），q 从 2.50 漂到 2.99；q' 只做 mm 级 offset。
- 当前 `LIDAR_RANGE=1.20m`；R1 太远时可能完全看不到物体，导致侧向速度/线拟合不可靠。
- `SharedMotionEstimator` 有协方差 `P`，但没有显式“估计是否可靠/可观测”门控。

## 2. 可见 support 语义词

- 不构造完整物体形状；只使用两侧机器人当前看到的 support 线段/端点/法向。
- 对每侧提取可见 support 参数区间 `[lo,hi]`、法向 `n_i`、可见长度/质量。
- 候选 q 必须落在至少一侧的可见 support 内；不补未知端点。
- 双机接触点分离 `||q0-q1|| >= 2R + margin`（例如 .40m）；同面/异面都用世界坐标判距。

## 3. 轻量接触点选择器

以每侧机器人**最接近的可见接触点**为唯一初始点，做单初始坐标下降（不做 multi-start）；同一面上两个 q 保持初始顺序，不允许互相越过（避免双机交叉）。

```text
cost = wF * ||sum(f_i) - F_des||^2
     + wM * (sum(cross(q_i - o_ref, f_i)) - M_des)^2
     + wSep * max(0, 2R+margin - ||q0-q1||)^2
     + wEdge * (edge/corner penalty)
     + wMove * travel_from_previous_q
```

- `f_i ≈ n_i * p_i`，p_i 由当前 P/推力方向决定；不使用隐藏摩擦/质量。
- `F_des, M_des` 来自参考推动方向（例如南向 `(0,-1,0)`）。
- **关键限制**：没有形状先验时 `o_ref` 不是真实质心；只能用可见 support 的参考点/接触点中点做近似，必须明确标注“力矩参考近似”，不能声称真实质心力矩。
- 双机点不能太近；若可见 support 放不下分离点，先一个 leader 接触/等待，不硬分。
- 参考 `mode opt` 的方向对齐、距离分配和坐标下降思想；**不使用它的 multi-start/随机起点**。候选集合来自可见边界而非 shape contact_points。

## 4. 跟踪：可靠估计时随动，不可靠时回投影

### 可靠
- 以 `SharedMotionEstimator` 的协方差/条件数、两侧可见线是否非平行、可见长度/接触质量作为门控。
- 通过门控时，q 按 material velocity `v + ω×r` 随动，并**每 tick clamp 到当前 visible support**；定期重选/重优化，避免漂到端点。

### 不可靠
- **停止 q 的 v/ω 积分**。
- 用当前自身观测：把 q 投影到当前可见 support 的参数区间，或按当前 patch 的相对位置重定位；保持 q 在 support 内。
- 仍执行接触点分离约束；如果看不到合适 support，保持/等待而不是外推。
- 这样 q 不会因为 R1 看不到物体/侧向速度估计差而积分漂移。

## 5. 雷达/可见性

- 当前 `LIDAR_RANGE=1.20m`；建议先做单变量实验提高到 1.8–2.0m，并记录两侧 visible support 长度/是否看到物体。
- 增大雷达不是替代估计门控；仍要在“看不到/单侧/平行面”时使用回投影策略。
- 不因看得更远就假定速度可观测。

## 6. 实施顺序

1. 先修 q 的 support clamp/reproject（最小改动，单独验证 q 不再漂到端点）。
2. 单变量增大 `LIDAR_RANGE`，确认 R1 可见；不影响成功判据。
3. 实现纯函数 `select_contact_pair(visible_supports, desired_wrench, ...)`，单测力/力矩/分离/端点。
4. 接入 `UnknownShapePush`，先关闭 q motion 积分，用当前 observation 重投影；再开门控随动。
5. 在 B/F、N+E、reposition 后推、弱推进上验证：q 端点距离、接触保持、object angle/progress；不宣称完整泛化。

## 7. 风险

- 力矩参考点非真实质心，可能选错；必须把 `o_ref` 近似和误差写进报告。
- 可见 support 短/部分遮挡时没有足够候选；应等待/单 leader，而不是外推。
- 力矩优化可能与分离/可达性冲突；需要明确优先级。
- 速度估计协方差门控阈值需要校准，不能只凭单场景。

## 8. 当前实现进度

- `LIDAR_RANGE` 1.20→1.80。
- `UnknownShapePush` 的 q motion 增加 support 投影/clamp；不可靠时默认关闭 v/ω 积分。
- 新增纯函数 `select_contact_pair`：每机最近可见点为初始，坐标下降，同面不越过另一机；纯测试 `tests/test_d013_contact_pair.py` 3 项通过。
- paired 且同面 allocator 失败时接入 selector fallback；N+E 仍未验证 selector 是否被实际选中（其 q reason 仍 support_extension）。
- 未完成：torque reference 的精确性、可靠估计门控、selector 在 N+E/双机压力场景的实测。

## 9. selector vs 速度积分（主从关系）

- **主跟踪**：可见边界 selector 重算 q。当前任务里更可靠，因为 R1 可能看不到物体/侧向速度不可观测；速度积分无 support 校正，会漂到端点。
- **速度积分**只作为短时预测/平滑：仅在 EKF 可观测、协方差门控通过、且预测结果不越出 support 时使用 1–2 tick，随后重新 selector/clamp。
- 不可观测时：不积分，保持/重投影到当前可见 support。
- 端点突变仍会造成 selector q 跳变；用 `travel_from_previous_q` 代价、上一个 q 热启动/限幅和事件重算抑制，但不要做无约束积分。

## 10. 决定：selector-only（暂不做速度积分）

- q 短预测在隔离测试中会跳到 0.93/1.01；selector-only q 连续，N+E maxangle .273、156 tick。
- 当前删除 q motion 积分和 D013_Q_PREDICT；保留 0.1s selector + 中间 hold q。
- 速度积分不再作为默认路径；若以后恢复，必须重新通过协方差/最大步长/support clamp 门控。

## 11. 力矩参考修正

- 旧 `o_ref` 为两 support 区间中心的中点，N+E 时 y≈1.72，偏下，selector 会偏好左下 q0。
- 改为两侧可见 support 点的 bounding-box 中心（y≈1.96–2.0）；N+E q0 移到左侧上半段（y≈2.21），task success 130 tick、angle .21。
- 仍是可见边界 bbox 近似，不是真实质心；后续需在可见面不完整时说明误差。

## 12. 双锚点实现

- selector 保存 world q + support-relative anchor（face n/rho、u、support span）。
- local 时用 anchor 映射到当前同面；面匹配/跨度门控失败则回退 world q。
- 修复与 env 原 `_last_anchor` 的命名冲突（改为 `_selector_anchor`）。
- N+E goal fixed：success 140 tick、angle .121，q 连续；reposition：success 535、angle .008。

## 13. EKF 不可观测子空间的速度先验

- 单面接触只给一个法向速度标量约束；切向速度、角速度在零空间中，默认取 0 是合理的**先验/正则**，不是观测。
- 第二个非平行面可以补一个独立法向约束；但应使用接触点 `r_i` 的刚体约束 `n_i·(v+ω×r_i)`，不能只看法向方向。两个非平行面一般仍可能留下一个自由度。
- 实现应做 null-space prior（零空间向 0 收缩、协方差相应变大），而不是硬编码每个世界分量=0；第二面/接触点约束可覆盖。
- 目的：抑制不可观测方向的 EKF 漂移，同时保留真实切向滑动/转动的可能性。
