# D006 侧向偏移问题简报（给规划侧）

- 状态：诊断/分析，**未改默认、未训练、未做正式验收**。
- 目的：请规划侧对控制算法的**符号语义 + 数学逻辑**做一轮系统检查；检查后再发有界任务书。
- 完整公式整理见：[D006_Control_Equations.md](D006_Control_Equations.md)。
- 可复现实验：`experiments/option_rl_v19/diag_rotation_feedback.py`；runs 已保存到 `outputs/testbed/runs/`，在测台 Runs 列表可见；摘要 `outputs/testbed/diagnostics/rotation_feedback_*.json`。

## 1. 现象

同一侧/相邻侧双推（`相邻侧双推 5s 右上`）中：

- object 发生明显偏转；
- target 接触点随之移动，但机器人相对物体的**横向 x 偏移很小**；
- 某些阶段 target 已明显右移，机器人并没有等量侧移；
- 7–7.5 s target 右移与 CW 偏转对应，但后续 CCW 时 target 没有明显左移。

结论限定：这是**控制链/目标跟踪问题**，不是“target 完全没动”。

## 2. 关键代码路径

| 环节 | 代码 |
|---|---|
| 法向变化率 | [`effect_push.py:645`](/home/tzl/repo/coimagine/src/coimagine/random_start/effect_push.py#L645) `_update_effect` |
| contact target / q_rot | [`effect_push.py:670`](/home/tzl/repo/coimagine/src/coimagine/random_start/effect_push.py#L670) `_choose_contact` |
| 横向误差/指令 | [`effect_push.py:827`](/home/tzl/repo/coimagine/src/coimagine/random_start/effect_push.py#L827) |
| 参考速度/力 | [`skills.py:340`](/home/tzl/repo/coimagine/src/coimagine/random_start/skills.py#L340)；[`control.py:176`](/home/tzl/repo/coimagine/src/coimagine/local_contact_option/control.py#L176) |
| contact_ready / recovery | [`push_controller.py:160`](/home/tzl/repo/coimagine/experiments/option_rl_v13/push_controller.py#L160) |
| pair coordinator | [`system.py:56`](/home/tzl/repo/coimagine/experiments/option_rl_v19/system.py#L56) |

## 3. 当前数学形式（摘要）

```text
n = unit(patch.normal_world)       # robot -> surface
t = rotate90(n)                    # documented as robot-left
ω_n = filtered dθ(n)/dt            # from fitted normal change

# target
point_offset += clamp(k_h heading + k_ω ω_n + k_l lateral - point_offset,
                      ±point_rate dt)
q_rot = k_rot ω_n
target = anchor + t * (sign L/2 + point_offset + role_offset + q_rot)

# lateral
e_t = <target - own, t>
lateral = clamp(k_t e_t - c_damp v_t, ±v_t_max)
v_ref = radial n + lateral t
F = clip_norm(KV (v_ref - v), F_max)
```

`k_rot = rotation_feedback_gain`，新增 signed 参数（−2..2）；默认 0。

## 4. 已跑对照（同一 preset/初始状态/序列）

| gain | angle_end | displacement | both contact | PUSH |
|---:|---:|---:|---:|---:|
| 0 | −0.111 | 0.967 | 62 | 69 |
| +0.1 | −0.132 | 0.965 | 62 | 69 |
| +0.5 | −0.181 | 0.927 | 62 | 69 |
| −0.1 | −0.073 | 0.974 | 62 | 69 |
| −0.15 | −0.034 | 0.981 | 62 | 69 |
| −0.18 | +0.005 | 0.979 | 62 | 69 |
| −0.2 | +0.085 | 0.974 | 62 | 69 |
| −0.5 | +0.220 | 0.957 | 62 | 69 |

所有 run `success=False`。这只证明接触/推进，不证明任务完成。

## 5. 需要规划侧检查的符号/数学问题

1. **`q_rot` 的符号链**
   - 当前 `q_rot = +k_rot ω_n`，`t=rotate90(n)` 被文档定义为机器人左侧；
   - 实测正 gain 放大 CW、负 gain 才阻尼；
   - 请检查 `n`、`t`、`heading`、`lateral`、`anchor_side`、`sign`、`offset` 的符号是否在同一坐标系中一致；
   - 特别是：`q_rot` 的符号、`t` 的左/右定义、target 偏移与接触力矩方向是否匹配。

2. **target 是不是正确的被控量**
   - `target` 是 endpoint-relative 接触点，但实际执行的是 `e_t=<target-own,t>` 与 `lateral`；
   - 机器人相对物体的横向 x 只从约 −0.074 变到 −0.026（gain −0.15），而 target-robot x 可到 0.3–0.5 m；
   - 请检查：target 偏移是否真的转化为接触点/力矩，而不是只移动参考点。

3. **target 的运动来源**
   - target 主要随 endpoint anchor、物体平移和 same-face reservation 移动；
   - `q_rot` 只是其中一项；请检查 endpoint 估计误差是否主导 target，及是否应先修 endpoint/tracker。

4. **横向执行权限**
   - `point_rate=0.16`、`v_t_max`、`contact_tangent_damping=0.7`、`F_max` 与 contact/force gate 是否共同把侧向指令吃掉；
   - 需要规划侧明确“target→lateral→实际 x”的哪一段是限制点。

5. **same-face allocation / reservation**
   - 绿色机器人长期被留在长边右侧；reservation 是否在 `q_rot` 之前覆盖 target；
   - 是否允许在旋转偏差超过阈值时重分配。

6. **near-contact 语义**
   - `contact_ready` 用 gap≤0.05，不是物理接触；它影响 pair gate/PUSH，不应与真实 contact 混读；请确认侧向反馈测试的基线是否受它影响。

## 6. 建议规划侧先回答的问题

- 在标准几何下，`q_rot = +kω` 与 `-kω` 哪一个应产生**反向力矩**？给出 `n/t/anchor/contact point` 的符号推导。
- 若目标是“抵抗当前 rotation”，`q_rot` 应正比于 `+ω` 还是 `-ω`？左/右应如何由 `cross(r_contact-r_obj, F)` 决定，而不是固定符号？
- target 偏移的物理作用点到底是**机器人接触点**还是**参考点**？若是参考点，是否需要把 `q_rot` 直接作用到 lateral/contact force，而不是 target？
- endpoint/tracker 与 same-face 是否会主导 target；是否应先做 endpoint 稳定性诊断，再改反馈？

## 7. 建议的审计后测试方式（供规划侧写任务书）

- 固定同一 preset/初始状态/序列，比较：
  - `q_rot` 符号翻转前后；
  - `target→lateral` 链路的 `e_t`、lateral 指令、`R1−object` x；
  - 端点估计误差与 target 位移的相关性；
  - same-face reservation 开启/关闭的对照。
- 评测分开：
  - **任务 success**（official 0.20/0.02/0.05/0.5）；
  - **侧向/接触诊断指标**（angle、target-robot、R1-object、contact continuity）。
- 不改默认、不训练；先只读/有界对照。

## 8. 结论

当前证据支持：**符号语义和 target→lateral 传递是主要问题，而不是“target 不够敏感”**。`rotation_feedback_gain` 的 signed sweep 已给出可复现的阻尼区间（约 −0.15 ~ −0.18），但该符号是否应写进实现、以及 target 是否应改成力/力矩目标，需要规划侧先做数学与语义审计，再发任务书。
