# D006 机器人控制公式与当前诊断

- 状态：分析/候选，不是正式验收；没有训练、没有改默认。
- 代码入口：`src/coimagine/random_start/effect_push.py`、`experiments/option_rl_v13/push_controller.py`、`experiments/option_rl_v19/system.py`、`src/coimagine/random_start/skills.py`。
- 数据入口：`outputs/testbed/runs/`（`DIAG rotfb ...`）、`outputs/testbed/diagnostics/rotation_feedback_*.json`。
- 记号：`n` 为机器人到表面的单位法向（patch 指向物体）；`t = R(π/2)n` 为机器人左侧切线；`d` 为 option 的 effect direction；`L` 为 face 长度；`R` 为机器人半径；`dt = 0.2 s`（5 Hz 控制）。

## 1. 观测与 face

对每个控制 tick，控制器只使用 `SkillObservation` 的合法输入（own、teammate、scan、patches、contact）：

```text
patch       = 最近的拟合表面
n           = unit(patch.normal_world)       # robot -> surface
t           = rotate90(n)                    # robot left
gap         = patch.nearest_distance - R     # edge-to-edge clearance
face.length = L
face.centre = patch.centre / tracked midpoint
face.anchor = endpoint tracker 的端点
```

法向变化率（`_update_effect`）：

```text
Δθ       = atan2(cross(n_prev, n), dot(n_prev, n))
ω_inst   = Δθ / dt
ω_n      = 0.65 ω_n + 0.35 ω_inst
```

`ω_n > 0` 表示表面法向逆时针转；它不是直接从 object angle 得到，而是从拟合边界的法向变化得到。

## 2. Contact target（`_choose_contact`）

```text
limit = max(0, L/2 - min(0.20, 0.20 L))

heading = sign(h) * max(0, |h| - deadband_h),
          h = atan2(cross(d, n), dot(d, n))
lateral = sign(λ) * max(0, |λ| - deadband_l),
          λ = dot(face.centre - reference_centre, tangent(d))

desired = k_h heading + k_ω ω_n + k_l lateral            # current feedback
desired = clamp(desired + role_offset, -limit, limit) - role_offset

point_offset += clamp(desired - point_offset, -point_rate dt, point_rate dt)
q_rot = clamp(k_rot ω_n, -limit, limit)                  # new explicit term
offset = point_offset + role_offset + q_rot

target = anchor + t * (sign * L/2 + offset)
q = clamp(dot(target, t), lo + margin, hi - margin)
contact_target = target + t * (q - dot(target, t))
```

然后 `SameFaceReservation.project(...)` 可能把 `contact_target` 改成同面保留点；绿色机器人一直在长边右侧，主要受这个 anchor/reservation 约束。

当前参数：`k_h=0.20`、`k_ω=0.10`、`k_l=0.35`、`point_rate=0.16 m/s`、`k_rot=rotation_feedback_gain`。

## 3. 横向与法向速度

```text
e_t      = dot(contact_target - own, t)
v_t      = dot(own.v, t)
lateral  = clamp(k_t e_t - c_damp v_t, -v_t_max, v_t_max)
```

当前 `k_t=tangential_gain`、`v_t_max=tangential_speed`、`c_damp=0.7`。

法向：

```text
# approach / ready
radial = clamp(gap_gain (gap - approach_gap), -0.08, approach_normal_speed)

# push, normal case
speed  = clamp(target_gain * remaining, min_push_speed, speed_command)
radial = clamp(speed + gap_gain * gap, 0, speed_command)

# push, joint pair but not active
radial = 0 if contacting else
         clamp(joint_engage_speed + gap_gain * gap, 0, joint_engage_speed)

raw = radial * n + lateral * t
```

## 4. 参考速度与力

```text
v_ref = RateLimit(raw, a_ref_max=0.5 m/s²)
F     = clip_norm(KV (v_ref - v), F_max),  KV=40 Ns/m
```

即执行层是“速度参考 + 比例力反馈”，不是直接力控制。`F_max` 和 `A_REF_MAX` 是硬限幅；参考速度每 tick 最多改变 `0.5*dt`。

## 5. Coordinator / contact

```text
contact_ready = sensor_contact OR gap <= 0.05
recovery_enter_gap = 0.07
both_ready   = all(skill.ready)
both_contact = all(skill.contact_ready)
phase:
  APPROACH if not both_ready
  READY    if no contact
  ENGAGE   if one-sided contact
  PUSH     if both_contact
  RECOVER/EXIT on loss/done
```

`joint_active = release and both_contact and phase==PUSH`；PUSH 才放行正常力。`contact_ready` 的 0.05 m 是 near-contact 语义，不是物理接触；`contact` 原始字段仍是 tick-start 传感器。

## 6. 任务 success（env，不是 option success）

```text
distance_ok = ||object_xy - goal_xy|| <= 0.20
linear_ok   = ||object_v|| <= 0.02
angular_ok  = |object_ω| <= 0.05
hold        = 以上连续 0.50 s
success     = hold satisfied
```

option 的 `done`/stage 只表示 option 执行结束；`success` 是 env task success。

## 7. 诊断数据（同一 preset、同一初始状态/序列）

`rotation_feedback_gain` signed sweep，object angle_end：

| gain | angle_end | displacement | both contact | PUSH |
|---:|---:|---:|---:|---:|
| 0 | −0.111 | 0.967 | 62 | 69 |
| +0.1 | −0.132 | 0.965 | 62 | 69 |
| +0.5 | −0.181 | 0.927 | 62 | 69 |
| −0.1 | −0.073 | 0.974 | 62 | 69 |
| −0.12 | −0.045 | 0.981 | 62 | 69 |
| −0.15 | −0.034 | 0.981 | 62 | 69 |
| −0.18 | +0.005 | 0.979 | 62 | 69 |
| −0.2 | +0.085 | 0.974 | 62 | 69 |
| −0.5 | +0.220 | 0.957 | 62 | 69 |
| −1.0 | +0.482 | 0.944 | 62 | 69 |

结论：

- 当前公式 `q_rot = +k_rot ω_n` 下，**正 gain 放大 CW，负 gain 才阻尼**；
- 零偏转附近约 `−0.15 ~ −0.18`；
- 全部 `success=False`；这只证明接触/推进，不证明任务成功；
- `R1−object` 的 x 从 gain 0 的约 −0.074 到 −0.15 的约 −0.026，侧向相对位移确实存在，但很小；
- target 相对机器人的 x 会随 gain 增大而增大（target 前移），但机器人没有等量跟踪；**target 移动 ≠ 机器人横向修正**。

## 8. 公式暴露的问题

1. **符号语义错误/不统一**：当前 `q_rot=+kω_n` 在这组几何下是正反馈；数据要求阻尼项为 `q_rot=-k_damp ω_n`（或 signed gain 取负）。参数说明里的“+ = CCW→left”与“阻尼”不是同一件事，必须拆开定义。
2. **target 不是被控量**：target 是 endpoint-relative 的接触点，机器人实际跟的是 `e_t = <target-own,t>`；target 随 anchor/object 平移时，`e_t` 可能很小，导致“target 动了但机器人相对物体没侧移”。
3. **端点/anchor 主导**：target 的主要运动来自 endpoint tracking 和物体平移，不是旋转反馈；端点估计误差会直接移动 target。
4. **same-face reservation**：它可能覆盖 `contact_target`，使左右重分配失效；绿色机器人一直在右侧。
5. **横向执行受限**：`v_t_max`、`point_rate`、接触/力门控和 `contact_tangent_damping` 共同限制实际侧移；不能只看 target。

## 9. 候选修复（未改默认）

最小候选：

```text
q_rot = -k_damp * ω_n,  k_damp >= 0
```

即把现有 signed gain 的推荐值改成负值，或把实现符号取反，让正参数表示阻尼。当前 signed 参数已经可以直接用 `−0.15` 测试，不需要改默认。

进一步（更大改动）：

- 把 `e_t` 和 `q_rot` 直接暴露到 run/曲线，检查 target→lateral 指令链路；
- 在 endpoint 估计上加稳定性/误差指标；
- 给 same-face reservation 一个“旋转偏差超过阈值时允许重分配”的条件；
- 用 object/face 的几何符号（不是固定 `+`）决定 `q_rot` 的符号，避免换边后反向。

## 10. 结论

现在的控制律可以写成：

```text
target = anchor + t (sign L/2 + point_offset + role_offset + q_rot)
e_t = <target-own,t>
v_ref = radial n + clamp(k_t e_t - c_damp v_t, ±v_tmax) t
F = clip(KV(v_ref - v), Fmax)
```

数据说明：**`q_rot` 的符号/方向和 target 到机器人横向的传递是当前两个主要问题**；不是简单“target 不够敏感”。下一步应先把 `q_rot` 的符号统一为阻尼，再确认 target→`e_t`→lateral 是否真的传递，最后再谈端点重分配。
