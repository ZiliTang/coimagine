# CONTRACT 修订（local_position_push_v1）

只列相对 push-v3 的**修订**。未列出的条款原样继续有效：物理参数、力上限 15 N、控制周期、
信息白名单、APPROACH/READY/BRAKE/SETTLE 语义（两臂共用）、判稳实现、延迟队列、root 与扰动。

## R1 新增臂 X：局部表面位置反馈推动

```
d = dot(c − p, n)        c = 该段观测质心（机器人自己的位姿变换到世界系）
q = p + d·n
p_contact = q − r·n
gap_signed = d − r        允许为负，不做 max(gap,0)
direction = cosα·n + sinα·t,  t = Rot90(n)
p_ref = p_contact + β·δ·direction
e = p_ref − p ;  e_b = clip_norm(e, e_max=0.10 m)
F_raw = Kp·e_b − Kd·v ;  F = clip_norm(F_raw, F_max=15 N)
```

固定配置（非增益搜索）：`Kp = 100 N/m`、`Kd = 25 N·s/m`（等于冻结基线 Kv）、`e_max = 0.10 m`、
`β` 在进入 PUSH 后 0.20 s 内由 0 线性升到 1、`δ ≤ 0.08 m`。力全程经真实受限力驱动，
限幅后不加任何单位力，不直接设置速度。

## R2 动作到偏置的映射

`δ = Kv·s/Kp`，本轮 Kv=25、Kp=100 故 s={0.08,0.16,0.26} → δ={0.020,0.040,0.065} m。
`local_parameters` 在 δ 超过 0.08 m 时**抛错**而不是静默截断。该映射只匹配零间隙下的名义推进项，
不保证相同实际速度、冲量或轨迹。

## R3 参考接触位置与支撑检查

- 支撑线穿过观测质心、用法向构造；垂足 `q`。
- 垂足离开观测线段切向范围超过 **0.01 m**，且该状态持续超过既有的丢观测容忍 **0.15 s**，
  则以 `patch_support_lost` 退出公共协议；不向无限延长面继续推动，也不无限保留旧面。
- 法向跳变/角点规则沿用既有 30° 与 `face_switch`。

## R4 阶段集成与状态重置

- APPROACH/READY/BRAKE/SETTLE 与 V **完全相同**（同一实现），只有 PUSH 的力不同。
- `PositionPushState.start_push` 在进入 PUSH 时调用：β 归零、冻结技能 δ/α；
  `end_push` 在离开 PUSH 时调用：撤销虚拟推进偏置并清空 p_contact/p_ref/q 缓存，
  不把侵入式参考带进 SETTLE。
- 不重置物理速度、位姿或接触历史；非活跃机器人继续执行共同制动。

## R5 共用传感修复（协议修复，不计入 X 的功劳）

1. `scan_surface` 的距离连续性改为与**相邻**射线比较（原先与本次 run 的第一条射线比较，
   使分组长度依赖斜视程度）。
2. 360° 扫描改为**环形**分组（原先线性分组，跨 0/2π 边界的片段被拆开）。

两者都改变了 V 与 X 共用的观测，因此本轮 V 的数值与上一轮 N 不完全可比。

## R6 日志与指标

新增记录：`beta`、`signed_distance`、`gap_signed`、`q`、`p_contact`、`p_ref`、`error`、
`error_bounded`、`force_raw`、`support`、`support_loss`、`tangential_offset`、
`position_active`、`segment_world`、`contact_point_robot`、`contact_impulse`。
`metrics` 新增 `terminal_robot_speeds`、`terminal_robot_omegas`、`approach_seconds`。
主要工程判断不再使用"速度跟踪接近 1"或"位移达到旧 oracle 的 80%"；改用完成率、接触、
运动量、扰动与接续的分层对照，并明确 2 cm / 3° 的运动门槛（事先固定，事后未改）。
