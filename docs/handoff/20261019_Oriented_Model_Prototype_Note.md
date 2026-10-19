# 备忘：朝向模型原型 A（机器人需要转向、稳定推面）与有界对照

日期：2026-10-19｜来源：option 测台 Round 003 后的模型保真度讨论｜状态：**原型已实现并对照，未改冻结 v3.1**

## 1. 问题

用户指出：当前机器人**推动过程中完全不转**（`backend._apply_forces` 对机器人设
`torque=0`、力施加在质心），朝向只影响 72 束雷达的索引帧，对推动行为几乎无影响。真实机器人
一般用稳定的一侧（如车头 bumper）推物体，需要先对准；差速驱动还只能沿朝向平移。这是一个
**模型保真度**问题，会决定 option 语义，晚改会作废已调结果。

## 2. 原型 A 实现（默认关闭，v3.1 逐值不变）

- `EnvConfig` 新增 `oriented=False`、`max_omega=1.5`、`align_gain=3.0`、
  `align_tolerance_deg=20.0`，并进入 `as_dict()`。
- `RandomStartEnv` 新增**运动学 heading** 状态（`heading`/`heading_rate`）：观测、雷达、
  `true.robots` 在 oriented 时使用该 heading；`scan_for`/`observe` 通过 `_legal_robot_states()`
  与 `dataclasses.replace` 注入。
- 每个控制周期：以**力方向**为目标方向，用比例控制转动 heading（`ω=clip(gain·error, ±max_ω)`）；
  朝向误差 > 容差时，力按 `cos(error)` 缩放（“稳定推面”的简化）；行日志新增
  `heading/heading_rate/align_error_deg/force_scale/force_gated`。
- `export_branch_state`/`restore_branch_state` 记录并校验 `oriented` 与 heading，防止跨模型恢复。
- **默认 `oriented=False` 时行为逐值不变**：Round 000 的 6 个 v3.1 分支最终物体状态
  6/6 完全复现；测台与 random_start 定向测试 21 项通过。

## 3. 有界对照（同一几何、同一动作序列）

命令：`timeout ... 120s .venv/bin/python experiments/option_testbench/oriented_prototype.py`
（输出 `outputs/option_testbench_v1/oriented_prototype/`，3.6 s）。

| 分支 | 模型 | 物理 s | 物体位移 | 转角 | 目标进展 | 接触 s (r0/r1) | 平均朝向误差 | 平均力缩放 |
|---|---|---:|---|---:|---:|---|---:|---:|
| face__push_push | v3.1 | 5.10 | (0,0) | −0.0 | 0.000 | 2.05/2.05 | — | — |
| face__push_push | oriented | 6.20 | (0,0) | −0.0 | 0.000 | 2.05/2.05 | 47.3° | 0.68 |
| corner_minus__push_push | v3.1 | 5.00 | (0,0) | +30.7 | 0.000 | 2.05/2.05 | — | — |
| corner_minus__push_push | oriented | 5.70 | (−0.001,+0.003) | +30.8 | 0.003 | 2.05/2.05 | 57.7° | 0.61 |
| corner_plus__push_push | v3.1 | 3.20 | (0,0) | +0.4 | 0.000 | **0.05/0.05** | — | — |
| corner_plus__push_push | oriented | 6.20 | (0,0) | +33.6 | 0.000 | **2.05/2.05** | 49.1° | 0.68 |
| teammate__push_push | v3.1 | 5.15 | (0,+0.231) | −0.0 | +0.231 | 2.05/2.05 | — | — |
| teammate__push_push | oriented | 6.20 | (0,+0.225) | −0.0 | +0.225 | 2.05/2.05 | 47.3° | 0.68 |
| edge_turn__transfer_cw6_then_push | v3.1 | 26.50 | (−0.037,−0.205) | −13.2 | +0.205 | 0.00/2.10 | — | — |
| edge_turn__transfer_cw6_then_push | oriented | 32.85 | (−0.021,−0.308) | −15.3 | +0.308 | 2.05/3.35 | 45.1° | 0.80 |

## 4. 结论

- **方向性结论不变**：对称角点双推仍是**力偶**（`face` 目标 0.000，两侧都转正后仍抵消）；
  `teammate` 同侧推仍有效（+0.225 对 +0.231）；`edge_turn` 换面后再推仍为正（+0.308 对 +0.205）。
- **朝向会改变部分接触结果**：`corner_plus` 从 `contact_lost`（0.05 s）变成 2.05 s 接触；
  `edge_turn` 的 r0 从 0.00 s 变成 2.05 s 接触。转向运动改变了接触几何，说明“先对准”不是
  纯开销，会改变行为。
- **代价是时间与力**：180° 对准使单技能 5.10→6.20 s、换面 26.50→32.85 s；平均朝向误差
  45–58°、平均力缩放 0.61–0.80（转向期间推力被 cos 缩放）。
- 因此：**朝向模型值得纳入下一版**，但当前 option 结论（力偶、换面）在方向上是稳健的；
  真正需要重做的是“对准阶段 + 稳定推面”的技能语义与 RL 输入。

## 5. 原型的限制（下一版必须补齐）

1. **不是力矩耦合模型**：heading 是运动学的，接触反力矩没有建模；“稳定推面”只是
   `cos(error)` 缩放，没有真正的前端接触点。
2. **观测缺少 `omega`**：324 维里只有 `cos/sin/vx/vy`，策略看不见自己的角速度；下一版
   必须加 `omega` 与朝向误差。
3. **技能没有 align 阶段**：当前是对准与推力同时进行；生产版应有显式 align/对准再推，
   并重新调参。
4. **未定驱动模型**：本原型是“全向平移 + 朝向”的折中；若目标平台是差速驱动，应改为
   非完整约束（只能沿朝向平移），换面/绕行变成“转—走—推”序列。
5. **单一分支、未重训、未覆盖全部 snapshot**；未使用 `segment-contact-v1`（两边都用 baseline
   技能，以隔离朝向效应）。

## 6. 需要规划端决定

- 目标驱动方式：**全向 + 稳定推面**（本原型路线）还是**差速/非完整**（需要重做运动学）？
- 是否把“前端接触点 + 接触反力矩”做成真实物理（而不是 `cos` 缩放）？
- 新版本需要：`ENV_VERSION` 升级、新 snapshot、新观测（含 `omega`）、显式 align 技能阶段、
  新契约与重训；v3.1 与现有 checkpoint 保留为对照。
- 在这些决定之前，**不建议继续在旧模型上调 option**；Round 004 的接触目标可视化可以照做，
  但接触语义的最终版应以新模型为准。
