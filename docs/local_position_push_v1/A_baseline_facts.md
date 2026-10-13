# A. 基线事实表（`local-contact-option-push-v3`）

核对时间 2026-10-13 17:00。用户给的基线参数与实际代码一致，未发现版本差异；本轮**冻结实际当前版本**，
没有拼装不存在的配置。

| 项 | 实际值 | 位置 |
|---|---|---|
| 提交 | 见 `manifest.json` 的 `git_head` | — |
| 配置 hash | `roots_{split}.json` 的 `frozen_config` + 每条执行的 `option_config` | `outputs/local_position_push_v1/<run>/` |
| Kv | **25.0 N·s/m** | `dev_kv_selection.json`（push-v3 选定，本轮未重选） |
| 参考加速度上限 `a_ref_max` | **3.0 m/s²** | 同上 |
| 接近上限 / 接触速度 / 增益 / 预接触 | 0.30 / 0.06 / 4.0 / 0.02 | 同上 |
| 力上限 | 15 N（= `PhysicsConfig.max_force`，冻结） | `contract.FORCE_LIMIT` |
| 控制周期 | 0.05 s（20 Hz），物理 1/120 s，6 substeps | `continuous2d` |
| 速度式 PUSH 公式 | `v_raw = s(cosα·n + sinα·t)`；`v_ref` 限幅 `a_ref_max·dt`；`F = clip_norm(Kv(v_ref−v), 15)` | `control.local_push_force` |
| APPROACH | `n·clip(g·gap, 0.02, 0.30)`，`gap=d_patch−r`，`gap≤5 mm` 时用 `v_contact` | 同上 |
| READY | `n·0.02`（本轮上调后的预接触速度）；掉接触退回 APPROACH | `option` 阶段机 |
| BRAKE | 入口用实测速度初始化 `v_ref`，随后以 `a_ref_max` 向零限幅；**施力用正在减速的参考**，不是零参考 | `control.brake_force` |
| SETTLE | `F=clip_norm(Kv(0−v),15)`，判稳：`|v|<0.03`、`|ω|<0.08`、表面法向速度 RMS `<0.02`，保持 0.30 s | 同上 |
| 观测延迟 | 每个变体各自设定；延迟变体 0.10 s。按到达时间投递「已到达」的最新帧，保留在途帧，启动期明确 unavailable | `delay.FixedDelayQueue` |
| 判稳实现 | 两次**采样时间戳差**求法向速度，逐点 RMS；帧龄 >0.20 s、重复/乱序时间戳、匹配点 <3 一律 unavailable 并记原因 | `control.update_surface_estimate` |

## 六项既有问题的回归证据

| # | 问题 | 回归测试 |
|---|---|---|
| 1 | 表面速度用采样时间戳差、逐点 RMS；重复/乱序不能累积新稳定证据 | `test_local_contact_option_repair.py::test_five_millimetres_in_fifty_milliseconds_is_point_one_metres_per_second`、`..._same_true_speed_...`、`..._repeated_and_out_of_order_frames_are_refused`、`..._symmetric_normal_motion_...` |
| 2 | 延迟队列不冻结最初帧、不删除未来帧 | `..._delay_queue_delivers_by_arrival_time`、`..._delay_queue_keeps_future_frames_and_advances` |
| 3 | BRAKE 实际施力使用正在减速的参考；阶段重置跳变明确 | `..._executed_brake_force_matches_the_rate_limited_reference`、`..._reference_starts_at_zero_...`、`test_local_contact_option.py::test_force_and_reference_acceleration_stay_bounded`（只允许 BRAKE 入口一次跳变） |
| 4 | 每台机器人独立 READY、双 ready 握手；有效接触无多加一周期 | `..._a_robot_can_be_ready_while_its_teammate_is_still_approaching`、`..._push_interval_bounds_and_effective_time_agree` |
| 5 | 接续不是 `len(steps)==3`；终态含两机器人与物体线/角速度 | `run_continuation.py` 的 `normal_completion`；`metrics` 的 `terminal_robot_speeds/omegas` |
| 6 | root 级配对 bootstrap 用真实 root_id，同时报告位置与角度 | `make_metrics.py::paired_bootstrap`（按 root 重采样，位置/角度分开） |

全部通过：`pytest -m "not slow"` 见 `STATUS.md`。

## 本轮新发现的两个共用传感缺陷（协议修复，不计入新控制器功劳）

1. **射线分组用「本次 run 的第一条射线」作距离连续性基准**，使分组长度依赖观测的斜视程度：
   0.18 m 处延伸 0.7 m 的平面被切成 7/3/5 条射线的三段（跨度仅 0.068 m）。改为与**相邻**射线比较后
   同一帧得到 11/10 条射线。
2. **360° 扫描按线性分组**，跨越 0/2π 边界的片段被拆开：机器人正对物体 0.25 m 端面时实际有 14 条
   射线命中，却以「8 条 + 6 条」两段出现。改为环形分组后合并成一段约 0.2 m 的面。

两者都使 X 的 `patch_support_lost` 在第一版实现下于 6 类中的 4 类误触发（0.15 s 即中止）。
修复后 V/X 在 dev 六类上都能跑满 2 s 有效推动。两臂共用同一传感，因此对比仍然公平，
但**本轮 V 的数值与上一轮 N 不完全可比**，这一点已记录在 errata。
