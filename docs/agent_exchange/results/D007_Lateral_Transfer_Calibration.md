# D007 执行回报：侧向控制减法与传递校准

- 状态：`complete`（36 条正式 run + 聚焦测试完成）；**候选 C 未通过门槛，结论是 not-supported / inconclusive，不自动 promote**。
- 时间：2026-10-30；实际正式轨迹 36 条，4 worker 未用满（runner 串行等效 1 worker），正式运行 **21.6 s**；聚焦测试 **7.44 s**；总分析回放约 40 s，远低于 900 s 硬上限。
- 任务书：`tasks/D007_Lateral_Transfer_Calibration.md`；配置：`tasks/D007_config.json`。
- 输出：`outputs/agent_exchange/D007_20261030_002/`（旧 `_001` 保留，含首轮缺 effect 记录的失败尝试；不作为最终证据）。
- 当前 server 未重启；用户正在使用的默认 server 未替换。没有训练、没有默认 profile 替换、没有改 reward/物理/成功判据。

## 先回答

**去掉哪些项后，机器人更能跟随目标了吗？** 没有。B（关闭 heading/角速度/横向/rotation target 外环）和 C（B + 去掉额外接触切向阻尼）在这组 adjacent 基准上反而让物体角度明显变差；C 的 `G_robot` 没有稳定高于 B，RMS e_t 也没有降到 B 的 0.9 倍。**C 候选未获支持**。

**是否牺牲推进/稳定性？** adjacent/镜像的 B/C 推进略降、最大角偏显著增大；same_face 系列基本持平。不能把 B/C 当作更好的默认。

**下一处断点在哪里？** 有效扰动只有 **4/8** 个 B/C 共同有效对（门槛 6/8）；robot0 的 target→robot 传递存在（G≈1.36–2.69），robot1 的 target 偏移没有形成 ≥0.02 m/0.5 s 的目标差（support=0），因此侧向传递的另一侧和 same-face/role 几何仍未闭合。**目标不是“换一个 gain”，而是先修 robot1 的目标偏移进入 target 的链路。**

## 1. Preflight 与输入

- 30 个冻结输入的 SHA256 在运行前全部匹配 `D007_config.json`。
- HEAD `22c1333027bbf2a39dea3246bee78638916b4171`；dirty 153（开始前记录）。
- 当前候选源码快照：`outputs/agent_exchange/D007_20261030_002/source_snapshot/`（12 个文件，含 hash）；基线源码快照未在记录字段修改前单独复制，只有 config 的基线 hash 可查。这是本轮协议偏差，已保留说明。
- 为了记录 target/e_t/lateral/q_rot 等字段，`effect_push.py`、`push_controller.py`、`contact_timing_audit.py`、`system.py` 加了**纯记录/诊断**字段；没有改变默认 profile、物理、success、动作频率、接触门限或安全投影。D007 的 profile 通过 runner 的 factory 应用，没有写回默认。

## 2. 实现

- A `legacy`：原配置 + `rotation_feedback_gain=0`。
- B `outer_off`：`feedback=False`；heading/angular_rate/lateral/rotation 全 0；`contact_tangent_damping=0.7`。
- C `minimal`：B + `contact_tangent_damping=0.0`。
- `diagnostic_tangent_offset`：仅 translate 且未 follow_started，在 `point_offset` 的 clamp/rate-limit 之前加入；默认 0；普通 run 不设置；probe 用仿真时钟在 `2.0≤t<3.0 s` 写入指定机器人。
- 记录字段按任务书补入 `effect`：`n/t/heading/lateral_error/omega_est/anchor/side/status/point_offset/role_offset/q_rot/target_before_visibility/target_before_reservation/target_final/e_t/v_t/first_clip/c_v_t/lateral/radial/raw_before_after_avoid/raw_before_after_norm_cap/v_ref/F_pre_post_norm_cap`；`p4` 加 `gate_pre/gate_post`。`CONTROL_DT=.05`，decision_dt=.2。

## 3. 矩阵

- baseline 12：adjacent / adjacent_mirror / same_face / same_face_closer × A/B/C。
- probe 16：adjacent/镜像 × B/C × robot0/1 × ±.04 m，2.0–3.0 s。
- witness 6：W_N/W_E × A/B/C，129 tick。
- non-interference 2：adjacent A/C 关闭记录复跑。
- 共 36 条；run 文件在 `runs/`，`matrix.csv`、`probes.csv`、`metrics.csv`、`analysis.json`。

## 4. 关键结果

### 4.1 Baseline（同一 preset/序列）

| case | profile | progress | max abs angle | angle end | both contact |
|---|---|---:|---:|---:|---:|
| adjacent | A | 0.966 | 0.1106 | −0.1106 | 62 |
| adjacent | B | 0.929 | 0.4047 | +0.4047 | 62 |
| adjacent | C | 0.899 | 0.4602 | +0.4602 | 62 |
| adjacent_mirror | A | 0.966 | 0.1106 | −3.031 | 62 |
| adjacent_mirror | B | 0.929 | 0.4047 | +2.737 | 62 |
| adjacent_mirror | C | 0.899 | 0.4600 | +2.682 | 62 |
| same_face | A | 1.4624 | 0.0148 | −0.0148 | 120 |
| same_face | B | 1.4620 | 0.0067 | −0.0067 | 120 |
| same_face | C | 1.4621 | 0.0098 | +0.0098 | 120 |
| same_face_closer | A | 1.5142 | 0.0417 | +0.0226 | 123 |
| same_face_closer | B | 1.5206 | 0.0487 | +0.0487 | 123 |
| same_face_closer | C | 1.5204 | 0.0685 | +0.0685 | 123 |

B/C 在 adjacent/镜像上把最大角偏从 0.11 rad 放大到 0.40–0.46 rad，推进损失约 0.067 m；C 的 `contact_tangent_damping=0` 没有带来跟踪改善。same_face 系列基本持平。

### 4.2 Probes（±4 cm / 1 s）

有效对只有 8 条单 profile（B 4、C 4），但**B/C 共同有效对 = 4/8**，未达门槛 6：

| case | profile | side | offset | G_robot | G_target | support | contact | valid |
|---|---|---:|---:|---:|---:|---:|---:|---|
| adjacent | B | 0 | −.04 | 1.36 | 1.16 | 18 | 63 | yes |
| adjacent | B | 0 | +.04 | 1.67 | 1.38 | 18 | 63 | yes |
| adjacent | C | 0 | −.04 | 1.46 | 0.82 | 18 | 62 | yes |
| adjacent | C | 0 | +.04 | 2.69 | 2.54 | 18 | 63 | yes |
| adjacent_mirror | B | 0 | ±.04 | −1.36/−1.67 | −1.16/−1.38 | 18 | 63 | yes |
| adjacent_mirror | C | 0 | ±.04 | −1.46/−2.69 | −0.82/−2.54 | 18 | 62/63 | yes |
| adjacent/镜像 | B/C | 1 | ±.04 | ~0.07–0.13 | ~0 | **0** | 62 | **no** |

- robot0 的 target→robot 传递明确存在；镜像方向符号翻转，符合镜像。
- robot1 的 ±4 cm target 偏移没有形成 ≥0.02 m 的 target 差，support=0；**这是当前主要阻断点**，不是 gain 大小。
- B/C 共同有效不足，C 的 G 中位数/误差门槛无法满足：C 的 RMS e_t 中位数约 0.02545，B 约 0.02395（C/B≈1.06，要求 ≤0.90）。

### 4.3 C 门槛

- 共同有效 4/8（要求 ≥6）：**不满足**。
- C vs B 的 G 中位数（signed，镜像抵消）约 0/−0.001，未显示 +0.10 改善；RMS 比 1.06（要求 ≤0.90）：**不满足**。
- adjacent/镜像 baseline 推进损失 0.067 m（阈值 max(.02,.05·|A|) ≈ .048）；最大角偏增加 0.35 rad（阈值 .05）：**不满足**。
- W_N/W_E A/B/C 全部 129 tick success；没有新增异常。
- 结论：**C 候选 not-supported**；B 单独也不比 A 好。不能把 C 的删除阻尼推广。

### 4.4 非干扰与 prefix

- probe 0–2 s 与同 profile 零扰动的位姿最大误差：**0.0**（所有 probe）。
- adjacent A/C 关闭记录复跑：采样 0.2 s 的 post-decision 位姿/机器人误差 **0.0**；注意 runner 的 record=False 只在 decision 边界采样，不是每个 0.05 s tick 的完整记录；任务要求的“所有 post-tick”未逐 tick 采样，这是本轮偏差。
- 记录字段本身没有改变物理轨迹（在采样点上）。

## 5. 测试

- 新增 `tests/test_d007_lateral_calibration.py`：7 项纯数学/配置。
- 聚焦回归：D003 / D004 / D005 / joint-contact-recovery / v13-recovery / same-face + D007 = **67 项通过，7.44 s**。
- 测试中的物理实例未混入 36 条正式轨迹；没有全仓库 pytest。

## 6. 对假设的回答

- **“外环修正互相干扰”**：证据不足。B 去掉外环后 adjacent 角偏更大，说明至少在这组几何外环不是单纯干扰；但不能证明它在所有几何必要。
- **“额外 c=.7 阻尼减慢跟踪”**：**反对/不支持的候选**。C 去掉阻尼后 RMS e_t 更差、G 没有稳定提高、baseline 角偏更大。
- **“target 动了但机器人没动”**：robot0 有传递（G≈1.4–2.7）；robot1 没有 target 支撑（support=0），所以“没动”主要是目标偏移未进入有效 target，而不是简单的执行层阻尼。
- **“符号翻转就能解决”**：不成立；本轮没有把 rotation gain 当默认，也没有用 angle_end 选符号。

## 7. 偏差/限制

- 为了补记录，修改了 `effect_push.py`/`push_controller.py` 等；这些是纯记录/诊断，但**不是零改动**；默认 profile 未变。
- `record=False` 的非干扰复跑只采到 0.2 s decision 边界，不是每个 0.05 s tick；报告不能声称逐 tick 非干扰。
- 有效共同扰动 4/8，未达 6；C 结论是 inconclusive/not-supported，不是统计泛化。
- 没有训练、没有 D008、没有默认替换；D002 B 仍冻结；完整 D005 8-case 当前版本验收仍未补齐。
- 图表：`chains/013_probe.png`、`014`、`017`、`018`、`021`、`022`（target/robot x、e_t/lateral、object angle）。

## 8. 交回 GPT

建议下一步（不自动执行）：

1. 先查 robot1 的 `diagnostic_tangent_offset` 为什么没有进入 `target_before_reservation/target_final`（support=0）：是 same-face/role/anchor 投影，还是 `offset` 加在错误侧。
2. 用同一 probe 结构只改 robot1 的 target 生成链路，确认 target→robot 的 G 是否恢复；不要先加新 gain。
3. C 候选暂不 promote；B 也不优于 A。若要继续减法，应先解决有效样本/robot1 目标传递，再重跑同门槛。
4. 完整 D005 8-case 当前版本验收仍需另一轮；本任务未冒充。

## 9. 产物

- `outputs/agent_exchange/D007_20261030_002/`：`manifest.json`、`summary.json`、`analysis.json`、`matrix.csv`、`probes.csv`、`metrics.csv`、`runs/`（36）、`chains/`（6）、`source_snapshot/`（12 文件+hash）。
- 旧 `D007_20261030_001/` 保留为首轮记录缺口尝试，不作为最终证据。
