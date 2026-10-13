# CONTRACT 修订（local_contact_option_repair_v2）

只列相对 [v1 CONTRACT](../local_contact_option_v1/CONTRACT.md) 的**修订**。未列出的条款原样继续有效。
Kv=20、旧控制巡航速度、新控制各档速度、推动目标 2 s、物理参数、传感范围 0.6 m / 72 射线、
原有 root 与扰动全部冻结未改。

## R1 表面速度的单位与时间戳

- `update_surface_estimate(state, patch, pose, sample_time, now)` 返回
  `(rms, signed_mean, matches, unavailable_reason)`；
- 速度 = 匹配点法向位移 / **两次有效扫描的采样时间戳之差**（不是控制周期，也不是消息接收时间）；
- 聚合用**逐匹配点法向速度绝对值的 RMS**，signed mean 仅作诊断；
- 拒绝条件：无片段 `no_patch`、无时间戳 `no_sample_time`、帧龄 > **0.20 s** `stale_frame`、
  重复时间戳 `repeated_timestamp`、乱序 `non_positive_dt`、匹配点 < 3 `insufficient_matches`；
- 只有产生有效估计的扫描才更新缓存；重复帧不能反复递增稳定持续时间；
- 观测新增 `sample_time` 字段（采样时刻，不是到达时刻）；状态缓存保存上一有效扫描、采样位姿与
  采样时间戳；跨 skill 使用单调仿真时间。

## R2 固定延迟队列

`FixedDelayQueue`：每帧带采样与**到达**时间；每周期投递「已在 `now - delay` 之前到达」的**最新**
帧，保留未到达帧，只清理被替代的历史帧；队友消息同一队列、同一处理。启动阶段没有到达帧时明确
`unavailable`：不推进阶段机、按零力步进物理、日志记录 `observation_unavailable_for`；**不回退为
零延迟新观测**。整个观测包一起延迟（自身状态与队友消息同时延迟），不保留任何字段为最新值。

## R3 制动渐变进入执行器

- `brake_force(observation, state, dt)` 先对参考做限幅更新（`a_ref_max` 上限），再用
  `clip_norm(Kv*(v_ref - v_actual), Fmax)` 施力；参考降到零后保持零速反馈；
- 参考**在进入 BRAKE 的当周期**从该机器人实测速度初始化（P 没有等价的自身速度参考，同样如此），
  之后每周期只更新一次（只在施力调用内），不再出现「先按巡航参考施力、下一周期再跳变」；
- 第一周期参考显式初始化为零，然后按 `a_ref_max*dt` 限幅，不存在 `None -> raw` 的绕道；
- 加速度上限是针对**参考**的，不是对实际加速度或力变化的限制；
- BRAKE+SETTLE 的 4 s 上限从进入 BRAKE 起算，两个阶段都生效。

## R4 接触握手与退出一致性

- READY 是**每机器人**状态：单机器人接触连续 0.10 s 后进入自身 READY，参考速度 0.01 m/s 法向
  预接触，队友可以仍在 APPROACH；双方都 READY 且接触连续保持 0.10 s 后同步进入 PUSH；
- READY 失去接触 → 清除 ready，回到接近/接触建立；技能启动起的接近/等待总上限仍为 5 s；
- 退出先记录 `exit_reason`，再协同制动；`push_completed`、`settled_observed`、
  `settled_after_abort` 三个标签分开，中止后判稳只加标志不覆盖原因；
- 有效推动 2 s 只在**该周期确实施力且所需机器人均接触**时累计，进入 PUSH 不预先加一周期；
  日志给出区间起止时间与接触判定来源（`sensed robot_object contact flag`）；
- 不在 `active_robots` 内的机器人执行制动 `F=clip_norm(Kv*(0-v),Fmax)`，不再恒零力；O 保持历史路径。

## R5 验收分母与统计

- G2/G4 主分母 = **solo_central + dual_same_side** 基准变体（40 root）；四可推动类（80 root）作
  次要组另列；旧 92.5% 不搬到新分母；
- G5 同时要求位置 P90 下降 ≥20% **且角度 P90 恶化 ≤10%**（wrap，报 rad 与度）；角度接近零时只报
  绝对差并标注数值敏感，不新增宽松阈值；
- bootstrap 按**真实 root_id** 重采样（同一 root 的所有臂/变体一起保留），每次重采样重算 P90 与
  N−P 差/改善率；断言 root 数等于目标集合大小；均值区间不冒充 P90 区间；
- 观测判稳 / 真值判稳 / 中止后判稳分别统计；覆盖率主分母 = **每臂全部 960 次**，子集覆盖率另标。

## R6 接续与终态完整性

- `attempted`（尝试了几步）、`push_completed`（几步有效推动）、`normal_completion`（三步都正常
  完成且判稳）三者分开；`len(steps)==3` 只表示尝试了三步；
- 入口资格检查两机器人与物体的线/角速度（仅评估，不反馈控制）；报告假判稳后仍继续的案例；
- 每步位移用**该步初态**计算，另报全链累计；终态取退出当刻的独立快照，不依赖最后一条抽样日志。
