# D011 option 级验收标准与首批分类

- 状态：`evaluation_proposal`；不改 controller，不改物理/成功判据。此文档只回答“option 本身是否按预期执行”，不把 goal success 当唯一标准。
- 当前 40 task 的 env success 是 22/40；这个数字不能直接等同于 22 个 option 真失败。

## 1. 评价对象

每个 case 只看 option 的行为链：

1. **stage 链**：是否按 `standoff/final/touch/push`（或当前 legacy 的 approach/push）转移；是否跳阶段、停在 standoff/approach、反复回退。
2. **真实接触**：raw `contacting_object` 是否发生、持续时间、是否保持；不是 `contact_ready`。
3. **push 兑现率**：接触期间物体沿请求方向的位移 / 理论最大位移（`push speed × contact time`）；不要求到达 goal。
4. **稳定性**：物体相对初角峰值、机器人 heading 变化/回摆、接触点/法向是否持续大幅跳变；不把目标外的小幅 heading 修正当失败。
5. **安全/异常**：碰撞、出界、NaN、卡死、无接触却持续 push。

## 2. 判定

- **option_effect_ok / task_incomplete**：stage 链成立、真实接触、有正推进、无持续振荡/异常；只是没到 goal 或固定预算结束。
- **true_fail**：没有真实接触、stage 链断裂、接触极短且无有效推进、持续振荡/卡死、安全异常。
- **incomplete**：记录不足，不能判。

默认阈值（工程初稿，可调）：
- 接触 ≥0.5s；
- 沿请求方向位移 >0.05m；
- 接触期间兑现率 >0.3；
- 物体相对初角峰值 ≤0.35rad；
- stage 不得在 `standoff/final/touch/push` 间持续反向跳变。

## 3. 两个回归 preset 的分类

### `D011 regression contact_2_p0`

- 帧数 128；**真实接触 40 tick = 2.0s**；
- 物体南向推进 **0.2397m**；
- 物体相对初角峰值 **0.0200rad**；
- phase：`standoff → final → touch`；接触保持；无阶段回退；
- push 兑现率约 `0.2397 / (0.16 × 2.0) ≈ 0.75`；
- goal 距离 .21，未成功，但这是 **16 个 push decision 的任务预算不足**。

**判定：option_effect_ok / task_incomplete，不是真失败。** 初始 heading 小波动不是核心。

### `D011 regression dual_2_p0`

- 帧数 172；真实接触只有 **4 tick = 0.2s**；
- 物体南向推进 **0.0131m**；
- 全程 phase 在 `approach` 与空值间反复，**没有进入有效双推**；
- 两机器人没有形成稳定 contact/push 链。

**判定：true_fail。**


## 双机接触分配实现后（D011 regression dual_2_p0）

- 新增同面支持 union + 切向分配：
  - 两机 q 从 2.375/2.375 变为 **2.71 / 2.29（间距 .42m）**；
  - 两机都在 tick 80 真实 contact；
  - 物体推进 **0.249m**，相对初角峰值≈0；
  - 最终 goal 距离 **0.201m**，只差 1mm 未过。
- 按 option 标准：**option_effect_ok / task_incomplete**，不再是“双推没有形成”的 true fail。
- `dual_0_p0` 跑出类似：两侧 contact 40 tick，物体沿东向推进约 .249m；任务距离未过。
- 剩余 true fail 仍是 `reposition_0_p2`（接触 1.5s、推进≈0）和 `reposition_2_p1`（接触 1.5s、方向/效果错误）。
- 尚未重跑完整 40；不能把 dual 单例结果写成总体 40 通过。


## dual 停顿修复

- 现象：dual_0_p0 约 2s 接触后停约 1s，`stage=final`，`desired` 被压到接近 0。
- 原因：`final→touch` 阈值 `dgoal≤.015m` 太紧；机器人停在目标落位几十毫米外，final 速度再被避障压低，无法进入 touch/contact。
- 修复：阈值放大到 `.035m`；final 最小接近速度提高到 `.10m/s`。
- 结果：
  - dual_0_p0 / dual_2_p0 均 success，129 ticks；tick32 开始持续接触，无 1s 停顿。
- 当前 40 task 回到 **30/40**；剩余失败：contact p1/p2（4）、dual p1/p2（4）、reposition p2/p1（2）。


## remaining reposition 复核（不是 true fail）

- `remaining reposition_2_p1`（run `20261102_163711_f1a720`）：
  - 真实接触 39 tick=1.95s；推进 0.238m；最大相对角 .0184rad；
  - 终点 (2.519,1.762)，goal (2.5,1.55)，距离 .212m；goal 半径 .2，只差 1.2cm；
  - **option_effect_ok / task_incomplete**，不是 option true fail。
- `remaining reposition_0_p2`（run `20261102_163914_b53076`）：
  - 接触 39 tick=1.95s；终点 (2.262,2.019)，goal (2.05,2.0)，距离 .212m；
  - **option_effect_ok / task_incomplete**。
- testbed 顶部 `no success` 仍是 env 的 task success，不是新 option verdict。当前 remaining 列表实际是“task 未进 goal 容差”，不等于 option 行为失败。

## 4. 对当前 40 task 的初步含义

- `dual_0/2` 全部 10 个：按新标准是 true fail（没有有效双推）。
- `contact_0/2` 的 p0–p2 共 6 个：从 contact_2_p0 的数据看是 option-effect ok、任务预算不足；其余 case 需同样逐 tick 核验。
- `reposition_0_p2`、`reposition_2_p1` 仍以 goal 失败记；是否 true fail 要看它们的真实 contact/push 阶段，不应沿用旧 success 表直接判。
- `approach_0/2` 全过，不是失败。

结论：当前 22/40 不能写成“22 个 option 真失败”；首批已确认 **dual 是真正失败**，contact 类更像 horizon 太短。下一步应按这个标准给剩余失败 case 逐条打标，再决定是否给 contact 增加预算或条件化 standoff。

## inside-band 修复

- safe 只作为带外目标；`dgoal≤R+band+.02` 时直接 final/touch，不再先退回 safe。
- dual_0_p1 success(129)，pre 方向转为 +x；当前 40 task 34/40；剩余 contact p1/p2（4）与 reposition p2/p1（2）。

## 当前 6 个 task 失败的 option 复核（全部 task-incomplete）

| case | 接触 | 推进 | 最大相对角 | option 判定 |
|---|---:|---:|---:|---|
| contact_0_p1 | 1.65s | ~0.204m | .074 | task-incomplete |
| contact_0_p2 | 1.65s | ~0.204m | .074 | task-incomplete |
| contact_2_p1 | 1.65s | ~0.204m | .074 | task-incomplete |
| contact_2_p2 | 1.65s | ~0.204m | .074 | task-incomplete |
| reposition_0_p2 | 1.95s | ~0.238m | .018 | task-incomplete |
| reposition_2_p1 | 1.95s | ~0.238m | .018 | task-incomplete |

结论：当前 40 task 的 6 个失败全部有真实接触、正推进、角度在阈值内；按新 option 标准 **没有 true-fail**。按用户要求，testbed 已移除这 6 个 preset，只保留 seed 基础场景。

## 全量 40 option 审计（heading 解绑后）

- task success：36/40。
- option passed：38/40。
- **true fail：2/40**，且都是 task success 但 option 失败：
  - `T_dual_0_p2`：物体最大相对角 **0.7212rad**，contact 3.1s；task success；
  - `T_dual_2_p2`：同样 0.7212rad。
- task incomplete（option 正常）：contact_0/2 p1、p2 四个；接触 1.65s、推进 .204m、角度 .074。
- testbed 当前只保留两个 true-fail preset：
  - `truefail dual_0_p2` id `20261102_164544_8e1e39`
  - `truefail dual_2_p2` id `20261102_164544_0e88e6`
- 审计脚本 `experiments/option_rl_v19/audit_options.py` 只跑 40 个 T 验收 case；结果 `outputs/agent_exchange/D011_20261030/option_audit_40.json`。

## peer gate + push label 修正后（最终当前 40）

- `truefail` 从 2 降为 **0**；task success 36/40。
- dual_0_p2 / dual_2_p2：最大相对角 **0.0902rad**（原 .7212），task success；不再 true-fail。
- R0 在 tick50 后 `contact_phase=push`，两侧均真实推进；不再出现“单机 push、另一机 wait”却已施加推力的标签错位。
- `final` 慢的原因不是切向：R0 的 `e_t≈.002`，raw 主要沿 support 法向；延迟来自 final 最小速度 .10 和等待 peer contact，不是切向反馈不足。
- testbed 已删除两个 truefail preset；当前只剩 seed 场景。

## N+E 双侧协推：peer gate 对齐修正

- 现象：R1 raw contact 后直接 `wait_peer_contact`，但自身 heading 尚未对齐；R0 随后单独 push。
- 修正：
  - 自身 heading 未对齐（>0.18rad）先进入 `orient`，不允许进入 wait/push；
  - pair latch 改为“双方 contact 且双方 heading 都已对齐”才释放。
- 验证（N+E preset `20261102_172943_1d04c5`，新 run `20261102_173249_5e0648`）：
  - tick29 R1 `orient`，R0 raw contact 但进入 `wait_peer_contact`；
  - tick35–45 R0 一直 wait，R1 从 85.7° 收到 17.9°；
  - tick50 两机都 push（heading error -0.5° / 7.8°），不存在单机先 push。
- 该场景 task 仍失败、后续最大相对角 .449rad；这是对齐 gate 之后的旋转/接触问题，不是原来的“未对齐就 wait”问题。40 case 尚未在该 gate 后重跑。

## N+E：共享 twist 驱动目标 q

- 共享 `SharedMotionEstimator` 本来已融合两侧观测；缺失的是 q 在 contact/wait/push 后不再更新。
- 新增 q 更新：`q += v_at(q)·dt`，其中 `v_at(q)=v+ω×r`，v/ω 来自双方观测共享的 EKF。
- N+E run `20261102_173704_0db840`：
  - tick35–50 R1 在 orient/wait，q 已从 2.4957 跟到 2.5073；
  - tick55 双方 push，R1 q≈2.512；
  - 最大相对角从 .449 降到 .285。
- 该场景仍 task success=False；当前属于 task-incomplete/继续观察，不是原来的 q 冻结问题。

## N+E 切向跟踪负结果（不要保留）

- 当前 best（q 随共享 twist 更新、u_adjust=1.5×err，±.08、FF=0）：max angle .285，contact 162 tick，task false。
- 侧向增益 4.0×err，±.20：max angle .667，contact 丢失，task false；出现过冲/旋转。
- paired 打开表面速度 FF + 1.5×err：max angle .676，contact 丢失，task false。
- 两次试验均已回退；当前保留 best 版本。
- 结论：不是单纯切向增益不足；裸位置误差高增益和直接 surface FF 都会激励旋转。下一步应做速度级切向跟踪（q 处 material velocity + 有界位置修正，重新核符号/参考点），而不是继续加大 K。

## 2D 位置 P 负结果（N+E）

- 按用户建议实现：`ref=q+outward*(R-.05)`，`raw=unit(ref-own)*speed`，去除 u_adjust/FF/切向分解。
- N+E run：max angle **.649**、task false；比此前 best `.285` 更差。
- 结论：两面同时接触时，简单归一化位置 P 仍会通过接触力注入旋转；这不是速度归一化或增益能解决的。该实验已回退，默认保留 best（q 跟踪 + peer gate + u=1.5×err/±.08）。
- 下一步需要 pair-level 力分配/接触速度协调，而不是单机位置 P。

## q'-offset P 探索版 testbed 典型失败

- 代码切到 q'=q+t*offset（offset 由 support 法向相对初始法向的角度误差积分），ref_C=q'+outward*(R-delta)，超过 5cm 时去掉 delta，再做归一化 2D P。
- testbed 只保留 5 个探索案例：
  - 单机东推·接触短推进 `20261102_175825_b3e03d`：contact 26 tick，推进 .0529，角度 .0145；
  - 绕行后南推·弱推进 `20261102_175825_aaa34c`；
  - 双机北推·低推进 `20261102_175825_17fdbc`；
  - 接近阶段东推·弱推进 `20261102_175825_f64481`；
  - N+E 双侧协推 `20261102_172943_1d04c5`：maxangle .155。
- 该版全量 task success=0、option true-fail=0；明确是探索版，不作为默认验收。

## P 探索版 K=6 结果（offset rate .02）

- contact P：`v=6.0*e`，速度上限 .30；offset rate 保持 .02。
- 弱推进 `单机东推·接触短推进`：contact 18 tick（原 26），final x=2.568（原 2.553），max angle .0117；task false。
- N+E：contact 128 tick，max angle **.0732**（原 .155）；task false。
- 说明增大法向 P 对 N+E 旋转有明显改善，但弱推进仍推不远；40 case 未重跑。

## 纯距离接触（patch_gap≤.02）失败记录

- 控制链路已移除真实 Pymunk contact 与 contact_point：`contact_ready = patch_gap≤.02`，estimator 只用拟合线，cp 不再来自真实接触。
- 弱推进 `20261102_181302_b74460`：est_contact=0，物体停在 (2.502,2.000)，没有推进。
- N+E `20261102_181303_80e5f3`：est_contact=0，物体停在 (2.501,2.001)，没有推进。
- 这是一个合法但失败的版本：说明当前 approach/P 没有把机器人带到 gap≤.02，接触从未触发；下一步必须修 approach/参考位置，而不是恢复真实 contact。

## 从 q 到力的链路与瓶颈

- q：可见 support 的接触点。
- q'：q + t·offset；offset 更新 .02×法向角误差，clamp ±.06。
- ref_C：q' + outward·(R−.05)，仅当 **surface gap=‖q′−own‖−R ≤.05**；否则 q′+outward·R。
- v_des：`6.0*(ref_C−own)`，cap .30。
- 然后 static avoidance、reference 限速、force。

真实数据（弱推进）：
- 修复前：tick15 gap=.0039，ref 实际用了 q+R，desired≈.023，推力弱，物体 x≈2.547。
- 修复后：tick15 gap=.0034，desired≈.299，contact 74 tick，最终 x=3.368，maxangle 0；goal 2.95，因此变成 **overshoot**。
- N+E：contact 188，maxangle .271。
- 结论：问题不是 K 小，而是 ref_C 的 .05 侵入判断写错；修后 P 足够强，弱推进转为过冲，下一步应做 terminal stop/brake，而不是继续加 gain。
