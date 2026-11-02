# D010 M3：接触候选闭环实现（离线阶段）

- 状态：**M3 离线实现完成，物理验证未跑**；D010 仍 **incomplete**。
- 依据：[M3 合同](../tasks/D010_M3_Candidate_Contract.md)、[continuation 复核](../reviews/D010_Continuation_Review.md)。
- 新增物理运行 **0**：精确 attempt 上界仍 unknown，按合同先完成离线实现/日志字段/纯测试，不用 326 下限计算剩余量。
- 复核后补齐：缺候选禁止 SHIFT、info/effect 候选/own/contact_point、接触时长单计、候选按当前线/支持再验证、ALIGN 位置/速度双条件持续达标/超时等待、真实 controller step fixture。
- 没有训练、没有默认替换、没有 D011。

## 实现

- 新增 `ContactCandidate`：
  - `id/q/n/t/selected_time/verified_time/support/margin/valid/updates/reason/view_ticks`；
  - q 是接触空间候选，不称物体中心。
- 新增 `choose_candidate_q(lo,hi,m=.06,min_span=.02)`：
  - 连续可见支持足够时 q 内缩 .06 m；
  - 支持短于 .02 m：`view_insufficient`，不伪造端点；
  - 中间情况记 `margin_limited`，不假装完整边距。
- 候选锁存于 `UnknownShapePush.candidate`：
  - 首帧建立；
  - 原候选 q 仍落在新支持内且支持未稳定扩展时保持；
  - 支持失效/稳定扩展（连续 3 tick、跨度增加 >.05 m）才显式换候选，记录旧/新 q；
  - 不再每 tick 追 `points.mean`。
- `candidate_err_t` 每 tick 从锁存 q 计算；缺失时为 `None`，不再默认 1.0。
- shift 方向由 `shift_sign_for_error(candidate_err_t)` 决定；`None → 0`，不侧移。
- RETREAT/SHIFT 仍在接触丢失后持续；shift 上限 .08 m/s、计时器进入时设置。
- 提前返回前刷新 `obs.contacting_object/own/candidate_err_t`；raw contact 不再靠 `contact_ready` 陈旧值。
- `_info_unknown` 增加 candidate/own/own_v/contact_point_robot/contact_point_world/phase/timers 等字段，供后续 pre/control/post 日志使用。
- 缺候选时 RETREAT/SHIFT 转 wait，shift_sign 清零，不沿旧方向侧移。
- 接触时长只在顶部 raw contact 刷新时累计一次。
- 候选再验证：当前线偏移 >.02m 或 q 离开支持 ±.02m 时显式换候选；小幅支持变化保持 id 并刷新 verified_time。
- ALIGN 统一按非双机 gap<.08 检查：|e_t|≤.015m 且相对切向速度≤.03m/s 持续 .15s 才 touch；位置或速度不达标保持 align 并计时；.60s 超时转 wait，不强行 touch。此前“外层 >.015、内层 <=.015”的逻辑错误已修。

## 测试

- `tests/test_d010_contact_state.py` 6 项通过；
- `tests/test_d010_candidate.py` 6 项通过；
- `tests/test_d010_controller_step.py` 9 项通过：缺候选不得 SHIFT、contact 只计一次、表面移动 2cm 换候选、小幅支持变化保持 id/刷新 verified_time、ALIGN 位置/速度单条件不达标、连续达标、达标中断、超时 wait。
- 共 21 项通过。仍未做 BRAKE 抢占/真实 Slot 全链路/真实 contact 时序的 fixture。

## 预算与停止原因

- 已知 case-run 下限 326；精确 attempt 数 unknown；原 deadline 未重置。
- 因次数上界无法核实，本轮没有新增物理运行；这是合同要求的停止条件，不是算法结论。
- 40 task/new/contract/260 比较、20Hz 正式 schema/scorer/replay：未跑。

## 下一步

先恢复可核实的 attempt 上界/剩余预算，再按 M3 验证顺序跑：
1. 两条 reposition p0 + 两条 dual p0，记录首触点、实际 shift 距离、新候选 q、重新接近路径；
2. 局部候选行为正确后，才扩 40；
3. 仍失败则按原授权做候选评分/有代价探视，不改 EKF/FF/物理/脚本。

当前不能写 D010 accepted；候选接线完成也不等于 contact-acquisition 通过。
