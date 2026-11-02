# FOLLOW 绑定 PUSH 生命周期

2026-10-28。仅修正 FOLLOW 的动作定义、组合合法性和退出时机，未禁止 PUSH+REPOSITION，未启动 RL。

## 当前候选行为

新 FOLLOW 是本轮 PUSH 的随动角色，不再有 2/6/12/24 s 独立时长：

- 仅允许与 PUSH 配对；FOLLOW+WAIT、FOLLOW+REPOSITION、双 FOLLOW 在新协议中不可选。
- PUSH 准备时，FOLLOW 可接近、跟踪物体；PUSH 推进和制动期间继续随动。跟随观测计时只作诊断，不触发正常退出。
- 收到上一控制周期的 PUSH 完成报告后，FOLLOW 制动，返回 `follow_push_complete`。失败的 PUSH 对应 `peer_push_failed`；FOLLOW 自身失败则下一控制周期取消 PUSH，返回 `peer_follow_failed`。
- 仍保留观测丢失、安全距离、总超时和任务终止约束，失败不会为了对齐而被标成成功。本轮没有新增自动接触恢复策略。
- 保留原 2 cm 目标间隙、同伴避让和混合接触点投影，没有修改推动控制或物理参数。

同步指“随同一轮 PUSH 完成后制动”，不是强迫两机同一瞬间静止。PUSH 保留原超时；FOLLOW 的有界窗口在 PUSH 超时上附加 0.05 s 状态通知与 0.8 s 制动余量，仍受原 joint/task 上限约束。额外真实执行照常计时计费，没有免费维护时间。

## 动作编码与信息边界

新增显式菜单 `ACTIONS_PUSH_BOUND_FOLLOW`，版本 `effect-options-v10-push-bound-follow`：**65 项**。前 64 项 PUSH/REPOSITION/WAIT 不变，FOLLOW 只保留一个 `SUPPORT_FOLLOW_PUSH`，`duration=0`、`scale_kind=peer_push`，14×5 因子表中的 FOLLOW 仅槽 0 有效。

配置使用 `follow_completion="peer_push"`，不允许把旧的有时长 FOLLOW 传入后静默忽略秒数；工厂会拒绝菜单/模式不匹配。环境也在物理执行前拒绝无 PUSH 同伴的 FOLLOW。`joint_action_mask` 和 `action_mask_given_peer` 提供采样前约束，其他组合不因本次修改被禁用。

旧 68 项菜单及计时协议保留用于历史回放和对照；当前 RL 适配器未迁移，旧 checkpoint 未覆盖。下一轮 RL 必须显式选择新菜单及联合合法性采样，不能只替换控制器后继续沿用旧策略的概率/动作含义。

开始/结束依靠前一控制周期已报告的执行状态，不在同周期读取另一侧私有扫描/接触估计，也不提前读取未来反馈。现有实验是集中式共享观测执行协议，不将这些状态事件宣称为已计入有限 B-bit 通信的部署方案。

## 真实场景对照

[交互测台](../../outputs/option_testbench_v1/push_bound_follow_v1/round_002/index.html) · [全部指标](../../outputs/option_testbench_v1/push_bound_follow_v1/round_002/metrics.json)

8 场景 × 原独立计时/新绑定 = **16 分支，全部有记录**。分支墙钟 8 s、批次 60 s 上限，最终批次约 24.16 s。PUSH 请求、原超时、感知、物理和奖励保持不变；FOLLOW 的动作编码和受约束退出窗口是本轮明确改变的协议。

| 场景 | 原计时 FOLLOW | 新绑定 FOLLOW |
|---|---|---|
| PUSH[0.3 m]+FOLLOW，主场景 | FOLLOW 7.50 s 退出，PUSH 8.90 s 完成 | PUSH 8.90 s 完成，FOLLOW 9.15 s 制动完成；无提前 hold |
| 交换机器人身份 | 同样提前退出 | 同样在 PUSH 后 0.25 s 结束，未引入执行顺序依赖 |
| PUSH[0.1 m] | PUSH 6.60 s 完成，FOLLOW 7.55 s 才结束 | FOLLOW 6.85 s 结束，不再额外跟满独立时长 |
| PUSH[0.5 m] | FOLLOW 7.50 s 退出，PUSH 11.25 s 才完成 | FOLLOW 11.50 s 结束，无提前 hold |
| 原 FOLLOW[2 s] / [24 s] | 一个过早退出，一个继续到 24 s 窗口上限 | 都映射到同一无时长动作；两条物理轨迹和施力完全一致，FOLLOW 均在 9.15 s 结束 |
| 不可用 PUSH 方向（故意直接请求的负例） | PUSH 立即失败，FOLLOW 仍继续至 7.15 s | PUSH 失败后 FOLLOW 0.05 s 接收状态并结束，原因明确为失败 |
| FOLLOW 看不到物体 | FOLLOW 0.95 s 丢失观测，PUSH 仍继续 | FOLLOW 0.95 s 失败，PUSH 1.00 s 收到失败通知后退出 |

以上时间采用日志的控制周期开始时刻；回放画面显示周期结束后的物理状态，时间相差一个 dt=0.05 s 属于既有记录规则。

主场景 FOLLOW 最终真实物体间隙由原约 **15.5 cm 降为 3.8 cm**；PUSH 本体法向行程仍约 0.312 m，完成原 0.3 m 请求。生命周期同步不保证全程精确 2 cm，FOLLOW 在这些分支中仍未记录到实际物体接触，不声称已产生支撑力。新的跟随质量/支撑控制不在本轮继续调参。

## 验证

- 51 项生命周期/合法性/故障/计时/接口检查与相关回归，73 项联合执行/推后随动/恢复/REPOSITION 混合回归，共 **124 项通过**，未运行全套。
- 16 分支原始位姿与上一周期法向独立复算推动行程；接触点投影及原奖励也独立复核，详见 contract_audit.json 和 validation.json。
- 行程复算最大误差 4.73e-17 m，奖励复算最大误差 2.45e-15；全部分支最大物理穿透约 0.571 mm，未触发异常阈值，不称为严格零穿透。
- 实际浏览器检查 96 个视图，无脚本错误；机器人切换保留帧位置，390 px 下无横向溢出。
- 新行为只有一个无时长 FOLLOW 编码，旧 2 s/24 s 的映射不存在两套不同物理行为。PUSH+REPOSITION 和双 PUSH 的合法性由单测明确保留。
- 状态通知时序由测试检查：停止通知只能在 PUSH 已报告退出的下一周期产生；FOLLOW 丢失观测时 PUSH 同样下一周期退出。制动连续使用原参考速度变化上限，未重置物理速度。
- round_001 是完整小试；round_002 冻结最终末端制动力峰值记录修正后的代码。原始产物保留，不覆盖上一轮结果。

代码：[FOLLOW 控制器](../../src/coimagine/random_start/skills.py)、[联合执行](../../src/coimagine/random_start/env.py)、[动作菜单与掩码](../../src/coimagine/random_start/effect_push.py)、[对照入口](../../experiments/option_testbench/push_bound_follow_eval.py)、[测试](../../tests/test_push_bound_follow.py)。
