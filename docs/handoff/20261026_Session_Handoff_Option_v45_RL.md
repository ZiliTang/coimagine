# 会话交接：Option v4.5 + RL 训练（2026-10-26）

本文件总结本次会话（从 option 测台到 v4.5 + RL 训练器）的结论、证据与未决项。配套文档：
`20261026_Option_v45_Handoff_and_RL_Training.md`（v4.5 参数/命令速查）。

## 1. 版本演进（一句话）

v4.1（760 维、52/2704）→ v4.2（**PUSH 只保留法向力**，删 ±15°，7 类型 × 4 = **28/784**）→ v4.3（FOLLOW 跟踪修复）
→ v4.4（接触连续性：推进记录修复、接触 seek、换面不中止）→ v4.5（接触点瞄准、队友避让、双推可行性过滤）。

## 2. 已验收的机制与证据

| 机制 | 证据 |
|---|---|
| **推进记录 bug 修复** | `PushSkill.step` 原先不调 `_note_position`，`signed_progress` 恒 0、push 永远“未完成”；修复后 `push_complete` 0→35+ |
| **接触 seek** | `CONTACT_LOSS_TOLERANCE` 0.6→2.5 s + 接触保持参考；`contact_lost` 33→2 |
| **换面不中止** | 法向跳变只记录并沿用新法向；`face_switch` 退出仅保留在 REPOSITION |
| **接触点瞄准** | approach 瞄准可见面中心；同面双推保持原对称位置；`safety_abort` 8→1 |
| **FOLLOW 验收** | 质心参考 + 增益 4.0 + 法向速度前馈 + 超时=准备+维持；`teammate` 场景位姿 0.051、容差内 0.80 → fulfilled |
| **双推可行性过滤** | 可见法向相反（dot<−0.5）的“对侧对夹”标为不可接受（`opposite_faces_both_push`） |
| **协同价值=稳定性** | 同侧对称双推：0.0° 转角、0 横向漂移；单推：1.3°、~0.97 m 漂移；相邻侧双推转 23–109°（非稳定） |

## 3. 当前实测（round_034，68 分支 / 120 option）

- 未过滤 option 级 **59/120 = 49.2%**；已接受请求 **57/92 = 62.0%**；branch 条件兑现 23/54 = 42.6%。
- 已知漏斗：14 条不可行对侧双推（已剔除）+ 12 个 FOLLOW 失败（跟随方站在推动侧/长时漂移，需影子侧迁移）。

## 4. RL 训练（本次新增）

- 训练器：`experiments/option_rl_v45/train.py`（760 维合法观测、28×28 因子化头、semi-Markov return、
  观测归一化 + 动作 mask + 每评估 checkpoint + 自动 dashboard）。
- 监控：`experiments/option_rl_v45/monitor.py` → 自包含 HTML 曲线（return/success/progress/loss/entropy/
  奖励分量/成本），训练器每次评估重写 `dashboard.html`。
- **run1**（4.5 h / 515 iter）：**无效**——外部 timeout 与训练预算同时到点，未写 checkpoint；val_success 全程 0。
- **run2**（修复后，~1.2 h / 120 iter，已主动停止）：val_success 0、val mean_progress 0.0、entropy 6.65→4.0；
  贪心动作分布 **WAIT_BRAKE 20 / REPOSITION_CW 12 / PUSH 0** → 策略坍缩到 do-nothing。
- 结论：当前不是“option 推不动”，而是**探索/信用分配（或奖励中距离/时间代价压过 progress）**让“推”不可学。

## 5. 未决项（需要用户决定）

1. **奖励设计**：`distance_cost=0.02/m·s`、`time_cost=0.002/period` 是否压过 `progress_weight=1.0/m`？
   若确认是，需授权调整奖励项（不得为凑正结果偷偷改）。
2. **FOLLOW 影子侧迁移**：12 个失败的主因，属 FOLLOW 机制（超出“PUSH 范围”）。
3. **60% 口径**：已接受 62.0% 达标；未过滤 72/120 需 FOLLOW 影子侧或调整期望集。
4. **RL 预算**：run1 无效、run2 停止，下一次算第一次有效长训练；最多两轮、总限 12 小时。

## 6. 常用命令

```bash
# 测台 / 指标 / 渲染
.venv/bin/python experiments/option_testbench/demo_showcase.py
.venv/bin/python experiments/option_testbench/cli.py run --out <dir> --branch-wall-seconds 10 --total-wall-seconds 80
.venv/bin/python experiments/option_testbench/metrics.py --out <dir> --suffix _v2
.venv/bin/python experiments/option_testbench/cli.py render --out <dir>
# 验收
.venv/bin/python experiments/option_testbench/follow_validation.py
.venv/bin/python experiments/option_testbench/anchor_boundary_eval.py
.venv/bin/python experiments/option_testbench/cooperation_eval.py
# 训练 + 监控
.venv/bin/python experiments/option_rl_v45/train.py --hours 4.0 --episodes 8 --eval-roots 16 --out outputs/option_rl_v45/run3
.venv/bin/python experiments/option_rl_v45/monitor.py outputs/option_rl_v45/run3
# 测试
.venv/bin/python -m pytest tests/test_option_testbench.py tests/test_random_start.py tests/test_random_start_anchor.py -q
```

## 7. 下一步建议（按顺序）

1. 跑“全 WAIT / 随机 PUSH / 朝目标推启发式”三条固定策略的 return 对照，确认推是否有正收益路径；
2. 若推有正收益：修探索（熵系数、mask 收窄、PUSH 优先采样），跑第一次有效长训练；
3. 若推被奖励淹没：先与你确认奖励设计，再训练；
4. 训练可用后，再按需做 FOLLOW 影子侧与相邻侧稳定构型。
