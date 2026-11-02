# D010 continuation 报告：状态机修复与接触链验证（incomplete）

- 状态：**incomplete**。M0 预算/attempt 账本、M1 显式 RETREAT/SHIFT 状态机、M2 两条 reposition + 两条 dual 探针已完成；后续 40 task 仍 **30/40**，reposition_0/2 各 **0/5**。40 new、contract、184/260 最终比较、正式 20Hz 全套交付未完成。
- 本报告是 [phase1 报告](D010_Contact_Recovery_Autonomous.md) 的补充；旧 phase1 结果/报告不覆盖。
- 没有训练、没有默认替换、没有 D011。

## M0：预算与记录

- 保守开始：`2026-10-30T12:59:24+08:00`；12h deadline `2026-10-31T00:59:24+08:00`。
- 在本次 continuation 记录时 wall 约 2.2h，剩余约 9.7h（以文件时间和命令为准；不重置预算）。
- 已知物理批次下限：7×40=280 条（phase1 的 7 张表）；continuation 4 条探针 + 2 次启动错误 + 40 task 回归 = 326 条已知 case-run 下限；**逐 attempt 历史不完整，精确总数 unknown**。
- 账本：`outputs/agent_exchange/D010_20261030_continuation/attempts_ledger.jsonl`、`budget.json`。
- 账本把 40 行表记为一个 batch，不冒充逐 case attempt；探针/启动错误单独登记。

## M1：状态机修复

- 新增纯状态函数 `persistent_phase_command`。
- RETREAT/SHIFT 已从 contact 分支提升到每 tick 顶部：
  - 接触丢失不会覆盖 retreat/shift；
  - retreat 定时到后进入 shift；
  - shift 定时到后才回 approach；
  - 计时器只在阶段进入时设置；
  - retry_count 不在每 tick 重设。
- shift 速度上限从 .18 降到 **.08 m/s**；方向使用候选切向误差，不再直接 `−sign(omega)`。
- 新增 `tests/test_d010_contact_state.py`：6 项纯状态测试通过（无物理）。
- 仍未完成：candidate 锁存/支持扩展触发、align 达标而非纯超时、视角更新账本。

## M2：局部真实探针

| case | 结果 | 观察 |
|---|---|---|
| T_reposition_0_p0 | fail (.403) | approach→align→retreat→shift→approach→align→touch→retreat→shift→... |
| T_reposition_2_p0 | fail (.389) | approach→align→touch→retreat→shift→... |
| T_dual_0_p0 | success | 无回归 |
| T_dual_2_p0 | success | 无回归 |

- 两条 reposition 已证明 **retreat/shift 确实执行**，不是“没有进入侧移”。
- 但执行链仍未产生成功；shift 后仍在偏心接触处重新接触。
- 逐 tick 日志：`contact_probe.json`（含 phase/contact/force/gate/action）。

## 40 task 回归

当前 M1 版本重跑 40 task：**30/40**。

| family | 成功/总数 |
|---|---:|
| contact_0/2 | 5/5 each |
| approach_0/2 | 5/5 each |
| dual_0/2 | 5/5 each |
| reposition_0/2 | **0/5 each** |
| total | **30/40** |

- 结果：`eval_after_state_machine.json`。
- 与冻结 B 的 30/40 同分，但版本/源码不同；不能把 30/40 写成 contact-acquisition 通过。

## 未完成/缺口

- 40 new validation、40 contract、最终 260 比较：未跑。
- 20Hz 正式 schema、独立 scorer、pre/post 分离、8 条 replay、6 图、消融：未完整交付。
- 候选跨 tick 锁存、支持不足处理、align 稳定条件、候选评分/有代价探视：仍未实现或未验证。
- 没有最终源码冻结/hash 绑定；当前仍是开发版。
- 没有训练、默认替换、D011。

## 结论

M1 的确定状态机 bug 已修复，M2 证明重试链能真实执行；但这没有使 reposition 成功。下一步应基于 `contact_probe.json` 的首次接触位置/候选变化，做 **单个** contact candidate/观察实验，而不是再改 retreat/shift 或扩大矩阵。D010 仍为 incomplete。
