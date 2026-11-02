# D010 M3四项集成修复后的复核

2026-10-30，GPT。重跑16项测试全部通过；补充真实factory/step/Slot的合成观测fixture，未实例化env/backend，新增物理运行0，运行算法未修改。

原件：`outputs/agent_exchange/D010_20261030_m3/gpt_review_integration/`；复核脚本check_integration.py、checks.json及当下源码/测试/manifest快照。

## 已确认修复

- 无候选且旧shift_sign=-1、当前SHIFT时，实际step转wait、sign=0、desired=(0,0)。
- 实际返回日志已有candidate/own/局部与世界接触点。
- 一个dt=.05的接触step只累计.05s，双计已修。
- 观测面偏移.03m，候选id改变、q到新观测面，reason=support_invalid；小偏移保持id并刷新验证时间的测试通过。
- 实际Slot在SHIFT收到BRAKE仍得到switching→brake→brake、最终hold且pending清空。

上述范围可以记为四项集成缺口已修；它不等于接触获取已通过物理验证，也不能绑定旧30/40。

## 必须再修的确定逻辑：ALIGN稳定门槛不可达

当前unknown_shape.py:441的外层条件为`abs(e_t) > .015`，进入后445行才判断`abs(e_t) <= .015`。
同一个e_t同时满足两者不可能，因此align_settle递增分支不可达。
当误差进入阈值，控制反而落到467–470行的gap<=.06快捷分支，立即TOUCH，绕过相对速度与.15s稳定要求。

真实step合成反例（物理可成立的输入，但不是物理实验）：

```
gap=.04m, e_t=0, relative_vt=.10m/s, elapsed=.05s
结果：contact_phase=touch，align_settle=0，normal_desired=.04m/s
```

应按阶段而不是“误差是否超阈值”决定是否检查ALIGN：

```
进入/处于近场ALIGN：
    settled = abs(e_t)<=.015 and abs(relative_vt)<=.03
    align_settle = align_settle+dt if settled else 0
    if align_settle >= .15: TOUCH
    elif alignment_deadline已到且未达标: WAIT/重新获取
    else: 保持安全间隙并切向对齐
```

不允许另一个gap快捷分支绕过该门槛。保留BRAKE/安全抢占、原时长/物理/FF/EKF不变。
一次补齐controller-step真值表：误差在阈值内但速度超限、速度达标但误差超限、两者连续达标3tick、达标中断需清零、未达标超时等待。不能只验证超时不touch而漏掉“误差小就直接touch”。
现有16项未覆盖此核心条件；通过16项不能宣称ALIGN稳定逻辑已验证。

## 交付一致性

manifest/current SHA256为a2925413…，unknown_shape_m3.py快照仍为fb1e3e76…，来自上一版。
保持旧快照/旧复核证据不改，最终修完与测试后用新目录/新版本冻结源码、manifest和测试结果；不要用新hash指向旧源码。
测试名test_surface_moved_two_cm_revalidates_candidate实际输入1.0→1.03m，属于3cm测试；需要边界测试时另测.02±epsilon，不借名称当精度证据。

## 处置

继续原M3离线工作，一次修正ALIGN并完成上述step矩阵、刷新交付版本；无新M4/重试策略。
物理验证仍未运行。历史attempt上界unknown照实保留；若历史账无法恢复，可由用户另行移交一笔独立、从零逐attempt记账的小验证预算，不能虚构旧剩余额度。当前复核不授予新物理预算。
D010仍incomplete；下一批物理只需先验证两个reposition p0和两个dual p0，再决定扩大，没必要立即跑新40/全部260。
