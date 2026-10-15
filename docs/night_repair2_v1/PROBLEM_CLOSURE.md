# PROBLEM_CLOSURE — 两轮审查问题的闭环表

第一轮审查（`night_repair_optionrl_v1_review.zip`）25 项，第二轮审查
（`night_repair2_v1_review.zip`）9 项。全部**确认**，无一项不成立。
下表给出每项的状态、证据与所在位置。

## A. 第二轮审查（本轮的修复对象）

| # | 问题 | 判定 | 修复 | 独立证据 |
|---|---|---|---|---|
| A1 | 成功奖励 +10 / 失败惩罚 −2 没有进入训练奖励 | **确认** | `rl.py` 改为 `shaping − time_cost + terminal`，terminal 来自环境的 `task_event`，落在技能末周期（γ^{n−1}） | `verification.json:success_bonus_enters_the_reward`、`reward_recomputed_from_recorded_state` |
| A2 | 奖励"手算验证"是循环论证 | **确认** | 新增从**记录的位置、周期数、事件**独立重算的检查；不再把奖励代回自己的公式 | `reward_recomputed_from_recorded_state`（max_error < 1e-9） |
| A3 | 混合采样重复加入随机（ε→(2ε−ε²)） | **确认**（0.1→0.19、0.3→0.51） | 删除第二条 `rng < ε` 分支，只从混合分布采样一次 | `exploration_rate_matches_label`（含 ε=0.1/0.3/0.5） |
| A4 | 观测时钟每个技能回零 | **确认**（166 次回零；32.60 s 存成 2.60 s） | `_inner_observations` 接受累计时间偏移，时钟改为 episode 物理时间 | `clock_is_cumulative_across_skills`（0 次回零，16.50 s 对 16.55 s） |
| A5 | 可恢复检查点保存 `len(history)` 而非迭代号 | **确认**（40 轮存成 9） | 新增 `save_resumable(iteration)`，周期保存 + 真实迭代号 + 内部最佳状态 | `resumable_iteration_is_the_optimisation_step`、`resume_delivered_checkpoint` |
| A6 | 正常训练没有周期保存 | **确认** | 每 `save_every` 轮原子保存 | 同上 |
| A7 | 恢复时不恢复历史最佳模型 | **确认** | `best` 随检查点保存并在恢复时载入 | `resumable_iteration_is_the_optimisation_step`（`best_state_saved`） |
| A8 | 恢复测试走特殊分支，未验证交付检查点 | **确认** | 新检查直接恢复**交付的** `policy_resumable_seed0.pt` | `resume_delivered_checkpoint` |
| A9 | 分支丢弃父前缀、计数器被清零 | **确认** | 保存 `parent_prefix_frames`、`parent_episode_id`、`parent_counters`，分支继续父计数器 | `branch_keeps_prefix_counters_and_marks_intervention` |
| A10 | 强制动作被当成策略概率 | **确认** | `forced_intervention` 标记 + `intervention_note`，ε 记 0 | 同上的 `forced_flag` |
| A11 | 数据卡沿用旧格式（声明 `trace`） | **确认** | `datacard.py` 重写为新 schema，并断言声明字段确实存在 | `data_card_matches_the_files` |
| A12 | 只用两独立比例检验配对设计 | **确认** | 记录逐 root 结果，改为 McNemar 精确检验 + 配对 bootstrap | `train_result_seed0.json` 的 `paired`、`per_root` |
| A13 | 缺更新前策略的表现 | **确认** | 保存初始参数并在同一批 root 上评估 | `train_result_seed0.json` 的 `initial_policy` |
| A14 | 世界模型同口径补齐只有文字、缺代码与结果文件 | **确认** | `train.py` 的修改已提交；结果写入 `world_train_eval.json` | `results/world_train_eval.json` |

## B. 第一轮审查（上一轮已修，本轮保持）

| # | 问题 | 状态 |
|---|---|---|
| B1 | 采样与训练历史不一致 | 已修（比值 1.000000） |
| B2 | 训练未应用动作屏蔽 | 已修 |
| B3 | 分支候选 ≠ 实际执行 | 已修（一致率 1.0） |
| B4 | 分支未从中间快照分叉 | 已修（起点来自真实快照） |
| B5 | 观测字段内部/边界含义不同 | 已修 |
| B6 | 时间轴每技能多一帧 | 已修（`frames = prefix + steps + 1`） |
| B7 | `net_progress` 符号写反 | 已修（两个显式属性） |
| B8 | 截断 bootstrap 用错价值 | 已修（用最终合法历史价值） |
| B9 | 技能内逐周期折扣未聚合 | 已修（手算核对） |
| B10 | 检查点被最终保存覆盖 | 已修（推理/可恢复分离） |
| B11 | 数据规模统计矛盾（140,415） | 已修（本轮如实报告，不再相加） |
| B12 | "任何策略都无法完成 solo_central"证据不足 | 已降级为"已测动作均未成功" |

## C. 仍然未定的问题（如实记录）

| 问题 | 为什么未定 |
|---|---|
| 修正管线在更多迭代下能否学到稳定策略 | 本轮 60 次迭代、480 episode；曲线无上升趋势，但与随机差异不显著 |
| 初始策略本身是否已优于随机 | 本轮记录 `initial_policy`，由 `paired.initial_vs_random_validation` 给出 |
| `solo_central` 是否对**所有**动作序列不可达 | 只测了随机、当前策略与贪心 |
| 技能能力缺口的确切边界 | 已确认"不能反向推、不能绕到另一侧"，未穷举 |
| 4M 容量点与长历史消融 | 不在本轮范围 |
