# STATUS — data_upgrade_v1（基线修复与成功轨迹分支数据升级）

任务书 `docs/tasks/20261012_2137_CoImagine_基线修复与成功轨迹分支数据升级任务书.md`。
run_id `data-upgrade-20261012`；Grid 审计 `grid-audit-v2`（新目录）。契约
[CONTRACT.md](CONTRACT.md)，Grid 修复明细 [A1_A2_grid_fixes.md](A1_A2_grid_fixes.md)。
本文每个数字都来自列出的命令；未运行的写 `NOT_RUN`。

## 环境与不可变项

- 起始 git HEAD `627d5b861bc55152cef40489d7351ec8777dd53b`（branch `main`）；本轮新增文件均为
  新文件，旧目录 `outputs/forecast_baseline_v1/` 与 `outputs/deterministic_grid_v1/` 只读引用、
  未覆盖。三场景（S0_dev/S0_v2/S0_v4）仍为 `closeout_v2`，物理参数、控制器、2 s 窗口、
  成功/失败判定均未改动。
- Python 3.11.4，numpy 2.4.6，torch 2.14.1+cpu，pymunk 6.9.0，32 核。
  评估与训练脚本显式 `torch.set_num_threads(1)`（单样本推理下默认线程数慢约 200 倍，
  且固定归约顺序便于复现）。
- Grid 三个 checkpoint 与 `samples.jsonl`/`splits.jsonl` 的 SHA256 在本轮 A 运行前后一致
  （`outputs/data_upgrade_v1/grid-audit-v2/hash_check.json: unchanged=true`）。

## 阶段 A — 已确认问题的修复

### A1 Grid 聚合

`deterministic_grid/audit.py::aggregate_by_state` 的 `model_seed="mean"` 行原来把精确模型
（`model_seed=-1`）当成一个学习种子一起平均。修复后只平均种子 0/1/2，精确行单列。
算术检查（`A1_A2_grid_fixes.md` §1.2）：精确行的偏差类指标恒为 0，因此“含 exact 的均值”
恰好等于学习均值 × 3/4；`chosen_true` 的精确行不是 0，不满足 3/4 关系，如实标注。
修复后（先状态内平均候选种子、再跨状态平均）：

| 验证集 | selected_bias | candidate_mean_bias | selection_excess | regret | chosen_true |
|---|---:|---:|---:|---:|---:|
| 新（无坐标偏置） | +0.867 | −1.719 | +2.586 | 1.075 | −5.650 |
| 旧 20 状态 | +1.250 | −1.450 | +2.700 | 0.664 | −5.214 |

旧表头条 `1.250 / −1.450 / 2.700` 与任务书给出的定位参考一致。**这不是“两种聚合口径”的
差异，而是精确行被错误混入**。

### A2 Grid 单候选反事实

`build_traced_case` 原来对**全部候选**各修一次再重排，却按“只修选中候选”描述。新主分析：
缓存全部 `J_hat`，只对原选中候选 c 的首次指定错误修一次、只替换 `J_hat[c]`（改动条目数=1），
其余逐项不变，再重排，并另报新选择的**真实**收益是否改善。`false_success` 干预改名
“首次虚假成功处的完整状态干预”。旧的“全候选干预”作为历史附件单列并更正因果表述。
结果：单候选 `full_state`/`object_anchor` 干预在 3/3 高 regret 案例改变排序，但真实收益
改善只有 1/3 —— “选择改变”不等于“修复有收益”。

### A3 连续终止与可行性协议

`continuous2d_learning/{closed_loop,rollout,contract}.py`：

- 闭环只执行 `exit_class ∈ {window_complete, success}`；`invalid_prediction`、`non_finite`、
  `physical_failure` 一律不执行；无可行序列 → `planner_no_feasible_plan`，`live_env` 不再前进一步，
  并记录停止时刻、已用时间、末端距离。
- 合法性优先：预测同时声称 success 且状态非法时判 invalid（新增测试覆盖）。
- 初态已成功/已失败直接吸收，不调用模型（新增测试覆盖）。
- 几何检查的 **y 越界改为按 arena height 判定**（原来一律用 width），与冻结环境
  `PymunkBackend.in_bounds` 的语义一致。
- 离线固定候选诊断保留全部候选的分项与 J，另加 `aggregate_feasible` / `metrics_table_feasible.md`
  两张独立表；`conditional_regret` 在无可行候选时写 `None`(NA)，不填 0、不删起点。
- 测试：`tests/test_data_upgrade.py` 12 项（全非法、混合、非法+success 冲突、全合法、
  终止吸收、y 越界、可行性 NA），全部通过。

**协议变化的实际影响**（旧 checkpoints，S0_v4/S1_v4，H=1/3，54 回合）：旧实现把预测为非法/
失败的序列照样执行——重算显示旧结果中有 **59 次非法序列执行、涉及 22 个回合**；修复后
36 个学习回合里有 **22 个**以 `planner_no_feasible_plan` 结束，正好是同一批回合。
精确 H=1 3/6、H=3 6/6 与 uniform 0/6 不变（本轮仍全部重跑，未复用）。

### A4 报告与旧测试

- Grid 递归累计终止率由原始 JSON 重算：A4 协议下新验证集 H=4 为 **0.137**、
  teacher 0.164；旧 20 状态为 0.104 / 0.128。上一轮 STATUS 的 0.918 来自连续平台的旧
  fallback 递归诊断（`forecast-20261012/diagnostic/recursive.json`，非法时**保留上一合法状态**，
  `model_seed=0/1/2` 为 0.917/0.941/0.931），不是 Grid A4 协议下的数字。已在
  `docs/forecast_baseline_v1/STATUS.md` 更正并注明来源。
- 连续状态弃用“完全相等错误率”，改用固定容差：任一体位置误差 >0.05 m、物体角度误差 >0.10 rad、
  错误终止或非法预测；三项另列独立发生率，teacher 与 recursive 各自独立累计，
  正确成功吸收不计错误（`chain_tolerance_metrics`，测试覆盖）。
- 过期测试更正：正向“简单自由空间可解”改用 **S0_v1**（depth1 成功、2 窗口）；
  新增 `test_fixed_s0_dev_is_legal_and_keeps_the_frozen_protocol` 验证 S0_dev 初态合法性与协议
  （重置/精确前缀恢复/窗口终止词表）。未改场景、未删正向检查。

## 阶段 B — 几何、拟合与旧基线验收

### B1 几何检查验收：`GEOMETRY_OK`

- 原数据 2373 个真实窗口末状态：**0 个被判非法**（`window_complete` 2369、`success` 4），
  逐条误判记录文件为空。
- 与引擎逐时刻对比（4800 个完整 step）：最大绝对差 0.0056 m、均值 0.00012 m，
  `systematic_disagreement=false`。
- 差异来源已用受控实验定位（`geometry/convention_probe.json`）：冻结环境的
  `penetration = max(0, -contact.distance)` 对 **circle-vs-segment**（机器人-边界）和
  **polygon-vs-polygon**（物体-障碍）成立，但对 **polygon-vs-segment**（物体-场地边界）
  Chipmunk 给出**正** distance，于是引擎自身的 bookkeeping 看不见物体压进场地边界；
  独立检查器按物理重叠计算，因此在该对上更严格（观测到的差 ≤5.6 mm，阈值 50 mm）。
  这是被记录的约定差异，不是系统性不一致，故不判 `GEOMETRY_BLOCKED`。
- 旧模型预测状态（teacher-forced，2373 窗口）：穿透 p50≈0.12–0.13 m、p99≈0.30–0.39 m，
  远高于 0.05 m 阈值；违反配对以 `robot_object` 为主（≈1150–1280），`object_wall`
  与 `robot_wall` 各 ≈600，`robot_robot` ≈230。**79% 的非法率不是边界符号约定的产物。**

### B2 编码与小样本拟合：`FIT_CHECK_OK`

- `encode_state_target → apply_state_delta` 往返：最大位置误差 4.5e-16、wrapped 角度 8.9e-16、
  速度 2.5e-16（2373 窗口），通过。
- 角度残差：原实现对角度 **差值**做普通差，π 的整倍数差被罚 2π；现改为把残差 wrap 到
  `(-π, π]`（`state_residual`），并统一用于 D 的三个正式组。满圈残差 2.0 → 0.0，
  半圈仍为 1.0。
- 32 样本拟合（从原 train 按固定 SHA256 抽 16 平移/转动 + 8 静止 + 4 成功 + 4 补齐，
  seed=0、全批 3000 更新）：物体位置 0.00085 m、机器人 0.00045/0.00082 m、物体角度
  0.00059 rad、终止 32/32、非有限 0、非法率 0.000。全部达标且远优于容差。
  该 checkpoint 不用于正式评估。

### B3 原模型分组与闭环复核

- 三个原 checkpoint、test 480 窗口（种子均值）：物体位置误差 0.191 m、机器人 0.439/0.443 m、
  物体角度 0.354 rad、非法率 0.791；与 persistence 的**逐样本配对差**（模型 − persistence）
  物体位置 +0.017…+0.033 m（模型更差），`fraction_model_better` <0.5。
  分组（平移/转动/静止/接触/无接触/S0/S1）与样本数见 `old_models/old_model_metrics.json`。
- 固定容差全链指标（真实 test 回合，40 链/模型）：recursive 累积终止率 4 步内达到 1.0，
  即**每个真实回合链在 4 窗口内就出现容差越界/错误终止**；teacher-forced 与 recursive 分母分列。
- 闭环复核见 A3 段；`closed_loop/episodes.jsonl` 与 `closed_loop/aggregate.json` 为本轮结果，
  旧的 60 回合首轮结果保留在 `outputs/forecast_baseline_v1/forecast-20261012/closed_loop_superseded_first_pass/`。

## 阶段 C — 成功轨迹附近的分支数据

### C1/C2 引导轨迹（精确模型只用于离线采集）

- 十个场景各用修复后的 `exact_fixed_pool`、H=3、K=128、最多 12 窗口，episode seeds 先 {0,1}，
  两条都失败才用 {2,3}。**共花 21 个精确规划回合（预算 40）**，得到 20 条真实成功引导：
  train 12、val 4、test 4（正好等于上限，无因超预算被丢弃的引导）。
- 每场景最多 2 条成功轨迹；所有失败尝试一并记录在 `guides.json`（含回合数、终止原因、墙钟）。
- 引导窗口本身记录但**不加入训练**，使 Random/Branch 的等数量比较严格成立
  （否则 Branch 组会额外多出 ≤144 个窗口而对照组没有）。

### C3 锚点

90 个锚点，全部来自真实非终止窗口起点；先取最后两个非终止窗口起点，再按窗口序号均匀补齐，
去重至最多 6 个/引导。使用完整精确快照恢复，保留 `success_counter`，未平移或扰动任何状态。

### C4/C5 分支与随机对照

| 数据 | 输入窗口 | 保留 | 与原数据完全相同 | 组内重复 | 同输入不同标签冲突 |
|---|---:|---:|---:|---:|---:|
| BranchAdd | 3902 | 3787 | 79 | 36 | 0 |
| RandomAdd | 2377 | 2281 | 60 | 36 | 0 |
| 引导窗口（仅留档） | 99 | 94 | 5 | 0 | 0 |

- RandomAdd 各 train 场景按 BranchAdd 的场景计数采集，shortfall 全为 0；
  等数量匹配后 **Branch 2271 = Random 2271**（逐场景取两者较小值，按固定哈希排序取样）。
- 分支首动作：1 条专家下一技能 + 1 条 hold/hold + 4 条按 `skill_features` 平方距离最近的
  不同候选 + 10 条 root 均匀随机；去重不足用 root 随机补齐；之后最多再执行 2 个窗口
  （0.5 重复 / 0.5 root 均匀），成功或物理失败立即终止。种子由
  split/scene/teacher_episode/anchor/branch 派生。预算：12×6×16×3=3456 分支窗口，
  实际 3787（含 val/test），未超预算。

### C6 覆盖

| 数据集 | 窗口 | 锚点 | 成功窗口 | 近成功窗口 | 物理失败 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| Old(train) | 1413 | 0 | 4 | 12 | 0 | SUCCESS_COVERAGE_LIMITED |
| RandomAdd(matched) | 2271 | 0 | 5 | 10 | 0 | SUCCESS_COVERAGE_LIMITED |
| **BranchAdd(matched)** | 2271 | 54 | **121** | 46 | 0 | SUCCESS_COVERAGE_OK |
| BranchAdd(全部 split) | 3787 | 90 | 233 | 66 | 0 | SUCCESS_COVERAGE_OK |

- 成功窗口从 4 升到 121、成功锚点从 <4 升到 54，**上一轮的“成功分组证据不足”被直接解除**；
  物理失败仍为 0（允许，未制造失败）。
- 同锚点不同动作的真实进展 spread：90 个锚点，平均 0.795 m、最大 2.075 m
  （`passage_cost_to_go` 的下降量）——同一起点选不同动作后果差异很大。
- 技能组合与接触类型逐项见 `coverage/coverage_table.md`。

## 阶段 D — 同结构数据对照

九次正式训练（三组 × 三种子），同网络/损失/Adam/batch/epochs/patience，同种子同初始化。
checkpoint 规则按合同冻结为 `0.5×原 val + 0.5×分支 val`。

**必须报告的一个后果**：该规则对没有分支训练数据的组退化——Old 组的分支 val 损失随训练
单调变差，于是 `best_epoch=0`（几乎未训练，train loss 0.77）。这与“数据好坏”无关，是规则本身
对 Old 组不利。因此除合同规则外，另做一次**对三组完全一致**的次级规则
（只用原 val 损失选 checkpoint），两组结果都报告，不按 test 选择任何东西：

| checkpoint 规则 | Old best epoch | Old+Random | Old+Branch |
|---|---|---|---|
| 合同（0.5 原 val + 0.5 分支 val） | 0 / 0 / 0 | 2 / 3 / 3 | 19 / 11 / 12 |
| 次级（只用原 val，三组一致） | 60 / 81 / 46 | 41 / 44 / 49 | 44 / 68 / 54 |

### D2 两套独立 test 表（次级规则；每格为三种子均值）

| 指标 | Old | Old+Random | Old+Branch |
|---|---:|---:|---:|
| 原 test 物体位置误差 (m) | 0.184–0.199 | **0.175–0.185** | 0.193–0.212 |
| 原 test 非法率 | 0.775–0.783 | 0.773–0.794 | 0.769–0.856 |
| 分支 test 物体位置误差 (m) | 0.211–0.217 | 0.202–0.223 | 0.217–0.221 |
| 分支 test 非法率 | 0.649–0.699 | **0.605–0.639** | 0.695–0.737 |
| 分支 test 成功召回 (分母) | 0.00 (24) | 0.00 (24) | 0.02 (24) |
| 分支 test 成功 precision (分母) | – (0) | – (0) | 1.00 (1–2) |

单步预测上 **Branch 没有超过等数量 Random**；原 test 上 Random 最好、Branch 反而略差。
成功召回仍接近 0，尽管训练集已有 121 个成功窗口——终止识别没有学会。

### D3 固定候选与闭环

固定候选复用原 12 起点与原候选池（`reference_verification.json: start_ids_match=true,
pool_mismatches=0, candidate_count_mismatches=0, reused=true`），只复核 H={1,3}、K={16,128}、
原三个候选种子；九个模型使用同一候选文件。全候选排序与“仅可行候选”两张表分列。

次级规则、H=3/K=128（种子均值）：

| 组 | chosen_true | uniform_true | candidate_best | regret | 候选拒绝率 | 仅可行 chosen_true | conditional regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old | −0.448 | −0.361 | +2.465 | 2.913 | 0.918 | −0.525 | 1.309 |
| Old+Random | −0.363 | −0.361 | +2.465 | 2.828 | 0.922 | −0.252 | 1.550 |
| Old+Branch | −0.262 | −0.361 | +2.465 | 2.727 | 0.917 | −0.166 | 1.096 |

Branch 在决策类指标上略好于 Old 与 Random，但差距很小，且三组的 chosen_true 仍低于
uniform，candidate_best（+2.465，含真实成功候选）远未被选中。

闭环（H=3，S0_v4/S1_v4，每组 18 回合）：

| 组 | 成功 | 平均窗口 | 平均末端距离 | 平均 J | no_feasible |
|---|---:|---:|---:|---:|---:|
| Old | 0/18 | 6.50 | 1.005 | −0.758 | 13 |
| Old+Random | 0/18 | 7.28 | 1.130 | −0.850 | 14 |
| Old+Branch | 1/18 | 7.78 | 1.032 | −0.312 | 13 |

对照案例：18 对配对回合中 **9 改善 / 9 退化**，平均 ΔJ +0.54；最好的一个是 S0_v4/seed0/模型 1
（−0.45 → +9.04，真实成功），最差的是 S0_v4/seed2（−0.58 → −1.36，末端距离 0.31 → 1.41）。
无 `planner_no_feasible_plan` 之外的物理失败。

## 停止状态

`GEOMETRY_BLOCKED` 否；`FIT_CHECK_FAILED` 否；`GUIDE_COVERAGE_LIMITED` 否（train 的 S0/S1
任务族都有成功引导）；`SUCCESS_COVERAGE_LIMITED` 仅对 Old/RandomAdd 成立，BranchAdd 为
`SUCCESS_COVERAGE_OK`。无 NOT_RUN。

## 自查（`selfcheck.json`）

精确模型不混入学习均值（19 个 mean 行均无 exact）；单候选反事实只改 1 个分数（文档 §2）；
全非法时 live_env 无新增 step 与成功不能绕过几何检查（测试覆盖）；train/val/test 场景、
锚点、回合三者互斥；Random/Branch 匹配（2271=2271，逐场景相等）；报告数字全部由脚本生成；
三种子全报、无按 test 选 checkpoint；旧目录文件哈希不变。
