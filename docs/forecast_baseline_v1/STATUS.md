# STATUS — forecast_baseline_v1（Grid 诊断收尾与连续单尺度世界模型基线）

任务书 `docs/tasks/20261012_2007_CoImagine_Grid诊断收尾与连续世界模型基线任务书.md`。
run_id `forecast-20261012`（Grid 审计 `grid-audit-v2`）。契约
[CONTRACT.md](CONTRACT.md)，报告 [report.md](report.md)，更正 [errata.md](errata.md)。
本文每个数字都来自列出的命令；不存在的运行写 `NOT_RUN`。

## 环境与保护

- 起始/结束 git HEAD：`627d5b861bc55152cef40489d7351ec8777dd53b`（branch `main`）。
- 本轮新增文件全部为新文件；未修改 `src/coimagine/continuous2d/`、
  `src/coimagine/deterministic_grid/` 的既有模块、既有实验脚本或任何
  `outputs/deterministic_grid_v1/` 下的旧结果。`pyproject.toml`/`README.md` 在本轮开始前
  已是 dirty，未再改动。
- Python 3.11.4，numpy 2.4.6，torch 2.14.1+cpu（无 GPU），pymunk 6.9.0，32 核。
- 单样本推理在评估脚本里显式 `torch.set_num_threads(1)`：默认线程数会把 57→256 的
  小矩阵乘拖慢约 200 倍（0.125 ms → ~30 ms/节点），这是实现层修正，不是协议改动。
- Grid 三个 checkpoint 与 `samples.jsonl`/`splits.jsonl` 的 SHA256 在 A 开始与结束时一致
  （`grid-audit-v2/hash_check.json: unchanged=true`）。

## 阶段 A — Grid 诊断更正：实现完成、测试通过、实验已运行

命令：`python experiments/deterministic_grid/run_audit_v2.py --run-id grid-audit-v2`
（60–90 s）+ `pytest tests/test_grid_audit_v2.py -q`（10 passed）。只读旧数据、不重训。

- **A1 更正口径**（test split，按真实转移分组，三种子均值）：`object_moved` n=270 完整状态
  0.536、非法率 0.086；`object_stationary` n=2820 完整状态 0.892、非法率 0.047；
  **新增** `full_state_unchanged` n=1322 完整状态 0.858、非法率 0.081；全体 n=3090 完整状态
  0.861。`full_state_unchanged ⊆ object_stationary` 成立。旧 0.53 就是 `object_moved`
  完整状态准确率（重算均值 0.5358），不是“推动准确率”。
- **A2 无坐标偏置验证集**：按 `SHA256("grid-audit-v2|"+state_key)` 每图取前 10；不用坐标、
  不用模型误差、不用回报。机器人格覆盖从旧集合的 robot0 1 格 / robot1 14 格升到 16 / 18 格。
  实际不同候选数 H=1 为 81（不是 256），H=2/4 为 256。H=4/K=256、先状态内平均候选种子再
  跨状态平均：新集 `selected_bias +0.650`、`candidate_mean_bias −1.289`、
  `selection_excess +1.939`、`regret 0.806`；旧集同口径 `+0.938 / −1.088 / +2.025`。
  旧表头条 `1.250 / −1.450 / 2.700` 是旧的按行平均口径，重算复现到 4.4e-16，两种口径都列出。
- **A3 拆解与追踪**：拆解恒等式 `10ΔS−Δ步数−0.5Δ距离−10Δ非法` 对 106,740 条逐候选记录成立，
  `max|Σ分项−总偏差| = 0.0`。H=4 候选均值（种子均值）：成功 −0.002、步数 +0.387、
  距离 −0.004、**非法 −2.101**，合计 −1.719；而**选中候选的非法罚项在所有 H/K/种子下都是
  0**。因此 `selection_excess` 主要来自候选池的非法预测罚项，不是选中方案的乐观误差。
- **A4 递归对照**：与主诊断同协议（非法即终止、成功吸收、不再伪造下一状态）。新集 320 链/模型：
  第 1 步 teacher 与 recursive 相同（无效链 6.4%–7.4%，存活链误差 0.126，分母 297）；
  第 4 步存活链 teacher 0.070–0.078 vs recursive 0.223–0.244；累计终止率递归 0.137、
  teacher 0.164（**更正**：本节初版误写 0.918–0.917；该数字来自连续平台的旧 fallback
  递归诊断 `outputs/forecast_baseline_v1/forecast-20261012/diagnostic/recursive.json`，
  其协议是“非法预测时保留上一合法状态”，与本文件 §A4 的“非法即终止”不同。Grid A4 协议下的
  新验证集末步累计终止率为 0.137、旧 20 状态为 0.104，重算过程见
  `docs/data_upgrade_v1/A1_A2_grid_fixes.md` §3 与
  `outputs/data_upgrade_v1/grid-audit-v2/summary.json:termination_rate_reconciliation`）。
  旧 fallback 曲线不再用于解释主流程。

## 阶段 B — 单尺度连续世界模型：实现完成、测试通过、实验已运行

命令：`collect_data.py`（29.1 s）、`train_models.py`（~12 s 三种子）、`model_eval.py`。

- **B3 数据**：200 回合、2373 窗口（合同上限 2400，未达上限是因为部分回合提前成功）。
  train 120 回合/1413 窗口、val 40/480、test 40/480；按整场景划分。
  train 覆盖：平移 869、转动 839、静止 519、机器人-物体接触 1141、成功 4、物理失败 0；
  val 283/282/189，接触 360，成功 0，失败 0；test 288/281/183，接触 382，成功 0，失败 0。
  技能组合 train：double_push 777、push_move 352、single_push 180、move_hold 56、
  move_move 41、hold_hold 7。**成功组与物理失败组证据不足**（成功仅 train 4 例，失败 0 例），
  按合同如实标注，未加采样、未改采集策略。
- **B2 训练**：2×256 ReLU MLP（86,551 参数），一个配置、三个种子，仅按 val loss 选 checkpoint。
  seed 0/1/2 最优 epoch 60/81/46，val loss 0.01913/0.01872/0.01965，用时 4.0/4.8/2.9 s。
- **B4 test 单窗口**（种子均值）：物体位置误差 0.191 m、robot0 0.439 m、robot1 0.443 m、
  物体角度 0.354 rad、duration 误差 0.008 s、success_hold 误差 0.0045 s；
  **raw 非法预测率 0.791**（全部是 `penetration_limit`，无越界/非有限）。
  分组：object_translation 0.234/0.505（n=288）、object_rotation 0.233/0.505（n=281）、
  stationary 0.126/0.332（n=183）。
- **persistence 对照**：物体 0.166 m、robot0 0.420 m、robot1 0.443 m、角度 0.305 rad、
  duration 0.0、success_hold 0.0。**模型在位置与角度上不比“状态不变”更好**；
  duration 上 persistence 反而更准（test 窗口 duration 恒为 2 s）。
- **终止预测**：480 窗口 × 3 种子中只预测了 3 次 success，全部为虚假成功；真实成功样本
  test 为 0，所以“虚假成功率”分母极小。分类头实际上退化为 window_complete。
  测试集真实成功与物理失败都是 0 → **该分组证据不足**，不据此判断任务能力。
- 与 persistence 比较、与分组错误率一起看：模型在接触密集的窗口更差（接触 1141/1413），
  这正是关键接触转移，模型没有学会。

## 阶段 C — 固定候选排序诊断：实现完成、测试通过、实验已运行

命令：`python experiments/forecast_baseline/run_diagnostic.py --run-id forecast-20261012`
（552 s）。12 起点（每 test 场景 1 个初态 + 5 个 SHA256 排序的非终止窗口起点），
`H={1,2,3}`、`K={16,64,128}`、candidate_seeds `{0,1,2}`、model_seeds `{0,1,2}`；
真值每条候选只算一次并被三个模型共用；精确对照行 bias/regret 恒为 0。

- **候选覆盖（先于模型报告）**：12 起点中 **11 个**存在“可行且 d 低于起点”的候选，
  5 个存在至少一条成功候选；实际去重候选数 H=1 为 121，H=2/3 为 128；
  精确候选真实失败率 0.000（所有 H / 候选种子）。**结论 COVERAGE_OK**，因此执行 D。
  成功候选仍少（精确最佳成功率 0.056–0.278），成功类结论只作进展排序，不当作完整任务能力。
- **模型排序**（种子均值，各 H/K）：`selected_bias` 基本为负（+0.04…−12.6），
  `candidate_mean_bias` 远更负（−12.6…−19.1），`selection_excess` 为正（+6.0…+16.9）；
  但拆解显示被选候选的偏差几乎全部是 **failure_penalty**（模型判其 invalid），
  成功/时间/距离分项都接近 0。所以正的 selection_excess 不是乐观高估，而是候选池里
  大量 −20 非法罚项造成的口径差。
- **模型有没有选对**：H=3/K=128 时 `chosen_true` 为 −0.353（三种子均值），
  同一池的 `uniform_candidate_true` 为 −0.361、`candidate_best_true` 为 +2.465；
  排序损失（regret）约 2.82，且随 K 与 H 上升。三个最大 regret 起点的案例里，
  精确最佳候选都是**真实成功**（J_ref ≈ +9.8…+9.9），而模型给它的 J_hat ≈ −20.1，
  因为模型把它标成 `invalid_prediction`；它在 J_hat 下只排到 105/128、121/128、17/128。
- **失败/成功虚构方向**：全部 13,572 条候选记录里，真实成功候选-模型配对各 149 个，
  其中 **72.5% 被模型判为 invalid_prediction**、27.5% 判为 window_complete；
  模型在整张候选表里**一次都没有预测过 success**（预测成功数 0，虚假成功 0）。
  也就是说：连续平台的错法不是 Grid 那种“虚构成功”，而是**把真正能成功的候选判成非法/失败**，
  再用 −20 罚项把它们排到后面。
- **C4 递归对照**（同 C 的固定序列，H=3，candidate_seed=0，1536 链/模型）：
  第 1 步 teacher 与 recursive 相同（非法率 0.635/0.743/0.724，存活链位置误差 0.45）；
  第 3 步存活链只剩 191/1536…280/1536，其位置误差 0.65–0.66，而 teacher-forced 同一批
  真实状态上的误差只有 0.41–0.42；累计终止率 0.917/0.941/0.931。
  “全部链累计失败/错误比例”按精确匹配口径饱和为 1.000（连续状态几乎不可能逐步精确相等），
  因此该口径不可用，可用的全链数字是累计终止率——两者都在 `recursive.json` 里列出。

## 阶段 D — 同协议闭环基线：实现完成、测试通过、实验已运行

命令：`python experiments/forecast_baseline/run_closed_loop.py --run-id forecast-20261012`。
共 **54 回合**（exact 12、learned 36、uniform 6），两 test 场景、H={1,3}、最大 K=128、
最多 12 个 live 窗口；每回合只执行所选序列的首窗口。

| 方法 | H | 模型种子 | 回合 | 真实成功 | 平均窗口 | 平均末端距离 | 平均 J | 物理失败 | no_feasible |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| exact_fixed_pool | 1 | – | 6 | 0.50 | 8.00 | 0.226 | +4.281 | 0 | 0 |
| exact_fixed_pool | 3 | – | 6 | **1.00** | 5.33 | 0.000 | +9.584 | 0 | 0 |
| learned_fixed_pool | 1 | 0 | 6 | 0.00 | 12.00 | 0.371 | −1.099 | 0 | 0 |
| learned_fixed_pool | 1 | 1 | 6 | 0.00 | 12.00 | 0.119 | −1.031 | 0 | 0 |
| learned_fixed_pool | 1 | 2 | 6 | 0.00 | 12.00 | 0.885 | −1.241 | 0 | 0 |
| learned_fixed_pool | 3 | 0 | 6 | 0.167 | 10.50 | 1.189 | +0.467 | 0 | 0 |
| learned_fixed_pool | 3 | 1 | 6 | 0.00 | 12.00 | 0.945 | −1.254 | 0 | 0 |
| learned_fixed_pool | 3 | 2 | 6 | 0.00 | 12.00 | 1.003 | −1.270 | 0 | 0 |
| uniform_root | – | – | 6 | 0.00 | 12.00 | 0.994 | −1.270 | 0 | 0 |

- 精确方法在新标量协议下成功（H=3 六回合全成），**新标量精确成绩不得与旧字典序
  `depth1 5/10、depth3 9/10` 合并**；这是重跑的独立结果。
- 学习模型 18 个 H=1 回合成功 0，18 个 H=3 回合成功 1；与 uniform_root 的 0/6 同一量级，
  没有证据表明学习模型优于随机候选选择。
- 三种方法都没有 `planner_no_feasible_plan`、没有物理失败：模型的“拒绝”不是拒绝，而是
  给候选打 −20；模型预测成功时本轮仍执行首窗口并由真实判据决定，未直接把回合标成成功。
- 两个场景、6–36 回合是小样本；只作基线复核，不声称统计泛化或新方法优势。

## 过程偏差（已修正，留有记录）

第一次 D 运行把 `uniform_root` 也按 `H` 循环，多跑了 6 个与 H 无关的相同回合（60 而不是 54）。
已把该次输出原样保留在
`outputs/forecast_baseline_v1/forecast-20261012/closed_loop_superseded_first_pass/`，
修正循环后重跑，正式结果 `closed_loop/` 为合同要求的 54 回合（uniform 的 `H=null`、6 回合）。
两次 uniform 结果完全相同（mean_J −1.2697、0/6 成功），说明该偏差不影响结论。

## 测试

- 新增：`tests/test_forecast_baseline.py`（16 项：评分手算与恒等式、吸收、零 regret、
  编码布局与技能敏感性、几何合法性、候选池确定性与模板、非法/成功吸收、episode 前缀可复现、
  训练可复现、累计分母语义）、`tests/test_grid_audit_v2.py`（10 项）。
- 相关既有测试：`tests/test_deterministic_grid.py`、`tests/test_continuous2d.py`、
  `tests/test_protocol_fix.py`、`tests/test_grid_selection_audit.py`。
- `pytest -q -m "not slow"`：全部通过，除了一个与本轮无关的既有随机性测试
  `tests/test_belief_model.py::test_the_release_transition_is_conditioned_on_whether_it_can_work`；
  它在 `OMP_NUM_THREADS=1` 下失败、默认线程数下通过（该测试属于旧 `learning` 模块，
  本轮未改动其任何文件），单独复跑通过。
- 既有 `tests/test_continuous2d.py::test_depth_one_mpc_solves_a_free_space_scene` 失败，
  原因是 closeout 修好 S0_dev 初态机器人位置后旧字典序 depth1 在该场景不再成功；
  `docs/closeout_v1/README.md` 已记录该结果，本轮不改测试期望、不隐藏。

## BLOCKED / COVERAGE_LIMITED / NOT_RUN

- BLOCKED：无。
- COVERAGE_LIMITED：否（11/12 起点有可行且降低 d 的候选）。
- NOT_RUN：无。多尺度、跨步约束、新损失、模型集成、分布式预测融合、通信、异步技能、
  第三台机器人、多物体、机械臂仿真均未实现（本轮范围之外）。
