# STATUS — CoImagine migration task v1 (phases A–D)

> **2026-10-19补充**：有界物理分支确认4个短循环状态中两处可由现有REPOSITION快速恢复，
> 一处有界尝试未解除，另一处已临近90秒预算；不作永久不可达判断。
> 已交付`docs/tasks/20261019_CoImagine_RL_Stability_LongRun_v1.md`：共同v3.1环境、
> critic共享梯度开/关的两臂单因素对照，每臂最多24小时或5000更新；交执行agent实施。
> 候选B的一批核验未胜A，任务书明确保留负结果。本会话未启动这轮长训练。

> **最新排查（2026-10-18，接续交接）**：`docs/random_start_rl_diagnosis_v4.md`。
> 已修复独立split（train几何保留；新validation/test版本v2）和PUSH寻物误用整局时钟
> （环境行为版本v3.1，奖励/90秒预算不变）。41项定向测试通过。
> 原8例回放initial/100/4600为3/0/1成功；新独立验证initial/4600为3/64、7/64，
> 差值配对区间包含零。已测固定轨迹梯度、一次更新干预与30步动作拟合；未开新长训练。
> 两条失败轨迹的4个短循环状态中，每机15动作均立即退出、仅3种首周期施力，
> 后续先查技能安全退出后的可恢复性。审查包：`outputs/random_start_rl_diagnosis_v4/`。
> 以下旧交接及历史STATUS保留；其中“split尚未修复”等状态已由上述结果更新。

> **最新交接（2026-10-18）**：`docs/handoff/20261018_RL_Progress_and_Next_Agent_Report.md`。
> 256例v3训练已到时间预算（4648更新，恢复状态4648，训练PID不存在），最佳仍为热启动更新0。
> manifest的running字段滞后。新确认validation64与train前64例完全重复，独立验证/泛化表述撤回，
> split修复尚未执行。下一步先做拆分修复、行为回放与梯度审计，不自动续训。

Updated at the end of the round. Every number in this file comes from a run whose command
is listed. Run roots are under `outputs/migration_v1/`.

## Environment

- git HEAD at start and end: `627d5b861bc55152cef40489d7351ec8777dd53b` (branch `main`).
  Working tree dirty files: the two untracked task documents under `docs/`. No tracked
  file was modified except the additions listed below; no checkpoint was overwritten and
  no user change was reverted.
- Python 3.11.4, numpy 2.4.6, matplotlib 3.11.1, PyYAML 6.0.3, pytest 9.1.1, and the newly
  pinned `pymunk 6.9.0` (official wheel; the only physics engine used).
- Manifest: `docs/migration_v1/manifest.json` (commit, dirty files, checkpoint SHA256,
  versions, scenes, seeds, scoring, action order, defaults, commands).

## Phase A — facts and frozen interfaces: DONE

- `docs/migration_v1/interface_table.md` (15 rows, legacy + new).
- `docs/migration_v1/manifest.json`.
- Acceptance run: `python experiments/grid_selection_audit/phase_a_check.py` →
  real `push:0:E` costs 1.0 and moves the object into the goal; `belief.npz` predicts
  `cost_hat=1.000` and `P(in goal)=0.502`. `PHASE_A_OK`.

## Phase B — legacy grid selection audit: DONE (not blocked)

- Code: `src/coimagine/diagnostics/grid_selection_audit.py`;
  runner `experiments/grid_selection_audit/run_audit.py`.
- Command: `python experiments/grid_selection_audit/run_audit.py --run-id phaseB-full`
  (15.7 s).
- Outputs: `outputs/migration_v1/phaseB-full/grid_selection_audit/` —
  `candidates.jsonl`, `summary.csv`, `checks.json`, `conclusion.md`, `figures/`.
- Result direction: selection bias is positive at every H/K and selection excess grows
  with H (0.41 → 0.76) while the candidate-mean bias goes slightly negative; regret grows
  with H. Verdict: **supports "selective overestimation"** within this diagnostic.
- Checks all pass: nesting, unique orders, mock `J_hat=J_ref` gives zero bias/regret,
  candidate order does not change true trajectories, checkpoint SHA256 unchanged.
- Limitation: the true reference is the exact expectation over the worlds consistent with
  the robot's information, not one realised world; `H=1` has only 6 and `H=2` only 36
  distinct sequences, so the larger K values reuse the same set (identical rows).

## Phase C — continuous 2D prototype: DONE, physics acceptance PASSED

- Code: `src/coimagine/continuous2d/` (`config`, `scenes`, `geometry`, `backend`,
  `skills`, `environment`, `candidates`, `scoring`, `mpc`, `data`, `render`).
- Command: `python experiments/continuous2d/run_physics_acceptance.py --out
  outputs/migration_v1/physics_check` (≈60 s) → `PHYSICS_ACCEPTANCE_OK`, 10/10 checks.
- Checked: all ten scenes start contact-free; a resting object never self-moves; centred
  vs off-centre pushes give distinguishable rotation; forces stay under the cap and a
  robot cannot reach a point inside the object; a contact push is not a failure; worst
  penetration 0.0011 m against a 1.0 m object; identical prefix ⇒ identical trajectory;
  fast snapshot clone matches exact prefix replay; candidate copies do not cross-
  contaminate; success requires an in-goal hold, not one frame; the S1 feasibility
  witness replays and completes in 5 windows.
- S1 witness: the accepted depth-3 plan's executed joint specs replayed open-loop from a
  fresh reset complete. A hand-designed single-robot two-phase rule was tried and did
  **not** complete (it rotated the object into the wall); this is recorded, not hidden,
  and the witness is a replay of a planner sequence rather than an independently
  hand-authored one.

## Phase D — exact-model MPC and data: DONE

- Commands:
  `python experiments/continuous2d/run_baseline.py --run-id phaseD-final` (full 10 scenes),
  `... --quick` (S0_dev + S1_dev smoke, 90 s).
- Outputs: `outputs/migration_v1/phaseD-final/continuous2d/` —
  `results.csv`, `results.json`, `summary.json`, `windows.jsonl`, `episodes.jsonl`,
  `results_table.md`, `media/` (per-step PNG frames + GIF for one S0 and one S1 success,
  regenerated by `experiments/continuous2d/render_showcase.py`).
- Headline (10 scenes): random control 0/30 episodes; depth-1 MPC 6/10
  (all five S0 + S1_v1); depth-3 MPC (beam 3) 7/10 (all five S0 + S1_dev + S1_v3). Mean wall
  clock per episode: random 0.1 s, depth-1 6.0 s, depth-3 42.4 s. On S1 alone: random 0/15,
  depth-1 1/5, depth-3 2/5.
- Engineering thresholds: at least one S0 and one S1 scene are solved by exact-model MPC
  (S0 all five by both depths; S1_dev and S1_v3 by depth 3). The S1 solutions require
  preparation (the object must be re-oriented before it fits through the passage).
- **Depth 3 is not uniformly better.** On S1_v1 depth-1 succeeds (7 windows) while
  depth-3 beam 3 fails (12 windows). A targeted beam sweep shows this is a beam-width
  artifact: depth-3 beam 1 succeeds (7 windows, 20 s), beam 2/3/5 all fail. The fixed
  lexicographic key over an imagined 3-window sequence chooses a first window that is not
  the best under receding-horizon execution; a wider beam keeps branches that look better
  in imagination. S1_v2 and S1_v4 fail at both depths. Reported as a negative result.
- Depth 1 and depth 3 have different compute budgets; nodes, physics steps and wall clock
  are reported per row, and no fairness claim is made.
- Reproducibility note: the candidate list order is part of the frozen protocol, because
  the random control selects by position (candidate_id). Reordering the per-face contact
  samples changes the random baseline while leaving the MPC rows unchanged. The
  authoritative run is `phaseD-final`; `phaseD-full` differed only in three random-control
  episodes for that reason. Within one frozen order the run is exactly reproducible.

## Not done / limits

- No learned world model, no multi-scale method, no distributed prediction; out of scope.
- The continuous prototype is a top-down planar approximation, not high-fidelity robot
  dynamics; the drag/push-speed controller is an engineering choice recorded in
  `continuous2d/config.py`.
- The S1 depth-3 planner uses the passage-aware geometric cost-to-go in the failure
  branch (documented in `scoring.passage_cost_to_go`); success still comes only from the
  real simulator reaching and holding in the goal. Without that geometric term the
  planner had no gradient towards preparation and never rotated the object.
- Only depth 1 and 3, beam 3, were run in the headline table. Depth-3 is a width-limited
  beam search, not an exhaustive optimum, and is named accordingly.
- The S1 witness is a replay of an accepted plan's executed skill specs, not an
  independently hand-authored sequence; a hand-designed single-robot two-phase rule was
  tried and did not complete.

## Next step (one only)

Freeze a scalar ``J`` contract for the continuous environment and re-run the *selection*
audit there with the same fixed-candidate protocol used in Phase B; that is the bridge to
the learned-model and multi-scale work, and it needs its own task document.

---

# STATUS — 协议修复与基线复核（protocol_fix_v1）

时间戳 2026-10-12 15:57 任务书。run_id `20261012-160943`。
详情见 `docs/protocol_fix_v1/`（PATCH_PLAN、report、errata、manifest.json、changelog）。

## 完成项

- S0 基线冻结与接口核实：`outputs/protocol_fix_v1/S0_baseline.json`、`PATCH_PLAN.md`。
- S1 快照与隔离：Snapshot 含 success_counter/termination/scene_config/physics/deep-copied
  controller_state/prefix；`restore_exact` 前缀重放并按四类量 1e-9 容差校验；
  `approximate_restore` 仅开发用；live/planner 分离；三项定向测试通过。
- S2 终态与搜索：成功吸收、失败终态不展开、`planner_no_feasible_plan`、每窗口成本只计一次、
  统计区分 live/候选模拟/精确重放步；URL 测试通过（含真实环境端到端与 mock 控制流）。
- S3 数据与覆盖：schema v2 结构化技能与双方施力、round-trip、segment 不跨 episode；
  四类穿透分别报告且共同参与阈值；`pymunk==6.9.0` 写入 pyproject。
- S4 复核：smoke 与 10 场景全量修复后基线（`outputs/protocol_fix_v1/20261012-160943/`），
  before/after 与首处分歧（`comparison/`）；Grid errata（`docs/protocol_fix_v1/errata.md`，
  旧数据未动）；passage 启发式声明写入报告。
- S5 交付：report/errata/manifest/changelog + 定向测试与物理验收 JSON + 小 ZIP 审查包。

## 结果（修复后）

- random 0/30；depth1 6/10；depth3(beam3) 8/10。
- 翻转：S1_v2 depth3、S1_v3 depth1、S1_v4 depth3 失败→成功；
  S0_v2 depth1/depth3 成功→`planner_no_feasible_plan`。
- **depth3 退化仍在**：S1_v1 depth1 成功、depth3 失败；旧 beam 伪影归因撤回。
- S0_v2 场景机器人初态与边界 capsule 重叠 0.06 m > 0.05 m 阈值，按禁令未改场景、未放宽阈值。

## 未做 / 限制

- 未修复旧 Grid 阈值化成功传播；纯动力学误差未隔离；单机器人对照未做。
- 三窗口候选计划与 beam 中间预测状态未落盘；只保存每次决策选中的完整预测序列与关键末态。
- depth3 为 beam=3 宽度受限搜索，非全局最优；精确重放使正式运行成本显著上升
  （depth3 均值 ≈77 s/回合，最长 S1_v1 depth3 ≈332 s）。

## 下一步（仅一项）

先修 S0_v2 初态边界重叠并复核该场景，再决定是否重启确定性 Grid 或学习模型任务。

---

# STATUS — 确定性 Grid 控制实验（B）

时间戳 2026-10-12 17:04 任务书。run_id `grid-20261012-194632`。
契约 `docs/deterministic_grid_v1/CONTRACT.md`，报告 `docs/deterministic_grid_v1/report.md`。

## 完成项

- B0 独立实现 `src/coimagine/deterministic_grid/`：6×6、两机器人、不可旋转 2×2 物体、两张
  固定地图（empty；obstacle 单格 (3,2)）、9 动作/机器人、81 联合动作、固定标量 J 与显式
  invalid_prediction。`transition_exact/state_is_valid/is_goal/score_trajectory` 有手算测试
  （同向双推、反向冲突、碰墙、移动冲突、同步回滚、目标吸收）。
- B1 精确链路：`run_env_checks.py` 全通过（witness 4/9 步；状态/顺序不变；完美模型
  J_hat=J_ref regret=0；吸收与初态成功；立即推 vs 换位再推手算候选）。
- B2 数据与模型：30,000 去重转移，80/10/10 按状态哈希切分；MLP 2×256 三头，3 种子；
  test 完整状态 0.860、非法率 0.05；物体移动样本 0.53 vs no-op 0.89。
- B3 固定候选诊断：20 个 test 起点（0 个在训练集），H=1/2/4、K=16/64/256、候选种子 0/1/2、
  模型种子 0/1/2；真值一次计算复用；精确对照行 bias/regret=0；teacher-forced vs recursive 对照。
- B4 报告与两张图；`deterministic_grid_review.zip`（2.82 MB，42 文件，SHA256 全对）。

## 结果要点

- 单步模型总体较准但推动后明显更差（0.53 vs 0.89）。
- **选择性高估成立**：selected_bias 与 selection_excess 全为正，excess 随 H 从 +1.03 升到 +2.70。
- **K 增大未降低实际选中收益**：chosen_true_return 平/略升，candidate_best 升更多，regret 上升。
  即候选变好而选择没跟上，不是“搜索越宽越差”。
- 候选覆盖不足：精确候选最佳成功率 H≤4 仅 0–5%，成功类结论受限。
- 递归使完整状态错误从 0.19 升到 0.34，teacher-forced 0.12–0.19；不据此断言递归是唯一原因。

## 未做 / 限制

- 点预测 argmax，无概率分布递归、无不确定性规划、无多尺度；20 起点只作描述性证据。
- 未知/未记录：无；合同要求的运行文件全部在包内。
- 不为改善趋势做超参搜索或挑案例。

## 下一步（仅一项）

在冻结候选与评分下，对“物体移动/no-op”做分头或重加权，复核 selection_excess 是否下降；
若不变，标注该简单确定系统不足以暴露该机制并回到连续平台。

---

# STATUS — 随机初态 RL 定向排查（2026-10-15）

> 最新更新（2026-10-16）：用户授权新稠密奖励v2与机制清理，详见 `docs/random_start_dense_reward_v2.md`。
> 默认奖励已改变，以下v1结果仅为历史记录。新版尚未长训练，不宣称成功率收益。
> v2采用真实距离进展＋距离积分成本＋终局奖励，gamma=1；删除终止归零、默认bandit和子集回退，
> 关闭熵奖励/隐式weight decay，保留PPO与安全保护。核心回归40 passed，短程日志奖励独立复算误差0。

## 2026-10-16：用户授权的24小时长训练已启动

> 最新状态（2026-10-17）：按用户要求停止8实例v3运行，启动
> `outputs/random_start_time_budget_v3_coverage256/`，热启动源为原最佳更新1550的best.pt，
> 权重复制哈希已核对；优化器重新初始化（旧best.pt无优化器）。
> 256个train实例按打乱遍历每批8个，保持v3奖励/网络/90秒预算；原8例跟踪、固定32训练面板、
> 64验证面板每50更新评价，最佳按验证面板选。上限5000更新/24小时；31项测试通过。
> 当前worker初始化和起始评价中。详见 `docs/random_start_v3_coverage256.md`。

> 后续状态：v2已按用户预算修正授权主动停止，非数值异常。
> 新v3删除30技能提前失败，统一90秒物理期限；保留稠密奖励系数，未新增补偿。
> 30项定向测试通过；真实90秒不同技能切分均付−3.6时间成本，总回报一致。
> 详见 `docs/random_start_time_budget_v3.md`；新版尚未长训练。

> 2026-10-16 后续：用户已授权并启动 v3 独立长训练，输出目录
> `outputs/random_start_time_budget_v3_long/`。冻结配置显示
> `random-start-time-budget-v3`、90 秒失败期限、8 个 train 实例、8 进程采样、
> 5000 更新或 24 小时先到为止；启动前定向测试 30 passed。
> 当前是刚启动状态，尚无成功率结论；旧 v2 检查点未复用。查看目录中的
> `monitor.json`、`updates.jsonl`、`curves.png`、`train.log` 获取后续进度。

- 计划：`docs/random_start_dense_reward_v2_long.md`；输出 `outputs/random_start_dense_reward_v2_long/`。
- 8独立采样进程＋1同步学习器，每更新8条完整episode；最多5000更新或24小时，先到为止。
- 原清单8个有成功证据的L0 train实例；新稠密奖励v2、网络与输入冻结；重新初始化。
- 启动前并行/串行一致性、正常检查点恢复与现有协议测试共27 passed。
- 初始状态：worker启动/初始基线评价中，当前不报告学习收益。
- 自动监测：每更新写 `monitor.json` 和README；每30分钟摘要/曲线；每50更新固定面板评价。
- 每5更新恢复检查点；异常停止；硬截止终止包含worker的整个进程组，不自动延长或进入S5。
- 原v1重训和奖励修改后的短程验证仍保留，下文为历史记录。

用户要求先排查上轮未建立可学习性的可能原因；本轮不是完整 S4 重跑。

- 最新报告：`docs/random_start_rl_audit_v1/report.md`。
- 最新审查结果：`outputs/random_start_rl_audit_v1/final/audit.json`。
- 已复现并修复：critic 决策前/后时间索引不一致、缺失声明的技能边界梯度截断、采样未使用局部 RNG。
- 当前定向测试：20 passed；另已运行包含连续物理环境的回归，33 passed（当时尚未添加最后两项定向测试）。
- 固定真实数据（2 条失败轨迹、60 决策）20 次 critic-only 拟合：修复路径解释方差 -0.094→0.889；
  旧路径也能拟合至 0.876，未建立这些修复对成功率的因果收益。
- 新确认：完整终止轨迹的 MC 势函数塑形会望远镜相消；不能将其误认为整轨迹稠密进度监督。
- 尚未修：324 维输入缺动作/阶段历史；S4 通过判据缺连续三点评价；S5 采集路径另有审计缺口。
- 奖励/折扣/物理/技能/输入 schema 均未改，历史交付包保留；原 S4 不通过、S5 阻塞状态不变。
- 下一步建议：版本化补全合法历史与门槛判据，再做固定条件的有界 S4 对照；不先改奖励或扩采。

## 修复后有界重训 — 已停止，时间预算耗尽（incomplete）

- 用户授权后于 2026-10-15 启动，计划 `docs/random_start_rl_audit_v1/retrain_plan.md`。
- 输出：`outputs/random_start_rl_retrain_v1/README.md`；已有曲线 `results/curves.png`、`results/curve.csv`。
- 已触发预设 45 分钟墙钟上限，完整记录 74/120 次更新（592 条训练轨迹、23 次成功）。
  最后恢复检查点在更新 70；71–74 有指标但无恢复模型，被打断更新的部分轨迹未计入 592。
- 最新完整评价为更新 70：确定性 0.000，随机执行 0.000；历史最高确定性 0.200（更新 5、15）。
  未过 S4；未完成计划末尾的最终验证评价，不能声称已完成 120 更新或有最终验证成绩。
- 更新 61–74 的采样成功率为零，回报未持续改善。没有延长训练或进入 S5。
- 硬上限：120 更新 × 8 episode、种子 0、45 分钟；不自动延长，不进入 S5。
- 原实际 5 实例、原奖励/折扣/技能/网络/输入；只带入三项语义修复。
- 每 5 更新确定性及随机评价、逐更新回报/成功率/critic EV、原子恢复检查点。
- 新增发现：原成功清单实际为 8 个，旧训练 scan=32 静默漏掉三个高编号实例。
  本轮明确沿用旧训练实际 5 个，显式清单不完整时现会报错。
- 启动前定向测试 21 passed；一更新 smoke 完成训练与周期评价，但工具 120 s 超时截断后续最终评价，
  因而不将 smoke 记为完整成功运行。正式后台任务独立保存日志、源码和配置，超时将显式标为 incomplete。
- 随后补充价值指标及导出测试后定向测试 23 passed；超时后仅离线整理已有曲线，未追加模拟。
