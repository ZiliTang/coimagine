# ERRATA — data_upgrade_v1

更正与撤回。所有更正都由本轮运行生成，不手改数字；旧表与旧运行目录原样保留。

## E1 Grid 的学习均值里混入了精确模型（更正）

`aggregate_by_state` 的 `model_seed="mean"` 行把精确模型（`model_seed=-1`）当成第四个“种子”
一起平均。因为它三个偏差类指标恒为 0，“含 exact 的均值”恰好是学习均值的 3/4。
修复后只平均学习种子 0/1/2，精确行单列。旧验证集 H=4/K=256 由 0.938/−1.088/2.025 回到
**+1.250/−1.450/2.700**，新验证集为 **+0.867/−1.719/2.586**。
上一轮把 1.250 与 0.938 的差异说成“两种聚合口径”，**撤回**：那是精确行混入造成的。
证据：`outputs/data_upgrade_v1/grid-audit-v2/summary.json`、`A1_A2_grid_fixes.md` §1。

## E2 Grid 单候选反事实的因果表述（撤回并重做）

`build_traced_case` 的旧实现对**全部候选**各自修复首次同类错误后重排，却写成“只修选中候选”。
新主分析只替换 `J_hat[c]` 一项（改动条目数 = 1），其余候选逐项不变，并另报新选择的真实收益。
旧结果作为历史附件保留并更正表述。更正后的结论：单候选干预在 3/3 高 regret 案例改变选择，
但**真实收益只有 1/3 改善**；“选择改变”不能读作“修复有收益”。

## E3 `false_success` 不是“只修成功分类”（撤回命名）

该干预通过替换整个真实下一步状态实现，现命名为“首次虚假成功处的完整状态干预”。

## E4 Grid 递归累计终止率 0.918 → 0.137（更正）

`docs/forecast_baseline_v1/STATUS.md` 在 Grid 段写了 0.918–0.917。从原始 JSON 重算：
A4 协议（非法即终止、成功吸收、累积 = 已终止链数/全部链数）下新验证集 H=4 为 **0.137**、
teacher 0.164，旧 20 状态为 0.104 / 0.128。0.917/0.941/0.931 出现在**连续平台**的旧 fallback
诊断 `outputs/forecast_baseline_v1/forecast-20261012/diagnostic/recursive.json`，其协议是
“非法预测时保留上一合法状态”，与 Grid A4 不同；本轮 A4 记录中扫描 0.918±5e-4 命中 0 次。
旧 STATUS 已就地更正并注明来源。

## E5 连续闭环此前执行了预测非法的序列（撤回旧闭环数字）

旧协议只把 `non_finite` 当拒绝条件，`invalid_prediction` 与 `physical_failure` 仍可被选中执行：
重算旧结果得 **59 次非法序列执行、涉及 22 个回合**。修复后只执行
`window_complete / success`，无可行序列时 `planner_no_feasible_plan` 且 `live_env` 不动。
因此上一轮“学习模型闭环 1/18 成功”**不能继续称为合同内结果**；本轮闭环表为协议内结果，
同一批 22 个回合现在以任务失败原因 `no_feasible_plan` 终止。

## E6 几何检查器的边界约定（更正并量化）

A3 修正了几何检查中 **y 用 arena width 判界**的错误（改用 height）。B1 进一步发现：冻结环境的
`penetration = max(0, -contact.distance)` 对 circle–segment 与 polygon–polygon 成立，但对
**polygon–segment（物体压向场地边界）** Chipmunk 返回正 distance，于是引擎自身的 bookkeeping
看不见该类重叠。独立检查器按物理重叠计算，因此在该对上更严格（观测差 ≤5.6 mm，阈值 50 mm）。
该差异被记录、量化，未据此放宽或收紧阈值；真实数据上两种口径都不产生非法状态。

## E7 角度残差未按 wrapped 计算（修复）

原损失的“角度误差”是对**角度差值**做普通差，π 的整倍数差会被罚 2π。现改为把残差 wrap 到
`(-π, π]`（`state_residual`），并统一用于本轮三个正式组。旧 checkpoint 仍作历史复核。

## E8 合同 checkpoint 规则对 Old 组退化（如实报告，不改规则）

`val_score = 0.5×原 val + 0.5×分支 val` 使没有分支训练数据的 Old 组 `best_epoch=0`
（几乎未训练）。规则按合同保留为主要结果；另做一次对三组一致的“只用原 val”次级规则，
两组结果都报告，且都未按 test 选择任何 checkpoint。这不改变任何协议或场景。

## E9 过期测试的更正（不是放宽标准）

`test_depth_one_mpc_solves_a_free_space_scene` 原先断言 S0_dev 可解；closeout 修好该场景
机器人初态后旧字典序 depth1 不再成功（`docs/closeout_v1/README.md` 已记录）。正向检查改用
已确认可解的 **S0_v1**，并新增 S0_dev 的初态合法性与协议测试。未改场景、未删正向检查。
