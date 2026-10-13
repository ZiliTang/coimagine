# STATUS — fit_capacity_v1（训练充分性与模型容量验证）

任务书 `docs/tasks/20261012_2256_CoImagine_训练充分性与模型容量验证任务书.md`。
run_id `fit-capacity-20261012`。契约 [CONTRACT.md](CONTRACT.md)、更正 [errata.md](errata.md)。
每个数字都来自列出的命令；未运行的写 `NOT_RUN`。

## 环境与保护

- git HEAD `627d5b861bc55152cef40489d7351ec8777dd53b`（branch `main`）；本轮全部为新增文件，
  `outputs/data_upgrade_v1/`、`outputs/forecast_baseline_v1/` 只读引用，检查前后哈希不变。
- Python 3.11.4、numpy 2.4.6、torch 2.14.1+cpu、pymunk 6.9.0；训练固定
  `torch.set_num_threads(4)`（任务书默认值）。
- 参数规模实测（代码计数）：2×256 = **86,551**、2×512 = **304,151**，与任务书一致。
- 冻结项：`data_upgrade_v1` 的数据/划分/场景/物理/控制器/2 s 窗口/57 维输入/编解码/几何容差/
  终止优先级/评分 J/候选与真实参照；损失用已修复的 wrapped 版本，系数不变。

## 阶段 A — 最小更正与现有 checkpoint 全训练集检查：完成

命令：`python experiments/fit_capacity/run_existing.py`（1 分 40 秒）。

- **A1**：见 [errata.md](errata.md) E1–E3。配对键更正后 18 对、改善/相等/退化 6/2/10、
  平均 ΔJ +0.446；uniform 比较按协议分开；训练成功锚点 22、分支 test 成功窗口 50。
- **A2**：两套规则 × 三数据组 × 三种子 = 18 个 checkpoint，全部存在（`missing` 0 个），
  逐个记录 SHA256、宽度、best_epoch、updates（`existing_checkpoints.csv`）。
  每个 checkpoint 在**其自身完整训练集**、原 val、分支 val 上重测（`existing_fit_metrics.csv`，
  完整四项损失与物理指标在 `existing_fit_details.json`）。摘要（三种子均值，宽度均 256）：

| 规则 | 组 | 训练集 n | best_epoch | 自身训练集 obj | 自身训练集非法率 | 原 val obj | 分支 val obj |
|---|---|---:|---|---:|---:|---:|---:|
| mixed_val | Old | 1413 | 0/0/0 | 0.369 | 0.829 | 0.361 | 0.359 |
| mixed_val | Old+Random | 3684 | 2/3/3 | 0.183 | 0.792 | 0.192 | 0.219 |
| mixed_val | Old+Branch | 3684 | 19/11/12 | 0.212 | 0.751 | 0.194 | 0.218 |
| original_val | Old | 1413 | 60/81/46 | 0.181 | 0.778 | 0.196 | 0.214 |
| original_val | Old+Random | 3684 | 41/44/49 | 0.165 | 0.763 | 0.187 | 0.212 |
| original_val | Old+Branch | 3684 | 44/68/54 | 0.196 | 0.763 | 0.196 | 0.211 |

  同一样本的 persistence 物体位置误差：训练集 0.150–0.207、原 val 0.158、分支 val 0.199
  —— 旧模型在训练集上也不优于 persistence。**训练集与验证集误差几乎相同**，所以旧模型的
  问题是训练拟合不足，不是验证泛化退化（见 errata E6）。

## 阶段 B — 固定预算训练 × 容量对照：完成，12 次新训练

命令：`python experiments/fit_capacity/run_matrix.py`。矩阵 = 2 数据组（Old+Random、Old+Branch，
各 3684 窗口）× 2 宽度（256/512）× 3 种子 = **12 次训练，全部完成**。
固定 6000 `optimizer.step`，不早停；Adam lr=1e-3、batch=128；评估点 26 个
（0,100,250,500,750,1000,1250,…,6000）。

- 正确性：同宽度同种子的初始参数 hash 跨数据组相同（256: `f3ad9435…`、512: `3bc34e24…`）；
  同数据组同种子的样本顺序与宽度无关（`sample_order` 逐条相同）。
- 预算：首 run 19.5 s → 预估整批 46.9 min，实际 **268.4 s（4.5 min）**，远低于 60 min 上限；
  每 run 的 `seen_samples = 762,232`。
- 可恢复：每 run 保存 `resume_state.pt`（model + optimizer + data RNG 状态 + updates +
  epoch/cursor + 配置 + 数据 hash）；`batch_loss.jsonl` 与 `eval_metrics.jsonl` 分开。
- 无 NaN/Inf；12/12 `stopped_reason=completed`。
- checkpoint：每 run 保存 `update1000/3000/6000.pt`、`best_mixed_val.pt`、`best_original_val.pt`；
  两种 best 均排除 update 0、严格更低才更新、并列保留较早。

## 阶段 C — 三张表：完成

`experiments/fit_capacity/run_tables.py` → `tables/{progress_table,capacity_table,selection_table}.json`
与 `tables/tables.md`。逐 seed 配对，报各值/均值/范围。

- **C1 进度**（1000 / 3000 / 6000，三种子均值，摘要）：
  Old+Random 256 train obj 0.1630→0.1496→0.1473、train loss 0.0144→0.0109→0.0086，
  原 val loss 0.0178→0.0174→**0.0192**；Old+Random 512 train obj 0.1602→0.1362→0.1290、
  train loss 0.0119→0.0075→0.0044，原 val loss 0.0170→0.0193→**0.0228**。
  Old+Branch 512 train obj 0.2066→0.1894→0.1467、train loss 0.0238→0.0103→0.0062，
  原 val loss 0.0223→0.0229→**0.0292**。即：**训练损失明显下降，原 val 损失反而上升**。
- **C2 容量**（512−256，同数据同种子，6000 updates）：训练集物体位置误差差值
  **6/6 为负**（−0.0006…−0.0737，均值 −0.034），训练总损失差值 6/6 为负；
  原 val 物体位置误差 4/6 为正、分支 val 3/6 为正；非法率改善不一致；
  墙钟 512 约 24–27 s vs 256 约 19–20 s。
- **C3 选择**：12/12 run 的 `best_mixed_val` 落在 update 100/250/500（见 errata E4），
  12/12 `loss_vs_physics_disagree = True`（见 errata E5）。

“明显改善”工程标记（物体位置误差相对降 ≥20% 且非法率绝对降 ≥10pp 且 ≥2/3 种子同向）：
以 1000→6000、512 宽为例，Old+Branch 训练集物体误差 −29% 但非法率仅 −0.8pp，**不满足**；
Old+Random 512 训练集物体误差 −19.5%（接近但未达 20%）且非法率 −5.0pp，**不满足**。
原 val/分支 val 上更无一组满足。**未达标仍完成 D。**

## 阶段 D — 固定留出集与候选复核：完成

- **D1**：12 个 run × {best_mixed_val, best_original_val, fixed_6000} 在原 test（480 窗口）与
  分支 test（750 窗口，其中 50 个真实成功窗口）上评估，重复 checkpoint 用 SHA256 缓存。
  摘要（三种子均值）：原 test 物体位置误差 0.173–0.224 m、非法率 0.78–0.84；
  分支 test 物体位置误差 0.200–0.241 m、非法率 0.66–0.74。
  `best_mixed_val` 的分支 test 总损失（0.53–0.57）明显低于 `fixed_6000`（1.38–1.47），
  但原 test 总损失更高（0.022–0.027 vs 0.020–0.031 区间内偏高）。
  成功识别基本为 0：原 test 上 12/12 模型成功召回为空（真实成功 0）；分支 test 上
  仅 0.00–0.06（分母 50），precision 0.33–1.00（分母 1–4）。真实 `physical_failure` 仍为 0
  → **未验证**。
- **D2**：12 个 `best_mixed_val`，沿用上一轮 12 起点与同一 H=3/K=128 候选缓存，
  核对结果 `start_ids_match=true, pools_checked=72, pool_mismatches=0,
  candidate_count_mismatches=0, reused=true`。
  未筛选：chosen_true −0.24…−0.56、candidate_best +2.465、uniform −0.361、
  selected_bias −9.35…+1.50、regret 2.71–3.02。
  执行协议：候选拒绝率 **0.79–0.92**，无可行候选起点 0–5/12；仅可行 chosen_true
  −0.53…+0.28（分母 7–12 起点）。
  分解（按起点先平均候选种子再跨起点）：误拒损失 +0.97…+2.62、集合内排序损失
  +0.37…+2.02、执行 regret +2.19…+4.26，三者恒等式逐行成立。
  256 与 512 的配对比较见 `candidates/paired_256_512.json`。
  人工核对：精确对照 `J_hat=J_ref` 的 bias/regret 为 0（`data_upgrade_v1` 已验证并在本轮复用）。
- 本轮不跑闭环、不扩大 H/K、不增种子、不为收益调参。

## 测试与自查

- `tests/test_fit_capacity.py` 9 项：配对键唯一且必须含 model_seed、四项损失相加一致、
  不同宽度样本顺序一致、初始参数只依赖宽度与种子、训练签名不含 test、final 不被 best 覆盖、
  best 排除 update 0、resume 状态往返一致、误拒/排序分解恒等式。全部通过。
- 既有非 slow 测试全量通过（`pytest -q -m "not slow"`，0 失败）。
- 旧文件未覆盖：`outputs/data_upgrade_v1/.../dataset/windows.jsonl` 与
  `outputs/forecast_baseline_v1/.../test_metrics.json` 的 SHA256 与本轮前一致
  （`selfcheck` 见 `a1_audit.json` 与 manifest）。

## BLOCKED / NOT_RUN

- BLOCKED：无。missing checkpoint：0。partial：无（12/12 新训练完成 6000 updates）。
- NOT_RUN：闭环、图网络/Transformer、多尺度、概率集成、几何投影、分布式融合、GPU 迁移
  —— 本轮范围之外，按任务书不执行。
