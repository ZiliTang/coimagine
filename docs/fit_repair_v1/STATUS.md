# STATUS — fit_repair_v1（训练协议修正与预测失效定位）

任务书 `docs/tasks/20261013_0157_CoImagine_训练协议修正与预测失效定位任务书.md`。
run_id `fit-repair-20261013`。契约 [CONTRACT.md](CONTRACT.md)、更正 [errata.md](errata.md)。
旧 `fit_capacity_v1`、`data_upgrade_v1`、`fit_repair_v1` 之前的目录全部保留。

## 环境与保护

- git HEAD `627d5b861bc55152cef40489d7351ec8777dd53b`（branch `main`）；本轮只改
  `src/coimagine/fit_capacity/`、`experiments/fit_capacity/`、新增 `experiments/fit_repair/`、
  `src/coimagine/fit_capacity/{termination,geometry_attribution,paired,checkpoints}.py` 与
  `tests/test_fit_repair.py`。未修改连续环境、数据采集器或 Grid。
- Python 3.11.4、numpy 2.4.6、torch 2.14.1+cpu；训练固定 `torch.set_num_threads(4)`。
- 旧文件校验：`data_hashes.json` 记录 5 个原始数据文件的完整 SHA256（见 manifest）。

## 阶段 A — 修正已确认的实现与统计问题：完成，测试通过

`tests/test_fit_repair.py` 11 项全部通过（`pytest tests/test_fit_repair.py -q`）。

- **A1 batch loss 归约**：旧 `torch.stack(components).sum()` 对样本也求和，实测等于正确值的
  batch size 倍（12/12）。改为 `torch.stack(..., dim=0).sum(dim=0).mean()`；与
  `data_upgrade.training.per_sample_loss(...).mean()` 的**标量 loss 与参数梯度**逐位一致；
  同批样本重复两遍均值 loss 与梯度不变。训练日志记录每批实际 `batch_size`。
- **A2 按时保存中间模型**：到达 1000/3000/6000 时立即深拷贝参数，checkpoint 内记录
  `parameter_hash`（参数内容）与 `file_sha256`（文件）**两个不同 hash**，best 两项使用独立复制。
  测试：20 步运行里的 `update10` 参数与独立跑到 10 步的模型逐位相同、与 `update20` 不同；
  重新加载 `update10` 重测物理指标与即时日志一致。
- **A3 真正的恢复**：每个评估点原子写 `resume_state.pt`（模型/Adam/数据 RNG/permutation/
  cursor/updates/seen_samples/best 状态与分数），新增 `resume=True` 入口，恢复前核对数据内容
  hash 与配置 hash；样本索引流**全量**写入 `sample_order.bin` 并对其完整 SHA256 作为流 hash。
  测试：连续跑到 20 步 与 跑到 10 步后恢复到 20 步，最终权重、optimizer 步数、样本流 hash、
  best 选择全部一致；数据 hash 不匹配时拒绝恢复。
- **A4 候选共同样本配对**：新增 `fit_capacity/paired.py`，按 `(start_id, candidate_seed)` 配对，
  只保留双方都有可行候选的行，先按起点平均候选种子再跨起点平均，并报告被丢弃的行数/无解率；
  uniform 与 candidate_best 在同一子集重算。旧 paired 文件作废。
- **A5 最小报告更正**：见 [errata.md](errata.md) E5（11/12、−0.252/−0.166 高于 −0.361、
  “较早 checkpoint”措辞、召回 `null`、“物理最优”适用范围）。

## 阶段 B — 正确协议下的 12 次训练：完成

`python experiments/fit_repair/run_matrix.py --run-id fit-repair-20261013`，输出
`outputs/fit_repair_v1/fit-repair-20261013/runs/`。

- 训练集实测 Old+Random 3684、Old+Branch 3684 窗口；原 val 480、分支 val 728。
- 12/12 完成 6000 updates、`seen_samples=762,232`，无 NaN/Inf，无 early stopping；
  墙钟 271.5 s（预算 60 min），首 run 19.8 s 外推 4.0 min。
- 不变量：同宽同 seed 的两数据组初始参数 hash 相同；同 seed 的两宽度与两数据组
  **样本流 hash 完全相同**（seed0 `d31796835d30`、seed1 `2bbb434ec886`、seed2 `b2997d541b8e`）；
  五个固定 checkpoint 的参数内容 hash 互不相同且 updates 正确。
- 评估点 0/100/250/…/6000（26 个），`batch_loss.jsonl`（训练期）与 `eval_metrics.jsonl`
  （checkpoint 全集评估）分开保存；可 `--skip-existing` 跳过已完成 run，任一 run 可 resume。

## 阶段 C — 两项定位诊断：完成

### C1 终止 CE 归因（`c1_termination/termination_attribution.json`，36 个 checkpoint）

按真实类别统计 CE 贡献（全数据分母，贡献相加 = 总 CE，恒等式残差 ≤1e-12）：

| 数据集 | 选择 | 总 CE | window_complete 贡献 | success 贡献 | 成功窗均值 CE | 成功窗真实标签概率 | 真成功被预测为 window_complete | 真普通窗被预测 success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | mixed_val | 0.0197 | 0.0067 | 0.0131 | 2.64 | 0.386 | 0.583 | 0.0016 |
| train | original_val | 0.0013 | 0.0002 | 0.0011 | 0.043 | 0.967 | 0.011 | 0.00005 |
| train | final6000 | 0.0001 | 0.0000 | 0.0000 | 0.0018 | 0.998 | 0.000 | 0.000 |
| 分支 val | mixed_val | 0.414 | 0.0093 | **0.405** | 4.75 | 0.200 | 0.793 | 0.0034 |
| 分支 val | original_val | 1.001 | 0.0052 | **0.995** | 11.69 | 0.232 | 0.769 | 0.0033 |
| 分支 val | final6000 | 1.002 | 0.0151 | **0.987** | 11.58 | 0.283 | 0.715 | 0.0043 |
| 原 val | 三种 | 0.0002–0.0078 | 同总 CE | 无成功样本 | – | – | – | – |

结论（观测归因）：CE 恶化来自**成功漏判**——真实成功窗口被高度自信地判成 `window_complete`
（分支 val 上 72%–79%，均值 CE 4.8–11.7），而普通窗口误判 success 只有 0.3%–0.4%。
问题集中在分支 val 场景族；训练集上成功窗几乎被完全拟合（均值 CE 0.0018，真标签概率 0.998），
说明这是**场景泛化缺口**，不是表达能力不足。原 val 没有成功样本，是“只用原 val”选择的实际盲点。
本轮**不**宣称终止头造成位置误差，也不宣称多头梯度竞争已被证实。

### C2 几何非法的来源（`c2_geometry/geometry_attribution.json`，固定两个 run）

原始预测：非法率 0.75–0.79（train 0.745–0.794、分支 val 0.745–0.782）；
穿透 p50 0.11–0.15 m、p99 0.31–0.49 m；违规 pair 比例（一个样本可多对）：
`robot_object` 0.48–0.58、`robot_wall` 0.31–0.48、`object_wall` 0.25–0.38、
`robot_robot` 0.05–0.15。阈值以上的 pair 同序。

离线 pose 替换（速度与终止类别保持预测值，只调用冻结几何检查）：

| 替换 | 非法→合法 | 合法→非法 | 总非法率（原值 → 替换后） |
|---|---:|---:|---|
| 仅物体 pose | 458–564 | 104–159 | 0.745–0.794 → **0.606–0.676** |
| 仅两机器人 pose | 1141–1282 | 156–234 | 0.745–0.794 → **0.456–0.515** |
| 三刚体 pose 全部 | 全部非法行 | 0 | → **0.000** |

即：**机器人 pose 误差是更大的单一贡献者**（单独纠正可去掉约 35%–40% 的违规），
物体 pose 次之（约 15%–20%），两者耦合；全 pose 真值参照下非法率为 0，
说明检查器在同一时刻自洽，不存在时刻/场景/解码错位。训练更多（mixed→final6000）
并没有降低非法率（0.745→0.762、0.769→0.782）。这些混合状态不可物理到达，未送入 rollout。

## 阶段 D — 留出回归与候选对照：完成

- **D1**（`holdout_metrics.csv`）：36 个逻辑 checkpoint（12 run × 3 选择）在原 test 与分支 test 上复测。
  原 test：物体位置误差 0.192–0.204 m、非法率 0.797–0.810、**真实成功 0 → 召回 null**；
  分支 test：0.214–0.227 m、非法率 0.670–0.721、真实成功 50、召回 0.0017–0.0167、
  precision 0.69–1.00（分母 1–4）。同一批样本的 persistence 物体位置误差 0.1665（原 test）/
  0.2165（分支 test），模型在两个集合上都不优于 persistence。
- **D2**（`candidates/rows.jsonl` 1296 行、`aggregate.json`、`paired.json`）：36 个 checkpoint
  沿用同一 12 起点、candidate seeds 0/1/2、H=3、K=128 与既有真实参照
  （`reference_verification.json: reused=true`）。逐行检查 `误拒损失+排序损失=执行 regret`，
  最大残差 **1.8e-15**。三类选择的 12-run 均值：

| 选择 | 未筛选 chosen_true | candidate_best | uniform | 未筛选 regret | 拒绝率 | 无解起点 | 仅可行 chosen_true (n) | 误拒损失 | 排序损失 | 执行 regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best_mixed_val | −0.441 | +2.465 | −0.361 | 2.906 | 0.881 | 1.0 | −0.376 (10.6) | 2.053 | 1.024 | 3.077 |
| best_original_val | −0.429 | +2.465 | −0.361 | 2.894 | 0.918 | 1.4 | −0.387 (9.9) | 1.925 | 1.357 | 3.282 |
| fixed_6000 | −0.366 | +2.465 | −0.361 | 2.831 | 0.959 | 1.1 | −0.326 (10.0) | 2.219 | 0.879 | 3.097 |

  共同样本配对（42 组，先按起点平均候选种子再跨起点平均，分母 7–12 起点）：

| 配对 | Δ仅可行 chosen_true | Δ拒绝率 | Δ误拒损失 | Δ排序损失 | Δ执行 regret | 有解→无解 | 无解→有解 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed→original（12 组） | −0.059 | +0.037 | −0.214 | +0.273 | +0.059 | 42 | 27 |
| mixed→final（12 组） | +0.040 | +0.087 | +0.115 | −0.155 | −0.040 | 53 | 36 |
| 256→512（18 组） | −0.076 | −0.001 | +0.063 | +0.013 | +0.076 | 47 | 64 |

  即：换到 `best_original_val` 或 `fixed_6000` **不能稳定改善**候选选择——
  拒绝率分别再升 3.7pp 与 8.7pp，可行选择真实收益分别为 −0.059 与 +0.040（范围跨 0），
  而且两组都有 40+ 行从“有解”变成“无解”。非可行行按 `null` 保留、未赋 0、未执行非法候选。

## 图表与交付

`figures/fig_progress.png`（训练/验证位置误差与非法率）、`fig_losses.png`（state loss 与终止 CE）、
`fig_selection.png`（三种选模的真实候选收益与拒绝率；分母逐行列在 CSV，不与不同条件混画）。

## BLOCKED / partial / NOT_RUN

- BLOCKED：无。missing：无（36 个逻辑 checkpoint 全部存在；A2 修复后 intermediate checkpoint 可信）。
- partial：无。训练 12/12 完成 6000 updates；候选评估 36/36 完成。
- NOT_RUN：闭环、多尺度、几何投影、分布式融合、GPU 迁移——按任务书本轮不执行。
- 旧 `fit_capacity_v1` 结果仅作历史附件，不与本轮结果混算种子均值。
