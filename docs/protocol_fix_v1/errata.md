# Grid 诊断 errata（只更正解释，不重算旧模型）

本文件由 `experiments/grid_selection_audit/write_errata.py` 从已有 `phaseB-full/grid_selection_audit/summary.csv` 与 `candidates.jsonl` **只读**生成；旧模型、旧动力学、旧评分、旧候选与旧原始数据均未改动，旧 `conclusion.md` 未覆盖。

## H=3 固定深度：K=16/64/128 的事实

| K | selected_bias | selection_excess | candidate_regret | chosen_true_return | uniform_candidate_return |
|---:|---:|---:|---:|---:|---:|
| 16 | +0.554 | +0.774 | 1.014 | -1.233 | -2.708 |
| 64 | +0.560 | +0.753 | 0.971 | -1.010 | -2.757 |
| 128 | +0.538 | +0.757 | 1.292 | -0.398 | -2.700 |

## Errata 声明

1. 当前诊断仍使用**隐藏参数、学习信念与原边缘状态传播**（`learning/belief_model.py:step_actions` 把同一 41 维边缘动态复制 16 份后按 b(θ) 混合）；未使用 `diagnostics/conditional_planner.py` 的逐参数条件状态。
2. **成功概率阈值化导致两边收益语义不一致**：模型侧把 `P(in goal) ≥ 0.5` 当作 100% 成功并截断后续代价（`diagnostics/grid_selection_audit.py:model_outcome`），真实侧是与观测一致世界上的精确期望（同文件 `true_outcome`）。
3. 代表 H=3 案例（scene=3, seed=0, H=3, K=128，选中 candidate_id=10）的模型分**只推演 1 步**后触发阈值；该场景 128 条候选中模型推演步数分布为 {1: 23, 2: 2, 3: 103}。 选中候选 J_hat=+8.79，J_ref=-2.50，一致世界数=2。
4. 固定 H=3 下**实际选中收益随 K 增加而提高**（chosen_true_return：K=16 -1.233 → K=64 -1.010 → K=128 -0.398）；不能据此声称“搜索越宽越差”。
5. 旧结论应限定为**当前管线的现象**；本轮未隔离纯动力学误差，也未修复概率版吸收传播。
6. 结论生成器 `_conclusion`/`conclusion_from_rows` 已改为只输出事实表与“评分协议待修正，暂不作机制判断”，不再写死“支持”或固定数字；本轮不重算旧数据。

（代表案例 regret=10.500；selection_excess 等逐场景值见原 `summary.csv`。）
