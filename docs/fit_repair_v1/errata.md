# ERRATA — fit_repair_v1

本轮 A 阶段只修已确认的实现与统计问题。旧 `fit_capacity_v1` 的输出与模型全部保留，
以下更正只说明它们哪些数字不可用，不覆盖任何文件。

## E1 `fit_capacity_v1` 的 batch loss 多乘了一个 batch（实现错误）

`fit_capacity/trainer.py` 写成 `loss = torch.stack(components).sum()`。`loss_components` 返回
四个形状 `[B]` 的逐样本向量，因此这是对样本也求和：实测 12 条样本时 legacy = 21.2910，
正确值 = 1.7742，比值 12.0 = batch size（`tests/test_fit_repair.py` 断言该恒等式）。
正确写法是 `torch.stack(components, dim=0).sum(dim=0).mean()`，与
`data_upgrade.training.per_sample_loss(...).mean()` 在标量 loss 与参数梯度上一致。
影响范围：`fit_capacity_v1` 的 12 次训练全部受影响，其 `batch_loss.jsonl` 的 total 与四项
均值、以及由此得到的训练轨迹数字都不可直接引用。**Adam 对整体梯度缩放有部分抵消，
所以不能把旧 bug 说成“学习率放大 128 倍”；末批大小不同也意味着两者并不严格等价。**
本轮 B 已用修正后的协议重跑全部 12 次训练。

## E2 `fit_capacity_v1` 的 update1000/3000/6000 都是最终参数（实现错误）

旧实现在训练**结束后**才写这三份 checkpoint，它们的内容都是 update 6000 的参数；
只有 best 两项是训练中复制的。因此旧目录里 `update1000.pt` / `update3000.pt`
**不能当作 1000 / 3000 步时的模型**，任何基于它们的中间进度评估（例如“早期 checkpoint
重测物理指标”）都不成立。本轮改为到达指定 update 时立即深拷贝快照，并在 checkpoint 内
同时记录文件 SHA256 与**参数内容 hash**（两者分开）。

## E3 `fit_capacity_v1` 并没有真正的续训（实现错误）

旧代码只在正常结束时写一份 `resume_state.pt`，没有恢复入口，训练中途被打断无法续跑。
本轮改为每个评估点用“临时文件 + rename”**原子保存**（模型、Adam、数据 RNG、
当前 permutation、cursor、updates/seen_samples、best 状态与分数、样本流 hash），
并新增 `resume=True` 入口，恢复前核对数据内容 hash 与配置 hash。
测试：连续跑到 20 步 vs 跑到 10 步后恢复到 20 步，最终权重、optimizer 步数、
样本流 hash、best 选择全部一致。

## E4 旧的“配对”比较不是配对（统计错误）

`fit_capacity/run_candidates.py` 的 paired 文件直接把两个聚合均值相减，两部分并不在同一组
`(start_id, candidate_seed)` 行上（各模型可行行集合不同）。本轮改为：只保留两个模型**都有
可行候选**的行，先按起点平均候选种子、再跨起点平均，并同时报告完整样本的拒绝/无解比例；
uniform 与 candidate_best 也在同一子集上重算。旧 paired 文件作废，不进入本轮任何结论。

## E5 上一轮报告的最小更正（只用旧原始记录）

| 旧表述 | 更正 | 证据 |
|---|---|---|
| 修正协议下“训练总损失 12/12 下降” | 逐 seed 是 **11/12**：Old+Branch/256/seed0 的训练总损失从 0.0253 升到 0.0458，只有按组均值才表现为下降 | `fit_capacity_v1/.../tables/progress_table.json` |
| “三组的仅可行 chosen_true 都低于 uniform −0.361” | **不成立**：Old+Random −0.252、Old+Branch −0.166 都**高于** −0.361，只有 Old −0.525 低于；且必须注明这是**可行性筛选后**协议，与未筛选协议不可混称 | `data_upgrade_v1/.../fixed_candidates/aggregate_origval.json`（H=3,K=128,model_seed=mean） |
| 把 `best_mixed_val` 选出的 checkpoint 称作“近初始化模型” | 改为**较早的 checkpoint**：它们完成过训练（update 100/250/500），只是被该规则选在很早的位置；是否有能力要由指标判断，不能由位置判断 | `fit_capacity_v1/.../tables/selection_table.json` |
| “原 test 12/12 模型无成功召回” | 原 test 的**真实成功数为 0**，因此召回是 `null`（分母 0），不是数值 0；precision 同理 | `fit_repair_v1/.../holdout_metrics.csv` 与 `fit_capacity_v1/.../holdout_metrics.csv` |
| “总损失最优与物理误差最优的 checkpoint 从不相同，说明存在严重冲突” | 该比较只在**若干已保存 checkpoint** 上、只用**两套 val 的物体位置误差之和**这一个指标；“选点不同”本身不构成冲突证据，应报告指标差值 | `fit_capacity_v1/errata.md` E5 更正；本轮 `tables/selection_table.json` 记录差值 |
