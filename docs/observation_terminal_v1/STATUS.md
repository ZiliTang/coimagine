# STATUS — observation_terminal_v1

任务书 `docs/tasks/20261014_0040_...md`。**完成后停止**，不自动修改研究主线、不追加新实验。

| 阶段 | 状态 | 产物 |
|---|---|---|
| A1 收尾（C 评分入口统一、允许负 progress、两入口一致、旧措辞修正） | 完成 | `contract_score.py`、`verify_scoring.py` → `VERIFY_C_SCORING_OK`；`../option_prediction_mechanism_v1/C_frozen_negative.md` 与 `report.md` 已更新 |
| A2 字段合同 | 完成 | `CONTRACT.md`、`FIELDS.md`（在训练前冻结） |
| A3 标签时刻 T | 完成，**5120/5120 重放一致** | `label_audit_ledger.json`、`replay_verification.json` |
| B1 八项检查 | 完成 | `tests/test_observation_terminal_v1.py`（8 项通过） |
| B2 两个基线 | 完成 | `baselines.py`（P0/P1） |
| B3 一档三种子训练 | 完成 | `checkpoints/deployable_seed{0,1,2}.pt`（97,573 参数 / 86 维） |
| C 评价 + 12 条抽查 | 完成 | `terminal_report_train.json/md`、`spot_checks.json`、`test_root_dump.json` |

## 一句话结论

三种子**同时超越 P0 与 P1**：中点二维误差 0.0376–0.0499 m（P0 0.1410 / P1 0.6323）、
法向角 2.85–3.36°（4.59 / 21.49）、时长 0.085–0.113 s（0.350 / 3.847）、退出准确率 0.971–0.973
（均 0.922）；动作差值 0.020 m（P0 0.109）、秩相关 0.73–0.75（置换≈0）。
**未超越 P0 的目标**：残余线速度（该目标动态范围≈0，无区分度）。
**弱项**：起始不可见、终态可见的机器人 0.1495 m，是可见类的 3.6×。

## 明确登记的限制

1. 归一化沿用**按组单标量**（mean/std shape `(1,)`，非逐列），几何组内米与 sin/cos 同一尺度。
2. 残余速度目标几乎无方差，本轮的该列不构成有效评价。
3. 起始不可见样本明显更难（未做输入侧专门建模）。
4. 本轮**未接规划器**，不主张可递归、分支合法或规划收益。
5. 每机器人仍有 **1141/10240** 个槽在 T 无可见片段，属真实观测缺失，不是数据缺口。

## 明确不做

新场景搜索、新数据采样、控制器调参、模型扩容、历史编码、多尺度、分布式融合、集成、
不确定性新损失、递归 rollout、规划器；C 的信息歧义场景未推进。
