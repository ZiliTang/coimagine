# STATUS — local_position_push_v2（范围收紧版）

本轮范围按用户 2026-10-13 的收紧要求执行：**交付环境复现状态、A1 分类审计、B 初步结果后停止**。
A2 只保留基于现有日志的门槛/预算反事实；C 未开始。

## 完成

| 项 | 状态 | 产物 |
|---|---|---|
| 环境复现 | **完成** | `ENVIRONMENT.md`；commit `6c7c177`，Python 3.10.12 venv，pymunk/torch 等已装 |
| 小规模基线复现 | **完成** | v1 dev 384 条：基准 48 子集 V 23/24、X 21/24，与已提交一致 |
| 规定回归测试 | **完成** | `pytest -m "not slow"`：426 收集 / **417 通过 / 9 失败**；9 项全部因缺少未入库的 `outputs/grid/world_model/*` |
| **A1 分类审计** | **完成** | `A1_classification.md` + `A1_tables.{md,json}` + `experiments/local_position_push_v2/audit_a1.py` |
| **A2 仅日志反事实** | **完成（有限）** | `A2_counterfactual.{md,json}` + `analyse_a2_counterfactual.py`，适用条件已写明 |
| **B 初步结果** | **完成（初步）** | `B_transfer.md` + `B_transfer_{V,X}.{md,json}` + `analyse_b.py` |
| 本轮代码改动 | 完成 | `CONTRACT_revision.md`；正式协议未改，新增项全部默认关闭 |
| errata | 完成 | `errata.md`（对上一会话 4 处结论的确认/收紧/再修正） |

## A1 一句话结论

判稳率 V 0.674 / X 0.617（判据决定集 0.761 / 0.768）。**每个未判稳退出在退出瞬间和整个 0.30 s
保持窗里真值都是静止的**（V 313/313、X 368/368），判稳了但真值仍在动 = 0。失败集中在
`noise` / `delay` 两个传感压力变体（均 0/120），原因是**估计量的分辨力**：真值静止时估计量中位
读到 0.029–0.030 m/s（无噪变体只有 0.0032–0.0036，`noise` 变体 0.033–0.037），骑在 0.02 门槛上。

## 未做 / 明确停在

- **A2 主动制动**：未采纳（代码里有默认关闭的 `brake_hold_force` 路径，见 `CONTRACT_revision.md` 第 2 节）；
- **正式门槛**：未改（保持 0.02 m/s）；
- **90% 目标**：未作为必须达成项；
- **C 楔形接触**：**未开始**，无需保留的部分实现；
- 世界模型训练、不确定性感知规划、新动画：未做。

## 已知缺口

1. 9 个回归测试需要 `outputs/grid/world_model/*`（旧轮次未入库产物）。恢复要重跑旧轮次世界模型
   训练，见 `ENVIRONMENT.md` 第 5 节。
2. `docs/local_position_push_v1/settle_audit.md` 的 B/C/D 节与已提交代码不自洽（旧版估计量），
   见 `errata.md` 第 8 条。
3. `settled_after_abort` 是死代码（结构性恒 False），本轮只记录未修。
