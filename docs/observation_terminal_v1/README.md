# 审查者导读 — observation_terminal_v1

1. `report.md` — 主报告（六问，≤1500 字）。
2. `CONTRACT.md` / `FIELDS.md` — 训练前冻结的协议与字段合同（含输入/目标/单位/坐标系/时刻/
   来源/mask/归一化/split/哈希/三种子/基线/停止判据）。
3. `STATUS.md` / `ENVIRONMENT.md` / `REPRODUCE.md` — 状态、环境与可运行复现命令。
4. `terminal_report_train.md` / `.json` — 基线与三种子的完整评价（含分组、动作差值、置换对照）。
5. `spot_checks.json` — 12 条固定哈希抽查；`test_root_dump.json` — 一个 test root 的八动作
   输入/标签/预测。
6. `checkpoints/` + `normalisation.json` — 三个正式检查点与归一化参数（**已直接打包**）。
7. `label_audit_ledger.json` / `replay_verification.json` — 标签时刻审计与重放一致性总账。

**核心结论**：可观测终态可被有效预测——三种子同时超越 P0（起始持久化）与 P1（命令分组统计），
并捕捉动作条件变化；**但**残余速度目标无区分度、起始不可见样本明显更难，
且本轮**不支持**任何关于可辨识性、分支合法性、可递归或规划收益的主张。
