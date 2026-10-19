# v3：最佳模型热启动，256实例均衡覆盖

2026-10-17 用户授权停止旧训练，从最好检查点继续训练。

- 旧运行 `outputs/random_start_time_budget_v3_long` 已发SIGTERM停止进程组，保留所有结果。
- 使用其 `checkpoints/best.pt`（停止时最佳为更新1550）模型权重；复制到新目录并记录SHA256。
  best.pt不包含优化器，因此明确重新初始化AdamW；这不是逐状态恢复或从头训练对照。
- 新目录：`outputs/random_start_time_budget_v3_coverage256`。
- 256个冻结L0 train实例每轮打乱、每批8个不重复，32更新遍历一轮；顺序由seed/epoch确定，可恢复。
- 8进程、5000更新或24小时先到为止；网络、输入、奖励、90秒预算、PPO超参数保持不变。
- 每50更新评价：原8实例（确定性与每实例3种子随机）、固定32训练实例（确定性）、
  完整64验证实例（确定性）；不使用test。32训练实例按预登记seed20261017无放回选择。
- 最佳选择按验证集确定性成功率，再按最终距离中位数；热启动模型也作为更新0候选，防止丢失。
  初始化在同面板评价，便于量化热启动后的增益；反复使用验证集不代表独立测试结论。
- 每5更新恢复检查点，每30分钟监测，评价点曲线增加train_panel/validation；不因低成功率提前停止。
- 增加逐轨迹实例、种子、成功、回报、终止、真实周期、技能次数和决策损失占比日志。
  本轮未加入独立actor/critic梯度归因，避免将未做的诊断称为完成；失败决策权重问题仍待验证。
- 该轮同时改变训练分布、采样覆盖并重置优化器，不能单独将变化归因于“256场景”一项。

启动命令：
```
.venv/bin/python experiments/random_start_reposition_v1/long_train.py \
  --out outputs/random_start_time_budget_v3_coverage256 \
  --warm-start-run outputs/random_start_time_budget_v3_long
```
