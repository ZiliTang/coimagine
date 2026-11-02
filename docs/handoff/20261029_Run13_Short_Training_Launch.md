# Run13：生命周期修复后的30分钟短训

2026-10-29 13:04:35（北京时间）启动。本轮由用户明确授权，从头训练，不续用 run12 权重。

任务目录：`outputs/option_rl_v13/short_lifecycle_20261029_130435/run13_001/`。

## 固定配置

- 30分钟训练预算，8个采样worker，每轮8个完整episode；90秒模拟任务预算。
- seed=0，逐episode动态生成训练根；64个固定validation，每10轮评估，首轮也评估；每20轮和首轮生成回放。固定test仅保留，不参与本轮调参。
- 46项持续模式、822维输入、共享两层256 MLP及双机器人因子化动作头、共享状态价值头。
- 高层5 Hz、控制20 Hz、物理120 Hz；使用已验收的run13失败/切换修复。
- PPO：学习率3e-4、4 epochs、minibatch256、clip0.2、value系数0.5、entropy0.02、gamma1；完整回报Monte Carlo advantage。
- 任务reward、物理参数、初始化分布保持；未加入GAE、学习式历史编码器或世界模型。
- 启动前复核源码与 `smoke_001/manifest.json` 的全部冻结指纹一致。

本轮先检验冻结执行修复的可学性，不同时调整熵、奖励或网络。原始成功率包含初始在目标内的场景，主判断使用目标外成功率、进度与行为。

## 观察与收尾

重点看持续推动/接触、切换频率，以及真实失败后是否通过新方向或换位恢复有效动作。Dashboard单列本机失败、等待同伴、感知等待及其并集，防止仅将失败改名为等待后误判改善。

本地supervisor每10秒检查一次，约每分钟保存监测摘要。保留原有非有限数值、心跳、持续物理异常、极端KL及熵崩溃停止规则；不会因为早期成功为零直接停训。预算到达后在当前迭代收尾保存退出；不自动延长或开启下一轮。

- [实时dashboard](../../outputs/option_rl_v13/short_lifecycle_20261029_130435/run13_001/dashboard.html)
- [本地监测摘要](../../outputs/option_rl_v13/short_lifecycle_20261029_130435/run13_001/monitor_summary.md)
- [行为回放](../../outputs/option_rl_v13/short_lifecycle_20261029_130435/run13_001/episodes.html)
- [启动记录](../../outputs/option_rl_v13/latest_launch.json)

短训完成后的判断应与run12按episode数、模拟时长和墙钟预算共同对齐，不能仅比较一个成功峰值或将固定验证反复测量当成独立泛化样本。
