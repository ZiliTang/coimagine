# Run13：持续模式的失败与切换修复

从 run12 独立冻结的版本。46 项动作、822 维合法输入、5 Hz 决策、20 Hz 控制、120 Hz 物理、任务 reward 和物理参数保持。未开始长训练；`smoke_001` 仅完成两轮框架检查。

## 执行语义

- 本机已失败后，重复同一意图或只改速度仍保持失败。不会每 0.2 秒自动重启。
- 显式改 PUSH 方向或 REPOSITION 绕行方向，可以提出新尝试：先真实制动，再新建控制器，重新执行原有感知、碰撞与通路检查。这不是可行性保证。
- 一方 PUSH 失败时，健康同伴制动等待，不再被永久写成 `peer_mode_failed`；对方改变模式后，健康方可继续。自身的安全检查仍生效。
- 新联合 PUSH 配对清除旧配对的就绪、接触与恢复信号。FOLLOW 等待对方实际提交 PUSH，不继承对方旧模式的失败。
- 已跟踪面短暂丢失时制动等待，最多 `min(endpoint_max_age, 0.25 s)`；有效关联恢复后继续，持续丢失仍失败。只处理已有面的短时丢失，不放宽面拟合、端点或换面检查。恢复后的速率估计使用真实观测间隔。
- 切换继承当前实际制动器的参考速度，避免拿失败瞬间的过时参考值重新启动斜坡。
- Actor 的模式保持偏置不加在本机明确失败的状态上。正常状态仍保留原来的可学习 2.5 初值，熵系数保持 0.02。

Dashboard 新增本机失败、等待同伴、感知等待、失败或等待并集、显式改意图重试的训练/验证曲线。等待占比单列，不能通过将失败改名为等待宣称任务改善。

## 验证与限制

- 最终测台：`outputs/option_testbench_v1/continuous_recovery_v2/round_004/index.html`，11 案例 × 原版/修复版，共 22 分支。
- 两个健康参考的物理状态与力逐步一致；窄通路仍停止；修复版最大穿透约 0.408 mm，单周期参考变化不超过 0.025 m/s，独立 reward 重算误差小于 6e-17。
- 19 项聚焦测试通过，包括旧版持续模式/PPO 的回归测试。完整仓库测试未运行。
- 1000 个初始化、48 个首物理步通过；两轮 × 8 episode、8-worker PPO 冒烟通过，包含固定验证与回放。
- 冻结旧权重的 8 个诊断场景，四种变体均为 0/8；修复后的平均 progress 更差。这是控制器反事实诊断，不是重训练结论，不宣称学习效果改善。
- 完整 branch 快照仍未实现；测台从相同初态完整重放指令。旧 checkpoint 不能直接续训新接口；诊断脚本的权重载入是明确标注的离线反事实。

## 入口

```bash
.venv/bin/python -m experiments.option_rl_v13.diagnose_recovery \
  --out outputs/option_testbench_v1/continuous_recovery_v2/round_NEW --seconds 60
.venv/bin/python -m pytest tests/test_option_rl_v13_recovery.py
.venv/bin/python experiments/option_rl_v13/train.py \
  --out outputs/option_rl_v13/smoke_NEW --iterations 2 --episodes 8 \
  --hours 0.025 --workers 8 --eval-roots 8 --eval-every 1 --render-every 1
```

详细诊断与未解决项：[20261029_Run12_Failure_Diagnosis_and_Run13_Repair.md](../../docs/handoff/20261029_Run12_Failure_Diagnosis_and_Run13_Repair.md)。
