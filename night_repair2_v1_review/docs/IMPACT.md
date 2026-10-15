# IMPACT — 受影响范围与旧资产处置

## 1. 受影响的结论

| 结论 | 状态 | 原因 |
|---|---|---|
| 世界模型四项返修 | **不受影响** | 缺陷在 RL 与采集侧 |
| latent 递推消融（递推有效） | **不受影响** | 评审已用包内逐样本预测独立复核 |
| 固定训练/验证面板同口径评估 | **本轮补齐** | 原先只记录随机批次损失 |
| "RL 优于同规则随机"（0.234 对 0.141） | **撤回** | 概率比 ≈0.05，PPO 更新基础不成立 |
| "扩采门槛通过（0.25 对 0.05）" | **撤回** | 依据是上一条；且无同面板对照 |
| "`net_progress` 正数 = 正向推进" | **撤回** | 代码定义 `结束 − 起始`，正数表示更远 |
| 数据规模 140,415 帧 | **撤回** | 49,794 + 138,424 = **188,218** |
| 71 个独立 root | **撤回** | 两集合来自同一 `build_roots("dev", 6)` 池，不能相加 |
| "任何策略都无法完成 solo_central" | **降级为"已测动作均未成功"** | 只测了随机、当前策略与贪心 |
| "自主终止未实现" | **保留** | 本轮仍未实现，如实标注 |

## 2. 受影响的资产

| 路径 | 处置 | 说明 |
|---|---|---|
| `outputs/night_repair_optionrl_v1/collection/`（140 episode） | **需重采** | 时间轴膨胀、字段歧义、观测历史与策略不一致 |
| `outputs/night_repair_optionrl_v1/collection_expanded/`（340 episode） | **需重采** | 同上 |
| 上述两者中的 `branch/`（合计 240 episode） | **仅诊断** | 候选标签 ≠ 实际执行（抽查 12/12） |
| `outputs/night_repair_optionrl_v1/option_rl/policy_seed0.pt` | **仅诊断** | 训练管线概率比错误 |
| `outputs/night_repair_optionrl_v1/option_rl/*.json` | **仅诊断** | 同上 |
| `outputs/night_repair_optionrl_v1/world_models/` | **可保留** | 世界模型侧正确 |
| `outputs/transformer_scale_latent_v1/` | **可保留** | 上一轮世界模型结果 |
| `night_repair_optionrl_v1_review.zip` | **保留但需标注** | 其 RL 与采集部分不可用于结论 |

**未删除、未覆盖、未改名**任何旧文件；本轮全部结果写入新目录
`outputs/night_repair2_v1/` 与 `docs/night_repair2_v1/`。

## 3. 本轮新增且可用的资产

| 路径 | 内容 |
|---|---|
| `outputs/night_repair2_v1/repair/verification.json` | 11 项机器可读检查，全通过 |
| `outputs/night_repair2_v1/option_rl/policy_best_seed0.pt` | 按验证成功率选出的推理检查点 |
| `outputs/night_repair2_v1/option_rl/policy_resumable_seed0.pt` | 含优化器/迭代号/RNG 的可恢复检查点 |
| `outputs/night_repair2_v1/option_rl/train_result_seed0.json` | 训练曲线 + 训练后/随机同面板对照 |
| `outputs/night_repair2_v1/option_rl/checkpoint_comparison.json` | 两个检查点 × 两个划分 × 随机的对照 |
| `outputs/night_repair2_v1/collection/` | 协议修复后的试采数据（时间轴与分支一致） |

## 4. 复现命令

```bash
PY=.venv/bin/python
$PY experiments/night_repair2_v1/verify.py                      # 11 项修复检查
$PY experiments/night_repair2_v1/ppo.py --iterations 40 --episodes 8 --seed 0 \
    --out outputs/night_repair2_v1/option_rl                    # RL 最小重跑
$PY experiments/night_repair2_v1/collect.py --policy \
    outputs/night_repair2_v1/option_rl/policy_best_seed0.pt \
    --episodes 20 --branch-snapshots 20 --out outputs/night_repair2_v1/collection
```
