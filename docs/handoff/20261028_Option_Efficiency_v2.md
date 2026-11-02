# Option 执行效率优化 v2（2026-10-28）

本轮目标是降低之后反复调整 RL 的采样成本。已完成计算快路径、平移 PUSH 准备阶段平滑、无进展退出和短 WAIT 动作表的实现，并完成局部质量对照。没有启动新一轮 RL。

## 结果与适用范围

主测台 19 个场景、两组共 38 个分支，48.72 秒完成，无未完成分支。基线成功完成主动技能的 16 个场景，新组全部保留完成并通过预先记录的相对质量门槛；另外 3 个基线失败场景仍然失败。补充的混合方向/同面测试中，3 个基线完成场景全部保留完成并通过补充门槛，1 个对向推动失败场景仍失败。这些是局部控制回归证据，不代表随机全任务成功率。

| 场景 | 基线仿真时间 | 新组仿真时间 |
|---|---:|---:|
| PUSH 0.1 m + FOLLOW | 6.20 s | 5.55 s |
| PUSH 0.3 m + FOLLOW | 8.50 s | 7.85 s |
| PUSH 0.5 m + FOLLOW | 10.85 s | 10.20 s |
| PUSH + 原 WAIT 24 s 对应的新 WAIT 2 s | 24.00 s | 6.50 s |
| REPOSITION 0.8 m + WAIT | 6.05 s | 5.80 s |
| REPOSITION 1.8 m（两个方向） | 12.40 s | 12.40 s |
| 单机器人旋转 ±0.1 rad | 5.90 s | 5.90 s |

WAIT 行是显式改变动作表后的比较，不是同一个“等待 24 秒”请求的等价提速。新组 WAIT 到期后仍制动保持到同伴主动技能结束；不会在同伴推动时直接终止整个联合技能。

## 计算成本下降

在同一常驻进程中预热、交替顺序重复 3 次，比较中保持动作、控制器、物理周期和任务时间一致；以下为墙钟时间中位数。没有包含 PPO 网络更新、多进程通信或渲染。

| 比较范围 | 优化前 | 优化后 | 采样速度比 |
|---|---:|---:|---:|
| 相同批量扫描，仅记录/诊断快路径：PUSH + FOLLOW | 0.235 s | 0.190 s | 1.24× |
| 相同批量扫描，仅记录/诊断快路径：固定 90 s 等待轨迹 | 1.985 s | 1.563 s | 1.27× |
| 标量扫描/旧记录路径 → 批量扫描/快路径：同一 PUSH + FOLLOW | 0.344 s | 0.189 s | 1.83× |
| 标量扫描/旧记录路径 → 批量扫描/快路径：固定 90 s、1800 控制周期 | 3.293 s | 1.511 s | 2.18× |

以上比较验证了物理轨迹前缀、边界状态和反馈一致。单个 PUSH 的 Python 分配峰值由 7,822,222 降至 376,760 字节（tracemalloc，不是整个进程 RSS）。不能据此宣称整轮 PPO 也提速 1.8–2.2 倍；需要下一轮训练另行测量。

## 实现调整

1. **采样时跳过无用数据构造。** `record=False` 不再逐控制周期生成完整日志行和未使用的策略观测向量；保留决策边界观测、累计奖励、退出和安全事件。物理层提供 `step_control(minimal=True)`，跳过重复的完整原生观测、接触明细及逐碰撞对调试统计；仍计算穿透、有限值、越界和成功保持计数。完整记录接口继续可用。
2. **保持真实扫描，改用 CPU 批量求交。** 新 profile 显式选择 `numpy_batch`，仍为 72 条本体固定真实射线、1.2 m 范围、20 Hz 扫描；没有用插值伪造新观测。标量后端继续用于历史复现。
3. **平移准备阶段更平滑。** 保留法向准备速度 0.15 m/s、切向上限 0.30 m/s；将原来的额外 3 cm 准备退让开关改为随切向误差/速度连续变化，减少临界点的靠近—退回。接触后加入切向速度阻尼 0.5，抑制准备阶段残余横向速度。旋转保留原控制参数，仅启用无进展检测。
4. **按有意义进展解除长时间占用。** 3 s 内必须累计改善准备误差、接近间隙或目标方向行程；抖动不能靠逐帧心跳无限续期。换阶段重新计时，就绪等待暂停计时。返回明确的 `prepare_stalled` / `translation_stalled` / `reposition_prepare_stalled` 等失败原因，供上层重新决策。
5. **独立的新动作表。** `effect-options-v11-efficient-options` 共 62 项：前 60 个 PUSH/REPOSITION 不变，WAIT 仅保留 2 s，FOLLOW 只有一个绑定 PUSH 生命周期的动作。PUSH+REPOSITION、双 PUSH 仍允许；不要求两机器人目标一致。

物理 120 Hz、控制 20 Hz、实际接触推动速度 0.12 m/s，以及力上限、加速度、安全阈值、技能完成定义和任务成本均保留。有效的长距离绕行没有统一砍成短超时。

## 旧训练阻塞状态重放

在旧 run10 的 3 个冻结决策边界上，仅打开无进展检测，保持旧 68 项动作表及控制配置：

| 冻结边界 | 原联合耗时 | 开启检测后 | 结论 |
|---|---:|---:|---|
| train_26309 / decision 8 | 24.00 s | 6.80 s | 准备阶段停滞提前返回，仍失败 |
| train_26976 / decision 3 | 20.70 s | 14.55 s | 后段推动停滞提前返回，仍失败 |
| train_26377 / decision 0 | 16.00 s | 16.00 s | 有持续进展的慢动作未提前截断，最终仍失败 |

这减少无效占用，不表示阻塞已被控制器全部解决。固定 90 s 任务在提前返回后还会继续决策，不能简单把单 option 节省时间换算为整个 episode 同比例提速。

## 质量门槛、负结果和局限

主测台门槛在 manifest 中记录：保留基线主动技能完成；前向位移差 ≤3 cm；侧偏 ≤max(基线侧偏+1.5 cm, 3 cm)；转角绝对值 ≤max(基线转角绝对值+2°, 5°)；穿透 ≤2 mm；机器人边缘间距 >2 cm。补充 4 场景使用保留完成、位移向量变化 ≤5 cm、转角变化 ≤5° 以及同样安全门槛，分母独立记录。

这是“相对基线不明显退化”，不是所有动作都纯平移：例如 +5° 初始倾角场景侧偏仍有 7.72 cm、转角变化约 −6.49°（基线 8.80 cm、−6.39°）；PUSH+WAIT 和不同尺度双 PUSH 也可能产生明显转动。该限制仍需世界模型/规划建模。

早期试验全部保留：固定首个接触目标虽快但显著增大侧偏，已拒绝；将准备间隙减至 2 cm 的方案在微扰场景不达门槛，也未采用。`round_001` 有 2 个相对质量失败和 1 个未完成分支（REPOSITION find→align 时空 gap 的异常，已修复并回归），不能与最终结果混合宣称全通过。降低物理频率的早期方案也没有采用。

## 验证与入口

- 聚焦与相关回归共 **157 项通过**（66 + 91）；没有运行或宣称完整仓库测试全通过。
- 独立从原始位姿重算主测台 38 分支的法向行程，最大重构误差 4.8e−17 m；接触点投影重构最大误差 8.9e−16 m。它不替代任务成功评估。
- 浏览器检查 19 场景 × 2 组 × 2 机器人 × 3 时刻，共 228 个视图，无脚本错误、无手机横向溢出。
- 最终测台 source SHA256 与当前实现一致。旧 run10 的 `best.pt` / `checkpoint.pt` 元数据校验通过；旧训练入口仍保留原 68 项接口，没有迁移或启动训练。

新环境入口：`coimagine.random_start.efficient_profile.make_env(instance)`，动作/联合 mask/完整配置分别由 `ACTIONS`、`mask(world)`、`metadata()` 提供。下一轮 RL 应明确迁移新 codebook 和 FOLLOW 联合 mask，重新训练，不能直接把旧动作编号套用到 62 项表。

```bash
.venv/bin/python experiments/option_testbench/option_efficiency_v2_eval.py \
  --out outputs/option_testbench_v1/option_efficiency_v2/round_NEW \
  --variants baseline efficient --seconds 60 --branch-seconds 8
.venv/bin/python -m experiments.option_testbench.mixed_contact_audit \
  --out outputs/option_testbench_v1/option_efficiency_v2/round_NEW
```

最终[交互测台](../../outputs/option_testbench_v1/option_efficiency_v2/round_002/index.html)、[逐场景表](../../outputs/option_testbench_v1/option_efficiency_v2/round_002/SUMMARY.md)、[相对质量结果](../../outputs/option_testbench_v1/option_efficiency_v2/round_002/quality_comparison.json)、[补充协同结果](../../outputs/option_testbench_v1/option_efficiency_v2/mixed_holdout/summary.json)、[阻塞重放](../../outputs/option_testbench_v1/option_efficiency_v2/stalled/summary.json)、[验证清单](../../outputs/option_testbench_v1/option_efficiency_v2/round_002/validation.json)。

![效率优化测台](../../outputs/option_testbench_v1/option_efficiency_v2/round_002/efficient.png)
