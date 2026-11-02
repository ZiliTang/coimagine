# V14：先验证小任务可学性

当前阶段为开发诊断，使用8个固定训练场景和8个沿初始接触面切向扰动1cm的场景。固定场景涵盖直推、接近后推、同面双推和换面后推；其真值反馈参考控制已全部实际完成。参考控制仅作离线可解性检查，没有BC、额外任务奖励或真值策略输入。正式全随机validation/test不用于本轮选例或训练。

## 实现

- 复用冻结的v13执行器与46项动作，保留20Hz控制、120Hz物理及任务reward。旧版本文件不修改。默认actor与critic分别使用两层256宽网络，独立optimizer和梯度裁剪；初始actor/value函数与共享版本完全一致。
- 可配置MC或GAE，gamma=1，默认`lambda(dt)=exp(-dt/10s)`。成功/90秒任务终止清零bootstrap；仅截断采样块的GAE函数可显式接收bootstrap。
- 默认一次actor采样保持0.8秒，内部仍每0.2秒运行原环境接口和记录观测；新局部失败可提前结束保持，底层安全规则始终生效。整个保持块只有一个logp、一条PPO transition。保持期间不重新抽动作，不自动重启失败模式。
- 默认944维合法输入：原822维加每机器人61维执行反馈，明确实际执行/请求模式、恢复/等待、失败分类、进度守卫、跟踪质量及接触目标误差。来自本机观测/执行历史与自身里程计，没有物体真值位姿/速度或对方私有历史。
- 可选共享雷达1D编码器和真正的动作条件第二机器人头（精确联合熵）；本轮四组对照统一使用flat编码器、独立合法动作头，便于归因。学习式历史/GRU和世界模型训练尚未接入。
- 场景固定但训练采样随机数按episode occurrence持续变化。每轮rollout使用同一冻结策略/normalizer，PPO之后才一次性更新统计。无跨版本续训。
- Replay以5Hz保存合法观测、实际dt、执行动作、reward和`actor_decision`标记。每31个episode occurrence选一条、最多256条，避免与8场景循环对齐导致只保存一类场景。PPO仅使用本轮数据。
- Dashboard每轮更新，包括固定训练/扰动成功、progress、梯度、GAE目标尺度、KL停止、实际actor样本和密集记录数。固定训练逐根保留情况在evaluation原始JSON中，不能用随机训练成功率替代确定性保留。

## 入口

```bash
.venv/bin/python -m experiments.option_rl_v14.learnability \
  --out outputs/option_rl_v14/learnability_NEW
.venv/bin/python -m pytest tests/test_option_rl_v14.py
.venv/bin/python -m experiments.option_rl_v14.short_suite \
  --out outputs/option_rl_v14/ablation_NEW \
  --panel outputs/option_rl_v14/learnability_002/panel.json \
  --iterations 40 --seconds-per-variant 240 --workers 8
.venv/bin/python -m experiments.option_rl_v14.analyze_suite \
  --batch outputs/option_rl_v14/ablation_002
```

`short_suite`依次只改一个主要设置：共享+MC+0.2秒 → 分离+MC+0.2秒 → 分离+GAE+0.2秒 → 分离+GAE+0.8秒。各组统一使用扩展输入、熵0.02、target KL0.03和原奖励；首组是本开发测台基线，不是run13精确复现。每组有独立目录、预算和故障停止规则，不自动续为长训。

当前`ablation_002`为四组并发、各4个采样进程（总16），每组最多40轮/4分钟。实际分母、墙钟与未完成情况按结果记录。单seed和小集合不足以作组件普遍优劣或全随机泛化结论。

首轮已结束，33项聚焦测试通过。详情见[实现与诊断报告](../../docs/handoff/20261029_Run14_Learnability_and_Pipeline_Update.md)。保持时间改变也会改变每物理秒施加熵正则的频次，不把此对照称为完全隔离了正则影响的纯时间效应。独立`capacity_probe.py`使用参考控制的特权离线标签，仅检查合法当前帧输入能否表达成功行为，其权重从未接入PPO。
