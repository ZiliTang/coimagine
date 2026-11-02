# run11：快速 Option / 62 动作 / 联合 FOLLOW 约束

独立于run10的新实验入口，从头训练；不可续用68动作checkpoint。

- 训练场景每episode attempt按新ordinal动态生成；64固定validation、64固定test，内容哈希审计。
- 62动作，type(14)/scale(5)稀疏编码，WAIT仅2s，FOLLOW仅配对PUSH。
- Policy先组合两个机器人的type/scale分布，再对合法联合动作归一化；采样P(a0)P(a1|a0)，PPO使用相同条件概率，熵为精确联合熵。可行性来源仅为本地观测及公开动作类型。
- 集中共享两机器人本地观测，不作B-bit部署或分布式策略声明。
- PUSH参考上限0.32m/s，准备上限0.30；单PUSH采用自身剩余法向行程渐减速。双PUSH保留联合比例调速，不在单方加入独立减速。旋转保留先前运动参数。
- FOLLOW与完成后接触随动速度上限配套提高到0.32，其余完成/安全/力/加速度/物理参数保留。这个协同适配与单机器人推荐参数的范围差异明确记录。
- 120Hz物理，20Hz控制/72射线扫描，NumPy批量扫描、record=False快路径。
- gamma=1，固定成本，无BC/课程；每轮8个完整episode，默认8持久worker。首批串行归一化预热，之后每批固定归一化快照。
- 训练曲线每轮更新；第1轮及每10轮完整验证；第1轮及每20轮行为回放。
- 除回报/成功/进度/PPO/奖励/异常等曲线，新增停滞退出、长option比例、option均值/p95物理耗时、采样控制周期/墙钟秒。

```bash
.venv/bin/python experiments/option_rl_v11/supervise.py \
 --job outputs/option_rl_v11/JOB_NEW --hours 12 --workers 8 \
 --episodes 8 --eval-roots 64 --eval-every 10 --render-every 20
```

supervisor在本机独立运行，每15s刷新dashboard、每分钟保存观测摘要、每小时保存阶段报告。
故障规则沿用经验证的run10：非有限数值、心跳停滞、高异常率、连续过大KL、熵坍塌或源码改变时请求停止并保留checkpoint。
不把早期零成功当作故障，不自动修改奖励或隐瞒异常。该机械监控不等于未来LLM复核承诺。

STOP：在run11_001目录写STOP.json，训练器在回合边界停止并保存。原始rollouts、全根验证（含异常根）、检查点及源码快照均保留。
