# 新 Option 的集中式 PPO 入口

这是 run9 之后的新训练入口，不能加载 run9 的 28 动作权重。动作表使用
`effect-options-v9-reposition-scales`，68 个动作；控制器使用当前 PUSH、圆弧 RP、
同伴外侧绕行、静态窄通道停止、双 PUSH 随动/联合恢复。目录 v10 不是新的动作协议版本。

每侧策略为 `P(type|obs) P(scale|type,obs)`：14 类，最多 5 档，显式映射 68 个 ID。
WAIT/FOLLOW 只有四档，第五档无效。采样、PPO log-prob、最大联合概率贪心评估与回放
共用 `interface.py`；掩码使用各机器人本侧可见几何，部署信息边界仍是集中式共享双方局部观测。
没有物体真值输入，也没有 B-bit 通信部署保证。

训练场景逐 episode 动态生成；验证/测试面板分别固定，物理内容哈希检查互不重复。
物体朝向随机后按真实旋转包围范围采样，机器人候选按实际圆盘—矩形间隙拒绝重采样，
没有坐标裁剪。机器人到物体的真实边缘间隙为 0.06–0.37 m；到墙外表面和同伴至少
0.05 m。新入口在 reset 前再次拒绝重叠初态。原 run9 生成器保留。

任务仍为 4×3 空场地、完整随机目标位置、90 s、原奖励与成功条件，无课程、无 BC。
目标可能初始就在容差内；逐根记录 initial_distance，解读训练成功时须单列这一情况。

```bash
# 检查新初态与动作接口（输出目录必须不存在）
.venv/bin/python experiments/option_rl_v10/preflight.py \
  --out outputs/option_rl_v10/preflight_NEW --count 10000 --physical-roots 32 --seconds 60

# 建议先做 30–60 分钟的首轮训练；下面为 1 小时上限
.venv/bin/python experiments/option_rl_v10/train.py \
  --out outputs/option_rl_v10/run10_001 \
  --hours 1 --episodes 8 --seed 0 --eval-roots 64 --eval-every 10 \
  --entropy 0.02 --render-every 20

# 同一 checkpoint、同一固定验证面板的贪心回放
.venv/bin/python experiments/option_rl_v10/render_episodes.py \
  --checkpoint outputs/option_rl_v10/run10_001/checkpoint.pt --episodes 4
```

训练器修正了 run9 的记账问题：无折扣 return 与环境 gamma=1 一致；决策保险上限由
90 s / dt 导出，不会因短 option 在 120 次决策时提前停任务。穿透/越界/数值异常轨迹
保留在 rollouts/，从 PPO 梯度中隔离；验证仍计入完整分母，不能删根后报告成功率。
每迭代最多尝试三倍请求回合数，不能补足合法轨迹时以 insufficient_valid_rollouts 结束。
奖励权重没有改。若异常频繁，应先诊断，隔离数据本身不能证明任务已被解决。

另修正新配置下独立 FOLLOW 的计时：请求时长是维护时间，准备时间只占超时预算；
旧配置仍使用 legacy_window，不静默改变旧测台。

产物：manifest.json（完整动作表/控制器、固定面板、源码哈希）、rollouts/（每个动态
场景及结果）、evaluations/（含逐根和距离分层）、history.json、dashboard.html、
checkpoint.pt（每迭代）、best.pt（验证成功率/进度排序）、policy.pt（最终）。
checkpoint 的 history 包括保存当轮；不兼容的动作/控制器描述会在加载时拒绝。
旧 28 动作权重不能迁移；同一新版动作/控制器配置的 checkpoint 支持 --resume-from，恢复权重、优化器、归一化和场景序号。

现阶段验收与局限见
`docs/handoff/20261028_New_Option_RL_Readiness.md`。

## 夜间无人值守（2026-10-28）

已增加 `supervise.py`：在独立进程中运行训练，每 15 秒巡检，每分钟更新
`monitor_summary.md`，每小时保留 `hourly_XX.json/md`。下面的入口默认 12 小时：

```bash
.venv/bin/python experiments/option_rl_v10/supervise.py \
  --job outputs/option_rl_v10/overnight_NEW --hours 12 \
  --episodes 8 --eval-roots 64 --eval-every 10 --entropy 0.02 --render-every 20
```

实际任务在 detached tmux `coimagine_rl10_20261028` 中运行，路径为
`outputs/option_rl_v10/overnight_20261028/run10_001/`。

新 dashboard 每轮在验证开始前就写入训练指标；评估期间显示已完成验证根数。
页面每 10 秒刷新，独立监控每 15 秒重绘当前状态，文件用原子替换避免读到半页。
18 张图覆盖回报、成功率、progress、policy/value loss、熵、奖励项、异常率、
actor/critic 梯度、KL/clip、采样概率误差、动作占比、阻塞/短 option、距离分层等。
验证曲线仅在实际完成固定面板后更新；中断的部分验证不会当作完整分数。
初始在目标容差内的成功与初始在目标外的成功另外显示。

停止规则冻结在 supervision_plan.json / supervise.fault：非有限值、心跳超过 300 秒、
连续三轮训练异常率 ≥25%、连续两次验证异常率 ≥10%、连续三轮 KL >0.5，或连续
10 轮熵 <0.1 且最近两次验证无非平凡成功、progress ≤0。不因早期成功率为零停训。
检查运行中的源码指纹；源码变化也会请求停止，避免把运行中修复混成一条曲线。

监控写 `STOP.json` 请求在回合边界停止，保留 checkpoint 和原结果；长时间不响应才终止
该任务自己的进程组。已用真实短任务验证停止路径。监控故障本身也会写停止请求。
本地程序不调用模型、不自动改代码或奖励；需要修复时由代理分析原始记录，另开新目录。

手动请求停止：

```bash
.venv/bin/python - <<'PY'
from experiments.option_rl_v10.live import atomic_json
atomic_json('outputs/option_rl_v10/overnight_20261028/run10_001/STOP.json',
            {'reason':'user_requested_stop'})
PY
```


## 8 worker 并行（2026-10-28 02:58）

当前活动任务为 `outputs/option_rl_v10/overnight_parallel_20261028/run10_001/`，
tmux `coimagine_rl10_parallel_20261028`。原串行任务保留，已为性能优化正常停止。
当前使用 8 个持久 spawn worker，共用于训练和验证；每个 worker 限 1 个计算线程。
训练主进程负责 PPO，另有独立监控进程。每次更新仍为 8 个有效 episode。

```bash
.venv/bin/python experiments/option_rl_v10/supervise.py \
  --job outputs/option_rl_v10/overnight_NEW --hours 12 --workers 8 \
  --episodes 8 --eval-roots 64 --eval-every 10 --entropy 0.02 --render-every 20 \
  --resume-from outputs/option_rl_v10/overnight_20261028/run10_001/policy.pt
```

当前从原串行第 8 轮恢复，场景序号从 68 开始，旧结果不覆盖；新目录从本地第 1 轮计数。
不传 resume-from 时从头训练，先用一轮串行采样建立归一化统计，再进入并行。
并行每批共享固定策略和归一化快照；worker 不改统计，主进程在 PPO 后合并合法原始观测。
按任务提交顺序组装 PPO 数据，不按完成先后组装；每 episode 的动作随机种子由 seed/ordinal
独立派生。采样概率和 learner 重算概率继续检查一致性。

同一快照、8 个完整 episode 的实测：串行 29.49 s，8 worker 冷启动 6.73 s、热池 6.00 s，
热池约 4.92 倍。回报、动作、物理结果相同；log-prob/value 最大差 0。
详见 `outputs/option_rl_v10/parallel_benchmark_001/benchmark.json`。
新调度还改变了归一化更新时序，不能把两段学习曲线当作纯并行度消融。
