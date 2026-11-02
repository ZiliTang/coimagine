# 12 小时事件进度并行训练：启动记录（运行中）

2026-11-03，用户要求在 GPT 的 v20 事件接口之后启动 12 小时 RL，并生成类似
`outputs/option_rl_v16/resume_smoke_001/dashboard.html` 的 dashboard；同时要求用 8 或 16 个
并行进程采样，每个 worker 多采样几个 episode 再批量更新。本文记录执行侧 preflight、
新增实现、吞吐对照、实际启动配置与当前状态。**结果只是运行中观察，不是学习结论。**

## 1. GPT 最后工作的复核

- `experiments/option_rl_v20/` 的 event/action/observation 契约、联合合法 mask、接触 0 点、
  0.20 m 进度事件、超时与 WAIT 均按 `reviews/Event_RL_Integration_20261103.md` 保留，未改语义。
- `tests/test_option_rl_v20_events.py` 7 项通过。
- 用 `--epochs 1` 重跑 `smoke_003` 的完全相同配置（2 training episodes / 97 decisions / 1 update）：
  `advantage_std=0.595899224281311`、`entropy=1.3152498006820679`、`initial_logp_max_error=0`，
  与 `smoke_003` 逐项一致；确认该检查可复现，不是只写在报告里。
- `smoke_003` 的 checkpoint/manifest/source_snapshot 与 47 项聚焦测试文件保持原样；v1/v2 运行不覆盖它们。

## 2. 新增的长跑实现

保留 v20 原始 smoke runner；新增四个文件，均不自动执行：

| 文件 | 作用 |
|---|---|
| `experiments/option_rl_v20/long_run.py` | 12h 墙钟预算、动态合法训练场景、固定验证/test 面板、每轮 history/dashboard、checkpoint、精确 resume、greedy 验证 |
| `experiments/option_rl_v20/parallel.py` | spawn 常驻 worker；每轮把冻结 policy/normalizer 写成 hash 命名快照文件（不是把 7 MB 模型随每个 job 复制 32 次）；worker 只采样，PPO/normalizer/checkpoint 留在主进程 |
| `experiments/option_rl_v20/monitor.py` | 自包含 dashboard；v16 风格卡片与自动刷新，事件原因/决策间隔/接触率等 v20 指标 |
| `experiments/option_rl_v20/supervise.py` | 脱离会话的 supervisor；每 10 s 重写 dashboard，每分钟写 monitor summary，显式故障写 STOP.json，不自动重启 |

新增 `tests/test_option_rl_v20_long_run.py`（7 项）：聚合、dashboard 数据、checkpoint/resume、
故障规则、快照文件与 2-worker 真实采样。连同事件测试共 14 项通过。

采样语义：一个 iteration 采样 `episodes` 个有效 episode（跨 workers 合计），每个 episode 使用
`behaviour_seed(seed, ordinal)` 独立种子；worker 在整轮内使用同一冻结快照；主进程收集所有
transition 后做一次 PPO，再用新快照做验证。PPO batch 顺序为声明的 job 顺序，不是完成顺序。
奖励、成功判据、控制器、物理、动作表、事件阈值均未修改。

## 3. 吞吐对照（固定负载，单机 CPU，不含正式长训）

| 配置 | 每轮 episodes | 轮时（首轮后） | episode/s | 用户态 CPU |
|---|---|---|---|---|
| 单进程（pilot） | 8 | ≈20.9 s | 0.38 | ≈1 核 |
| 8 workers | 24（3/worker） | 9.4–9.9 s | ≈2.5 | ≈6.4 核 |
| 16 workers | 48（3/worker） | 13.7–14.2 s | ≈3.45 | ≈12.5 核 |
| **16 workers** | **32（2/worker）** | **8.5–9.6 s** | **≈3.6** | **≈12.3 核** |

16×32 在三种配置里 episode/s 最高；每 worker 2 个 episode，且环境每回合固定 90 s 时间预算，
回合时长差异本来就小。16 个 worker 合计 RSS 约 4.3 GB；机器 15 GB 内存，余量足够。
32-root greedy 验证在 16 worker 下约 7.4–7.6 s（两波），每 10 轮一次，开销约 6%。

## 4. 正式运行

- 目录：`outputs/option_rl_v20/event_12h_20261103_v2/`（v1 因补 `parallel.py` 源码哈希主动停止，
  12 轮记录保留在 `event_12h_20261103/`，不混入 v2 曲线）。
- 启动：2026-11-03 04:13:48（Asia/Shanghai）；预算 12 h，预计 16:13 左右。
- supervisor PID 1384284；trainer PID 1384308；16 个 spawn worker。
- 配置：seed 20261103；hidden 256；4 PPO epochs；lr 3e-4；entropy 0.02；gamma=1；
  trace_seconds=10；target_kl=0.03；每轮 32 个有效 episode；每 10 轮在 32 个固定 validation
  roots 上做 greedy 验证；32 个 test roots 保留给最后分析，不参与训练/验证。
- 训练场景：`simple-legal-v2` 动态合法根，每个 episode 一个新 index；验证/test 与训练根做
  `content_hash` 去重；warmup 64 个训练根。
- checkpoint 每轮写 `checkpoint.pt`；验证更好时写 `best.pt`；每 50 轮写 `snapshots/`；
  `history.json` 每轮原子写；`evaluations/iteration_XXXXXX.json` 保存完整逐回合验证。
- 初始（第 1 轮）验证：32 roots，success **1/32**，progress −0.00056，0 anomaly；greedy 策略
  早期大量 WAIT，训练 return 仍以噪声为主。**这不是学习效果结论。**

## 5. 监控与停止

supervisor 规则：非有限数值；心跳 >300 s；连续三轮训练异常 ≥25%；连续两次验证异常 ≥10%；
连续三轮 PPO KL >0.5；持续低熵且无非平凡验证成功；运行中 `manifest.source_sha256` 任一文件改变；
硬截止超过 600 s。触发时写 `STOP.json`，trainer 在当前回合边界停止并保存 `policy.pt` 和 `result.json`。
supervisor 不自动重启、不改奖励、不改控制器。

**运行期间不要修改** `long_run.py` 的 `SOURCES` 中任何文件（包括 `event_env.py`、`interface.py`、
`parallel.py`、env/controller 源码），否则 supervisor 会按源码漂移停止。文档、测试、dashboard 以外的
只读分析不受影响。

## 6. 链接

- dashboard：[文件](../../../outputs/option_rl_v20/event_12h_20261103_v2/dashboard.html) ·
  http://127.0.0.1:8771/event_12h_20261103_v2/dashboard.html
- [monitor summary](../../../outputs/option_rl_v20/event_12h_20261103_v2/monitor_summary.md) ·
  [training.log](../../../outputs/option_rl_v20/event_12h_20261103_v2/training.log) ·
  [plan.json](../../../outputs/option_rl_v20/event_12h_20261103_v2/plan.json) ·
  [manifest.json](../../../outputs/option_rl_v20/event_12h_20261103_v2/run/manifest.json)
- [history.json](../../../outputs/option_rl_v20/event_12h_20261103_v2/run/history.json) ·
  [逐轮验证](../../../outputs/option_rl_v20/event_12h_20261103_v2/run/evaluations/) ·
  [source snapshot](../../../outputs/option_rl_v20/event_12h_20261103_v2/run/source_snapshot/)
- 单进程 20 轮 pilot：[目录](../../../outputs/option_rl_v20/pilot_20261103/)（4.0 分钟档，仅用于接线/吞吐，不是学习证据）

## 7. 未完成 / 边界

- 没有跑 test 集最终评估；没有做与固定时间决策的成对对照；没有多 seed。
- 没有使用旧 v16 的 retention/多 worker PPO 合并协议；PPO 仍在主进程单线程完成。
- worker 快照通过本地文件共享；`parallel.py` 已加入 v2 的 `SOURCES`（v1 漏了，因此重启）。
- 运行时间到达后由 trainer 自己写 `result.json`；supervisor 再写最终 dashboard/summary。
  醒来后先看这两个文件和曲线，再决定是否继续、停止或另开新轮；不要在运行中改源码。
