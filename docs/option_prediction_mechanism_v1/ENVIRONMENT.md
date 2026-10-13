# 环境、提交与数据总账（round 3）

## 1. 起始状态与提交

| 项 | 值 |
|---|---|
| 上游基线 | `6c7c177`（上一台机器交接） |
| 本轮起点 | `aedffae`（round-2 交付）→ phase A `cad4f63` → phase B `5a8702a` |
| 分支 / remote | `main` / `https://github.com/ZiliTang/coimagine`（HTTPS 本机可达，未改 remote） |
| 起始工作树 | round-2 交付未提交，先提交为 `aedffae`，之后每阶段一个检查点提交 |

## 2. 环境（新 PC）

| 包 | 版本 |
|---|---|
| Python | 3.10.12（`/usr/bin/python3`） |
| venv | `/home/tzl/repo/coimagine/.venv` |
| numpy / matplotlib / PyYAML | 2.2.6 / 3.10.9 / 6.0.3 |
| pymunk | 6.9.0（`pyproject.toml` 钉版） |
| torch | 2.14.1+cpu |
| imageio-ffmpeg | 0.6.0（动画 H.264，无需系统 ffmpeg） |
| pytest | 9.1.1 |

这台机器**没有** venv 的 ensurepip：`python3 -m venv .venv` 最后一步失败，用
`get-pip.py` 引导 pip 后再 `pip install -e '.[dev,video]'` 与
`-e '.[deterministic-grid]' --extra-index-url https://download.pytorch.org/whl/cpu`。
本轮**未升级任何依赖**。

## 3. 数据总账

| 阶段 | 内容 | 数量 | 墙钟 |
|---|---|---:|---:|
| A dev 选型 | old / 0.20 / 0.30 三档 | 3 × 192 = 576 | ~10 s |
| A test 复核 | old + 冻结 0.20（X 不重跑） | 2 × 960 = 1920 | ~24 s |
| B 32-root 检查 | 32 train root × 8 命令 | 256 | 3.8 s |
| B 第一档 | 256/64/64 root × 8 命令 | **3072** | 35.3 s |
| B 第二档 | 512/64/64 root × 8 命令 | **5120** | 60.7 s |
| B 训练 | 2 档 × (持久化 + 2 模型臂 × 3 种子) | 2 × (1 + 6) | ~4 min |

无执行被丢弃，无 root 采样失败（B 两档均为 0）。总墙钟约 **20 分钟**，远低于 12 小时预算；
60 分钟无产物的情况未发生。

## 4. 关键产物哈希（复核用）

| 产物 | sha256（前 16 位） |
|---|---|
| `phase_b/roots.json` | `691a9a795728b2c6` |
| `phase_b/commands.json` | `efe890b019863b83` |
| `phase_b/executions_train.jsonl`（一档） | `160e9574fdb13914` |
| `phase_b_tier2/executions_train.jsonl`（二档） | `af057e682918ce0b` |
| `phase_b/executions_val.jsonl` | `711bbe2fad8e3567` |
| `phase_b/executions_test.jsonl` | `f9bc881478e6e8e2` |

两档的 val/test 哈希完全相同，所以档间比较在同一测试集上进行。

## 5. 不在 git 里的东西

`outputs/**`（本机 682 MB，主要是逐周期日志）按仓库既有约定不入库；`roots/commands/ledger/
train_report_*.json/*.md` 等**小体积结论文件已复制到** `docs/option_prediction_mechanism_v1/`。
恢复方式见 `REPRODUCE.md`：全部执行都可从已提交代码重跑，B 两档合计约 100 s。
