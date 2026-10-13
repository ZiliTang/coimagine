# ENVIRONMENT — observation_terminal_v1

| 项 | 值 |
|---|---|
| 起始提交 | `c216be6`（工作区差异：仅新增任务书 `docs/tasks/20261014_0040_...md`，未改已冻结配置） |
| Python | 3.10.12（`.venv/bin/python`） |
| numpy / torch / matplotlib | 2.2.6 / 2.14.1+cpu / 3.10.9 |
| pymunk | 6.9.0 |
| pytest | 9.1.1 |
| 平台 | Linux x86_64 |

## 冻结配置哈希（本轮起点，未改动）

| 项 | sha256 前缀 | 路径 |
|---|---|---|
| 第二档 roots | `56202ef5718f2ba2` | `outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/phase_b_tier2/roots.json` |
| 第二档 train 执行 | `af057e682918ce0b` | 同目录 `executions_train.jsonl` |
| 协议版本 | `option-prediction-mechanism-v1/phase_b-v1` | `phase_b_tier2/ledger.json` |
| 判稳协议 | `settle_window=0.20, min_baseline=0.10, max_sample_gap=0.15` | `ledger.json` |

## 本轮新增/改动的代码

| 文件 | 类型 | 说明 |
|---|---|---|
| `src/coimagine/local_contact_option/option.py` | 改 | `ExecutionResult.terminal_observations`：在退出周期内捕获 T 的观测集（2 行 + 字段声明） |
| `experiments/observation_terminal_v1/replay.py` | 新 | 精确重放 |
| `experiments/observation_terminal_v1/audit_labels.py` | 新 | A3 批量重放与分类 |
| `experiments/observation_terminal_v1/dataset_terminal.py` | 新 | R_i0 标签与数据集 |
| `experiments/observation_terminal_v1/baselines.py` | 新 | P0 / P1 |
| `experiments/observation_terminal_v1/evaluate_terminal.py` | 新 | 本轮评价 |
| `experiments/observation_terminal_v1/train_terminal.py` | 新 | 检查与训练驱动 |
| `experiments/observation_terminal_v1/deliverables.py` | 新 | 抽查/单 root 转储/归一化导出 |
| `experiments/option_prediction_mechanism_v1/phase_c_frozen/*` | 改/新 | A1：合同评分统一到 `contract_score.py` |
| `tests/test_observation_terminal_v1.py` | 新 | B1 八项检查 |

## 预算使用

总墙钟上限 6 小时。实际：重放 48 s、32-root 检查 4 s、三种子训练 37 s、评价与交付 ~3 min，
其余为文档与核对。
