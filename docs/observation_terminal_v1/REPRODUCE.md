# REPRODUCE — observation_terminal_v1

```bash
cd /home/tzl/repo/coimagine
PY=.venv/bin/python
OUT=outputs/observation_terminal_v1
TIER2=outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/phase_b_tier2

# A3-1) 先核验 16 条重放（覆盖四种退出类别），再批量重放 5120 条既有执行
$PY experiments/observation_terminal_v1/audit_labels.py --verify 16      # -> VERIFY_REPLAY_OK
$PY experiments/observation_terminal_v1/audit_labels.py --all --workers 12  # ~48 s

# A2/B1) 八项合成与手算检查
$PY -m pytest tests/test_observation_terminal_v1.py -q

# B2/B3) 32-root 管道检查，然后一档三种子正式训练（同一命令同时评价 P0/P1）
$PY experiments/observation_terminal_v1/train_terminal.py --check --check-roots 32 --epochs 200 --seeds 0
$PY experiments/observation_terminal_v1/train_terminal.py --train --epochs 200 --seeds 0,1,2  # ~40 s

# C) 12 条固定抽查、单 test root 八动作转储、归一化导出
$PY experiments/observation_terminal_v1/deliverables.py

# A1) C 评分入口一致性（不重新仿真）
$PY experiments/option_prediction_mechanism_v1/phase_c_frozen/verify_scoring.py  # -> VERIFY_C_SCORING_OK

# 构包并自检
$PY experiments/observation_terminal_v1/make_bundle.py
```

**未打包的大文件与恢复方式**（`MANIFEST.md` 列出路径、大小与 sha256）：

| 文件 | 大小 | 恢复命令 |
|---|---:|---|
| `$OUT/terminal_observations_train.jsonl` | 10.5 MB | `audit_labels.py --all --workers 12`（48 s，不重新仿真物理以外的东西，只重放既有执行） |
| `$OUT/terminal_observations_val.jsonl` | 1.30 MB | 同上 |
| `$TIER2/executions_train.jsonl` 等 | 27 MB | 上一轮 `collect_b.py`（**不重跑**，仓库内已存在） |

检查点（1.2 MB）与归一化参数**已直接打包**。
