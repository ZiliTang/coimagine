# RUN_REVIEW

外部审查者的最小运行说明。**本次打包未重跑测试**；下面引用的通过数字来自打包之前仓库内的
一次运行，不代表外部环境必然可复现。

## 依赖安装

- Python 3.11.4；`pyproject.toml` 声明 `matplotlib>=3.7`、`numpy>=1.24`、`PyYAML>=6.0`，
  dev extra 为 `pytest>=8.0`。仓库**没有** `requirements*.txt` 或锁文件。
- 连续环境需要 `pymunk==6.9.0`（及其依赖 cffi、pycparser），该版本未写入 `pyproject.toml`。
- 安装：

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pip install 'pymunk==6.9.0'
```

本轮打包未安装、升级或降级任何依赖，未改变实验环境。

## 最小 smoke

```bash
python -c "import coimagine; from coimagine.continuous2d import default_config; \
print(default_config().config_hash())"           # 期望 765ede7cf67318f1
python experiments/grid_selection_audit/phase_a_check.py   # 期望结尾 PHASE_A_OK
```

## 测试运行命令与 slow 标记

```bash
pytest tests/test_grid_selection_audit.py -q
pytest tests/test_continuous2d.py -q                 # 约 60 s（含 depth-3 S1）
pytest tests/test_continuous2d.py -q -m "not slow"   # 跳过两项 slow
```

`tests/test_continuous2d.py` 中标记 `slow` 的是
`test_depth_one_mpc_solves_a_free_space_scene` 与
`test_depth_three_plan_contains_a_preparation_window_on_s1`。打包前的一次运行：新文件 19 项
全部通过（含 2 项 slow）；`pytest tests/ -m "not slow"` 在仓库内 246 项通过。**这些不是本轮
重新运行的结果**，也不是外部环境的验证。

## 仅从已有原始数据再生成汇总

- Phase B：`summary.csv` 由 `diagnostics/grid_selection_audit.py:_summarise`（397）从逐候选
  记录生成。`experiments/grid_selection_audit/run_audit.py` 会**重跑**全部候选评估（约 16 s），
  不是只读汇总入口。包内**没有**“仅读 candidates.jsonl 重建 summary.csv”的独立脚本（未提供）。
- Phase D：`python experiments/continuous2d/make_report_table.py \
  --run outputs/migration_v1/phaseD-final/continuous2d` 只读 `results.csv` 生成
  `results_table.md`。`summary.json` 由 `run_baseline.py` 在运行时写出，没有独立重建入口。

## 依赖包外/未打包的资产

- 主检查点已打包：`outputs/grid/world_model/belief.npz`（Phase B 只读它）。
- 另外 6 个历史检查点（`latent_released.npz` 等）未打包，本包内容不需要它们。
- 所有脚本以仓库根为工作目录、使用相对路径 `outputs/...`；无网络依赖。
- 未打包：S1_v1 失败的 GIF（本就不存在，且按要求未重新渲染）；三窗口候选计划与 beam 中间
  预测状态（**未记录**，本轮不补日志不重跑）；`quick`/`smoke`/`phaseD-full` 等早期运行目录
  （`phaseD-full` 仅保留其运行日志以说明一次可复现性差异）。
- 需要显示 GIF/PNG 时使用 matplotlib；无 GUI 依赖。
