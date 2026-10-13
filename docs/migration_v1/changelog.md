# 本轮变更清单（Migration task v1，A–D）

基线：`627d5b861bc55152cef40489d7351ec8777dd53b`（main）。工作区起始 dirty 文件只有两份
未跟踪任务书。本清单只列新增；**没有修改任何旧生产文件、检查点、评分器或 `_old/`**。

## 新增源码

- `src/coimagine/diagnostics/grid_selection_audit.py` — 阶段 B 固定候选选择偏差诊断。
- `src/coimagine/continuous2d/`（新包）：`__init__`、`config`、`scenes`、`geometry`、
  `backend`（pymunk 6.9.0）、`skills`、`environment`、`candidates`、`scoring`、`mpc`、
  `data`、`render`。

## 新增脚本

- `experiments/grid_selection_audit/phase_a_check.py` — 阶段 A 验收。
- `experiments/grid_selection_audit/run_audit.py` — 阶段 B 运行器。
- `experiments/grid_selection_audit/write_manifest.py` — 生成 manifest。
- `experiments/continuous2d/run_physics_acceptance.py` — 阶段 C 物理验收 + S1 witness。
- `experiments/continuous2d/run_baseline.py` — 阶段 D 随机/depth1/depth3 与数据导出。
- `experiments/continuous2d/make_report_table.py` — 由 results.csv 生成结果表。
- `experiments/continuous2d/render_showcase.py` — 重新渲染 S0/S1 真实执行帧与 GIF。

## 新增测试

- `tests/test_continuous2d.py`（17 项，含 2 项 slow）。
- `tests/test_grid_selection_audit.py`（含一致性世界与“模型不读隐藏真值”检查）。

## 新增文档

- `docs/migration_v1/interface_table.md`、`manifest.json`、`report.md`、`changelog.md`（本文）。
- 根目录 `STATUS.md`；`README.md` 增加 “Migration task v1” 运行命令一节。

## 新增产物（`outputs/migration_v1/`，未覆盖任何旧目录）

- `phaseB-full/grid_selection_audit/`：candidates.jsonl、summary.csv、checks.json、
  conclusion.md、figures/。
- `phaseD-final/continuous2d/`：results.csv/json、summary.json、results_table.md、
  windows.jsonl、episodes.jsonl、media/（逐帧 PNG + GIF）。
- `physics_check/physics_acceptance.json`、`beam_diagnostic.log`。

## 依赖变更

- 安装并锁定 `pymunk==6.9.0`（含 cffi、pycparser），写入 manifest 的 versions。
  未升级 numpy/matplotlib/PyYAML/pytest。
