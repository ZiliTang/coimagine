# 复现命令

```bash
# 定向测试（含零间隙等价性）
python -m pytest tests/test_local_position_push.py -q
python -m pytest tests/test_local_contact_option.py tests/test_local_contact_option_repair.py -q

# C：dev 执行验收（24 root x 8 变体 x 2 臂 = 384，基准 48 次子集单独报告）
python experiments/local_position_push/run_pair.py --split dev --workers 12

# C：test 配对实验（120 root x 8 变体 x 2 臂 = 1920）
python experiments/local_position_push/run_pair.py --split test --workers 12

# C：接续（12 root x 3 技能 x 2 臂 = 24 条链）
python experiments/local_position_push/run_continuation.py --workers 6

# C：六类指标 + root 级配对 bootstrap
python experiments/local_position_push/make_metrics.py

# C：三个必交时序诊断（逐周期曲线 + CSV）
python experiments/local_position_push/make_timing_diagnostics.py

# D：16 单臂 + 8 并排 + 8 GIF + 离线索引（1x 实时）
python experiments/local_position_push/make_animations.py

# D：两个 ZIP
python experiments/local_position_push/make_bundle.py
```

依赖：与上一轮相同，另需 `imageio-ffmpeg`（提供 H.264 编码器；系统 PATH 中没有 ffmpeg）。

运行产物目录 `outputs/local_position_push_v1/local-position-push-v1/`：
`executions_{dev,test}.jsonl`、`metrics_test.json`、`tables.md`、`continuation.json`、
`roots_{dev,test}.json`（含 `frozen_config` 与扰动定义）、`logs/{dev,test}/`、
`timing/`、`figures/{singles,pairs,gif}/`、`animation_index.{csv,json}`、`index.html`。
