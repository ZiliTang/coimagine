# 改动文件与复现命令

## 新增源码

- `src/coimagine/local_contact_option/delay.py`（固定延迟队列）
- `experiments/local_contact_option/make_comparison.py`（同批复核对表）
- `experiments/local_contact_option/make_animations.py`（C 动画包）
- `experiments/local_contact_option/make_bundle_repair.py`（审查/动画 ZIP）
- `tests/test_local_contact_option_repair.py`（A1–A6 定向测试 16 项）

## 修改源码（向后兼容，默认行为不变）

| 文件 | 改动 |
|---|---|
| `src/coimagine/local_contact_option/control.py` | 表面速度单位/时间戳/RMS；参考速度显式初始化；`brake_force` 用限幅参考 |
| `src/coimagine/local_contact_option/sensing.py` | `LocalContactObservation.sample_time` 与表面速度诊断字段 |
| `src/coimagine/local_contact_option/option.py` | 延迟队列接入；逐机器人 READY；BRAKE 入口初始化；被动机器人制动；有效推动按区间累计；`phase` 字段；三标签拆分；终态快照 |
| `src/coimagine/local_contact_option/metrics.py` | 终态取执行器快照；推动区间/延迟/表面速度字段；阶段分解用 `phase` |
| `src/coimagine/local_contact_option/runner.py` | 按变体注入延迟与噪声；导出 `arm_outcome`/终态 |
| `experiments/local_contact_option/make_metrics.py` | A5：主/次分母、角度门槛、按 root bootstrap、覆盖率分母 960 |
| `experiments/local_contact_option/run_continuation.py` | A6：attempted/push_completed/normal、入口资格、每步位移、终态快照 |
| `tests/test_local_contact_option.py` | 4 项测试适配修复后的接口语义 |

`src/coimagine/continuous2d/`、旧控制、旧数据、旧 checkpoint、v1 结果目录均未修改。

## 复现命令

```bash
# 定向测试
python -m pytest tests/test_local_contact_option_repair.py tests/test_local_contact_option.py -q

# B 同批实验（root/扰动与 v1 逐字节相同，Kv=20 冻结）
RUN="--run-id local-contact-option-repair-v2 --out-root outputs/local_contact_option_repair_v2"
python experiments/local_contact_option/run_test.py $RUN --split dev  --workers 12
python experiments/local_contact_option/run_test.py $RUN --split test --workers 12
python experiments/local_contact_option/run_continuation.py $RUN --workers 6
python experiments/local_contact_option/make_metrics.py $RUN
python experiments/local_contact_option/make_comparison.py

# C 动画与离线索引
python experiments/local_contact_option/make_animations.py

# D 交付
python experiments/local_contact_option/make_bundle_repair.py
```

依赖：与 v1 相同，另加 `imageio-ffmpeg`（提供 H.264 编码器；`ffmpeg` 二进制本身不在系统 PATH 中）。
