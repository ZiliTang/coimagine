# STATUS — local_contact_option_v1

run_id `local-contact-option-v1`。四个阶段全部完成；代码可执行，无 BLOCKED。

## A 实现事实与信息边界 — 完成

- `A_implementation_audit.md/json`：12 项事实逐条给出文件与函数名，`all_match_task_description = true`。
- `LocalContactObservation` 白名单 + 理想局部扫描（0.6 m / 72 线 / ≥3 点才拟合）。

## B 控制律与 option 协议 — 完成

- `LocalSurfacePush`、法向滤波与跳变判据、参考速度限幅、有界速度反馈、APPROACH/READY/PUSH/
  BRAKE/SETTLE 阶段机与 14 个显式退出。
- 开发集选 Kv：20/40/80 全部满足验收（retention 0.9375、无穿透、无振荡）→ 冻结最小增益 **Kv=20**。

## C 配对实验 — 完成

| 项 | 数值 |
|---|---|
| 开发 root | 24（每类 4） |
| 测试 root | 120（每类 20） |
| 变体 | 8（基准 / 2 mm / 10 mm / 0.02 / 0.05 / 噪声 / 100 ms 延迟 / 质量·摩擦 ±10%） |
| 测试执行 | **3840**（120×8×4），另开发 768 |
| 接续测试 | 12 root × 3 技能 × 2 臂 |
| 墙钟 | 采集与仿真合计约 2 min（12 worker），报告与交付约 15 min |

## D 门槛（测试前冻结，未改阈值）

| 门槛 | 结果 |
|---|---|
| G1 信息隔离/力限幅/无瞬移/计时/快照隔离 | 通过（16 项测试） |
| G2 N 有效推动 2 s 比例 ≥0.90 | **0.925 通过**（分母 80） |
| G3 假稳定 ≤5%，并报覆盖率 | **0.0% 通过**（897 次判稳，覆盖率 92.8%） |
| G4 N 位移中位数 ≥ P 的 80% | **0.437 未通过**（匹配命令速度 0.26 时 0.658） |
| G5 扰动终态 P90 下降 ≥20% | **55.6% 通过**（0.0126 → 0.0056 m） |

## 交付物

`docs/local_contact_option_v1/`、`experiments/local_contact_option/`、
`src/coimagine/local_contact_option/`、`outputs/local_contact_option_v1/local-contact-option-v1/`
（`executions_{dev,test}.jsonl`、`logs/`、`metrics.json`、`tables.md`、`continuation.json`、
`roots_*.json`、`dev_kv_selection.json`、`figures/*.gif`）、`tests/test_local_contact_option.py`、
`local_contact_option_v1_review.zip`。

## 下一步（只一项）

保留分阶段协议；先用参考速度/力量级把 G4 拉回 ≥0.80，再采小规模新 option 数据。
