# phase_c — 信息歧义机制实验（可复现入口）

全部命令都在仓库根目录用项目 venv 运行（不要新建 venv）：

```bash
PY=.venv/bin/python

# 1) 完整搜索：32 对 T1 + 32 对 T2 = 64 对，每对 8 个联合技能
$PY experiments/option_prediction_mechanism_v1/phase_c/run_phase_c.py --repeat-check
#    产物：outputs/option_prediction_mechanism_v1/phase_c/phase_c_records.json
#          outputs/option_prediction_mechanism_v1/phase_c/phase_c_table.md
#    --repeat-check 额外把 T1-03-one_sided 完整跑第二遍并逐字段比较（应 max |dQ| = 0.0）

# 2) 只重跑一对（调试用）
$PY experiments/option_prediction_mechanism_v1/phase_c/run_phase_c.py --only T1-12-symmetric

# 3) C4：有限候选信念规划参照（全部 18 个正例；默认取正例列表）
$PY experiments/option_prediction_mechanism_v1/phase_c/belief_reference.py
$PY experiments/option_prediction_mechanism_v1/phase_c/belief_reference.py T1-12-symmetric
#    产物：outputs/option_prediction_mechanism_v1/phase_c/phase_c_belief_reference.json

# 4) 独立复核（不重用搜索循环；从记录里重建参数与技能并重算）
$PY experiments/option_prediction_mechanism_v1/verify_c_positive.py
#    期望最后一行：VERIFY_C_OK
```

模块职责：

| 文件 | 作用 |
|---|---|
| `mechanism_scene.py` | 长矩形机制 `SceneSpec` 与 per-candidate `Continuous2DConfig`（物体尺寸 + 窗口长度） |
| `perception.py` | 声明的传感模型（前向 ±60° 锥、0.6 m / 72 射线）、原始扫描与 patch、观测同一性比较（缺失量记 mismatch，不返回 inf） |
| `actions.py` | 参考点锚定的 8 技能集合（`along_*`、`spin_*` / `lower_*`、`upper_*`） |
| `run_phase_c.py` | 候选网格、收益矩阵、`δ=0.02` 近优集合、正例判据、可重复性检查 |
| `belief_reference.py` | C4 的 all-knowing / common / legal observation-conditioned 参照与 post-skill-1 可区分性 |

结果与解释见 `docs/option_prediction_mechanism_v1/C_ambiguity.md`（数字摘要
`C_ambiguity.json`）。本目录**不修改** `src/coimagine/`，也不修改 phase_a / phase_b。
