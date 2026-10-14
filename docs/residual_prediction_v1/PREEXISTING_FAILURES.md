# PREEXISTING_FAILURES — 既有 9 个失败测试的逐条记录

任务书第 8 节要求列出名称、错误与基线复现结果。**基线版本 `dd6c7e2`（本轮开始前的提交）
在完全相同的两个文件上同样得到 `9 failed, 20 passed`**，因此这 9 个失败与本轮的数据、时间、
mask、预测接口**无关**。按任务书"无关失败记录后不扩大任务去修复"，本轮不改它们。

复现命令：

```bash
# 当前工作树
.venv/bin/python -m pytest -m "not slow" -q
# 基线提交
mkdir -p /tmp/base && git archive dd6c7e2 | tar -x -C /tmp/base
cd /tmp/base && /home/tzl/repo/coimagine/.venv/bin/python -m pytest tests/test_docs.py \
    tests/test_grid_selection_audit.py -q
```

| # | 测试 | 错误 | 基线 `dd6c7e2` |
|---|---|---|---|
| 1 | `test_docs.py::test_image_links_resolve[CoImagine_2D_世界模型_v1.md]` | `AssertionError`：该 md 链接到 5 张缺失图（`branches_showcase_t00.png`、`imagined_vs_real_t00.png`、`rollout_tree_showcase.png`、`belief_probe.png`、`prepare_probe.png`） | 同样失败 |
| 2 | `test_docs.py::test_image_links_resolve[CoImagine_2D_最终结论_v1.md]` | 同上，另有 `episode_showcase.gif` | 同样失败 |
| 3 | `test_docs.py::test_referenced_figures_exist[outputs/grid/world_model/branches_showcase_t00.png]` | `AssertionError`：路径不存在 | 同样失败 |
| 4 | `...[imagined_vs_real_t00.png]` | 路径不存在 | 同样失败 |
| 5 | `...[rollout_tree_showcase.png]` | 路径不存在 | 同样失败 |
| 6 | `...[belief_probe.png]` | 路径不存在 | 同样失败 |
| 7 | `...[prepare_probe.png]` | 路径不存在 | 同样失败 |
| 8 | `...[episode_showcase.gif]` | 路径不存在 | 同样失败 |
| 9 | `test_grid_selection_audit.py::test_model_branch_ignores_hidden_truth` | `FileNotFoundError: outputs/grid/world_model/belief.npz` | 同样失败 |

根因：`outputs/grid/world_model/` 下的产物在本仓库中从未提交（是早期世界的生成物），
而两份历史文档仍引用它们。与本轮的网络、数据、时间、mask、预测接口均无关。

## 与本轮相关的测试

`tests/test_residual_prediction_v1.py`（本轮新增）**10 项全部通过**；
全量 `-m "not slow"` 为 **484 passed / 9 failed**，失败集合与基线完全一致。
