# 环境复现状态（新 PC）

仓库：`/home/tzl/repo/coimagine`。基线提交 **`6c7c177`**（`0c1f3a3` 前缀见下），`main`，与
`origin/main` 一致。本文件记录这台机器重建环境的过程、结果和**已知缺口**。

## 1. 基础提交

| 项 | 值 |
|---|---|
| commit | `6c7c1770c1f3a344d57991c7d12765a75c563471` |
| branch / remote | `main` / `https://github.com/ZiliTang/coimagine` |
| 拉取方式 | 本机 HTTPS 可达（未复现上一台机器 "github.com 不可达" 的现象，remote 保持 HTTPS） |

## 2. 环境重建（本机实际执行的顺序）

这台机器**没有** `python3-venv` 的 `ensurepip`，`python3 -m venv .venv` 会在最后一步失败
（"ensurepip is not available"）。实际采用的顺序：

```bash
python3 -m venv .venv                                   # 建出的 venv 没有 pip（预期）
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py                        # 只有这一步需要网络
.venv/bin/python -m pip install -e '.[dev,video]'                    # pymunk 等核心依赖
.venv/bin/python -m pip install -e '.[deterministic-grid]' \
    --extra-index-url https://download.pytorch.org/whl/cpu           # torch（测试收集需要）
```

`pyproject.toml` 里的 `[video]`（`imageio-ffmpeg`）与 `[deterministic-grid]`（`torch`）都装了。
`torch` 是**测试收集的硬依赖**：`tests/test_grid_audit_v2.py` 等 5 个文件在 import 阶段就需要它，
不装则 pytest 直接 collection error。

## 3. 实测版本

| 包 | 版本 |
|---|---|
| Python | 3.10.12（`/usr/bin/python3`；**不是**上一台机器的 anaconda 3.11.4） |
| venv | `/home/tzl/repo/coimagine/.venv`（`include-system-site-packages=false`） |
| pip | 26.2.1 |
| numpy | 2.2.6 |
| matplotlib | 3.10.9 |
| pymunk | 6.9.0（== `pyproject.toml` 的钉版） |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| torch | 2.14.1+cpu |
| imageio-ffmpeg | 0.6.0 |
| coimagine | 0.1.0（editable） |

注意 Python 版本与上一台机器不同（3.10 vs 3.11.4）。目前没有观察到由此产生的行为差异：
下面第 4 节的小规模基线复现与已提交结论**逐位一致**。

## 4. 小规模基线复现

`pip install -e '.[dev]'` 之后先跑相关测试，再跑规定的回归测试。

**定向测试**（新代码）：

```bash
.venv/bin/python -m pytest tests/test_local_position_push_v2.py -q     # 10 passed
.venv/bin/python -m pytest tests/test_local_position_push.py -q        # 18 passed
```

**小规模执行复现**（`outputs/` 不在 git 里，必须自己重跑）：

```bash
.venv/bin/python experiments/local_position_push/run_pair.py --split dev --workers 12
```

结果与已提交的 `docs/local_position_push_v1/STATUS.md` 一致：

| 指标 | 已提交 | 本机复现 |
|---|---|---|
| dev 执行总数 | 24 root × 8 变体 × 2 臂 = 384 | 384 |
| 基准 48 次子集，V 判稳 | 23/24 | **23/24** |
| 基准 48 次子集，X 判稳 | 21/24 | **21/24** |

`--split test`（1920 条）也重跑了，判稳率 V **0.674** / X **0.617**、未判稳 V **313** / X **368**，
与 `docs/local_position_push_v1/settle_audit.md` 的 A 节逐位相同（见 `errata.md` 关于 B/C/D 节
为何不同的说明）。

## 5. 规定的回归测试

```bash
.venv/bin/python -m pytest -m "not slow"
```

| 项 | 数值 |
|---|---|
| 收集 | **426**（416 原有 + 10 本轮新增） |
| 通过 | **417** |
| 失败 | **9** |
| deselected | 21（`slow`） |

**9 个失败与本轮改动无关，全部是"缺少未纳入 git 的 `outputs/`"**：

- `tests/test_docs.py` 8 项：引用并检查 `outputs/grid/world_model/{branches_showcase_t00.png,
  imagined_vs_real_t00.png, rollout_tree_showcase.png, episode_showcase.gif, belief_probe.png,
  prepare_probe.png}` 是否存在；
- `tests/test_grid_selection_audit.py::test_model_branch_ignores_hidden_truth`：读取
  `outputs/grid/world_model/belief.npz`。

这些是**上一台机器遗留的、从未提交过的旧轮次产物**（`docs/handoff/README.md` 第 3 节已经声明
`outputs/**` 不在 git 里）。在新 PC 上它们必然不存在。恢复它们需要重跑旧轮次的世界模型训练，
不属于本轮范围；命令见 `docs/CoImagine_2D_世界模型_v1.md` 第 264–274 行。**在缺少这 9 项输入
的前提下，其余 417 项全通过。**

## 6. 复现顺序建议

```bash
git status                      # 必须干净
python3 -m venv .venv           # 若 venv 失败见第 2 节
#  bootstrap pip + pip install -e '.[dev,video,deterministic-grid]'（torch 用 CPU 源）
.venv/bin/python -m pytest tests/test_local_position_push_v2.py -q
.venv/bin/python experiments/local_position_push/run_pair.py --split dev  --workers 12
.venv/bin/python experiments/local_position_push/run_pair.py --split test --workers 12
.venv/bin/python experiments/local_position_push_v2/audit_a1.py --split test
.venv/bin/python experiments/local_position_push_v2/analyse_a2_counterfactual.py --split test
.venv/bin/python experiments/local_position_push_v2/analyse_b.py --split test --arm V
.venv/bin/python -m pytest -m "not slow"
```
