# 换机接续指引

仓库已经推到 GitHub，`main` = `314b458`（本文件所在提交）。另一台 PC 按下面三步接上。

## 1. 另一台 PC 拉取

```bash
cd <coimagine 目录>
git status                 # 必须干净；有本地改动就先 git stash 或先提交
git remote -v              # 见下方"关于 remote"
git pull --ff-only origin main
```

**关于 remote。** 本机把 `origin` 从 HTTPS 改成了 SSH：
`git@github.com:ZiliTang/coimagine.git`。原因是本机所在网络**只有 `github.com` 这一个主机不可达**
（`api.github.com` / `codeload.github.com` / `raw.githubusercontent.com` 都正常返回，SSH 22 和 443 都通），
HTTPS push 会卡死。另一台 PC 如果 HTTPS 正常，可以保持它自己的 remote 不动 —— remote 是每个 clone 的本地配置，不随 git 传输。

如果另一台 PC 也连不上 GitHub，用离线 bundle（本机已生成，1.6 MB，`git bundle verify` 通过"complete history"）：

```bash
git pull ~/coimagine-handoff.bundle main      # 把 bundle 拷过来即可，不需要网络
```

## 2. 重建环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'              # pymunk 是核心依赖，没有它连续环境跑不起来
pip install -e '.[video]'            # 只有要跑动画脚本才需要（imageio-ffmpeg 自带 H.264 编码器）
python -m pytest -m "not slow"       # 期望 416 passed
```

本机用的是 `.venv/bin/python`（Python 3.11.4，由 anaconda 的 python 创建，`include-system-site-packages=false`）。
仓库里不带 venv；`python` 不带路径时指向的是 anaconda base，不是项目环境。

## 3. 知道什么**不在** git 里

`.gitignore` 排除 `outputs/`、`data/`、`*.zip`、`.venv/`。这是仓库一直以来的约定（从未提交过任何 output 或审查包，
跟踪的最大文件是 0.10 MB）。所以另一台 PC 拉完之后：

| 不在 git 里 | 影响 | 恢复方式 |
|---|---|---|
| `outputs/**`（本机 6.2 GB） | 读历史运行产物的脚本会失败 | 重跑对应实验，见下 |
| 根目录 15 个 `*_review.zip`（116 MB） | 审查包不在 | 跑该轮次的 `make_bundle*.py` |
| `data/` | — | 按需生成 |

**重跑成本差别很大，这是最需要注意的一点：**

- 控制线（本会话的四轮）很便宜：`run_pair.py --split test` 是 1920 次执行，约 1 分钟；
  动画脚本约 8 分钟。
- 早期训练轮次很贵：`outputs/coverage_scale_v1` 有 2.9 GB，是 128k/256k 数据 × 多模型宽度的训练，
  以小时计。那些轮次的**报告在 `docs/coverage_scale_v1/` 里已提交**，结论可用；原始 run 产物如果要，
  得重跑。

各轮次的复现命令在对应的 `docs/<round>/REPRODUCE.md` 或该轮 `STATUS.md` 里；
控制线四轮的入口分别是 `experiments/local_contact_option/` 和 `experiments/local_position_push/` 下的
`run_test.py` / `run_pair.py` / `make_metrics.py` / `make_animations.py`。

## 4. 现在的状态

- 提交历史：82 个提交随本次推送上线，`main` 快进到 `314b458`，没有 force push。
- 测试：`416 passed`。
- 本会话新增 5 个提交（`d4d4a3f`、`912ad89`、`3448206`、`c94c35e`、`314b458`），把此前**从未提交**的
  若干整模块（`continuous2d`、`coverage_scale`、`local_contact_option` 等）及其实验、测试、报告一并入库。
- 下一步任务书见 `docs/tasks/20261013_1900_CoImagine_判稳可靠性与可观测转移任务书.md`；
  本会话的压缩上下文见 `docs/handoff/session_context.md`。
