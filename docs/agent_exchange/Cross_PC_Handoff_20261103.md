# 跨 PC 工作交接：Option、事件 RL 与经验增强 LLM 路线

> **后续更新（2026-11-03晚）：** 原训练已结束并冻结为[第一版V1基线](../rl_baselines/README.md)，最终V1迁移包为`outputs/rl_baselines/V1_event_progress_20261103.tar.gz`。当前8765已更新[Option V2](reviews/Option_V2_Yield_20261103.md)，含YIELD和PUSH方向修正，尚未训练。本文以下12:50包和权重是历史中期快照；其重复解包暂存已清理，压缩包仍保留。

冻结时间：**2026-11-03 12:50:13，Asia/Shanghai**。本文件是当前接续入口；实时状态仍以原机器上的进程与输出为准。最新讨论见[研究路线记录](../research/LLM_Option_Experience_Research_Direction_20261103.md)，阶段汇报见[导师稿](../option_optimization_and_event_rl_report_20261103.md)。

## 1. 已经完成与尚未完成

| 工作 | 当前证据 | 限制 |
|---|---|---|
| q 连续性与同段法向 | 固定八边形 800 tick，>5cm 跳变 125→0；已进入测台 | task 仍失败，不代表其他形状/遮挡均解决 |
| 删除轮廓 SupportMemory | 初选 q 只取自身当前观测；六例小对照无明显异常 | EKF 历史和锁定 q 状态保留；严重遮挡未覆盖 |
| 速度语义 | REPOSITION CW/CCW 统一 .30m/s；PUSH .10/.20/.30/.40 | 实际速度受力/阻力约束；历史 ID 兼容不等于可选动作 |
| 观测/测台性能 | 历史索引；同刻感知复用；射线归约等价优化 | 固定负载收益，不是所有场景/PPO加速承诺 |
| v20 事件 RL | 正在 16 worker、32 episode/轮采样；已有明显学习 | 单训练 seed、固定矩形分布、集中式；无固定时间公平消融 |
| 行为质量 | 仍有挡路、被推走、失去成功、停止/恢复异常 | 成功率高不等于协作行为合理 |
| LLM 经验策略 | 已保存研究问题、对照和单例验证顺序 | **未接 API、未新采数据、未实施 benchmark** |

以 option 行为合理性和计算效率为执行器优化目标，task complete 是附加评价。后续按用户要求先锁定一个失败实例，完整可视化再逐步修改；不要每想到一个点就直接跑全矩阵。

## 2. 12:50 的训练冻结证据

本次只读保存，未新训练/评测，未停止原训练。可随 Git 获取的紧凑原始证据在[assets/20261103_cross_pc](assets/20261103_cross_pc/summary.json)。

- v2 已完成 **3937 轮、125,984 个有效训练 episode、5,233,200 个事件 transition**；训练异常累计 0。
- 固定 32 根验证重复完成 394 次；不是 12,608 个独立场景。
- validation 选择的 best：**第 3500 轮 32/32**，平均相对距离改善 .82384，平均回报 +8.7456。排除初始目标内一例为 31/31。
- 本快照最近完整验证：**第 3930 轮 30/32**，改善 .68977，回报 +7.3851；排除初始目标内一例为 29/31。
- 已有中期 test：**旧 best@2580 为 26/32**，0 异常。不能将它绑定到 best@3500；后者尚无本次交接核实的 test 结果。
- 当前 test 已被查看，后续不能称完全未见或用于调参挑 checkpoint。32/32 是反复监控的 validation 峰值，不是未知动力学泛化证明。
- [导师稿](../option_optimization_and_event_rl_report_20261103.md)主体冻结在 **11:18、3124轮**，保留原曲线与数字；12:50 更新见本文件和导师稿顶部说明。

训练目录：`outputs/option_rl_v20/event_12h_20261103_v2/`；冻结 best@3500 和 latest checkpoint@3937 已另行拷贝并用 torch 读取确认，可查[checkpoint 哈希](assets/20261103_cross_pc/summary.json)。二者均是**中期**，此时还没有最终 `policy.pt`/`result.json`。

原运行 04:13:48 开始，计划 12h、约 16:13 截止。曾发现墙钟与进程/monotonic 累计时间存在不一致，故实际停止应查 STOP/result 和 supervisor，不用人工相减推断最终状态。进程不能通过 Git 或迁移包转移，也不会在新 PC 自动继续。

## 3. 接口与版本：修改前必须知道

- 控制 20Hz，单 tick .05s；事件条件包含接触保持 .15s、进度 .20m、接触丢失 .25s、获取接触超时12s、推进/绕行超时8s、全 WAIT 1s。
- PUSH 进度是接触期间自身里程计沿请求方向的投影；REPOSITION 是有符号边界进度。不是物体到目标的有效位移。
- 联合菜单允许 P+P、P+W、R+R、R+W、W+W 和角色互换；排除 P+R。35 个有效单机动作、1097 个静态合法有序组合。修改菜单会改变实验版本。
- 958维集中式输入，前760维归一化；分离 actor/critic、双256 Tanh；gamma=1、GAE时间尺度10s；实际时间累计奖励。
- WAIT 是主动制动，不能当作主动让路。PUSH 没有匹配法向时不应自动扩展成绕行；用户所见挡路尚需锁定 run/tick，区分请求、面选择、控制阶段与真实运动。
- 事件接触来自 `SkillObservation.contacting_object` 的仿真碰撞反馈；执行器 contact-ready 主要用 gap，语义不同。
- **形状先验仍未彻底剥离**：`v19 spec.mask → C.action_mask → estimate_face` 使用 `rectangle_size=(1.0,.25)`。q/control 不直接依赖完整尺寸，不代表整个策略栈无尺寸先验。
- 一次只改一个主要机制；新的接口/控制器实验与当前 v20 结果分版保存。D002 切换惩罚 B 仍冻结，不能宣布惩罚无效。

源代码已随 Git 管理。36 项训练保护文件在本次冻结时哈希均匹配、均已 tracked，详见[source_hashes.json](assets/20261103_cross_pc/source_hashes.json)。原机器训练仍存活期间不要改这些文件，supervisor 会触发源码漂移停止。

## 4. Git 与独立迁移包的分工

Git 提供当前源码、测试、报告、研究讨论、图表、12:50 紧凑证据、validation/test 面板及哈希；`.gitignore` 明确排除 `outputs/`、权重与压缩包。

**本次远端同步状态：受认证阻塞。** `git push origin main` 返回 `could not read Username for 'https://github.com'`；当前机器没有 Git credential helper、GitHub CLI、登录 token 或 SSH 密钥。文档已在本地提交，远端尚未包含本次更新。完成 GitHub 登录后运行 `git push origin main` 即可。不要将 token 写入文档或聊天。

同时生成增量 Git bundle：`outputs/handoff/20261103_cross_pc/coimagine_docs_20261103.bundle`，包含本次本地提交，以原远端 main 为前置版本。若暂时无法 push，新 PC 可先 clone 现有远端，再拷贝该 bundle 并执行：

```bash
git bundle verify /tmp/coimagine_docs_20261103.bundle
git fetch /tmp/coimagine_docs_20261103.bundle main
git merge --ff-only FETCH_HEAD
```

该操作恢复本次文档和证据；模型与回放还需下面的独立迁移包。Git bundle 也只在原 PC，不会自动出现在另一台机器。

已在**原 PC**生成可单独拷贝的迁移包：

```text
/home/tzl/repo/coimagine/outputs/handoff/20261103_cross_pc/coimagine_state_20261103_1250.tar.gz
```

准确大小/SHA256见[transfer_archive.json](assets/20261103_cross_pc/transfer_archive.json)，文件清单与逐文件 SHA256 见[transfer_files.json](assets/20261103_cross_pc/transfer_files.json)。**该包未上传 GitHub；仅 clone/pull 不会获得模型和回放。**

包内：best@3500、checkpoint@3937、对应源码快照与 manifest/面板、完整冻结 history、best/latest 原始验证、中期 test@2580、已发布六条 RL 回放、保存的测台 presets，以及导师稿11:18证据。

包内没有整个历史 outputs、旧的所有实验、虚拟环境、完整回放库，也没有训练未来才产生的最终结果。未归档的旧报告输出链接在新 PC 可能缺失，不能由报告文字代替原始数据。原训练结束后若要迁移最终版本，需要再次拷贝并校验 final `policy.pt`、best、result/history/manifest，不能把本包称作最终归档。

## 5. 新 PC 接续步骤

先同步代码，再从原 PC 复制上述压缩包到新 PC。`OLD_PC` 替换为实际 SSH 主机，不包含任何凭据：

```bash
git clone https://github.com/ZiliTang/coimagine.git
cd coimagine
# 已有仓库则先确认无未保存修改，再 git pull --ff-only。
scp OLD_PC:/home/tzl/repo/coimagine/outputs/handoff/20261103_cross_pc/coimagine_state_20261103_1250.tar.gz /tmp/
```

若没有原 PC 的远程访问，使用其他文件传输方式复制同一个包。校验后**在新 PC**解压；不要覆盖原 PC 的活跃训练目录：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
meta = json.loads(Path('docs/agent_exchange/assets/20261103_cross_pc/transfer_archive.json').read_text())
p = Path('/tmp') / meta['filename']
assert hashlib.sha256(p.read_bytes()).hexdigest() == meta['sha256']
print('archive SHA256 OK')
PY
tar -xzf /tmp/coimagine_state_20261103_1250.tar.gz -C .
python3 - <<'PY'
import hashlib, json
from pathlib import Path
for f in json.loads(Path('TRANSFER_FILES.json').read_text()):
    p = Path(f['path'])
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == f['sha256'], str(p)
print('all transferred files OK')
PY
```

`TRANSFER_FILES.json` 是归档清单，保持本地即可，不需提交。安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,deterministic-grid]' --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest tests/test_option_rl_v20_events.py
.venv/bin/python experiments/option_rl_v19/testbed_server.py --host 127.0.0.1 --port 8765
```

环境实测版本在[environment.json](assets/20261103_cross_pc/environment.json)：Python3.10.12、numpy2.2.6、torch2.14.1+cpu、pymunk6.9.0、matplotlib3.10.9。上述安装命令不锁定所有依赖，跨版本不承诺 bitwise；复现实验须对齐该环境并核对结果。

新 PC 打开 `http://127.0.0.1:8765/`。恢复回放后可查看 `?run=20261103_111051_rl06`（成功）、`rl00`（近目标未成功）、`rl24`（停滞/绕行）。回放编辑序列经过 .2s 量化，直接 Run 不等于精确回放；部分 RL 场景未注册会提示 unknown scene。前端拖拽/视觉没有完整 headless 验证。

仅检查模型/normalizer/优化器能读取，不采样或训练：

```bash
.venv/bin/python - <<'PY'
from experiments.option_rl_v20.long_run import load_state
from pathlib import Path
base = Path('outputs/option_rl_v20/event_12h_20261103_v2/run')
for name in ('best.pt', 'checkpoint.pt'):
    policy, norm, optim, generator, data = load_state(base / name, lr=3e-4)
    print(name, data['iteration'], policy.obs_dim, data['best'])
PY
```

**不要在新 PC 自动 resume 或另起12h训练。** 先确认原 PC 状态及用户希望继续什么；当前方案是调优/研究接续，不是新增训练授权。v20 提供 `--resume` 并校验配置，但跨机器并行确定性、全部环境依赖、剩余预算和原运行是否已终止需另行检查。

## 6. 下一会话应该先做什么

1. 读本文件、CURRENT、导师稿及新的[LLM路线讨论](../research/LLM_Option_Experience_Research_Direction_20261103.md)，核实本地源代码/权重/面板齐全。
2. 确认原12h训练是否结束；有最终结果再单独报告 best/latest/test，保留中期截点与退化实例。
3. 请用户在测台选定一条“反面阻挡”失败轨迹，以完整中间量区分选择问题与执行问题；先验证可行序列，再决定动作组合是否需要变化。
4. 对齐冻结 LLM 的观测、动作、调用预算与经验表示对照，再另写小实验配置。研究讨论不等于已有 API 实现或允许自动扩大数据采集。

当前由 GPT 与用户直接调测台，DS 人工交接协议仍适用于交给 DS 的任务；不自动给其他 agent 发消息。历史报告与负结果保持原样。
