# 交接：Option 小场景测台与三轮调优（2026-10-19）

本文交接给规划端/下一位执行 agent。范围：option 小场景测台（Round 000–002）的设施、
核心改动、实测结论、下一版提案与使用说明。**没有启动 PPO、世界模型或规模测试。**

相关文档：

- 任务书：[`docs/tasks/20261019_CoImagine_Option_Testbench_HumanLoop_v1.md`](../tasks/20261019_CoImagine_Option_Testbench_HumanLoop_v1.md)
- 测台契约与命令：[`experiments/option_testbench/README.md`](../../experiments/option_testbench/README.md)
- 仓库 README 的复用入口：[`README.md`](../../README.md) 的 “Option small-scene testbench”
- 设计备忘（RL 输入编码）：[`20261019_RL_Lidar_Encoding_and_InputSchema_Note.md`](20261019_RL_Lidar_Encoding_and_InputSchema_Note.md)
- 设计备忘（macro-action/时长维度）：[`20261019_Option_Duration_MacroAction_Design_Note.md`](20261019_Option_Duration_MacroAction_Design_Note.md)
- 每轮结论：`outputs/option_testbench_v1/round_00{0,1,2}/REVIEW.md`

---

## 1. 测台是什么、为什么这样设计

在 8 个**冻结的开发 snapshot** 上做 option 级诊断，用**预登记的动作分支**、**硬墙钟预算**、
**三层指标**和**本地 HTML 真实轨迹回放**。目的不是“跑出成功”，而是每轮只做一个可归因的
控制器改动，并用同一起点、同一时间轴的 baseline/candidate 对照回答“它到底改了什么”。

- **三层指标分开**：① 控制器声称完成（退出字符串）；② 独立重算的契约兑现（假完成记为
  `violated`）；③ 离线真值的任务效果（物体位移/转角/目标方向进展）。安全/受阻退出单列。
- **预算硬限制**：单分支 ≤10 s 墙钟；单批量 ≤60 s；单渲染/测试命令 ≤30 s；单轮
  模拟+测试+渲染累计 ≤180 s。`metrics`/`render`/`summarize` 不重新模拟。
- **超预算例外**：只有用户明确授权才可放宽，且写入该轮 `manifest.json` 的
  `authorized_exception`（Round 002 是唯一一次）。
- **不改物理/奖励/成功条件/安全阈值/观测布局**；候选控制器用
  `RandomStartEnv(instance, config, skill_factory=...)` 显式注入，默认 factory 是冻结 baseline。

## 2. 使用说明（可复制）

```bash
cd /home/tzl/repo/coimagine
TB=experiments/option_testbench
OUT=outputs/option_testbench_v1/round_000

timeout --signal=TERM --kill-after=2s 60s .venv/bin/python $TB/panel.py --out $OUT
timeout --signal=TERM --kill-after=2s 60s .venv/bin/python $TB/cli.py run --out $OUT \
  --branch-wall-seconds 10 --total-wall-seconds 55
timeout --signal=TERM --kill-after=2s 30s .venv/bin/python $TB/metrics.py --out $OUT
timeout --signal=TERM --kill-after=2s 30s .venv/bin/python $TB/cli.py render --out $OUT
timeout --signal=TERM --kill-after=2s 30s .venv/bin/python -m pytest tests/test_option_testbench.py -q
```

- 打开 `OUT/index.html`（全部本地资源；逐分支帧用静态 `<script src>` 加载，无需服务）。
  左栏选 snapshot/分支；“对比 / Compare”下拉做 baseline/candidate 并排同时间轴；图层开关含
  雷达（1.20 m 内无回波的射线按量程长度画成暗灰）、拟合片段、接触、法向/切向、参考速度、
  力、轨迹、真值层。
- **新增一轮**：冻结新的 `requests.json`（≤24 分支、每 snapshot ≤3），可给每个请求加
  `controller_version`（runner 会记录版本与源码哈希）；候选控制器放在独立模块里，用
  `skill_factory` 注入；跑完写 `REVIEW.md`（改动、逐例 baseline→candidate、副作用、一个
  具体反馈问题）。面板/请求菜单变更要升版本并保留旧结果。

## 3. 三轮核心改动与结果

### Round 000 — 测台与诊断基线

- 8 个冻结 snapshot（face / corner_minus / corner_plus / edge_turn / corridor_wide /
  corridor_tight / teammate / moving_recovery），24 个预登记分支。
- 预算：批量 9.05 s、渲染 4.45 s、定向测试 2.29 s；无超时。
- 覆盖率：24/24 accepted、8/8 states；分支 fulfilled 19 / violated 0 / partial 4 / benign 1；
  保守区间 [0.792, 0.792]。
- 主要发现：`face` 对称双推互相抵消（物体位移 0，双方仍 `push_complete`）；`corner_plus`
  小偏移导致 `contact_lost`；`edge_turn` 受阻；队友同侧推可正常推动（+0.231 m）。

### Round 001 — `segment-contact-v1`（接触点移入可见线段）

- 一次改动：PUSH 在 approach/push 阶段加有界切向参考，把机器人中心保持在可见线段端点
  内侧 ≥ `ROBOT_RADIUS + 0.02`；触发量是**切向到端点的偏移**（不是垂直间隙，也不是
  `nearest−distance`，后者会漏掉 `corner_plus`）。
- 结果：`corner_plus` `contact_lost`（3.20 s/接触 0.05 s）→ `push_complete`（5.30 s/2.05 s）；
  `edge_turn` r0 `blocked_no_progress` → `push_complete`；`face`/`corridor_wide`/`teammate`
  逐周期不变；`corner_minus` 结论不变但 +0.40 s。
- 基线复现：round_001 的 baseline 与 round_000 同一分支最终状态**逐值相同（6/6）**。
- 预算：12 分支、批量 4.53 s、渲染 3.19 s、测试 5.46 s。

### Round 002 — 授权超预算的多步换面测量

- 用户授权例外：每分支 ≤8 技能 / ≤45 s 物理 / ≤15 s 墙钟；控制器固定 `segment-contact-v1`，
  唯一变量是动作序列；7 分支，批量 7.92 s、渲染 2.46 s。
- 结果（`edge_turn`）：

| 分支 | 技能 | 物理 | r1 终点/所在侧 | 物体位移 | 转角 | 目标进展 |
|---|---:|---:|---|---:|---:|---:|
| 对称双推 (4,4) | 1 | 5.85 | 下（目标侧） | (+0.006,−0.007) | **−62.2°** | +0.007 |
| CW 转移 ×6（只转移） | 6 | 22.15 | **上（r0 侧）(1.71,1.51)** | (0,0) | 0° | 0 |
| CCW 转移 ×6（只转移） | 6 | 19.70 | 下 (2.39,0.96) | (0,0) | 0° | 0 |
| CW 转移 ×6 后协同推 | 7 | 26.50 | 上 (1.70,1.18) | **(−0.037,−0.205)** | **−13.2°** | **+0.205** |
| CCW 转移 ×6 后协同推 | 7 | 25.75 | 下（未换面） | (−0.284,−0.034) | −27.7° | +0.034 |

- 结论：对称角点双推是**力偶**（旋转 −62.2°、净平移≈0），不是控制器 bug。在这个 snapshot 上
  **存在**一条可行序列：CW 转移 6 次 / 22.15 s 绕到另一侧，再推得到 **−0.205 m 朝目标、
  仅 −13.2° 旋转**。收紧表述：6 次/22 s 是**一条可行序列**，不是必要次数或最短时间；
  最终推动主要来自换面后的 r1（r0 仍 `blocked_no_progress`），**双机器人协同增益尚未证明**，
  可能主要是让一台机器人获得更好的接触位置；转移期物体不动**不等于奖励中性**（仍有时间/
  距离成本），准备动作需要后续收益补偿，但这不直接证明“加 macro 就能学会”。

### Round 003 — 指标校准 v1 → v2（只改测量）

按规划端意见修正六处口径并从现有日志重算（不重新模拟）：PUSH 分 `contact_established`
（0.50 s）与 `full_push`（2.00 s）；统计截断到个体技能结束；REPOSITION 用请求方向正向累计
（不再 `abs`）；队友间隙只减自身半径；法向跳变从 `normal_world` 序列重算；条件兑现率分子
限定 accepted。v1 文件保留，v2 写入 `metrics_v2.json`/`METRICS_v2.md`；汇总数不变
（19/10/4 fulfilled、0 violated），逐项数字校准。详见
`outputs/option_testbench_v1/round_003/CALIBRATION.md` 与 `REVIEW.md`。

### 模型保真度原型 A（朝向 + 稳定推面，2026-10-19）

用户指出机器人推动时完全不转，与真实机器人（先对准、用稳定的一侧推）出入过大。已实现
**默认关闭**的运动学朝向原型（`EnvConfig(oriented=True)`：机器人需转向推力方向，未对准时
力按 `cos(error)` 缩放），v3.1 逐值不变（6/6 复现）。有界对照（5 个分支）表明：方向性结论
不变（对称角点仍是力偶、换面后仍为正），但朝向改变部分接触结果（`corner_plus` 0.05→2.05 s
接触、`edge_turn` r0 0.00→2.05 s），且 180° 对准使单技能 +1.1 s、换面 +6.35 s。详见
[`20261019_Oriented_Model_Prototype_Note.md`](20261019_Oriented_Model_Prototype_Note.md)。
**规划端需先决定驱动模型（全向+推面 vs 差速）再继续调 option。**

### Round 004 — 小范围诊断（按规划端决策，2026-10-19）

规划端决定：暂用“全向底盘＋有朝向的前端推面”，`oriented=True` 仅作诊断原型；先验证“从反馈力
反推期望朝向”是否会误门控制动/避障，并拆分 `corner_plus` 的改善来源。本轮 10 个诊断分支，
只改 `oriented_mode`（`off`/`lidar_only`/`scale_only`/`full`），批量 3.05 s、渲染 2.12 s。

- **制动门控未复现**：`moving_recovery` 上 v3.1 与 full 逐值相同（0.049→0.000、朝向不变、缩放
  1.0）——该状态制动力方向恰与朝向一致；风险仍列为正式候选要求。
- **`corner_plus` 改善来自“转朝向 + 随体雷达”**：`lidar_only` 得到与 `full` 相同的
  `push_complete`/2.05 s 接触；`scale_only` 反而 `blocked_no_progress`、零接触。圆盘碰撞形状
  与朝向无关，改善来自感知/施力时序。
- **表述边界**：`face` 位移与转角≈0 → 对向推动抵消；本轮不证明新模型更真实/更易学习/有协同增益。
- **推荐下一步**：Round 005 最小候选“技能指定期望朝向 + 制动/避障不受门控”，前端推面与转向
  力矩核验放后续独立轮。详见 `outputs/option_testbench_v1/round_004/DIAGNOSTICS.md` 与 `REVIEW.md`。

## 4. 代码改动清单（本轮新增/修改）

新增（测台）：

- `experiments/option_testbench/panel.py`：8 个 fixture、动态恢复 snapshot、契约、请求菜单、
  双语详细标签。
- `experiments/option_testbench/runner.py`：单分支隔离执行、内部 monotonic deadline、
  `controller_version` 注入与源码哈希。
- `experiments/option_testbench/metrics.py`：三层指标、覆盖率、三态与保守区间。
- `experiments/option_testbench/renderer.py` + `replay_template.html`：本地 HTML 回放
  （静态帧脚本、Compare 并排、图层、真值标注）与关键帧。
- `experiments/option_testbench/cli.py`：`prepare / run / render / summarize`，外部进程组
  超时（`SIGTERM`→`SIGKILL`，不用 `pkill -f`），可配 `--branch-wall-seconds` /
  `--recovery-horizon-seconds` / `--total-wall-seconds`。
- `experiments/option_testbench/candidate_push.py`：`segment-contact-v1` 候选控制器。
- `experiments/option_testbench/README.md`：契约与命令。
- `tests/test_option_testbench.py`：7 项定向测试（重复一致、分支隔离、单周期推进、
  信息边界、独立重算、分母/accepted=0、超时清理、baseline 菜单）。

修改（最小、默认行为不变）：

- `src/coimagine/random_start/env.py`：`RandomStartEnv(..., skill_factory=None)`，默认仍是
  `make_skill`；`joint_step` 用 `self.skill_factory`。
- `src/coimagine/random_start/skills.py`：`PushSkill._tangential_reference(...)` 默认返回
  `(0,0)`，在 approach/push 且无退出原因时叠加；baseline 行为逐值不变（已由 6/6 复现验证）。

文档：`docs/handoff/20261019_RL_Lidar_Encoding_and_InputSchema_Note.md`、
`docs/handoff/20261019_Option_Duration_MacroAction_Design_Note.md`、本文。

## 5. 下一版提案（交给规划端）

1. **option 时长/尺度（macro-action/repeat）维度**：动作 = `(类型, 时长/尺度)`，两台联合决策，
   8 档 one-hot（1x…8x），从 repeat=1（等价现有 225 动作）warm start 再解冻。理由：Round 002
   量化了长程承诺（约 22 s）与奖励 gap。PPO 支持：联合 logp = 因子化各项之和，ratio/clip
   不变，半马尔可夫目标已支持变长技能。详见 macro-action 备忘。
2. **RL 输入表示**：雷达量程外现在编码为 `(lidar=0.0, class=0.0)`；建议改为
   `min(d,R)/R` + 显式 valid + 类别 one-hot，解决非单调、不可达区间与类别序数问题。
   属 schema 变更，需新契约+重训。详见 RL 编码备忘。
3. **队友协调/动作选择**：`edge_turn` 的任务效果杠杆是“与目标同侧的机器人换面后再协同推”，
   属动作选择/规划轴（本测台已用脚本化序列量化）；可考虑一个去中心化的“朝目标才推/否则
   让位”规则，但要先解决“两侧都不 helpful 的 tie-break”和“队友挡在目标方向”的问题。
4. **低层端部语义**：显式换面退出（`corner_reached`/`face_switch`）与绕角连续性；与
   `segment-contact-v1` 正交，属低层控制器语义改动。

## 6. 当前状态、未完成与注意事项

- **未启动**：PPO/世界模型训练、规模测试、3D；没有用新动作维度重训。
- **Round 003 已完成**（指标校准 v2）；**Round 004 已完成**（小范围诊断，等待反馈）；
  下一步 Round 005 最小候选“技能指定期望朝向 + 制动/避障不受门控”。接触目标可视化/片段跟踪
  按规划端要求继续保留在待办中。
- 浏览器交互仍无自动验收工具；已用 `node --check`、资源路径、帧数与时间索引核对。
- 工作树未提交；本轮改动清单见第 4 节，源码哈希在各轮 `manifest.json`。
- 旧 RL 长训练（`outputs/random_start_rl_stability_long_v1/`）与本测台互不影响；本测台只
  新增了 env 的显式 factory 与 skills 的 no-op hook，baseline 默认行为未变。

## 7. 关键数字速查

| | Round 000 | Round 001 | Round 002 | Round 003 |
|---|---:|---:|---:|---:|
| 分支数 | 24 | 12 | 7 | 0（只重算） |
| 批量模拟 | 9.05 s | 4.53 s | 7.92 s | — |
| 渲染 | 4.45 s | 3.19 s | 2.46 s | 3×重渲染 |
| 定向测试 | 2.29 s | 5.46 s | — | 7 项通过 |
| 最大物理/技能 | ≤8 s / 1 | ≤8 s / 1 | 38.15 s / 8（授权例外） | — |
| 覆盖率 | 24/24, 8/8 | 12/12 | 7/7 | — |
| 兑现（v2） | 19 fulfilled, 0 violated | 10 fulfilled, 0 violated | 4 fulfilled, 0 violated | 校准后不变 |

## 8. 复现与文件索引

- 每轮：`outputs/option_testbench_v1/round_00{0,1,2}/`，含 `manifest.json`、`panel.json`、
  `contracts/`、`requests.json`、`logs/`、`metrics.json`、`METRICS.md`、`index.html`、
  `data/`、`keyframes/`、`REVIEW.md`、`ROUND_PLAN.md`。
- 复现命令见第 2 节与 `experiments/option_testbench/README.md`。
- 任务书与历史：`docs/tasks/20261019_CoImagine_Option_Testbench_HumanLoop_v1.md`；
  RL 诊断背景：`docs/random_start_rl_diagnosis_v4.md`。
