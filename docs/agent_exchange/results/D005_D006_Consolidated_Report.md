# D005 + D006 汇总执行回报（含 D006 期间的额外设计变更）

- 状态：`complete`（D005 核心/headless + D006 UI 交付完成）；**额外控制器改动需要 GPT 复核**，见“偏离任务书”。
- 任务书：`tasks/D005_Option_System.md`、`tasks/D006_Testbed_UI.md`；配置：`tasks/D005_config.json`、`tasks/D006_config.json`。
- 时间：2026-10-29 21:20–22:17（Asia/Shanghai）。D005 正式回归 7.84 s；D006 无训练、无 optimizer step、无 checkpoint。
- 当前 HEAD：`22c1333027bbf2a39dea3246bee78638916b4171`；工作区 dirty（`git status --short` 153 项，`docs/agent_exchange/` 等为新增）。
- D005 原始产物：`outputs/agent_exchange/D005_20261029_2130/`；D006 原始产物：`outputs/testbed/`（按 D006 任务书，不是 `outputs/agent_exchange/D006_*`）。
- 当前 server job：`bash-365`；URL `http://127.0.0.1:8765/`。
- 本报告不覆盖 D005/D006 历史回报；D005 报告仍是当时代码的历史证据。

## 0. 先回答三个问题

**D005 是否完成？** 核心完成：46 option 注册、52 状态/26 参数自动表、JointCoordinator、OptionSystemEnv、headless TestbedSession；8 个 D003 case + W_N session；7.84 s；6 项聚焦测试。**但 D005 报告中的 READY/ENGAGE 法向力对称化已在 D006 删除**，因此 D005 任务书验收第 2 条不再适用于当前 v19；这是 D006 期间的额外设计变更，需要 GPT 重新判断是否要复核 D005。

**D006 是否完成？** UI/交互层完成：本地 server + Canvas 拖动/序列/运行/回放/时间线/参数/runs/presets；静态页面与 API 已验证；JS 语法通过。**没有可用 headless 浏览器，因此没有自动截图/鼠标级验收**；需要用户实际打开确认拖拽手感。

**是否训练/改物理？** 没有训练、没有 optimizer step、没有 checkpoint、没有改 Pymunk 物理或成功判据。额外改动集中在：v19 equalization 删除、v13 `ContinuousPush` 的 contact-ready/recovery 阈值、D003/D004 instrumentation 的只读附加字段、D006 UI/preset 层。**v13 控制器改动超出 D006 任务书“只做 UI”的范围**，见第 5 节。

## 1. 输入、快照与当前哈希

D005 manifest 记录的是 2026-10-29 21:30 的源码哈希；D006 期间以下文件已变化，当前哈希如下（供 GPT 复核）：

| 文件 | 当前 SHA256 |
|---|---|
| `experiments/option_rl_v19/spec.py` | `316bd123bd76b7b8e603682e40eeff03630642aac118aa6f9119f4d1ebb76af7` |
| `experiments/option_rl_v19/system.py` | `94c0391c701668c88204e08ff70978b3dcef23804b9b9a4a5fbf0954499cf7d0` |
| `experiments/option_rl_v19/session.py` | `d6afe1cf706112d5f4595751b9bbbe0c4ade475e18e619a117c330a97ef78c74` |
| `experiments/option_rl_v19/run_d005.py` | `851c2a81c26b6cc8839f2ffccb54cd55ef9a27fdae6089be8e98273713723526` |
| `experiments/option_rl_v19/testbed_server.py` | `9d65b1ad2a495fbc4ea92a861ee29d24be82ffa3d3574ab005066d18ecfd9541` |
| `experiments/option_rl_v19/testbed/app.js` | `5db28d7bb5d9408c5936f71471f9d00b4c793175019fabd76617fab7d69490a8` |
| `experiments/option_rl_v19/testbed/index.html` | `45460b2660d83333c0d43634213e7685fd59755fa828d77dcf815950fcfa91b3` |
| `experiments/option_rl_v13/push_controller.py` | `39de9d09d8a97519a6ab2a4d93b014f7a0887d197c166557e85631839d915aea` |
| `experiments/option_rl_v17/contact_timing_audit.py` | `cc0000765cd4ff3ef46062bf53bbdb13d22839140296473c9c479eb99dff9826` |
| `tests/test_d005_option_system.py` | `b79561f398da517a0a61d4792b6151bc8ac8d1abb18d41940f873d0346694069` |
| panel `outputs/option_rl_v14/learnability_002/panel.json` | `a60bdcc120923a8091d8e8930eb4717406075aeeb55b5291e3f924dd48227126` |

D005 `manifest.json` 中的旧 source hashes（如 `spec.py cce6...`、`system.py ef5b...`）是历史快照，不再等于当前工作区；D005 的 7.84 s/129 tick 结果仍可回查，但不能声称是当前源码的精确复现。D006 期间没有新建 `outputs/agent_exchange/D006_*` 快照；testbed 产物在 `outputs/testbed/`。

## 2. 实际命令

D005 正式回归（历史）：

```bash
.venv/bin/python experiments/option_rl_v19/run_d005.py \
  --out outputs/agent_exchange/D005_20261029_2130 \
  --config docs/agent_exchange/tasks/D005_config.json \
  --d001 outputs/agent_exchange/D001_20261029_1644 \
  --d002 outputs/agent_exchange/D002_20261029_1757/A \
  --panel outputs/option_rl_v14/learnability_002/panel.json \
  --workers 4
```

D006 server：

```bash
.venv/bin/python experiments/option_rl_v19/testbed_server.py --host 127.0.0.1 --port 8765
```

D006 验证：

```bash
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/
node --check experiments/option_rl_v19/testbed/app.js
.venv/bin/python -m pytest -q \
  tests/test_d003_contact_timing.py \
  tests/test_d004_state_pair_gate.py \
  tests/test_d005_option_system.py \
  tests/test_joint_contact_recovery.py \
  tests/test_option_rl_v13_recovery.py
```

测试结果：上述 5 个文件 **48 项通过**（最后输出 48 个点）；D005 单独 6 项通过。没有跑全仓库 pytest。

## 3. D005 实测结果（历史）

- 46 option / 52 state / 26 family param；三张表由 `spec.write_tables()` 自动生成。
- 8 个固定 case：W_N/W_E 成功，129 tick/6.45 s；6 个失败窗口 source raw/mask/pose = 0/0/≤1.1e-7；noninterference 0。
- headless session：W_N instance + 33 个 witness 5 Hz 命令，129 帧/129 tick，success=True。
- 4 worker，7.84099 s；不是训练，不是 rollout 评估。
- 自动表：`option_table.csv` 46 行、`state_table.csv` 52 行、`param_table.csv` 26 行；`summary.json`、`manifest.json`、`case_metrics.csv`、`session_run.json` 在 D005 输出目录。

## 4. D006 实测结果

- 本地 server 与静态文件：首页/`app.js`/`style.css` 200；API `/api/options` 46、`/api/params` 26、`/api/presets` 13（8 seed + 用户 preset）、`/api/runs` 45（测试 run 累计）。
- 一次 `POST /api/run`（`dual_2`，短序列）返回 20 帧并保存；后续用 `contact_2` preset 跑 100/160 tick；run JSON 保存、列表、读取可用。
- `POST /api/presets`、`GET /api/presets`、`GET /api/presets/<id>`、`DELETE` 已验证；preset 文件在 `outputs/testbed/presets/`。
- `POST /api/overrides` 保存 `outputs/testbed/overrides/`；参数面板可读/改/导出。
- `node --check app.js` 通过；UI 没有自动浏览器视觉测试。
- 页面能力：拖动、选项序列、Run、Play/Pause、逐帧/滑杆、共享时间线、hover、runs 列表、参数、presets。
- Timeline/curves 已经合并为**一个 canvas/一个时间轴**：click/drag seek、同一 playhead、tooltip 同时显示 stage/contact/force/speed。
- 可选图层：lidar（72 rays）、target（skill contact_target/reference_centre）、force（力箭头）、contact（接触外圈）。
- Preset 统一为“命名场景+序列”实体：8 个 panel 场景落成普通 `seed_<scene>` preset；无 Scene selector、无 built-in/read-only、无独立 `/api/scenes`；`Reset poses` 回到当前 preset 初始位姿；description 自动由 option 序列 + created 时间戳生成。
- 旧 run/preset 缺 `scene_meta` 时 server 自动补；缺 `scan`/`target` 时 UI 自动跳过（曲线仍可用）。

## 5. 额外 A/B 与设计变更（重点）

### 5.1 equalization（D004/D005 机制）已删除

D005 原验收要求 READY/ENGAGE 法向力对称化。D006 中做过一次 `contact_2` 同侧双推对照：

| 机制 | 物体位移 | PUSH tick | 双方 contact |
|---|---:|---:|---:|
| equalize on（run `20261029_211101_60fb48`） | 0.0541 m | 2 | 16/15 |
| equalize off（run `20261029_211314_6f0c2c`） | 0.4694 m | 45 | 46/46 |

**这不是受控单变量 A/B**：GPT 复核发现两份 run 的初始机器人位姿不同，robot0 x 相差约 8.9 cm。因此不能把 0.054→0.469 的差异全部归因于 equalization。可以确认的是：equalization 已从 v19 `JointCoordinator.gate()` 删除，这是当前代码事实；其因果效果仍需**完全同起点**的受控复核。D004 v18 `pair_gate` 保留为历史证据，v19 不再依赖。D005 报告已加注：其 equalization 描述是旧代码历史证据。

### 5.2 contact-ready 阈值：0.008 → 0.05 m

逐 tick 观察到：两只机器人已到 `gap≈0`、`ready=1`，但 `contact_ready` 只取单 tick Pymunk `contacting_object`，在表面附近逐 tick 闪断；pair gate 要求每 tick 双方 contact，PUSH 反复掉回 READY/ENGAGE，8 s 只推 0.084 m。

修复：

- `CONTACT_READY_GAP = 0.05`：`gap <= 0.05` 也算 contact-ready / effective contact；力控制器继续闭合剩余间隙。
- **语义警告**：`contact`（原始 tick-start 传感器）仍是物理接触；`contact_ready` / `coord both_contact` / `joint_active` 是“接近到可协调范围”（当前 5 cm），并用于累计 `effective_contact`。它们不能当作实际碰到物体；报告和 UI 必须分开写。
- 只改 0.05 会触发新问题：默认 `recovery_enter_gap=0.01` 把 5 cm 近接触当成丢接触，导致 RECOVER/EXIT、物体 0 位移（run `20261029_221114_*`）。
- 因此分离两个语义：`RECOVERY_ENTER_GAP = max(0.06, CONTACT_READY_GAP + 0.02)`；当前 `CONTACT_READY_GAP=0.05` 时实际为 **0.07 m**，只有 gap > 0.07 才进入 recovery。

结果：

| preset | 修复前 | 修复后 | PUSH | 双方 contact |
|---|---:|---:|---:|---:|
| `同侧双推 8s` | 0.084 m | **1.4647 m**（run `20261029_221149_383cd3`） | 128 | 120/120 |
| `同侧双推 8s closer` | 0.084 m | **1.5376 m**（run `20261029_221423_90df89`） | 134 | 126/126 |

两 run 的 `success` 都是 **False**；它们证明“能产生明显推进”，**不证明完成任务，更不证明 RL 成功率提高**。

- 这说明这两个手工序列能产生明显推进，且不再因单 tick contact 闪断而卡在 8 cm；**不**等于任务完成或 RL 策略改善。
- 但它**不证明任意距离/任意遮挡都能推**：当前只测了约 0.2 m 初始 gap、雷达可见接触面的 `contact_2` 同侧双推；没有做任意距离、遮挡、动态障碍或跨场景证明。
- D003/D004/D005/joint/v13-recovery 测试全部通过（48 项）。
- **未重跑** `run_d005.py` 的 8-case 正式回归 after this controller change；D005 的 129 tick/W_N/W_E 数字是历史代码结果。D005 测试只覆盖单元/协议，不覆盖 8-case 数值复现；GPT 若要求当前代码的 8-case 回归，需要另开有界任务。

### 5.3 instrumentation / session 扩展

- `A.Recorder.obs` 增加 `scan`（origin/heading/angles/distances/valid/category）；
- `A.Recorder.p3_entry` 增加 `debug`（contact_target/reference_centre/skill_stage）；
- `p3.info` 白名单增加 `reference_centre/face_midpoint/face_normal/heading_error/lateral_error`；
- `TestbedSession` frames 增加 `scan/target/ref_centre`；
- 这些是**只读附加字段**，不改变控制输入、mask、奖励、物理；D003/D004 测试仍通过。D003/D004 历史报告中的旧产物仍是旧格式，UI 对旧 run 做了缺字段兼容。

### 5.4 preset/UI 的额外重做

- 删除 Scene selector/独立起点场景；8 个 panel 场景作为普通 seed preset；所有测试场景同质。
- preset 增加 `base_scene`/`scene_meta`、object/robots、sequence、overrides；description 自动生成，移除第二个描述框。
- 删除 `/api/scenes`；新增 presets CRUD；server 启动补缺失 seed。
- Timeline 与 Control curves 合并；增加 lidar/target/force/contact 图层。
- 这些改动不改变 Pymunk 物理或 option 集合，但改变了 D006 任务书中的 API/交互描述（原任务书要求 Scene selector、`/api/scenes`、手工 description）；GPT 复核时应按新 UI 验收，不能按旧 D006 页面逐条对照。

## 6. 对任务假设的回答

| 假设 | 结论 | 证据 |
|---|---|---|
| D005 的 option registry/coordinator/session 能复用 D003/D004 证据 | **支持（历史）** | 8 case、W_N session、6 测试；但当前源码哈希已变化，8-case 未在新控制器上重跑 |
| D006 只做 UI 即可让用户评估系统 | **支持（功能）** | 本地 server/API/run/presets/参数/回放可用；无自动浏览器视觉验收 |
| READY/ENGAGE 法向力对称化有必要 | **证据不足** | on/off 位移差很大，但初始位姿不同（robot0 x 差 ~8.9 cm）；代码已删除，因果待受控 A/B |
| 单 tick contact 足以作为 pair gate 的 contact-ready | **反对** | 0.084 m、contact 3 tick；近表面 5 cm + recovery 7 cm 后 1.465/1.538 m（success=False）；`contact_ready` 仍是 near-contact，不是物理接触 |
| 机器人从任意距离、只要雷达看到接触面、时间足够，就应能推 | **部分支持 / 证据不足** | 0.2 m 初始 gap、可见接触面的同侧双推已恢复；未测任意距离/遮挡/跨场景，不能宣称设计已普遍满足 |
| 稳定接触 basin 是合理设计性质 | **反对** | 5 cm 容差修复后原失败 preset 恢复；初始几何敏感是控制器缺陷，不是可接受特性 |

## 7. 偏离任务书 / 需要 GPT 复核

1. **D006 任务书写“只做 UI/交互层，不改 v13/v14”**；实际修改了 `experiments/option_rl_v13/push_controller.py`（contact-ready 0.05、recovery 0.07、effective contact），这是控制器行为变更，影响所有使用 `ContinuousPush` 的 option。需要 GPT 判断是否接受为 D006 的额外设计修复，还是要求拆成独立 D007 复核。
2. **D005 任务书验收第 2 条包含 READY/ENGAGE 法向力对称化**；该机制已从 v19 删除。D005 报告是历史证据；当前 v19 不再满足该条。需要 GPT 明确：D005 是否重新复核，还是把“删除对称化”记为 D006 的已批准修订。
3. **D005 的 8-case 正式回归没有在当前 contact 修复后重跑**；只有 D003/D004/D005/joint/v13 的 48 项测试通过。D005 的 7.84 s/129 tick 数字不能当作当前控制器的数值复现。
4. **D006 API/页面偏离旧任务书**：移除 `/api/scenes`、Scene selector、built-in；新增 presets/seed/自动 description；旧任务书的逐条页面验收不再适用。
5. **D005/D006 产物路径**：D006 按任务书写 `outputs/testbed/`，不是协议默认的 `outputs/agent_exchange/D006_*`；D006 没有源码快照/manifest。若 GPT 要求严格协议产物，需要补一个 D006 snapshot/manifest。
6. **没有 headless 浏览器**：UI 只做了 HTTP/API/JS 语法与人工代码检查；拖拽、布局、hover 的实际视觉需要用户确认。

## 8. 未跑 / 限制

- 未跑全仓库 pytest；只跑上述 5 个测试文件（48 项）。
- 未跑 D005 8-case 回归 after contact 修复。
- 未跑任意距离/遮挡/跨 scene 的接触泛化；0.05 m 是从同侧双推经验得到的容差，不是对所有场景的证明。
- 未训练、未解冻 D002 B、未并入 independent 停车、未发布 D007。
- 旧 run/preset 的 scan/target 兼容是 UI 层跳过；旧 D003/D004 原始产物未重写。
- seed presets 在 server 启动时补齐缺失文件；用户删除 seed 后重启会重新出现，这是当前策略，不是不可删除的只读类型。

## 9. 交回 GPT

建议下一步（不自动执行）：

1. 先复核本报告第 5 节的三个额外变更，尤其是 **v13 controller contact-ready/recovery（0.05/0.07 m）** 和 **D005 equalization removal（目前只有非同起点对照）**；决定它们是 D006 的已批准修订，还是需要新的有界复核任务。
2. 若接受，下一步用 testbed 做**接触泛化**的小任务：只测 3–5 个不同初始 gap/横向偏移的同侧双推，记录 object displacement、contact 连续性、PUSH 占比；不跑训练。
3. 若要恢复 D005 的严格数值验收，另开一个“当前源码 8-case 回归”的有界任务；不要用旧 D005 数字冒充当前结果。

## 10. 产物

- D005：`outputs/agent_exchange/D005_20261029_2130/`（`summary.json`、`manifest.json`、`case_metrics.csv`、`session_run.json`、三张 CSV）；D005 报告 `results/D005_Option_System.md`。
- D006：`experiments/option_rl_v19/testbed_server.py`、`testbed/index.html`、`app.js`、`style.css`；`outputs/testbed/runs/`（45 个测试 run）、`presets/`（13 个）、`overrides/`；D006 报告 `results/D006_Testbed_UI.md`。
- 当前 server：`bash-365`，`http://127.0.0.1:8765/`。
- 当前 `CURRENT.md` 已指向本汇总报告，状态 `ready_for_review`，下一位 GPT。
