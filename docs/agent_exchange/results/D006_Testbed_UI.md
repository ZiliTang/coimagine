# D006 执行回报：Web option testbed

- 状态：`complete`（server/API/UI 已启动并验证）；本地 URL：`http://127.0.0.1:8765/`
- 任务：只做 UI/交互层，消费 D005 `TestbedSession`/`OptionSystemEnv`；没有另写 Pymunk/env/option，没有训练，没有发布 D007。
- 实现：`experiments/option_rl_v19/testbed_server.py`、`testbed/index.html`、`app.js`、`style.css`。
- server job：`bash-272`（managed background，127.0.0.1:8765）。

## 已验证

- `/`、`/app.js`、`/style.css` 均 200。
- `/api/scenes` 8 个 panel 场景；`/api/options` 46；`/api/params` 26。
- `POST /api/run`：scene=`dual_2`，初始位姿 + 1 段 PUSH 序列，返回 20 帧/20 strip/coordinator；run 保存到 `outputs/testbed/runs/`；`GET /api/runs` 能读回。
- `POST /api/overrides`：保存 `outputs/testbed/overrides/smoke_test.json`。
- built-in/user presets：`GET /api/presets`、`GET /api/presets/builtin:contact_2`、`POST /api/presets`、`DELETE /api/presets/<id>` 已验证；built-in 不可删除。
- `node --check app.js` 通过。

## 页面能力

- Scene selector 已移除：8 个 panel 场景现在是 built-in read-only presets；load built-in → 拖动/改序列 → Save 成新的 user preset。
- 两侧 option 下拉（46 个）+ 持续时间；Add/Clear/删除段；序列列表。
- Run：服务端同源 Pymunk 计算；画布回放；Play/Pause、上一帧/下一帧、滑杆。
- Timeline：R0 stage / R1 stage / joint phase / contact+joint_active 四行颜色条带；点击跳帧；图例。
- Timeline hover：鼠标所在帧显示 tooltip（tick/time、两侧 stage、joint phase、contact、joint active、force），并画 hover 竖线；颜色改为更区分的阶段色，不需要为每个颜色下去查 legend。
- State：时间、物体/机器人、stage、joint phase、contact、joint active、force、success/termination/reward。
- Runs：保存多轮、列表切换重放。
- Parameters：按 family 分组；可改 default→override；Save override 导出 JSON；Run 使用当前 override。

## 限制

- 没有可用的 headless Chromium/Playwright，因此本轮只做了 HTTP/API/JS 语法验证，没有自动截图/鼠标交互测试；需要用户在浏览器中手动确认拖拽和播放体验。
- scene/sequence 当前来自 D005 panel；没有任意新场景创建、没有多机器人 >2、没有分布式。
- UI 只消费 D005；没有改 D005 env/物理/参数默认值；override 仍是文件，不自动写回原环境。

## 下一步

用户打开 `http://127.0.0.1:8765/`，先选 `dual_2`/`contact_2`，拖到合理位姿，加入 1–3 段 PUSH/REPOSITION，Run 后看画布、逐帧、颜色条带和 State。若交互/布局有问题，下一轮只修 UI；若要 D007，再讨论 scene/sequence 持久化和参数写回。

## 2026-10-29 补充：删除 equalization，新增 presets

- **equalization 已从 v19 删除**（不是默认 off）：`JointCoordinator.gate` 不再在 READY/ENGAGE 取 `min(n0,n1)`；UI 的 `equalize normal` 复选框和 API 的 `equalize` 字段一并删除。D004 的 v18 pair_gate 保留为历史证据，但 v19/testbed 不再依赖。
- 同侧双推对照（**不是同起点单变量 A/B**）：equalize on 0.054 m/PUSH 2，删除后 0.469 m/PUSH 45；两份 run 的初始 robot0 x 相差约 8.9 cm，因此不能把差异全部归因于 equalization。删除是当前代码事实，因果待受控复核。
- **Presets**：
  - 保存当前 scene + object/robots 初始位姿 + option sequence + overrides；如果序列已经执行，`S.object/robots` 仍是 run 之前的初始状态，保存的是执行前状态。
  - 默认命名 `${scene}_${YYYYMMDD_HHMMSS}`，输入框可编辑；保存记录 `created`。
  - UI Presets 卡：Save、列表、按 newest/name 排序、点击 load、× 删除。
  - API：`GET/POST /api/presets`、`GET/DELETE /api/presets/<id>`；文件保存在 `outputs/testbed/presets/`。
  - 已验证：保存 `contact_2_double_manual`、list/get 成功；重启 server 后仍在。
- 当前 server job：`bash-302`；URL 不变。
- Scene selector 已删除；preset 是唯一的“场景+序列”命名单元；`Reset poses` 回到当前加载 preset 的初始位姿；preset 有手动 description，避免用 base scene 名误导。

## 2026-10-29 最终简化：description 自动、所有 preset 同质

- 删除了第二个 description 输入框；preset 的 `description` 由 server 自动生成：
  - 前 3 段 option 的 `a0/a1` 名称 + 每段持续时间；
  - 加 preset `created` 时间戳（`YYYY-MM-DD HH:MM:SS`）；
  - 空序列显示 `empty sequence · timestamp`。
- 不再有 built-in/read-only/起点场景特殊类型：
  - 8 个 panel 场景在启动时落成普通 `seed_<scene>` preset（空 sequence），可 load、改、Save 成新 preset、删除；
  - 删除了 `/api/scenes` 端点；`SCENES` 只作为 server 内部 geometry source；
  - UI 不再显示 Scene selector、`built-in` 标签或第二描述框。
- `Presets` 列表现在所有条目同质：load / Save / delete / sort by newest/name；`Reset poses` 回到当前 preset 的初始位姿。
- 验证：`/api/presets` 返回 12 个 preset（3 个旧用户 + 8 个 seed + 1 个测试）；旧 preset 缺 `scene_meta` 时 server 自动补；旧 description 也按 sequence+created 重新生成。
- 当前 server job：`bash-314`；URL 不变。

## 2026-10-29 可视化图层 + 控制曲线

- Scene 卡新增可选 `view` 图层（默认关）：
  - `lidar`：两侧 72-ray LidarScan，按 object/static/teammate 着色；
  - `target`：skill 的 `contact_target` / `reference_centre`，画 T0/T1 十字与机器人到目标的虚线；
  - `force`：该帧 strip 的力向量箭头；
  - `contact`：该帧 contact 侧机器人外圈高亮。
- 新增 **Control curves** 卡：
  - `force`：|F0| / |F1| 随时间；
  - `speed`：|v0| / |v1| / |v_obj| 随时间；
  - 与 timeline 共用 frame/playhead，seek/play 时同步刷新。
- 数据来自 session：
  - `A.Recorder.obs` 增加 `scan`（origin/heading/angles/distances/valid/category）；
  - `p3.debug` 记录 skill 的 `contact_target`/`reference_centre`/`skill_stage`；
  - `frames` 增加 `scan`/`target`/`ref_centre`；旧 run 没有这些字段时 UI 自动不画。
- 当前 server job：`bash-344`；URL 不变。

## 2026-10-29 时间线合并

- Timeline 和 Control curves 合并为一个 canvas、一个时间轴：
  - 同一个 `frame -> x` 映射、同一个 playhead、同一个 hover 线；
  - 点击/拖动 timeline 任意位置都会 seek，曲线同步；
  - 底部只有一个时间刻度轴。
- `force` / `speed` 复选框移到 Timeline 卡头；隐藏某个曲线时 canvas 高度自动缩短。
- hover tooltip 现在同时包含：
  - frame/time/tick、joint phase、R0/R1 stage、contact、coord both_contact/release/active；
  - `|F0| / |F1|`（N）；
  - `|v0| / |v1| / |vobj|`（m/s）。
- 曲线仍用同一帧数据：`strip.forces` 与 `frames.robots/object` 速度；
- 当前 server job：`bash-344`；URL 不变。

## 2026-10-29 设计修复：contact-ready 不能只看单 tick 接触

- 现象不是“preset 初始几何不好”，而是控制器设计缺陷：
  - 两只机器人都能到 `gap≈0`、`ready=True`；
  - 但 `contact_ready` 只取单个 tick 的 Pymunk `contacting_object`，在表面附近会逐 tick 闪断；
  - pair gate 要求两侧每 tick 同时 contact，于是 PUSH 反复掉回 READY/ENGAGE，8s 只推 ~8cm。
- 修复：`ContinuousPush` 在 `stage=push` 时把 `gap <= 0.008 m`（后来为 0.05 m）视为 contact-ready / effective contact；力控制器仍继续闭合剩余间隙。这样 pair gate 不会因为单 tick 传感器闪断而退出 PUSH。
- **语义**：`contact` 仍是物理 tick-start 传感器；`contact_ready`/`coord both_contact`/`joint_active` 是 near-contact 协调量，不能读成实际接触。
- 验证（同一 testbed preset）：
  - `同侧双推 8s`：0.084 → **1.285 m**，PUSH 111 tick，contact 108/108；
  - `同侧双推 8s closer`：0.084 → **1.363 m**，PUSH 117 tick，contact 114/114；
  - D003 / D004 / D005 / joint-contact-recovery / v13-recovery 测试全部通过。
- 当前 server job：`bash-355`。
- 这符合设计目标：只要雷达能持续看到接触面，远处接近后应能收敛到稳定推动，而不是依赖狭窄的初始稳定 basin。

## 2026-10-29 接触容差改为 5 cm

- 按真实定位噪声，将 `CONTACT_READY_GAP` 从 0.008 调到 **0.05 m**（命名常量，位于 `experiments/option_rl_v13/push_controller.py`）。
- 只改 0.05 时出现新问题：`recovery_enter_gap` 默认 0.01，5 cm 的“近接触”会被当成丢接触，触发 RECOVER/EXIT，物体 0 位移。
- 因此把两个语义分开：
  - `CONTACT_READY_GAP = 0.05`：表面附近/定位噪声下仍算 contact-ready；
  - `RECOVERY_ENTER_GAP = max(0.06, CONTACT_READY_GAP + 0.02)`：当前 0.05 时实际为 **0.07 m**，只有拟合 gap 超过 7 cm 才认为真正丢接触。
- 验证：
  - `同侧双推 8s`：**1.465 m**，PUSH 128 tick，contact 120/120；success=False；
  - `同侧双推 8s closer`：**1.538 m**，PUSH 134 tick，contact 126/126；success=False；
  - 两 run 证明推进能力，不证明任务完成或 RL 成功率。
  - D003 / D004 / D005 / joint-contact-recovery / v13-recovery 测试全部通过。
- 当前 server job：`bash-365`。
- 设计要点：**contact-ready 容差**和**丢接触恢复阈值**不能共用一个数；前者应覆盖定位噪声，后者应明显更大。

## 2026-10-29 调优候选：相邻侧双推的侧向恢复

- 用户保存的 override：`outputs/testbed/overrides/相邻侧双推 5s 右上_override_mvtmq727.json`；关键是 `PUSH.tangential_gain=8.8`（默认 4.8）和 `PUSH.tangential_speed=0.5`（默认 0.213）。
- 同一场景/序列的三次 testbed run：
  - `20261029_221731_5cedfd` 默认：双方 contact 56/62，最长连续 56 tick，位移 0.839 m；
  - `20261029_221821_3a19e0` 只加 gain：58/65，最长 57 tick，位移 0.877 m；
  - `20261029_221857_d31a97` gain+speed：**63/62，最长 62 tick，位移 0.967 m**。
- 这说明侧向 gain/speed 确实改善了接触连续性；但三 run 都 `success=False`，只证明接触保持/推进，不证明任务完成。
- 该 override 是 **PUSH family-wide**，会影响所有 PUSH 方向和速度；在 promote 前应至少测 PUSH_E/PUSH_NE 和单机器人，观察侧向过冲/抖振。
- 语义注意：override 里的 `recovery_enter_gap=0.01` 不再是当前丢接触触发阈值；当前触发用 `RECOVERY_ENTER_GAP=0.07`。这个字段现在容易误导，promote 前应决定参数化还是移除。
- 尚未改默认、未训练、未进入正式验收。

## 2026-10-29 Parameters 存档可用性

- 之前 `Save override` 只写 `outputs/testbed/overrides/<name>.json`，UI 没有列出/加载入口；现已补：
  - Parameters 卡显示已存 profile 名称；
  - 点击 profile → `GET /api/overrides/<name>` → 把 `overrides` 写回参数面板；
  - `Save override` 后列表刷新。
- 语义区分：
  - **override/profile**：只存参数，不存 scene/sequence；
  - **preset**：存 scene + object/robots 初始位姿 + option sequence + 当前参数；用 Presets 卡的 Save 保存，load 时全部恢复。
- 要“参数 + 场景 + 序列一起存”：先设好参数（可加载 profile），再设好场景/序列，然后在 Presets 卡命名并 Save；不需要单独 Save override。
- 当前 server job：`bash-387`。

## 2026-10-29 布局微调

- Preset 列表不再重复显示 `created`（description 已含时间戳）；`preset-item` 改成 `minmax(0,1fr)` + ellipsis，删除按钮不再被长描述推远；列表禁用横向滚动。
- 主布局右侧栏 450 → **520 px**，`max-width` 1600 → 1700；左侧 Scene/Timeline 相对收窄。
- Timeline 高度 360 → **260 px**（无曲线 150），行高 30 → 24；stage/phase/contact 条带和 force/speed 曲线整体压扁，时间轴/playhead 不变。
- 静态改动，无需重启 server；当前 job 仍 `bash-387`。

## 2026-10-29 目标区域可视化与拖动

- Goal 从绿色实线圈改为 **amber 虚线圈 + `GOAL` 标签**，和机器人/物体颜色区分。
- 目标圈支持拖动：点击圈内或圈边（object/robot 优先）即可移动 goal；拖动后标记 preset 为 modified。
- goal 现在随 run/preset 进入 server：
  - `POST /api/run`、`POST /api/presets` 接受 `goal={x,y,radius}`；
  - `_scene_instance` 用 `goal_centre/goal_radius` 覆盖 base scene；
  - run/preset 的 `scene_meta.goal` 保存移动后的目标；加载 preset 会恢复该 goal。
- 已验证 goal 偏移 0.2/0.1 后 run 与 preset 的 `scene_meta.goal` 正确；临时 preset 已删除。
- 当前 server job：`bash-397`。

## 2026-10-29 rotation_feedback_gain（试调参数）

- 新增 PUSH family 参数 `rotation_feedback_gain`（默认 0，范围 0–2），直接加在 contact target 的切向偏移：
  - `rotation_feedback = clamp(gain * angular_rate, -limit, limit)`；
  - `t = rotate90(face.normal)` 是机器人左侧；正号定义为 CCW→左、CW→右；
  - 该量直接进入 target offset，不经过 `point_rate` 混合反馈。
- 参数已进入 `param_table.csv`（27 行）、`/api/params` 和 UI Parameters 面板。
- 初扫（同一 preset，gain 0 / 0.5 / 1.0）：
  - T1 确实随 gain 更明显右移；
  - 但 object 的 CW 角反而更大（gain 0.5/1.0 的 angle 比 0 更负），位移 0.967→0.927/0.905；
  - 说明当前符号/作用方向还不能证明“右移抵抗 CW”，或者 target 移动没有产生预期的反向力矩。**不能 promote，需要先做 sign/geometry 的受控验证。**
- 当前 server job：`bash-410`。

## 2026-10-30 rotation_feedback 自动诊断（testbed runs）

- `rotation_feedback_gain` 已允许有符号范围 `[-2, 2]`；默认 0。
- 同一 preset（`相邻侧双推 5s 右上`）、同一初始状态/序列，自动跑诊断并保存到测台 Runs：
  - `DIAG rotfb 0`：angle_end −0.111，位移 0.967；
  - `+0.1`：−0.132；`+0.5`：−0.181；`+1.0`：−0.173；
  - `−0.1`：−0.073；`−0.12`：−0.045；`−0.15`：**−0.034**；`−0.18`：**+0.005**；`−0.2`：+0.085；`−0.5`：+0.220；`−1.0`：+0.482。
- 结论：**当前实现下负 gain 才在阻尼**，正 gain 会放大 CW；零偏转附近约在 `−0.15 ~ −0.18`。`−0.15` 位移 0.981，双方 contact 62，PUSH 69；仍 `success=False`。
- 机器人相对物体的横向 x 也随阻尼 gain 改变：gain 0 终点约 −0.074，−0.15 约 −0.026；不是“侧向完全不动”。
- 所有诊断都是普通 testbed run，名字 `DIAG rotfb ...`，可在 Runs 列表直接打开；摘要 JSON 在 `outputs/testbed/diagnostics/rotation_feedback_*.json`。
- 当前 server job：`bash-419`。没有改默认，没有训练。
