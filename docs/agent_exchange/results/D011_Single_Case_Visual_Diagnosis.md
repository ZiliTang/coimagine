# D011 Stage A：T_reposition_2_p0 逐帧诊断页（人在环前）

- 状态：**Stage A 完成，等待用户查看并介入**。未修改控制算法/EKF/FF/物理/脚本；没有训练、默认替换或 D011 B/C 自动运行。
- 页面入口：**http://127.0.0.1:8766/**（只读静态服务，独立端口；不替换用户已有 8765/3080）。
  - 启动命令：`cd outputs/agent_exchange/D011_20261030 && python3 -m http.server 8766 --bind 127.0.0.1`
  - 无浏览器自动验收；`node --check`、数据 schema/事件定位通过，视觉交互仍需用户确认。
- 唯一实例：`T_reposition_2_p0`，原脚本 14.4s REPOSITION→8s PUSH_S→3.2s BRAKE，共 25.6s。
- B/F 均为该实例冻结版本；B hash `29b05254…`，F hash `c3f515b6…`。

## 页面包含什么

- B/F 并排、同一仿真时间同步；另有“首触对齐”模式。
- 播放/暂停、前后单 tick、0.1/0.25/1×、拖动时间轴；世界视图缩放/平移。
- 同一时间轴：原始 ray、patch/拟合线、法向/切向、candidate q/历史、实际接触点、own/真值、路径、速度/力向量。
- 曲线：gap/e_t、desired/reference/actual 速度、force/gate、角/推进；阶段轨：requested/executing action、control source、contact_phase、coordinator、raw contact。
- 事件可跳转到前 .25s：换面/PUSH 开始、候选新建/换选、phase、首触、制动；可留备注并导出 JSON。
- 中文逐帧解释和原始字段；真值标“评估真值”，不进入控制器。

## 先看这 5 个事件

1. **14.40s / tick 288**：REPOSITION 结束、PUSH_S 开始；关注机器人位置、可见支持和候选建立前的状态。
2. **14.65s / tick 293**：F 建立 candidate #1，`q=(2.2378,2.125)`，支持 `[2.0019,2.3795]`，margin `.06`；此时机器人中心离 q 还很远。
3. **15.55s / tick 311（B）**：B 首次 raw contact；B 没有 m3b candidate 字段，stage=push，首触 cp `(2.0022,2.1211)`。
4. **15.65s / tick 313（F）**：F 首次 raw contact；**contact_phase=align，stage=push，e_t=.3432 m**；cp `(2.0005,2.1239)`。这是本轮最该先看的矛盾：候选误差仍大，接触已经发生并进入 push，而不是由 align 条件确认。
5. **16.75–17.80s / tick 335–356**：F 进入 retreat→shift；由 truth 机器人位置重算的**实际 SHIFT 路径 = .0620 m**（旧报告的 shift=0 是分析错误，已纠正）；候选随后多次 `support_invalid/support_extension` 重选，q 移到 x≈2.32–2.43，但 e_t 仍约 .21 m。

## 已知事实（本轮）

- B：首触 tick 311；最终向南推进 .1086 m、最大相对角 .702 rad、距离 .3599 m；fail。
- F：首触 tick 313；最终向南推进 .0306 m、最大相对角 .154 rad、距离 .4207 m；fail。
- F 首触 cp 与 B 只差约 2–3 mm；F 的 q 不是首触 cp，q 距首触位置约 0.34 m。
- F 的 candidate 变更记录：293 (#1 初始) →321 (#2 support_invalid) →357 (#3 support_invalid) →369/386/396/423 (#4–7 support_extension)。
- F 首触时 `stage=push` 与 `contact_phase=align` 同时出现；页面可逐帧看到该状态。
- B/F 新富记录与 D010 旧记录逐 tick truth 最大误差 **0.0**，periods/动作/success 一致；记录未改变控制。

## 尚未核验/不要当结论

- q 与机器人半径/gap 的公式是否相容；页面展示的是记录值，未重算控制器语义。
- align 的位置/速度条件在首触 tick 的 settle/速度值为何没有阻止 contact；需要看页面侧栏和原始字段。
- 为什么 shift 实际 .062 m 后 e_t 仍约 .21 m；候选重选是否在同一接触空间、重选触发是否合理。
- 是否应由 contact gating 阻止 align 状态下的 full PUSH，还是应让 approach 先到 q；这两者都只是候选解释，尚未实现。
- 单实例仅说明可实现性与诊断线索，不证明普遍未知形态或 RL 可用。

## 顺序故障表

| 时间段 | 可见中间量 | 实际执行 | 下游影响 | 尚待核验 |
|---|---|---|---|---|
| 14.40–14.65s | PUSH_S 开始；q=(2.2378,2.125)，support 右侧 | 机器人从换面终点接近 | q 远离首触位置 | 候选是否应由支持/安全约束先移动 |
| 15.65s | raw contact=True；phase=align；stage=push；e_t=.343 | 走近后接触，直接进入 push 分支 | 物体开始转偏 | 为什么 contact 不等待 align；条件值 |
| 16.75–17.8s | retreat→shift 21 tick；实际路径 .062m；q 重选 2–7 | 侧移/重选/再接近 | 仍未到达 q，仍失败 | shift 目标与重选是否同一 q；实际 e_t |
| 22.4s–end | brake；F 南向 .0306m，B .1086m | 制动 | 两者 fail | 首个应改的控制门控/候选行为 |

## 产物

- `outputs/agent_exchange/D011_20261030/index.html`、`app.js`、`style.css`、`data.json`、`events.json`、`schema.json`、`fault_table.json`、`README.md`
- B/F rich 数据：`rich/B.json`、`rich/F.json`
- 4 次新物理 attempt（2 启动错误 + B/F 各1；新预算 4，已用满）：`attempts_ledger.jsonl`
- 非干扰比对：truth 最大误差 0.0；记录器日志 tick/time bug 已离线按行序纠正。


## 长期 testbed preset

- 8765 testbed 已新增 preset：`D011 T_reposition_2_p0 · 72R→40PUSH_S→16BRAKE`
  - id: `20261030_220220_f69190`
  - URL: http://127.0.0.1:8765/（Presets 列表选择）
  - 内容：scene `reposition_2`，初始位姿同 D011，sequence `42/44×72 → 26/44×40 → 44/44×16`，无 overrides。
  - 复制件：`outputs/agent_exchange/D011_20261030/testbed_preset.json`。
- **m3b 已接入 8765**（controller_version）：
  - `TestbedSession`/API/UI 增加 `controller_version: legacy|m3b`；默认 legacy，旧 preset 不变。
  - m3b preset：`D011 T_reposition_2_p0 · m3b · 72R→40PUSH_S→16BRAKE`，id `20261102_121304_cb379f`；复制件 `outputs/agent_exchange/D011_20261030/testbed_m3b_preset.json`。
  - UI 加载 preset 后会在 controller 下拉选 m3b；run 记录 `controller_version` 与 `unknown_shape.py` hash。
  - 8765 已重启加载新代码；只做了**不步进**构造检查（m3b session 有 estimator，factory 返回 `UnknownShapePush`），没有新增物理轨迹。
  - 详细 candidate/phase/ray 仍看 8766 的 D011 页面；testbed 的通用 UI 不是逐帧候选页。


## 8765 全量迁移 m3b

- `TestbedSession` 与 `/api/run` 现在**强制 m3b**；UI controller 仅 m3b，legacy 不再可选。
- legacy runs 已从 `outputs/testbed/runs` 移出（归档到 `outputs/testbed/runs_legacy_archive/`，不在 testbed 列表）；活动 run 仅保留 m3b 的 `20261102_122345_ad8cb2`。
- m3b run 已补 `candidate / candidate_err_t / contact_phase / contact / cp / desired / reference / force / gate` 每帧字段；UI 新增 `m3b debug` 面板和 q/接触点叠加。
- 8765 已重启；静态资源 `Cache-Control: no-store`。仍无新增物理轨迹；没有改算法/EKF/FF。


## Stage B 单实例修复结果（用户授权后）

- 修复 approach 语义：不再以“到 q 的相对 waypoint 距离”触发 touch；改为：
  - q 随可见支持持续更新；
  - 导航目标为动态 q 外侧安全距离带，安全偏移 `R+0.10`；
  - 只有 q 稳定 ≥3 tick、且到目标机器人落位 `q+outward*R` 的距离 ≤.12m 时，才 final/touch。
- 同时修复 final 触发条件：此前 `safe` 外缘≤.035 会让机器人在安全带内侧永远无法 final；现在按目标落位距离触发。
- 同一 testbed preset、同一脚本：m3b run **success，467 ticks**，run id `20261102_141113_55f9da`。
  - tick 293 q≈(2.238,2.125)；tick 350 后 q 更新到约 (2.503,2.125) 的中点；
  - tick 370 进入 final；tick 410 首次真实接触；之后推进到目标。
- 这是**单实例**结果，不证明其他场景/D011 全体验收；没有跑 40/new/contract。
- testbed 仍强制 m3b，legacy runs 已归档；m3b debug 面板可见 q/e_t/phase/contact/force。


## Stage B 侧向反馈修正

- 之前 contact 分支只有 `t·v_surface` FF，`u_adjust=0`；EKF 表面 FF 在接触后饱和（约 .16），是物体转偏的主导项。
- 加入有界接触点位置反馈：`contact_err_t = t·(q-cp)`，`u_adjust=clip(1.5*err, ±.08)`；并与 FF 分开记录。
- 单变量对照：
  - FF 开：success 473 ticks，物体最大相对角 **0.172 rad**，`u_adjust` 饱和 −.08。
  - **FF 关、只保留位置反馈：success 472 ticks，物体最大相对角 0.004 rad**（run `20261102_141512_5737c7`）。
- 因此 m3b 默认改为 `D009_SURFACE_FF=0`；env 仍可设 1 做旧 FF 对照。testbed 已重启。
- 仍为单实例结果，不证明全场景；下一步才做镜像/dual/扩展。


## Stage B 角度控制

- REPOSITION 继续用 `desired_heading=self.tangent`（切向运动方向）。
- PUSH：approach/waypoint/final 的 heading 取切向运动方向；真实接触后先进入 `orient`，要求 heading 与 support 法向误差 ≤.18 rad 才允许 full push；orient 期间只原地调整、不发力。
- testbed run：`20261102_142038_7965b3`，success 472 ticks；物体最大相对角 **0.00236 rad**。
  - tick 400 touch 前 robot heading≈−1.547 rad（接近 support 法向），接触后保持−1.57；
  - 横向反馈 u_adjust 已量级很小，不再被 FF 主导。
- 仍只验证 T_reposition_2_p0 单实例；不证明其他场景。


## Stage B 安全距离与 heading

- `waypoint` 不是路径规划器；它只是 `safe = q + outward*(R+band)` 的 q 相对安全间距反馈目标，追它用速度反馈 + 现有扫描避障。
- 单变量距离试验：
  - band=.10：success 472，最大角 .004；
  - band=.16：success=False（contact tick 420，固定 8s PUSH 只剩 1.4s），最大角 .0077；
  - **band=.12：success 472，最大角 .0111**，run `20261102_144029_75d0fe`。
- 当前默认 band 取 .12；.16 更安全但当前脚本窗口不够，不能同时宣称成功。
- heading：REPOSITION 跟切向；PUSH approach 跟切向/运动；接触后先 orient 到 support 法向再 push。

## 下一步（等用户）

请打开页面，优先停在 tick 313（F 首触）和 tick 335–356（shift），确认/否定“align 状态下已 contact 并 push”以及“q 与实际接触点不一致”。确认后我们再围绕同一实例逐项改控制；本轮不改算法，不扩 40/new/contract。
