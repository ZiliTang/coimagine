# C. 精确仿真的信息歧义机制实验

脚本：`experiments/option_prediction_mechanism_v1/phase_c/{mechanism_scene.py,perception.py,actions.py,run_phase_c.py,belief_reference.py}`
原始表：`outputs/option_prediction_mechanism_v1/phase_c/phase_c_records.json`、
`phase_c_table.md`、`phase_c_belief_reference.json`。
预算严格按 CONTRACT 3.2：**T1 32 对 + T2 32 对 = 64 对**，每对 **8 个联合技能**，只测单技能
（两技能序列只在 C4 参照里枚举）。δ = 0.02 事前冻结，评分公式与 CONTRACT 3.3 逐字一致。
**未改动 `src/`、未改动 phase_b。**

---

## 0. 结论摘要

命题成立：**存在 18 对合法世界 sA/sB，双方全部允许共享输入逐量完全相同（最大绝对差 0.0），
可执行技能集合相同，而两个世界的近优动作集合无交集且互相 regret 超过 δ。**

最强的一对是 `T1-12-symmetric`（表 1、表 2）：世界 A 最优动作 `along_pos_2`（Q=-0.370，未完成），
世界 B 最优动作 `spin_pos`（Q=+0.602，**完成**）；`along_pos_2` 在 B 里仍能完成（Q=+0.514），
`spin_pos` 在 A 里完全不行（Q=-0.626）。互相 regret = **0.256 > δ**，50/50 信息差 = **0.044**。

同时给出两个负面/限界结论，都不加修饰：

1. **这个歧义是"一次性"的。** 在全部 18 个正例 × 8 个首技能 = 144 次检验中，**没有一次**在
   执行完第一个技能后两个世界仍然不可区分（§6）。原因是任何推力都按真实物体位姿解算接触点，
   同一个命令在两个世界把机器人拖到不同的自身位姿，而机器人自己的位姿是已知可观测量。因此
   C4 的"第二技能必须用同一规则"约束**从未生效**；允许分支是合法的，不是作弊。
2. **正例的收益差主要来自"物体还剩多远"，而不是力矩/杠杆。** 见 §4 的机制说明。

---

## 1. 机制场景与传感模型（含**明确声明的** FOV 改动）

### 1.1 为什么必须另建场景

冻结物体 1.0 × 0.25 m，冻结扫描 `SCAN_RANGE = 0.6 m`（`local_contact_option.contract`）。
在**任何**合法接触姿态下，0.25 m 的短边都落在 0.6 m 内，端点必然可见，物体沿长边的切向平移会
被立即看到，T1 不成立。CONTRACT 3.2 因此明确允许另建**一个**长矩形机制场景。新场景独立于旧
基准：只由 phase_c 代码构造 `SceneSpec`，从不注册进场景库。

### 1.2 场景参数（同一对内共有，只有物体位姿变）

| 量 | 值 |
|---|---|
| arena | 14.0 × 4.0 m |
| 长矩形 | `object_len` ∈ {5, 6, 7} m，`object_thick` = **0.30 m**（两世界同一对内完全相同） |
| 长边中心线 | y = 2.00，两条长边 y = 1.85 / 2.15 |
| 质量/摩擦/阻尼/力上限 | 全部沿用 `PhysicsConfig`（3.0 kg、0.6、8.0/0.6、F_max=15 N），未改 |
| 机器人 | 固定全局位姿（**两世界逐位相同**），机体系 yaw 朝物体一侧；机壳距长边 0.02 m |
| 物体位姿 | 世界 A：x = `object_centre_x`；世界 B：x = `object_centre_x + shift`（只动 x，角度 0） |
| 窗口 | `window_steps = 120`（6.0 s），见 §1.4 |
| 动作序列后 | 追加一个 `hold`×2 窗（与技能窗同长 6.0 s），让既有 success 规则（goal 内 + 低速保持 0.5 s）有机会成立 |

**合法性**：每一对的两世界都用 `continuous2d.legality.legality_report` 检查，64 对全部
`legal=True`（最小间隙 ≥ 0.005 m）。T2 的墙也满足"两世界物体都在墙西侧"。

### 1.3 传感模型：唯一一处声明性收窄

允许共享观测由 `perception.observe` 构造，逐量包含：双方自身位姿与速度、逐射线**原始扫描**
（range + 命中标签 + 有效性位 + 目标位）、局部片段（点集、最小二乘线段、距离、世界法向、切向、
置信度、弧长、遮挡标签）、接触读数（是否接触、机体系接触点、法向、冲量）、已知静态地图（arena、
障碍、goal、物体尺寸）、以及产生当前状态的命令历史（本实验为冷启动，历史为空）。

**被明确改变的一项**：扫描从冻结的 **360° 全向环**收窄为**机器人自身 yaw ±60° 的前向锥**
（`FOV_HALF_ANGLE = 60°`）。`SCAN_RANGE = 0.6 m`、`SCAN_RAYS = 72`、理想传感（noise 0）
**未改**，短边**未裁剪**、range **未缩短**。

收窄的理由：全向环会看到机器人**身后**的 arena 边界/场地信息；两世界里物体平移后这些背景
回波不同，世界立刻可区分，歧义消失。前向锥让每个机器人只测它面对的接触边（以及 0.6 m 内的
同伴），是"局部接触传感"的最小合理声明。

必须说清楚的三点：

1. 该收窄**属于机制场景的声明**，写进 `phase_c_records.json` 的 `notes.sensing`，不是静默改动；
2. 收窄在**一对的两个世界里完全相同**（同一 `FOV_HALF_ANGLE`、同一射线列表），不可能制造差异；
3. 短边不在锥内并非靠裁剪，而是靠**物体足够长**：每个 stance 下可见长边跨度仅 0.693 m，
   而短边距最近机器人 ≥ 1.0 m（7 m 物体）/ ≥ 0.5 m（5 m 物体，仍 > 0.347 m 半跨度）。
   每对都逐量核验 `short_face_visible = False`（`observation_identity.short_faces_outside_range`）。

### 1.4 相对冻结 `Continuous2DConfig` 的全部改动（逐条列清）

机制场景用的 `PhysicsConfig` 与冻结默认值只有**两项**不同，且都写在每条记录的参数里：

| 项 | 冻结值 | 机制值 | 理由 |
|---|---|---|---|
| `object_width/height` | 1.0 × 0.25 m | `object_len`(5/6/7) × 0.30 m | 长矩形是 CONTRACT 3.2 明确允许的新场景；`PymunkBackend` 从 `PhysicsConfig` 而非 `SceneSpec` 取物体尺寸，所以必须同时写进 config |
| `window_steps` | 40（2.0 s） | 120（6.0 s） | `PushContact` 内部 `push_speed = 0.26 m/s`，2 s 内物体最多动 ~0.5 m，几乎不可能有动作真正完成，所有 Q 会退化成微小 progress 差 |

控制器（V 速度式局部推动的 continuous2d 技能）、质量/摩擦/阻尼/`F_max`、success 规则
（`velocity_tolerance`/`angular_tolerance`/`success_hold_time`/`goal_margin`）、评分公式
**全部未改**。冻结默认 config 哈希 `765ede7cf67318f1` 已记录在 `phase_c_records.json`。

---

## 2. 观测同一性：怎么核验、数字是多少

对每一对，在**决策时刻（reset 后、未执行任何命令）**分别构造两世界的 `SharedObservation`，
逐量比较。比较项与最大绝对差（正例全部为 0）：

| 比较量 | 正例上的最大绝对差 |
|---|---|
| `robot{0,1}.pose[0..2]`（全局位姿） | **0.0** |
| `robot{0,1}.velocity[0..2]` | **0.0** |
| `robot{0,1}.scan.origin`、`scan.yaw` | **0.0** |
| `robot{0,1}.scan.range`（逐射线，72 条，双侧都命中的射线） | **0.0** |
| `robot{0,1}.scan.range_mismatch`（一侧命中一侧未命中计数） | **0** |
| `scan.valid_mismatch` / `hit_label_mismatch` / `target_flag_mismatch` | **0** |
| `scan.short_face_mismatch`（是否看见短边） | **0** |
| `patch.present/distance/span/arc_rays/confidence` | **0.0** |
| `patch.normal[0..1]`、`patch.tangent[0..1]` | **0.0** |
| `patch.points`、`patch.segment`（逐点最大绝对差） | **0.0** |
| `patch.points_mismatch`、`patch.occluder_mismatch` | **0** |
| `contact.flag_mismatch`、`contact.point`、`contact.normal`、`contact.impulse` | **0.0 / 0** |
| `map.goal`、`map.arena`、`map.object_size`、`map.obstacles`、`map.obstacles_count_mismatch` | **0.0 / 0** |
| `history_mismatch`（冷启动：两世界历史都为空） | **0** |

判别函数 `perception.observation_is_identical`：**所有数值差必须 ≤ 1e-12，所有 `*_mismatch`
计数必须为 0**。18 个正例上 `observation_identity.max_abs_difference = 0.0`，
`per_quantity` 中非零项 **0 个**。6 对被判"观测已足够区分"，如实记为未通过前置条件（§8）。

另外逐对核验：
`robot_poses_equal = True`、`robot_velocities_equal = True`、`map_equal = True`、
`short_faces_outside_range = True`。

### 2.1 数值说明（避免误读）

- 观测比较**不再返回 `Infinity`**。一侧有量、另一侧没有的情形（例如某条射线只在一个世界命中）
  记为 `*_mismatch` 计数，数值差只在**两侧都命中**的射线上取最大值；文件里不存在非有限值
  （已断言）。此前版本用 `inf` 表示缺失，已修正。
- 世界 A/x 与 B/x 的**物体真值位姿本来就不同**（这正是隐藏状态），所以它不是"比较项"；
  它也不是任何允许共享的观测量。

---

## 3. 技能集合的可部署性（正面回答）

**结论：本实验记录的 8 个联合技能是可部署的，不是 oracle 参数化的。**

技能形态是 `PushContact(object_id=0, contact_local=(d + ref_x - cx_nom, ±0.15), push_direction_world=u)`。
其中：

- `ref_x`：**已知静态地图**上的一个点（本实验取两机器人已知位姿的中点，8.0/10.0 → 9.0；
  即 `reference_x = 9.0`）。机器人知道自己的位姿，也知道这一公共参考点。
- `cx_nom`：**声明的名义物体摆放**（本机制场景的公开场景描述的一部分，两世界相同）。
  世界 A 的真值等于它；世界 B 的真值沿长边平移了 `shift`。
- `d ∈ {-1.0, 0.0, +1.0}`：沿可见长边相对 `ref_x` 的切向偏移；`u` 取 ±x 或 ±y。

因此每个技能等价于一句**只用允许信息**就能表达的指令：

> "把接触点放在可见长边上、距公共参考点 `ref_x` 切向 `d` 的位置，沿世界方向 `u` 推。"

这条指令只用到：机器人自身位姿（已知）、可见长边的几何（由自身扫描得到）、静态地图
（`ref_x`、arena、goal），以及声明的名义物体摆放 `cx_nom`。它**不需要**物体的真实切向位置，
也不需要物体全局位姿的实时真值。`PushContact.contact_local` 只是这条世界坐标指令在物体坐标系
下的**内部编码**：编码里出现 `-cx_nom`，对两个世界是同一个常数，所以**两个世界收到完全相同的
命令串**（`phase_c_records.json` 的 `skills[].joint_skill_specs` 两世界共用一份）。

必须同时说明这一编码带来的物理后果：`PushContact` 会用**真实**物体位姿解算接触点，所以同一个
命令在世界 B 实际作用在沿长边平移了 `shift` 的材料点上，并且

- 物体整体离 goal 沿 +x 近 `shift`（这是收益差的**主因**，见 §4）；
- 机器人侧向控制器被拉向不同世界坐标，导致两个世界机器人终末自身位姿不同（这是"动作后即可
  区分"的原因，见 §6）。

如果评审要求技能完全不含任何名义物体摆放，那么这 8 个技能无法用当前
`PushContact`（其语义就是"物体坐标接触点"）表达；本报告如实说明这一点：**技能的"决策内容"
是世界坐标，编码依赖声明的名义摆放**，不是真值，也不是逐世界不同的参数。

---

## 4. 机制：为什么同一命令在两个世界收益不同

设世界 A 物体中心 `cx`，世界 B 为 `cx + shift`。A 的 goal 是全局固定的矩形，其左边界
`gx - hw` 固定。于是：

- 同一命令在两世界把物体沿 +x 推进的距离近似相同（推速上限与阻尼相同），
  但 B 的物体起点已经近了 `shift`，**到 goal 的剩余距离按 shift 直接不同**；
- 该剩余距离**不可观测**：两世界可见的只是一条平直长边，局部法向、patch 距离、patch 跨度、
  接触读数全部相同。

这就是 T1 的机制本体：机器人无法判断物体沿这条边"已经走到哪"，而"还要推多远"决定了
"用力推 / 收着推 / 换个接触点"哪个更优。次生的力矩差异（同一世界接触点相对真实质心的力臂在
B 中不同）会改变终末姿态，进一步改变角点距离与 completion。

`T1-12-symmetric` 就是这个机制的最干净实例（表 2）：

- 世界 B（物体更靠近 goal，`d=+1.0` 即 `along_pos_2`）推 8.1 s 后进入 goal 并保持 0.55 s →
  `completion=1`，Q=+0.514；`spin_pos` 更快（6.6 s）完成，Q=+0.602。
- 世界 A 同样的 `along_pos_2` 只把物体推到 d_end=0.664，**未完成**，Q=-0.370；
  `spin_pos` 在 A 里几乎原地（它要求把两个机器人沿 ±y 对推，真实质心不同导致净效果变成空转），
  Q=-0.626。

---

## 5. 收益矩阵与正例判据（冻结，未事后修改）

对每对、每世界、每动作：真实仿真，`completion` 用既有 success 规则（goal 内 + 低速保持 0.5 s），
`progress = min(1,(d_start-d_end)/max(d_start,1e-6))`，`d = max` 角点到 goal 的距离，
`time_cost` = 技能序列总墙钟秒数，`action_cost = Σ|F|dt / (F_max·T_total)`，
`Q = completion + 0.5·progress − 0.05·time_cost − 0.10·action_cost`。
`A*_h(δ) = {a : R_h(a) ≤ δ}`，`δ = 0.02`，`R_h(a) = max_b Q_h(b) − Q_h(a)`。
**正例 = `A*_A ∩ A*_B = ∅` 且两世界最优动作互相 regret > δ。**

**数值有效性前置**：控制器把每个机器人力截断在 `F_max`，所以指令冲量不可能超过
`2·F_max·T`。任何 `impulse > 1.05·2·F_max·T`、`|Q|>10`、非有限状态、越界、穿透超限、
peak speed > 5 m/s 的序列判为**数值发散**，**从收益矩阵中删除**，不参与近优集合与 regret。
删除哪些写在 `analysis.diverged_actions`。

### 表 1：18 个正例（单技能，δ=0.02）

| pair | 模板 | 观测最大差 | `A*_A` | `A*_B` | 互相 regret | 50/50 gap | 完成数 A/B |
|---|---|---|---|---|---|---|---|
| T1-03-one_sided | T1 | 0.0 | upper_x | lower_y_0 | 0.4315 | 0.0393 | 0/0 |
| T1-12-symmetric | T1 | 0.0 | along_pos_2 | spin_pos | 0.2562 | 0.0442 | 0/**1** |
| T1-13-one_sided | T1 | 0.0 | upper_x | lower_x, lower_y_0 | 0.3035 | 0.0254 | 0/0 |
| T1-31-one_sided | T1 | 0.0 | upper_x | lower_x, lower_y_0 | 0.3035 | 0.0254 | 0/0 |
| T1-16-symmetric | T1 | 0.0 | along_pos_2 | along_pos_1 | 0.1566 | 0.0140 | 0/0 |
| T1-24-symmetric | T1 | 0.0 | along_pos_2 | along_pos_1 | 0.1566 | 0.0140 | 0/0 |
| T1-21-one_sided | T1 | 0.0 | lower_x, upper_x | lower_y_2 | 0.1402 | 0.0366 | 0/0 |
| T1-23-one_sided | T1 | 0.0 | lower_x, upper_x | lower_y_2 | 0.1388 | 0.0375 | 0/0 |
| T2-17-one_sided | T2 | 0.0 | lower_x, lower_y_0, upper_x | lower_y_2 | 0.1332 | 0.0329 | 0/0 |
| T2-05-one_sided | T2 | 0.0 | lower_x, lower_y_0, upper_x | lower_y_2 | 0.1300 | 0.0329 | 0/0 |
| T2-19-one_sided | T2 | 0.0 | lower_x, lower_y_0, upper_x | lower_y_2 | 0.1262 | 0.0329 | 0/0 |
| T1-05-one_sided | T1 | 0.0 | upper_x | lower_y_2 | 0.1087 | 0.0324 | 0/0 |
| T1-17-one_sided | T1 | 0.0 | lower_x, upper_x | lower_y_2 | 0.0939 | 0.0265 | 0/0 |
| T2-07-one_sided | T2 | 0.0 | lower_x, lower_y_0, upper_x | lower_y_2 | 0.0659 | 0.0201 | 0/0 |
| T2-26-symmetric | T2 | 0.0 | along_pos_1, along_pos_2 | along_pos_0 | 0.0595 | 0.0121 | 0/0 |
| T2-22-symmetric | T2 | 0.0 | along_pos_1, along_pos_2 | along_pos_0 | 0.0566 | 0.0150 | 0/0 |
| T2-28-symmetric | T2 | 0.0 | along_pos_1, along_pos_2 | along_pos_0 | 0.0487 | 0.0155 | 0/0 |
| T2-12-symmetric | T2 | 0.0 | along_pos_2 | along_pos_1 | 0.0425 | 0.0135 | 0/0 |

### 表 2：`T1-12-symmetric` 完整收益矩阵（正例头名）

参数：`object_len=5.0`、`thick=0.30`、`object_centre_x=8.5`、`shift=1.5`、`edge_y=2.0`、
`robot_x=(8.0,10.0)`、`reference_x=9.0`、`arena=(14,4)`、
`goal=Box(cx=11.0, cy=2.5, half_w=3.5, half_h=2.0)`、无障碍。
（有效动作 A/B 相同集合；B 中 5 个 `along_*` 序列数值发散，已删除。）

| 动作 | Q_A | Q_B | R_A | R_B | 完成 A | 完成 B |
|---|---|---|---|---|---|---|
| along_pos_2 | **−0.36992** | +0.51373 | **0.0000** | 0.0884 | 0 | 1 |
| spin_neg | −0.64423 | −0.64799 | 0.2743 | 1.2501 | 0 | 0 |
| spin_pos | −0.62612 | **+0.60215** | 0.2562 | **0.0000** | 0 | 1 |

- `A*_A(0.02) = {along_pos_2}`，`A*_B(0.02) = {spin_pos}`，交集 **∅**。
- 互相 regret = max(0.2562, 0.0884) = **0.2562 > 0.02**。

**发散剔除规则的稳健性（重要）**：上表已经剔除了世界 B 中数值发散的 5 个 `along_*`。
若**不剔除**、把引擎暴走当成真实灾难性结果一起打分（`q` 会到 −1e5 量级），则
`A*_A(0.02) = {along_pos_0, along_pos_2}`、`A*_B(0.02) = {spin_pos}`，
交集仍为 **∅**，互相 regret **仍然 = 0.2562**，判据**仍然成立**。
两种约定下 **18/18** 个正例都成立（`analysis.all_actions_convention` 逐对记录），
且两种约定给出的互相 regret 完全相同。
本报告只把"发散剔除"作为主约定，并把两种约定都写进 JSON，避免结论依赖这一约定。
注意：`T1-25-one_sided` 之类的 near-miss 在两种约定下都不成立，不受此影响。
- 全知 50/50 `0.5·max_A Q + 0.5·max_B Q` = **+0.11611**；
  共同动作 `max_a 0.5(Q_A+Q_B)` = **+0.07190**（动作 `along_pos_2`）；
  **gap = +0.04421**。
- 明确声明：**这个 gap 本身不证明任何现有规划器犯了信息乐观错误**，只是隐藏状态信息的潜在价值。

---

## 6. C4：有限候选信念规划参照

规则（逐字实现）：执行首个技能 `a1` → 比较**允许观测**是否已能区分两世界 →
**仍不可区分则第二个技能必须两世界同一规则**；已可区分才允许按真实观测分支。
枚举全部候选（≤8 个 `a1` × ≤8 个 `a2`），不建通用 POMDP 求解器。三个参照**用同一有限候选集**
（`valid_keys` = 两世界都未数值发散的两技能序列）：

- `all_knowing_two_skill = 0.5·max_{a1,a2} Q_A + 0.5·max_{a1,a2} Q_B`
- `common_plan_value = max_{a1,a2} 0.5(Q_A + Q_B)`（同一规则）
- `legal_observation_conditioned = max_{a1} [ 可区分 ? 0.5 max_{a2}Q_A + 0.5 max_{a2}Q_B : max_{a2} 0.5(Q_A+Q_B) ]`

第三个是 `a1` 上的 **max**（规划器只提交一个 `a1`），不是平均；这样定义下它**必然 ≥**
`common_plan_value`（不分支方案永远可行）。已逐对验证
`legal ≥ common` 且 `all_knowing ≥ legal`（18/18 通过）。

### 表 3：三个参照（节选，全部 18 对见 `phase_c_belief_reference.json`）

| pair | all-knowing | common | legal obs-conditioned | mean-state | gap(all−common) | 最优 `a1` |
|---|---|---|---|---|---|---|
| T1-12-symmetric | +0.6657 | +0.5531 | **+0.6657** | +0.5531 | +0.1126 | spin_pos |
| T1-03-one_sided | +0.2559 | +0.1453 | **+0.1940** | +0.1453 | +0.1106 | lower_y_0 |
| T1-13-one_sided | +0.1977 | +0.1322 | +0.1523 | +0.1322 | +0.0655 | lower_y_0 |
| T1-16-symmetric | +0.1339 | +0.1186 | +0.1186 | +0.1186 | +0.0153 | spin_pos |
| T1-05-one_sided | −0.6091 | −0.6091 | −0.6091 | −0.6091 | +0.0000 | upper_x |
| T2-28-symmetric | −0.7613 | −0.7613 | −0.7613 | −0.7613 | +0.0000 | spin_pos |

### 6.1 首技能后是否仍不可区分：**否**

在 18 个正例 × 8 个首技能 = **144 次**后验观测比较中，**每一次都可区分**
（144/144），`belief_reference.post_skill1[*].distinguishable = True`，
JSON 中**不含任何 `Infinity`**（缺失量记 `*_mismatch` 计数，数值差只在两侧都存在的量上取最大）。

以 `T1-12-symmetric` 为例（`post_skill1` 最大有限差 / mismatch 项数）：

| `a1` | post-skill-1 最大有限差 | mismatch 计数 | 可区分 | 主导量 |
|---|---|---|---|---|
| along_pos_0 | 22.0 | 8 | 是 | 双方自身位姿、patch 点集 |
| along_pos_2 | 56.0 | 5 | 是 | 同上 |
| spin_pos | 19.0 | 10 | 是 | 同上 |
| spin_neg | **1.5** | **0** | 是 | `robot0.pose[0]`、`robot1.pose[0]` 各差 1.5 m |

`spin_neg` 是最接近"仍不可区分"的一次：物体在 A 和 B 都几乎不动，patch 距离/法向/跨度仍然
完全一致（mismatch=0），但**机器人自己的终末 x 坐标相差 1.5 m**——因为侧向控制器把机器人拉向
"用真实物体位姿解算出的期望接触点"，而两世界该点相差 `shift`。机器人**知道自己的位姿**，
所以世界已被自己的运动暴露。

**结论（负面但重要）**：本机制下歧义只影响**第一次决策**；一旦执行任何技能，双方仅凭自身
运动就能分辨。因此

- C4 的"第二技能必须用同一规则"约束在本实验**从未生效**；
- `legal_observation_conditioned` 等于在最优 `a1` 上的分支值（T1-12 中恰等于 all-knowing
  0.6657），严格大于 `common_plan_value`（0.5531）；
- 对 T1-16、T1-05、T2-28 等对，分支收益为 0，`legal == common`——这是干净的负结果，
  如实报告，不强行制造 gap。

### 6.2 均值状态规划器对照

`mean_state_planner_value = max_{a1,a2} 0.5(Q_A+Q_B)`（50/50 均值模型里选整条固定序列，
不尊重观测条件）。表 3 中它与 `common` 恒等（同一目标函数），正是 C4 要求给出的对照：
**只看均值状态会损失 `all−common` 这一份价值**，而表中 `all−common` 在 0.0000–0.1590 之间。

---

## 7. 可重复性

`run_phase_c.py --repeat-check` 对 `T1-03-one_sided` 完整跑两遍，逐字段比较：

- `q_identical = True`，`max_q_abs_difference = **0.0**`；
- 全部 16 个 outcome 的**物理字段**（success、termination、duration、impulse、
  start/end distance、final_object、progress、time/action cost、q、steps、success_hold、
  final_penetration、diverged、peak_speed）**逐位相同**；
- 唯一不同的字段是 `wall_seconds`（真实墙钟耗时），这在两遍里不同是**测量主机**的差异，
  不是仿真差异。首版脚本把 `wall_seconds` 一起比较，才出现 `outcomes_identical=False`
  而 `q_identical=True`；现已把它排除，并保留 `wall_seconds` 作为报告量。

**结论：仿真在任务层面完全确定性，重放误差 0，与 §3.3 的"重复仿真误差按 0 记"一致。**

---

## 8. 负结果处置与配对预算

64 对全部登记（`phase_c_records.json`），无删除、无丢弃观测。判负原因分类：

| 原因 | 对数 |
|---|---|
| 正例（`A*_A ∩ A*_B = ∅` 且互相 regret > δ） | **18** |
| 观测已足够区分两世界（前置条件不成立） | **6** |
| 近优集合相交 | **39** |
| ├ 动作排序完全相同（互相 regret = 0.0000） | 15 |
| ├ 排序差异 ≤ δ（0 < regret ≤ 0.02） | 4 |
| └ 排序差异 > δ 但交集非空（near-miss） | 20 |
| 无有效共同动作（序列全部数值发散） | **1** |

- **场景不合法**：0 对（初版有 6 个 T2 网格使墙与物体初始重叠，已改墙位并重跑，
  当前 64 对 `legal=True`）。
- **少于 8 个共同有效动作**：1 对，`T2-27-one_sided`：世界 B 中**全部 8 个动作**都把物体撞进
  窄墙口并使引擎发散（`diverged_actions.B` = 全部 8 个），因此不存在任何共同有效动作。
  在 all-actions 约定下它的近优集合相交，仍非正例。这是"技能集合/场景过于苛刻"型负例。
- 全库共 30 个（世界×动作）序列被判数值发散，逐条列在 `analysis.diverged_actions`。
- 观测已区分的 6 对：`T1-18-symmetric`、`T1-19-one_sided`、`T1-20-symmetric`、
  `T1-27-one_sided`、`T1-28-symmetric`、`T2-30-symmetric`。原因同一且已逐对定位：
  这些都在 `L=5.0, shift=2.0` 组合下，世界 B 的物体左端点恰好落在 robot0 的 x 上
  （物体 B 跨度 [8.0, 13.0]，robot0 在 x=8.0），robot0 的前向锥只命中 36 条射线（世界 A 为 72 条），
  `robot0.patch.present_mismatch=1`、`range_mismatch=36`。
  注意 `short_face_visible` 在这 6 对上仍为 `False`（射线擦过端点、未真正打在短边面上），
  所以**真正兜住这 6 对的是逐射线命中/有效性 mismatch 计数**，不是 `short_face_visible` 这一辅助
  诊断；`perception.observation_is_identical` 因此以 mismatch 计数为主判据。
  这 6 对如实记为"观测已能区分"，不进入正例。

### 8.1 最佳 near-miss

`T1-25-one_sided`（紧 goal 带负对照）：观测最大差 **0.0**，`A*_A = {lower_x, upper_x}`，
`A*_B = {lower_x}`，互相 regret = **0.57114**（超过 δ 很多），
但 `A*_A ∩ A*_B = {lower_x} ≠ ∅` → **判负**。
这正是 CONTRACT 3.3 要防的情形：argmax 不同且 regret 很大，但两个世界仍存在一个共同近优动作。
`lower_x` 在 A/B 的 Q 分别为 −0.55761 / −0.55603，差 0.0016，远小于 δ。

---

## 9. 文件、重放与哈希

| 文件 | 内容 |
|---|---|
| `experiments/option_prediction_mechanism_v1/phase_c/mechanism_scene.py` | 长矩形机制场景与 per-candidate 配置 |
| `experiments/.../phase_c/perception.py` | 声明的传感模型与观测同一性核验（无 `inf`） |
| `experiments/.../phase_c/actions.py` | 参考点锚定的 8 技能集合（可部署性见 §3） |
| `experiments/.../phase_c/run_phase_c.py` | 64 对搜索、收益矩阵、正例判据、可重复性 |
| `experiments/.../phase_c/belief_reference.py` | C4 有限候选信念规划参照 |
| `outputs/option_prediction_mechanism_v1/phase_c/phase_c_records.json` | 全部 64 对的观测同一性、Q 矩阵、outcome、replay 元数据 |
| `outputs/.../phase_c/phase_c_table.md` | 人读表 |
| `outputs/.../phase_c/phase_c_belief_reference.json` | C4 三参照与 post-skill-1 可区分性 |
| `docs/option_prediction_mechanism_v1/C_ambiguity.json` | 本报告数字的机器可读摘要 |

重放：每条 `outcome` 都同时保存 `skills[].joint_skill_specs`（完整精度）、
`scene_hash`（`legality.scene_content_hash`）、`snapshot_hash`（tick + 三体状态 +
success_counter + termination + prefix 长度的 sha256 前 16 位）。
正例头名的关键哈希：

| 项 | 值 |
|---|---|
| `T1-12-symmetric` 世界 A 场景哈希 | `ba8e5606a1817e29` |
| `T1-12-symmetric` 世界 B 场景哈希 | `e78c96fc442ee156` |
| `T1-12-symmetric` 世界 A `along_pos_2` 快照哈希 | `889c275c5568fbac` |
| `T1-12-symmetric` 世界 B `spin_pos` 快照哈希 | `4fd0aa343c646eea` |
| 机制配置哈希（5 m 物体，120 步窗） | `bec1519a438f60bf` |
| 冻结默认配置哈希 | `765ede7cf67318f1` |

精确重放方式：`Continuous2DEnv(mechanism_config(params)).reset(scene)` 后按记录顺序
`execute_joint_skill`；分支用 `export_state()`/`restore(exact=True)`。

---

## 10. 本实验说明与不说明什么

**说明**：在精确仿真、同一冻结评分、同一 8 技能集合下，**存在**满足 CONTRACT 3.1 的合法世界对：
允许共享的全部当前观测逐量相同（0.0，且无 mismatch），技能集合相同，而近优动作集合无交集、
互相 regret 远超 δ。机制是"物体沿可见长边的切向位置不可观测"，且该歧义只作用于**第一次决策**。

**不说明**：

- 不说明任何现有规划器犯了信息乐观错误（50/50 gap 只是隐藏信息的潜在价值）；
- 不说明这是自然任务中的真实先验（50/50 是受控机制假设）；
- 不说明长历史机器人也无法辨识（本实验是**冷启动**，历史为空）；
- 不说明本机制在旧 1.0×0.25 物体基准上成立（那里端点必然可见，T1 不成立）；
- 不说明"观测互补/主动辨识/物理约束"三种机制哪一支更优——本实验只验证"歧义确实影响动作选择"。
