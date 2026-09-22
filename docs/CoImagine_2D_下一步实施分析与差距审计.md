# CoImagine 2D 第一阶段：现状审计、差距分析与下一步实施建议

依据：`docs/CoImagine_v6_研究方案与创新评估.md`（长线方案）与
`docs/CoImagine_2D_Agent_执行提示_v1.md`（第一轮验收范围）。
方法：通读 `src/`、`tests/`、`scripts/`、`configs/`、`docs/` 并实跑测试。未修改任何源码。

---

## 0. 一句话结论

既有仓库实现的是 **v4/v5 的 Gate A**：*分布式局部信念想象 + 演化协作拓扑*，其世界是**确定性几何**的、信念是**三值符号 + 乐观自由空间闭合**的、**没有任何通信**。

v6 + 2D 执行提示所要求的第一轮核心——**隐藏物理类型 θ、贝叶斯后验、有代价的辨识、B bits 消息与码本、`O_full/O_central/O_codec/O_B` oracle 体系、未来观测/消息分支**——在代码中**几乎全部不存在**（只存在于两份方案文档里）。

因此下一步不应在 Gate A 的概率缺失处打补丁，而应：

1. 冻结现有 Gate A 作为已完成的另一条研究结果；
2. **新建一个概率化协议核心**（类型层 + 微型生成模型 + 精确 Bayes/oracle）；
3. 复用环境几何、候选生成、评测/统计/绘图/CLI 等基础设施；
4. 第一优先、信息量最大、最快能证伪的动作是 **阶段一的微型精确 oracle（M0–M3）**，而不是 grid、更不是学习模型。

---

## 1. 两份文档的关系与第一轮应遵循的范围

v6 是研究背景与长线路线；2D 执行提示 v1 定义第一轮的验收边界。二者冲突时以 v1 为准：

| 主题 | v6 方案 | 2D 执行提示 v1 | 第一轮遵循 |
|---|---|---|---|
| 3D 主仿真 | 核心实验 B | 明确不做，停止在 2D | v1 |
| 学习 pilot | 5k/20k/50k 技能转移 | 本轮不引入深度学习/神经世界模型 | v1 |
| 通用先验 / 大模型 API | E1 工程验证 | 只保留可替换接口，规则排序即可，默认不开 API | v1 |
| 双臂 / 实机 | 后续实验 C/D | 不涉及 | v1 |
| 交付物 | 完整方法论文证据链 | 微型 oracle + grid 可回放 + 最小对照 + 诊断报告 | v1 |

> 结论：**第一轮 = spec + 微型精确诊断 + 小 grid 对照 + 诊断报告**。不要一次实现 v6。

---

## 2. 既有代码审计

### 2.1 已实现、可复用的资产

- **几何与动力学原语**：遮挡感知 ray-cast 可见性 `visible_cells_from_pose`、`_line_cells`
  （`src/coimagine/envs/grid/environment.py:86-119`）、swept-region/占位/边界检查、
  `shortest_path`（换位路径非瞬移）、同向协推校验 `_validate_push_group`（`:251-289`）。
- **动作/候选层**：`OptionKind` 六种（`core/types.py:30-36`）；`LocalOption`/`JointOption`
  每机一动作且强制完整（`types.py:91-156`）；候选 top-K + 分类型多样化
  （`options/generator.py:376-411`）；程序化场景生成器（`envs/grid/generation/procedural.py`，2065 行）。
- **评测/报告基础设施**：多进程 runner、实时进度与 ETA、Wilson/分位数/配对 bootstrap/
  McNemar、CSV/JSON/MD 表导出、matplotlib 图、`--render-only` 恢复、manifest/resume、CLI
  （`evaluation/`、`reporting/`、`cli.py`、`scripts/`）。
- **测试基建**：36 个测试，pytest 配置就绪（`pyproject.toml:28-31`）。

### 2.2 与 v6/2D 需求的逐项差距

| v6 / 2D 需求 | 代码现状 | 证据 |
|---|---|---|
| 隐藏物理类型 θ（阻力 r∈{1,2}、约束模式、局部牵引） | **不存在**。`required_pushers` 是公开确定值；`mass_class` 只是渲染标签 | `state.py:15-23`；`procedural.py:1536` |
| 随机/联合观测模型 P(o_A,o_B,y_pub∣s,a,s′) | **不存在**。确定性 Bresenham ray-cast，无噪声、无似然表、无 RNG | `environment.py:86-119,121-165` |
| 精确贝叶斯后验 + 私有历史 h_i | **不存在**。三值符号 filter，纯集合赋值 | `belief/symbolic.py:13-16,35-57,147-215` |
| 联合公共状态 P(s,h_A,h_B∣C) | **不存在**。planner 直接拿真值 `GridState` | `environment.py:301-313,485` |
| B bits 消息 / 码本 / 实际位计数 | **完全不存在**（全仓库无 bits/bandwidth/message 计量） | grep 0 hits |
| O_full / O_central / O_codec / O_B / TrueModelLimitedSearch | **不存在**（仅文档）。只有 `PRIVILEGED_IET_ORACLE` 与 `FULL_TUPLE_ORACLE`（后者是笛卡尔 admission 参照） | `core/types.py:53-67`；`topology/admission.py:339-349` |
| 未来观测/消息分支（信息价值） | **不存在**。值递推是纯 `max`，单一后继信念 | `planning/exact_search.py:801,857,414-467` |
| 有代价的试探 / 主动辨识 | **不存在**。`OBSERVE` 是确定性 reveal 门，只作平局 tie-break | `environment.py:355-358`；`exact_search.py:341-350` |
| TrueState / PublicContext / PrivateHistory / OracleContext 类型隔离 | **不存在**。仅靠约定；真值直达 planner，且 `TaskSpec` 把 `hidden_entities` 广播给所有机器人 | 文档 `:81`；`observation.py:34` |
| 信息指标（bits / Brier / NLL / oracle regret） | **不存在** | 审计结论 |
| 通信预算曲线 / 试探成本扫描 / 回放 GIF | **不存在** | 审计结论 |

### 2.3 顺带发现的语义与工程问题

- `allowed_push_directions` 在 `_validate_push_group` 检查，但在 `execute_optimistic` 中**不检查**
  （`environment.py:265-266` vs `:373-395`）——两套执行路径不一致。
- `TaskSpec.activation_dependencies` 已声明并被生成器填充，但**动力学从不使用**
  （`state.py:40`；`procedural.py:1871`）——只进缓存键与渲染标签。
- `reporting/main_gate_a.py`（649 行）无任何 importer，是死代码。
- README 描述的输出目录与磁盘实际不符；controlled 系列的 `episode_results.csv` 是 0 字节
  （历史运行都带 `--skip-controlled`），manifest 中 `open_instances=0`。
- **术语陷阱**：现有 "hidden" 指"被遮挡、需 `Observe` 揭示的实体"，**不是**隐藏物理参数。
  新 spec 必须用不同词汇，避免与 v6 的"隐藏接触证据"混淆。

---

## 3. 复用 vs 新建的边界

**复用（不改语义）**
- 几何/碰撞/换位路径/协推校验的实现思路与工具函数；
- 候选生成骨架（但必须改成"只依赖公共几何"）；
- 程序化地图/障碍/机器人布局生成；
- 评测统计、绘图、CLI、进度、manifest、测试基建。

**新建（不要改造旧 core）**
- `protocol/`：`TrueState` / `PublicContext` / `PrivateHistory` / `OracleContext` 等强隔离类型；
- `toy/`：M0–M3 微型有限状态生成模型与转移/观测表；
- `exact/`：精确 Bayes 滤波器、离散码本、`O_full/O_central/O_codec/O_B/TrueModelLimitedSearch`；
- codec 与位计数、oracle 差距分解、预算/成本/noise 扫描。

**冻结保留（不删、不合并）**
- `topology/`（factors/admission/exact_verifier）、CST/IET 主对比、scaling。
  理由：旧 belief 是"集合 + 乐观闭合"，旧 planner 是"max + 确定 rollout"；
  把概率硬塞进去会让 **Gate A 的"想象未来观测"主张**与 **v6 的"消息/信念表示"主张**相互纠缠。

---

## 4. 下一步实施顺序与验收

### Stage 0 — 协议与可复现入口（0.5–1 天）

1. **冻结 baseline**：把当前工作树纳入 git 并打 tag（如 `gateA-v0.9.1`），否则结果无法注明代码版本。
2. 修掉/标记 2 个过期测试（见 §5）。
3. 写 `docs/2d_protocol_spec.md`（即提示要求的 `spec.md`，一页即可），必须写清：
   - 状态 `s=(x,c,θ)`：x 公共粗几何/运动；c 接触模式；θ 隐藏物理（阻力、约束、牵引）；
   - **完整的联合观测表** `P(o_A,o_B,y_pub ∣ s,a,s')`，并包含一个观测相关实例——**禁止未验证地假设两侧独立**；
   - 严格时序（提示 §三 的 6 步，含 t=0 是否有局部观测）；
   - 技能契约、代价量纲、终止条件、失败/截断代价（失败不得靠提前退出变便宜）；
   - 消息协议：每机每边界一次 ≤B bits，离散码本，候选 ID/缺失掩码计入有效载荷；
   - **oracle 适用范围**：`O_B` 只覆盖"一次同步消息轮次 + 公共规则提交"策略类。
4. 目录落位：`src/coimagine/{protocol,toy,exact}/`。
5. 记录 `AGENTS.md` 缺失（提示要求遵守但仓库中不存在）。

### Stage 1 — 微型精确诊断 M0–M3（3–5 天，**关键路径**）

- `toy/model.py`：显式有限生成模型（2–4 隐藏状态、2–3 合法联合 option、时域 2–3），
  转移表 / 联合观测表 / 代价 / 终止，全部可查询，概率归一化有测试。
- `toy/instances.py`：
  - **M0** 公共信息足够 → 通信无额外收益（负面对照）；
  - **M1** 公共"停滞"相同但局部反馈指向不同恢复方式，且恢复方式随原因不同；
  - **M2** 短试探有明确动作成本、反馈能改变后续最优动作；**同时**给出"试探无信息"和"试探过贵"两个负对照；
  - **M3（可选）** 准备动作改变接触模式，使后续联合技能可行——与"只获取信息"区分。
  - 必须报告实际时域与穷举规模。
- `belief/bayes.py`：每机精确 Bayes 更新 + 私有历史；公共协调用 `P(s,h_A,h_B|C)`；
  明确禁止"两个后验相乘"。
- `exact/oracles.py`：五种参照，各自输出**实际时域、策略类别、码本是否固定、遍历节点数、耗时**；
  预算耗尽返回**显式未完成状态**，不输出伪精确值。
- `exact/codec.py`：B=0/1/2 的离散码本、解码语义、实际位数。
- **性质测试**（提示 §七）：概率非负归一；Bayes 与手算一致；同公共记录 → 同共同动作；
  同轮不读对方新消息；未来反馈不提前用于分支；重复转发不重复计似然；位数正确；
  `J_full* ≤ J_central* ≤ J_B* ≤ J_codec*`（在适用条件内）；B 增大且策略类嵌套时 `O_B` 不劣；
  试探无用/过贵时不选摸索、有用时由求解结果决定；公共相同/私有不同的配对例子。
- 产出 **oracle 差距分解表**：物理不可见 / 有限共享 / 消息表示 / 计算限制。

> **为什么 Stage 1 最优先**：v6 末尾 Q&A 把第一轮定义为"瓶颈替换实验"。精确 oracle 一旦跑通，
> 就能在**零学习成本**下回答其中第 2、3、4 问（精确搜索 vs 有限搜索、完整历史 vs 紧凑后验 vs 消息、
> 是否显式考虑未来反馈），并直接验证 RQ1（私有证据是否真的改变决策）。
> 如果 RQ1 在这些实例上不成立，整个 v6 的 2D 前提就被证伪，越早发现越好。

### Stage 2 — 6×6 grid 协作推动（1–2 周）

- 在现有几何引擎上加入**隐藏阻力 r∈{1,2}**、公开低频物体位姿、局部驱动/滑移反馈似然、
  由转移定义的短推试探、改变接触模式的准备动作（约束—释放用显式表，注明符号抽象）。
- 候选生成**只依赖公共几何**：不得因隐藏 r 预筛推动候选（否则掩码本身泄漏真值）。
- 先交付 4 个明确机制场景 A0–A3（对应 M0–M3），不要一上来堆 20 张图。
- 六类最小基线：无消息公共策略 / 紧凑后验或模式消息 + `O_codec` / 后果摘要消息 /
  完整历史 `O_central` / 同码本但不展开未来分支 + 执行后重规划 / 真模型有限搜索对照精确规划。
- 大 grid 明确降级为 `TrueModelLimitedSearch`，**不得**再标"精确最优"。
- 交付：oracle 差距表、通信预算曲线、试探成本或反馈噪声扫描图、2–3 段回放、
  `results.md`（回答提示 §八 的 5 个问题）、`next_steps.md`、README 真实可执行命令。

### 本轮明确不做

学习世界模型、神经消息编码、`option_prior`/大模型 API、3D、双臂、实机、删除旧模块。
即使结果没有方法优势，也不得为制造优势添加新模块或改奖励。

---

## 5. 必须先处理的仓库健康问题（Stage 0）

1. **pytest 被 ROS 环境污染**：`/opt/ros/iron` 的 pytest 插件在导入 `launch_testing` 时
   因缺 `lark` 崩溃。实测 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 可正常收集 36 个测试。
   需在 README 或 `pyproject.toml` 中固化该变通（否则新 agent 一上来就跑不动测试）。
2. **绘图后端**：无 DISPLAY 时 matplotlib 退到 Tk 会失败，需 `MPLBACKEND=Agg`。
3. **2 个过期测试失败**（已复现，属预期漂移而非逻辑崩溃）：
   - `test_h3_iet_and_oracle_find_reposition_approach_push_chain`
     —— trace step 1 期望 `{APPROACH, IDLE}`，实际 `{APPROACH}`（`tests/test_local_imagination_gate_a.py:55`）；
   - `test_v010_randomized_channel_generator_has_two_sided_unscreened_variation`
     —— 断言 `benchmark_version=="v0.10"`，生成器已输出 `"v0.14"`（`:300`）。
   需判定是"测试过期"还是"行为回归"后再更新，不能直接改断言掩盖回归。
4. **全量 pytest 很慢**：`test_local_imagination_gate_a.py` 单文件实测 >280s 未跑完（超时中断）。
   Stage 0 应拆分/标记慢测试，给出"关键测试"与"全量"两条命令。
5. **git 几乎为空**：`git ls-files` 只有 `Readme.md`（且处于删除状态），无 tag。
   → 无法在结果中注明代码版本，必须先提交 baseline。
6. **`AGENTS.md` 不存在**，但执行提示要求遵守它；需新建或确认其位置。

---

## 6. 风险与禁区（写进 spec，逐条测试）

- **候选掩码泄漏**：固定一组低层技能契约与代价，所有方法复用；隐藏条件不得用于过滤候选。
- **观测相关**：直接用联合表，不假设两侧独立；旧 belief 恰是"无概率的乐观闭合"，不可沿用。
- **状态-历史耦合**：消息依赖私有历史，公共协调一般需要 `P(s,h_A,h_B|C)`；
  只缓存 `P(s|C)` 不足以预测未来消息。
- **oracle 命名纪律**：大 grid 有限搜索 ≠ 精确最优；`O_B` ≠ 无限制 Dec-POMDP 全局最优。
- **信息价值计算**：准备/辨识实验必须有真实未来观测/消息分支；
  固定动作序列 rollout 后重规划**不算**事前信息价值。
- **禁止作弊式奖励**：不得加信息增益奖励、使能奖励；不得为迁就先验改奖励或测试条件。
- **公平计量**：消息 bits 计入候选 ID、掩码、置信度、动作建议；排序广播与 API 字节/时延另列；
  为消息生成而预测全部候选的成本要计入，不能只统计最后搜索到的候选。
- **不许悄悄缩范围**：若精确 `O_B` 多步太贵，交付"已验证的单边界精确 `O_B` + 短时域 `O_codec`
  + grid 有限搜索"，并**明确列出未完成项与复杂度原因**。

---

## 7. 建议的决策点（需确认）

1. **旧 Gate A（CST/IET）定位**：建议冻结保留为"结构化世界模型/多步想象"的补充结果，
   **不与 v6 的概率核心合并**。是否同意？
2. **是否现在开工 Stage 0 + Stage 1**：建议立即做，Stage 1 是最高信息量、最快证伪的一步。
3. **是否新建 `AGENTS.md`** 以固定仓库约定（依赖、测试命令、信息边界、命名纪律）？

---

## 附：一句话记忆

> 旧代码答的是"乐观地想象未来观测能否解锁多步协作"；
> 新方案问的是"隐藏接触物理下，多少 bits 的什么消息足以支撑准备—辨识—执行的联合决策"。
> 二者共用几何与工具，但**核心必须新写**；先做微型精确 oracle。
