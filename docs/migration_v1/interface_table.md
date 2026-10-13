# 阶段 A：接口事实表（≤15 行）

核实时间 2026-10-12；git HEAD `627d5b861bc55152cef40489d7351ec8777dd53b`（main，工作区只有
两份未跟踪任务书）。旧 Grid = 离散 6×6 协同推动环境（`coimagine.grid`）+ 学到的世界模型
（`coimagine.learning`）。连续环境在阶段 C 新建，旧接口一律不改。

| # | 功能 | 实际文件 / 函数 | 输入 | 输出 | 本轮改动 |
|---|---|---|---|---|---|
| 1 | 旧环境 step | `grid/dynamics.py:execute` | `Scene`, `JointSkill`, `FeedbackNoise=NO_FEEDBACK_NOISE`, `rng` | `(Scene, StepResult{cost,y_public,feedback,success,failures})` | 否 |
| 2 | 旧环境 reset | `grid/scenes.py:scene_from_ascii` / `learning/data.py:probe_scene` | ASCII 行 / `(origin, HiddenPhysics)` | `Scene`（frozen） | 否 |
| 3 | 旧状态导出/恢复 | `grid/state.py:Scene`（frozen dataclass）；`Scene.with_robots/with_object`；`dataclasses.replace` | `Scene` | 可重建的 `Scene` 副本 | 否 |
| 4 | 旧模型加载 | `learning/belief_model.py:load_belief_model` | `outputs/grid/world_model/belief.npz` | `BeliefWorldModel` | 否 |
| 5 | 旧模型预测 | `BeliefWorldModel.dynamic_state/parameter_belief/step/step_actions/cost_actions/predict_outcome` | 188 维观测 / 41 维动态 / 16 维 θ 后验 | 下一步状态、代价、符号分布 | 否 |
| 6 | 旧模型评分/规划 | `diagnostics/belief_probe.py:plan_with_search`（`_cut_value`,`_beam_value`） | 观测、goal、belief、depth、beam | `(action_index, branches)` | 否 |
| 7 | 动作 ID 顺序 | `learning/features.py:ACTION_IDS` | — | `(push:0:E, push:1:E, co_push:E, probe:0:E, release:0:E, wait)` | 否 |
| 8 | 观测构造 | `learning/features.py:observation_vector` | `OccupancyScan, own_pose, feedback_a, public_events` | 188 维向量 | 否 |
| 9 | 旧场景库 | `grid/scenes.py:default_scenes/mechanism_scenes` | — | 4+4 个 `Scene` | 否 |
| 10 | 旧随机种子 | 各调用点局部 `random.Random(seed)` / `np.random.default_rng(seed)` | seed | 无全局 RNG 状态 | 否 |
| 11 | 旧观测/反馈噪声 | `grid/sensing.py:SensorNoise` / `grid/dynamics.py:FeedbackNoise` | 概率参数 | 默认无噪声 | 否 |
| 12 | 旧检查点 | `outputs/grid/world_model/{belief,latent_released}.npz` 等 7 个 | — | 参数 | 否（只读，hash 见 manifest） |
| 13 | 连续后端 | 新增 `coimagine.continuous2d.backend`（pymunk 6.9.0） | `SceneConfig`,`seed` | `Space` + 刚体句柄 | **新增** |
| 14 | 连续接口 | 新增 `continuous2d.environment.Continuous2DEnv`：`reset/export_state/restore/step_control/execute_joint_skill/rollout` | 见 C2 合同 | observation / snapshot / SkillResult / TrajectoryResult | **新增** |
| 15 | 连续规划 | 新增 `continuous2d.mpc.plan` + `continuous2d.candidates` | snapshot、候选模板 | 排序后的候选与逐步评估 | **新增** |

验收 A（已运行，未改动仓库资产）：在 `a0_public_sufficient` 上执行一次真实
`push:0:E` → `cost=1.0, y_public=moved, obj (1,3)→(2,3), goal=True`；同一场景用
`belief.npz` 预测 → `cost_hat=1.000`，物体落在目标区的概率 `0.502`（argmax 格正是 (2,3)）。
尺度（代价 0/0.3/1.0）、收益方向（距目标距离、是否在目标区）与动作身份（`ACTION_IDS`
固定顺序）均可对应。复现命令 `python experiments/grid_selection_audit/phase_a_check.py`。
旧资产齐全，阶段 B 不 BLOCKED。
