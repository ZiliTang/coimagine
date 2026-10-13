# REVIEW_INDEX

路径相对仓库根，行号对应当前工作区（HEAD `627d5b8` + 未跟踪新文件）。不确定处直接写
“不确定/未记录”。

**1. 旧 Grid 未知量与确定性.** 未知量是 `src/coimagine/grid/state.py:HiddenPhysics`
（resistance∈{1,2}、constraint∈{free,constrained}、traction∈{grip,slip}²，96–199）与隐藏的
`Scene.released`。给定完整 `Scene`（含 physics、released）后
`grid/dynamics.py:execute`（298）是确定的：默认 `NO_FEEDBACK_NOISE`，
`sample_feedback`（277）在 `flip_prob<=0` 时直接返回 nominal，`rng` 只在噪声开启时使用；
诊断不注入任何噪声。观测 `sensing.py:ray_cast`（103）默认无噪声。动作由固定序列给定，无策略随机性。

**2. “观测一致世界集合”.** `diagnostics/grid_selection_audit.py:consistent_worlds`（97）
枚举全部 16 个 `enumerate_hidden_physics`，从 `situation.initial_scene` 以 `released=False`
重放该情形的 lead-in 技能，保留同时满足三项者：(a) A 的私有反馈序列等于 `history_a`，
(b) 公共 moved/stalled 等于 `public_history`，(c) 末态物体 origin 等于被观测 origin。先验：
场景生成器对 16 型均匀采样，故集合内取均匀权重 `1/n`（`build_decision_point` 139 用
`np.full(n, 1.0/n)`）。使用的历史只有 A 自己的反馈、公共事件与无噪扫描，不含 B 的私有反馈。
**模型侧不是同一分布**：`belief_from_history` 用学习似然做 Bayes 更新（`model_belief`），
与精确集合可以不同。

**3. 传播方式.** 本诊断走**原边缘状态传播**：`learning/belief_model.py:step_actions`（157）
与 `cost_actions`（188）把同一 41 维边缘动态复制 16 份、用 one-hot θ 过转移头后按 `b(θ)`
混合。**未调用** `diagnostics/conditional_planner.py` 的逐参数条件状态 `s_t^θ`；该文件是
此前的修复线，与本审计无调用关系。

**4. J_hat/J_ref.** 两者共用 `joint_score`（205）：`J = 10·成功 − 代价 − 0.5·距离`。
`true_outcome`（211）：对一致世界逐一 `execute`，成功即吸收（停止计费并冻结终态），距离用
最坏物体格到最近目标格的曼哈顿；按权重取精确期望。`model_outcome`（267）：从初始输入用
`step_actions` 混合推演整条序列，`P(in goal) ≥ 0.5` 视为成功并停止计费，距离用
`_cell_distance_distribution` 的期望；成功判定是模型阈值而非真实指示。与历史
`belief_probe.plan_with_search` 的差别：后者用 `_cut_value` 且按私有反馈分支、取最优续接，
不是固定序列的 J。

**5. 通道几何代价.** `continuous2d/scoring.py:passage_cost_to_go`（73），辅以
`extent_across_travel`（60）：对每个严格位于物体与目标之间的竖直墙，
`deficit = max(0, extent(θ) − (opening − 0.02))`，返回
`deficit + |wall_x − x| + |y − gap_center| + |goal.cx − wall_x|`；物体已在墙东侧时用角点距离
`distance_to_box`（32）。开口由同一 x 的两个障碍箱求得，无场景 ID/类别分支，但隐含
“沿 +x 穿越竖直墙”的假设。

**6. 候选生成.** `continuous2d/candidates.py:per_robot_options`（47）按当前观测生成 11 个
选项（8 接触推 + 2 站位 + 1 hold）；`joint_candidates`（114）做 11×11 笛卡尔积并在
`max_candidates=128` 处提前返回。`mpc.py:_lookahead`（104）在根节点生成一次，并在每个保留的
beam 节点按其末态**重新生成**。128 限制**单个节点的联合列表**，不是整次规划或完整序列。

**7. beam.** `mpc.py:_keep_best`（174）按 `ordering_key` 降序取前 `beam`，在每层末尾截断。
键为字典序：成功 `(1, −累计时间, −累计控制, …)`；失败 `(0,0,0, −距离, −朝向, −累计控制, …)`；
末位固定为 `−root_candidate_id`。`evaluate`（54）把父节点 `cumulative_time/cumulative_usage`
累加进键。并列按较小**根** candidate_id；后续窗口的 id 不参与并列（同根序列靠 Python 稳定
排序保序）——这是设计选择，不是已验证的最优做法。

**8. export/restore.** `backend.py:Snapshot` 存 scene_name、config_hash、tick、物体与两机器人
BodyState（x,y,angle,vx,vy,omega）、`controller_state`、`prefix`（已施加控制序列）。
`restore`（186）重建空 Space 并写入位姿/速度（丢弃 Chipmunk 缓存的接触/arbiter）；
`restore_exact`（198）重建并重放 prefix。**未恢复**：接触缓存、成功保持计时器
（`environment.py:146` 的 restore 将 `success_counter=0.0`）、随机数（环境无 RNG，物理确定）。

**9. witness.** `experiments/continuous2d/run_physics_acceptance.py:witness`（254）先运行
depth-3 MPC，再取 `episode.trace[].results[0].joint_spec` 作为固定序列、reset 后开环重放，
因此**来自 MPC 成功轨迹，不是独立人工指定序列**。它不被候选生成器读取；测试
`tests/test_continuous2d.py:test_depth_three_plan_contains_a_preparation_window_on_s1`
单独调用规划器验证准备动作。序列记录在
`physics_check/physics_acceptance.json` 的 `s1_witness.replayed_specs`。

**10. 单机器人对照.** 未做。本平台只运行双机器人联合技能，没有单机器人/联合执行的消融对照。
