# Changelog — 协议修复与基线复核（protocol_fix_v1）

基线 HEAD `627d5b861bc55152cef40489d7351ec8777dd53b`。所有改动都在任务书允许范围内；
物理参数、场景布局、技能控制律、候选模板与顺序、depth/beam、episode 上限、成功容差、
原评分与 passage 公式、旧检查点与旧实验产物均未改动。

## 源码（src/coimagine/continuous2d/）

- `backend.py`
  - `Snapshot` 增加 `scene_config`、`physics`、`success_counter`、`termination`；
    `prefix`/`config_hash` 保留。
  - `restore` 拆为 `approximate_restore`（开发用，明确命名）与 `restore_exact`
    （重建初态 + 重放控制前缀 + 按 position/angle/linear_velocity/angular_velocity
    分项容差校验，超差抛 `SnapshotRestoreError`）。
  - `export_state` 接收 `success_counter`/`termination`。
  - `contacts()` 分类为 robot_object / robot_robot / robot_wall / object_wall；
    新增 `penetration_by_pair()`；`max_penetration()` 覆盖全部四类。
  - 存储 `boundary_shapes`/`obstacle_shapes`/`static_shapes` 以便墙接触检测。
- `environment.py`
  - `export_state` 深复制控制器状态并携带任务计时/终止。
  - `restore(exact=True)` 默认精确恢复，校验物理配置兼容，支持自定义 `SceneSpec`；
    不再无条件清零 `success_counter`/`termination`（只有 `reset` 初始化）。
  - `step_control`/`execute_joint_skill` 在已终止时返回零增量终态结果，不再推进物理。
  - `StepRecord` 增加控制步前/后状态与 `penetration_by_pair`。
- `mpc.py`（重写搜索语义）
  - 成功吸收：成功节点原样带入深层、不展开、不计费，允许返回短于 depth 的序列。
  - 物理失败（numeric/out_of_bounds/penetration_limit）为终态，不展开、不入选；
    全根候选失败返回 `planner_no_feasible_plan`。
  - `live_env` 与 `planner_env` 分离；`run_mpc_episode` 在 live_env 上直接执行首窗口，
    规划只读快照；`run_random_episode` 同样在 live_env 上直接执行。
  - 统计区分节点候选、候选模拟步、精确重放步、实时物理步、规划墙钟。
  - `DecisionRecord` 保存每次决策选中的完整预测序列 label、预计完成时间与关键末态。
- `data.py`
  - `schema_version=2`：结构化全精度技能、双方逐控制步施力、step 级 state/next_state、
    success_counter/termination/controller_state、场景/物理/控制器/代码版本引用。
  - `segments` 拒绝跨 episode 拼接。
- `skills.py`：新增 `skill_to_dict`/`skill_from_dict` 与 joint 版本，标签不再被解析。
- `pyproject.toml`：`dependencies` 加入 `pymunk==6.9.0`。

## Grid（仅报告生成器，不重算）

- `diagnostics/grid_selection_audit.py`：`_conclusion` 改为 `conclusion_from_rows`，
  只输出事实表与“评分协议待修正，暂不作机制判断”，删除写死的“支持”与固定数字。
- 新增 `experiments/grid_selection_audit/write_errata.py`，从旧 `summary.csv`/`candidates.jsonl`
  只读生成 `docs/protocol_fix_v1/errata.md`。旧原始数据与旧 `conclusion.md` 未改动。

## 脚本与测试

- 新增/更新：`experiments/continuous2d/{run_baseline,run_physics_acceptance,compare_baselines,write_protocol_manifest}.py`。
- 新增 `tests/test_protocol_fix.py`（13 项 S1/S2/S3 定向回归）；`tests/test_continuous2d.py` 更新为
  新快照/数据契约。
- 新增文档：`docs/protocol_fix_v1/{PATCH_PLAN,report,errata,manifest.json,changelog}.md`。

## README

- 安装说明标注 `pymunk==6.9.0` 为核心依赖；新增协议修复轮的命令一节。

## 未改动

旧检查点、`outputs/migration_v1/**` 旧产物、`_old/`、Grid 模型/动力学/评分/候选、
`condition_planner` 等历史线均未触碰。`passage_cost_to_go` 公式保留，只在报告中声明为启发式。
