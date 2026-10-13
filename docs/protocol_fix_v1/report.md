# 协议修复与基线复核报告（protocol_fix_v1）

时间戳 2026-10-12。run_id `20261012-160943`。新结果只在
`outputs/protocol_fix_v1/20261012-160943/`；旧产物未覆盖。

## 阶段完成情况

S0–S5 全部完成，无 PARTIAL/BLOCKED。缺失输入：任务书引用的
`CoImagine_快速代码审查_v1.md` 不在仓库，按任务书第 1 节的 8 条发现执行。

## 修复的确认问题

1. **快照合同**（已修）：`Snapshot` 现含 scene_config、physics、success_counter、
   termination、深复制 controller_state 与控制前缀；`restore` 不再清零计时/终止，
   只有 `reset` 初始化；恢复时校验物理配置；自定义 SceneSpec 可恢复。
2. **精确恢复**（已修）：主链路只用 `restore_exact`（重建初态+重放控制前缀），
   按 position/angle/linear_velocity/angular_velocity 分项容差（1e-9）校验，超差抛错；
   快速恢复更名 `approximate_restore`，主实验与 exact backend 不再调用。
3. **真实世界隔离**（已修）：`run_mpc_episode`/`run_random_episode` 使用整回合持续的
   `live_env` 直接 `execute_joint_skill`，规划只用独立 `planner_env` 与快照；
   live_env 不再经 `rollout` 恢复。
4. **终态与搜索语义**（已修）：成功吸收（原样带入、不展开、不计费，允许短于 depth）；
   物理失败为终态、不展开、不入选；全候选失败返回 `planner_no_feasible_plan` 且
   不执行失败候选；环境已终止后返回零增量终态结果。
5. **数据 schema v2**（已修）：结构化全精度技能、双方逐控制步施力、step 级
   state/next_state、success_counter/termination/controller/版本引用；技能序列化
   不解析 label；片段不跨 episode。
6. **穿透覆盖**（已修）：四类碰撞（robot-object/robot-robot/robot-wall/object-wall）
   分别报告并共同参与阈值判断；新增墙接触检查证实覆盖。
7. **依赖声明**（已修）：`pyproject.toml` 加入 `pymunk==6.9.0`。
8. **Grid 解释**（已更正）：结论生成器改为事实表 + “评分协议待修正，暂不作机制判断”；
   errata 由旧 `summary.csv`/`candidates.jsonl` 只读生成。

未量化：旧 Grid 的阈值化成功传播未修复；纯动力学误差未隔离；单机器人对照未做；
三窗口候选计划与 beam 中间状态仍未逐步落盘（本轮只保存每次决策选中的完整预测序列）。

## 主表（before=旧协议，after=修复后，分母完整）

| 条件 | before | after |
|---|---:|---:|
| random（3 种子） | 0/30 | 0/30 |
| mpc depth1 | 6/10 | 6/10 |
| mpc depth3 beam3 | 7/10 | 8/10 |

逐场景表（成功/termination/窗口/实际时间/控制用量/终态距离/节点/候选模拟步/精确重放步/
规划墙钟）见 `continuous2d/results_table.md`；逐场景逐种子 before/after 见
`comparison/before_after.csv`。5 处成功翻转的首处分歧见
`comparison/first_divergence.jsonl`。

**翻转**：S1_v2 depth3、S1_v3 depth1、S1_v4 depth3 由失败转成功（协议修复后）；
S0_v2 depth1/depth3 由成功转 `planner_no_feasible_plan`。

**S0_v2 场景缺陷**：其机器人初态与场地边界 capsule 重叠 0.06 m > 0.05 m 阈值；
新增墙穿透覆盖后，所有根候选即刻物理失败。按“禁止改场景布局、禁止放宽阈值”，
场景未改、阈值未放宽，如实记为失败场景。

**depth3 退化**：修复后仍存在。S1_v1 上 depth1 成功（7 窗口）、depth3 失败（12 窗口）；
旧“beam 宽度伪影”归因**撤回**，改以修复后真实结果为准：同一场景 beam=1 成功、
beam=2/3/5 失败（旧诊断），但本轮不重扫 beam；结论限定为“该实现下 depth3 不保证更好”。

## 启发式声明（S4.3）

`scoring.passage_cost_to_go` 保留未改。它显式包含“姿态投影宽度 vs 通道开口”的几何
启发式，因此准备性转动可获得即时启发式收益；**本轮不以成功轨迹证明“模型自主发现
长期准备价值”**。

## 复现命令

```bash
pytest tests/test_protocol_fix.py tests/test_continuous2d.py -q
python experiments/continuous2d/run_physics_acceptance.py --out outputs/protocol_fix_v1/<id>/physics_check
python experiments/continuous2d/run_baseline.py --run-id <id> --out-root outputs/protocol_fix_v1
python experiments/continuous2d/compare_baselines.py --after outputs/protocol_fix_v1/<id>/continuous2d --out outputs/protocol_fix_v1/<id>/comparison
python experiments/grid_selection_audit/write_errata.py
```

## 未改动的科学条件 / 下一步

未改：物理参数与尺寸、摩擦/阻尼/时间步/力限幅、场景布局、技能控制律、候选模板与顺序、
depth(1/3)、beam(3)、episode 上限(12)、成功容差、原评分与 passage 公式、旧模型/数据/检查点。

下一步（仅一项建议）：先修 S0_v2 的初态边界重叠并复核该场景，再决定是否重启确定性 Grid 或
学习模型任务；本轮不自动开始。
