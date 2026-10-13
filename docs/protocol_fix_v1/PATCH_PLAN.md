# PATCH_PLAN — 连续环境协议修复（S0）

冻结时间 2026-10-12 16:09（Asia/Shanghai）。HEAD `627d5b861bc55152cef40489d7351ec8777dd53b`；
工作区含本轮未跟踪新文件与 `README.md` 修改（见 `outputs/protocol_fix_v1/S0_baseline.json`）。
关键源码 SHA256（前 16 位）与配置哈希 `765ede7cf67318f1`、检查点
`belief.npz 5d041eafc1d7170e` 已记录。新结果写入 `outputs/protocol_fix_v1/20261012-160943/`。

**缺失输入**：任务书依据的 `CoImagine_快速代码审查_v1.md` 在仓库中不存在（`docs/` 下无此文件）。
本节按任务书第 1 节列出的 8 条发现执行；未据猜测补写审查报告内容。

| # | 发现（现场核实） | 文件·函数 | 修复方案 | 回归检查 |
|---|---|---|---|---|
| 1 | `model_outcome` 以 `P(in goal)≥0.5` 当作 100% 成功并 `break`，截断后续代价；真实侧是世界期望 | `diagnostics/grid_selection_audit.py:model_outcome`（267） | 只改**结论生成文字**，不重建旧信念；注明“H=3 代表案例只推演一步” | 新 errata 由已有 `summary.csv` 只读生成；旧数据不动 |
| 2 | snapshot 不存 `success_counter`/`termination`，`restore` 清零 | `continuous2d/environment.py:141/146` | Snapshot 增 `success_counter`、`termination`；`restore` 恢复、`reset` 才初始化 | S1.4-1：保持 0.40 s 后导出恢复，续跑 0.15 s 两者同步成功 |
| 3 | 真实执行与规划共用 `scratch.rollout`；真实世界在技能边界被快速恢复，接触缓存丢失 | `continuous2d/mpc.py:run_mpc_episode`（199）；`environment.rollout`（307） | 拆 `live_env`（整回合持续、直接 `execute_joint_skill`）与 `planner_env`；快速恢复改名 `approximate_restore`，主链路只用 `restore_exact` | S1.4-3：规划前后 live_env 状态/计时/终止及后续响应不变；两分支互不污染 |
| 4 | `_lookahead` 对成功/物理失败节点仍继续展开；节点 `success` 与本次 `result.success` 可能不一致 | `continuous2d/mpc.py:_lookahead`（104） | 成功节点原样带入、不展开不计费；失败节点不展开；键先按可行/成功分层再排序 | S2.4：成功吸收、失败不复活、全失败 `no_feasible_plan` |
| 5 | `passage_cost_to_go` 显式奖励姿态变得可通过，是启发式 | `continuous2d/scoring.py:passage_cost_to_go`（73） | 公式不动；报告中显式声明，并撤回“模型自主发现准备价值”的表述 | S4.3：报告文字核对 |
| 6 | `window_records` 顶层 `applied_control` 只存机器人 0；技能只存舍入 label | `continuous2d/data.py:window_records`（29） | schema v2：双方逐控制步力、结构化技能（kind+参数全精度）、`next_state`、计时/终止/版本 | S3 验收 4 条 |
| 7 | `max_penetration` 只查 robot-object / robot-robot | `continuous2d/backend.py:contacts/max_penetration`（241） | 纳入 robot/object 对静态墙与边界；分类报告四类最大穿透 | S3.2：墙接触检查证实覆盖 |
| 8 | `pymunk==6.9.0` 未进依赖声明 | `pyproject.toml` | 加入 `dependencies`，README 给准确安装命令 | 干净环境 import 检查（不在本机重装） |

不做：不改物理参数/尺寸/时间步/力限幅、场景布局、控制律、候选模板与顺序、depth/beam 主设置、
episode 上限、成功容差、原评分与 passage 公式、旧检查点与旧产物。旧结果目录保留为 legacy。

时间/控制成本口径：每窗口只累计一次（S2.3）；搜索统计区分节点候选、完整序列、候选模拟步、
精确重放步、实时物理步与规划墙钟（S4.1）。
