# RL 契约 v4（统一环境模型）设计草案

决策（用户 2026-10-19）：**只维护一个环境模型**；雷达截取、雷达随体转动、option 内置尺度
三项修改**直接进入 RL 契约**；**旧 RL 结果正式作废**；不做备选/回退分支。

## 1. 统一模型的规格

### 1.1 朝向（正式，不再默认关闭）
- 机器人有 heading 自由度；技能在 `info` 里声明 `desired_heading`
  （PUSH = 拟合表面法向、REPOSITION = 切向、WAIT_BRAKE/制动 = `None`）。
- 环境按有界角速度 `max_omega` 转向该目标；**制动、安全、避障不受朝向门控**（全向底盘）。
- 姿态、本体雷达、碰撞几何使用**同一个权威 heading**。
- 力仍施加在质心（圆盘）；**前端推面几何与接触反力矩**是同一模型的下一次演进（后续轮），
  不是备选模型。

### 1.2 雷达（本体固定、随体转动、按上限截取）
- 72 束按机器人坐标系排列，第 0 条朝前，安装偏角 0；用采样时刻的机体姿态；一次 360° 扫描
  视为同一时刻快照，不做扫描运动畸变；**不重插值成固定世界角序列**。
- 每束编码 **3 组通道**：
  - `distance = min(d, R) / R`（无回波 = **1.0**，即按上限截取）；
  - `hit`（1 = 有回波，0 = 无回波）；
  - 类别 **one-hot {object, static, teammate}**（无回波 = 全 0）。
- 每束 5 维 → 2 机器人 × 72 束 × 5 = **720**；非雷达部分 36 → **观测共 756 维**。

### 1.3 Option 内置尺度（连续执行，不是重复 N 次）
- 新增一个**尺度维度**：每个 option 内部带一个关键尺度参数，技能连续执行到该尺度完成或自然结束。
- 类型与尺度**因子化**：`(type, scale)`。
  - PUSH：类型 (alpha, speed) 9 种；尺度 = **有效接触时长预算**。
  - REPOSITION：类型 (CW/CCW) 2 种；尺度 = **转移时长预算**（距离 = 有效速度 × 预算）。
  - WAIT_BRAKE：类型 1 种；尺度 = **保持时长**。
- 尺度档位（建议 4 档）：**{1, 2, 4, 8} 秒**。
- 动作空间：每机器人 `(9+2+1) × 4 = 48`，联合 **2304**（因子化头，不是 2304 路拍平）。

### 1.4 窗口与计时
- 单技能窗口与尺度预算一致（最大 8 秒，保持现有 `JOINT_MAX_SECONDS`）。
- 若要让“一次 option 完成约 1.5 m 换面”，需要把窗口扩到约 16–24 秒（见待确认项）。

### 1.5 作废与历史
- 旧 324 维契约、旧 checkpoint、旧 RL 报告与输出**正式作废**；保留为历史并加
  `INVALIDATED.md` 标记，不再作为对照。
- 测台（option testbench）的快照、契约、指标随之升级到 v4；旧测台轮次保留为历史。

## 2. 与现状的差异

| 项目 | 现状（v3.1） | v4 |
|---|---|---|
| 雷达无回波 | `distance=0.0, valid=False, class=0`，观测编码 0.0 | `distance=1.0`（上限截取）+ `hit=0` + one-hot 全 0 |
| 雷达随体 | 机制上随体，但机器人不转 | 朝向正式化，雷达真正随体转动 |
| 朝向 | 无转矩通道、朝向恒定 | 技能指定朝向 + 有界转向；制动/避障不门控 |
| Option 尺度 | PUSH 固定 2.0 s；REPOSITION/WAIT 各 2 档 | 统一尺度维度 4 档，连续执行 |
| 动作空间 | 15/机器人、225 联合 | 48/机器人、2304 联合（因子化） |
| 观测 | 324 维 | 756 维 |

## 3. 待确认（不可逆参数）

1. **尺度档位与窗口**：建议 `{1,2,4,8} 秒`、窗口保持 8 秒；若要“一次 option 完成换面”，
   需要把窗口扩到 16–24 秒。
2. **雷达通道**：建议 `distance + hit + one-hot(3)` → 756 维；若想更省，可用
   `distance + hit + 类别标量` → 468 维。
3. **前端推面**：本轮先不做（同一模型后续演进）。

---

## 4. 实施结果（2026-10-19，已落地）

用户确认：尺度 `{2,6,12,24}` 秒、窗口=预算（最大 24 s）；雷达 `distance(min(d,R)/R)+hit+one-hot(3)`
→ 756 维；前端推面本轮不做。已实现：

- `spec.py`：`ENV_VERSION="random-start-oriented-v4"`；`OPTION_SCALES=(2,6,12,24)`；
  `ROBOT_ACTION_TYPE_COUNT=12`、`ROBOT_ACTION_COUNT=48`、`JOINT_ACTION_COUNT=2304`
  （`index = type_index*4 + scale_index`）；`JOINT_MAX_SECONDS=24`；新增 `OPTION_BUDGET_EXIT`。
- `skills.py`：`Skill.time_budget = action.scale`；预算耗尽时退出 `option_budget`（中性退出，
  已加入契约 benign 列表）。
- `env.py`：只有一种驱动模型（技能声明 `desired_heading`、有界转向、力从不门控）；
  观测 756 维（每束 `min(d,R)/R + hit + 类别 one-hot(3)`，无回波 = `1.0/0/0/0/0`）；
  联合窗口 = 两条 option 预算的较大者（≤24 s）。
- 契约/指标：REPOSITION 兑现改为**时间预算制**（MOVE 阶段 ≥0.8×scale 且有正向切向进展，
  或达到名义距离）；PUSH 仍以 2.0 s 有效接触为自然完成。
- 测台：`panel.py` 的基线动作改为 v4 索引（PUSH=17/REPOSITION_CCW=37/CW=41/WAIT=44），
  动态快照前缀改为 `(17,44)`。

### 验证

- 全量测试：random_start / option testbench / RL / continuous2d / grid 等**全部通过**；
  仅剩 8 项 `test_docs.py` 与 1 项 `test_grid_selection_audit.py` 因
  `outputs/grid/world_model/*`（gitignored、本 checkout 未生成）缺失而失败，**与 v4 无关**。
- 冒烟：24 s REPOSITION_CCW 分支实际运行 24.0 s（480 周期）并以 `option_budget` 退出；
  观测 756 维；无回波编码为 `1.0/0/0/0/0`；heading 随技能目标转动。

### 备注

- RL 策略沿用现有条件头，动作维自动变为 48/机器人、2304 联合；`(type, scale)` 因子化头
  （12+4 logits）是后续可选的**效率优化**，不改变契约。
- 旧 324 维契约、checkpoint、RL 报告与输出已加 `INVALIDATED.md`，仅作历史。
