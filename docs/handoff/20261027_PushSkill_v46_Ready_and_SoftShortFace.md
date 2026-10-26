# PushSkill v4.6：协同 staging + 短面软内缩（2026-10-27）

本文件记录本轮对 `PushSkill` 的两项行为修改、验收数据与未决项。

## 1. 修改

### 1.1 协同 staging（ready/release）

`spec.py` 新增：

- `COOP_READY_GAP = 0.03 m`：两侧各自到约 3 cm 边到边裕度；
- `COOP_READY_BRAKE_GAP = 0.03 m`：进入 6 cm 内改用慢速径向伺服，避免惯性冲进接触；
- `COOP_READY_ALIGN_DEG = 10°`：朝向与接触法向的对齐容差。

`skills.py`：

- `PushSkill` 新增 `ready` 阶段；cooperative 时在 3 cm 裕度、朝向对齐后零力等待；
- `ready` 等待不触发 `no_progress`/`contact_loss` 计时；
- 环境 `joint_step` 在**两侧都 ready** 时同一周期设置 `release=True`，两侧下一周期同时进入 `push`；
- 旧的 1.5 s light-contact hold 不再使用；`hold` 字段保留为 False 以兼容旧日志。

### 1.2 短面软内缩

- 旧硬 guard：可见线段 < `2*(R+0.02)=0.40 m` 直接拒绝，导致 0.25 m 短面永远不可推。
- 新 `_contact_inset`：宽面保持 `R+0.02` 内缩；窄面把内缩缩小，最小允许 `0.1R=0.018 m` 的可见线段；目标点为可见片段中心。
- `insufficient_visible_segment` 只在可见线段短于 `0.1R` 时出现。
- 注意：这是“接触优先”策略；窄面仍可能因接触点偏置产生力矩。

## 2. 验收

### 短面 `L1_train_0002`

旧：`insufficient_visible_segment`，停在 gap≈0.125 m，无接触。
新（PUSH 0.39 m/s / 1.00 m）：进入 push、接触，signed 0.564 m，物体位移 0.585 m，转角 −8.8°，exit `option_budget`；首次接触点相对真实面中心约 0.002 m。
新（PUSH 0.24 m/s / 0.50 m）：`push_complete`，signed 0.508 m，物体位移 0.518 m，转角 +4.0°。

### 协同（4.0×3.0 m 空场，无静态障碍）

9 个 `coop_*` 场景，同侧长面，3 种速度/距离（部分 1.0 m）：

- 两侧 ready 时刻一致，`sync delay = 0.00 s`；两侧在同一周期进入 push；
- staging gap 约 0.026–0.035 m；
- 同侧 pair 的 **after-engagement contact fraction = 100%**；无 `contact_lost`；
- 两侧接触前物体移动 < 0.02 m；最终转角 ≈0°（角点场景 ±4–6°）；
- 一侧远距离（gap 0.45 m）也等到双方 ready，近侧不再提前推；
- `coop_adjacent`（下侧 + 右侧）虽然能 ready/push，但右侧是 0.25 m 短面且两法向垂直：出现 `safety_abort` / `contact_lost`，右侧 after-engagement 只有 57–60%（一档 100% 但双方 safety_abort）。相邻侧仍不是稳定协同构型。

## 3. 测试与页面

```bash
pytest tests/test_push_v46.py tests/test_option_testbench.py \
       tests/test_random_start.py tests/test_random_start_anchor.py \
       tests/test_random_start_rl.py -q          # 40 passed
```

全量 `pytest -m "not slow"` 仅有既有的 `test_observation_terminal_v1` 失败：缺少
`outputs/observation_terminal_v1/terminal_observations_train.jsonl`（outputs 未入库），与本轮改动无关。

可视化：`outputs/option_testbench_v1/round_035_pusheval/index.html`（27 场景 / 93 分支，静态 frame scripts）。
`SUMMARY.md` 的协同表使用 after-first-contact 的 `ct*` 列；页面同时显示 staging gap、ready/release、after-engagement。

## 4. 未决

- 相邻侧（长面+短面，法向垂直）协同不稳定；是否需要专门的相邻侧 staging/力分配。
- 短面 1.0 m 快推仍会在 option budget 内结束（0.564/1.0 m）；是 timeout/速度问题，不是接触问题。
- `ENV_VERSION` 仍为 `random-start-oriented-v4.1`（历史状态）；本轮行为变化没有 bump 版本字符串。
