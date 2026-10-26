# Option v4.5 交接与 RL 训练说明（2026-10-26）

## 1. 当前冻结版本

- `ENV_VERSION = "random-start-oriented-v4.5"`（`src/coimagine/random_start/spec.py`）。
- 动作空间：**7 类型 × 4 尺度 = 28/机器人，784 联合**；PUSH 只有**法向力**（±15° 已删除），
  PUSH 尺度 = 请求距离 `{0.10, 0.25, 0.50, 1.00} m`；REPOSITION `{0.10, 0.25, 0.50, 1.00} m`；
  WAIT `{2,6,12,24} s`；FOLLOW `{2,6,12,24} s`。类型顺序：3 PUSH、2 REPOSITION、1 WAIT、1 FOLLOW；
  `index = type_index*4 + scale_index`。
- 观测：**760 维** = 2×(18 非雷达) + 2×2 进度 + 2×72×5 雷达（`min(d,R)/R` + hit + one-hot(3)）。
- 超时：`3.0 + 6.0·D/v_ref`，上限 24 s（`JOINT_MAX_SECONDS`）；FOLLOW 超时 = `prepare + duration`。
- 参数：`PUSH_SPEEDS=(0.12,0.24,0.39)`、`REPOSITION_SPEED=0.18`、`MAX_FORCE=15`、
  `KV=40`、`CONTACT_LOSS_TOLERANCE=2.5 s`、`CONTACT_SEEK_SECONDS=2.0`、`APPROACH_TIMEOUT=8.0`、
  `APPROACH_MIN_SPEED=0.10`、`APPROACH_MAX_SPEED=0.20`、`FOLLOW_GAP=0.05`、
  `FOLLOW_GAP_GAIN=4.0`、`FOLLOW_TANGENT_GAIN=4.0`、`FOLLOW_POSE_TOLERANCE=0.06`、
  `HOLD_MAX=1.5 s`（协同推等待上限）。

## 2. 本轮关键修复（按影响）

1. **推进记录 bug**：`PushSkill.step` 不调用 `_note_position`，`signed_progress` 永远 0，
   所有 push 只能跑到超时。修复后 `push_complete` 0→35+。
2. **接触 seek**：丢失接触不再 0.6 s 中止；0.6→2.5 s 窗口 + 接触保持参考（速度 + 间隙 P +
   间隙变化前馈）。`contact_lost` 33→2。
3. **换面不中止**：法向跳变只记录、继续推（`face_switch` 退出仅保留在 REPOSITION）。
4. **接触点瞄准**：approach 阶段瞄准可见面中心，避免先撞角点；同面双推保持原对称位置
   （重新定心会挤在一起）；push/approach 的避障含队友，`safety_abort` 8→1。
5. **FOLLOW**：参考点改可见点质心、增益 4.0、法向速度前馈、超时=准备+维持；独立验收
   `teammate` 场景位姿误差 0.051、容差内 0.80 → fulfilled。
6. **双推可行性过滤**：可见法向相反（dot < −0.5）的“对侧对夹”标为不可接受
   （`opposite_faces_both_push`），不计入期望集。

## 3. 当前实测（round_034，10×5 + 长程，68 分支 / 120 option 评估）

| 口径 | 值 |
|---|---|
| 未过滤 option 级 | 59/120 = 49.2% |
| 已接受请求 option 级 | 57/92 = **62.0%** |
| branch 条件兑现 | 23/54 = 42.6% |

## 4. 已知问题（下一位调优者优先看）

- **1 m 长程推**：接触在但推进行程不足，主要是物体转向消耗行程 + 有效推进速度低；
  建议把接触点固定在质心线、限制姿态随动。
- **同侧双推**：对称时零转角（teammate 双推 0.0°/0），非对称会转 20–70°，需要按质心对称选点。
- **相邻侧双推**：力矩叠加，物体原地转 23–109°；这是“稳定性”而非“位移”问题，应由世界模型学。
- **FOLLOW 影子侧**：跟随方站在推动侧时会接触（face 场景 6.05 s），需实现沿外轮廓到影子侧的迁移；
  这是当前 12 个 FOLLOW 失败的主因。
- **对侧对夹双推**：已剔除；不要为了数字把它加回期望集。

## 5. 常用命令

```bash
# 测台（当前最佳配置）
.venv/bin/python experiments/option_testbench/demo_showcase.py
.venv/bin/python experiments/option_testbench/cli.py run --out <dir> --branch-wall-seconds 10 --total-wall-seconds 80
.venv/bin/python experiments/option_testbench/metrics.py --out <dir> --suffix _v2
.venv/bin/python experiments/option_testbench/cli.py render --out <dir>
# 独立验收
.venv/bin/python experiments/option_testbench/follow_validation.py
.venv/bin/python experiments/option_testbench/anchor_boundary_eval.py
.venv/bin/python experiments/option_testbench/cooperation_eval.py
# 测试
.venv/bin/python -m pytest tests/test_option_testbench.py tests/test_random_start.py tests/test_random_start_anchor.py -q
```

## 6. RL 训练器（新增）

`experiments/option_rl_v45/train.py`：
- 观测直接取 `RandomStartEnv.observation_vector`（760 维，合法视图，无特权量）；
- 动作 = 两个 28 类因子化头（共享 trunk），联合动作 784；
- return 为 semi-Markov：技能 n 个周期贡献其 dense reward，下一步按 `gamma_step**n` 折现；
  技能退出不是 episode 结束，只有任务终止（成功/失败/时间预算）停止 return；
- 奖励用环境 `boundary["reward"]["total"]`（progress - distance_cost - time_cost + 事件）。
- 冒烟：`--iterations 2 --episodes 2 --eval-roots 2` 通过（obs 760，动作 28）。

```bash
.venv/bin/python experiments/option_rl_v45/train.py --hours 4.5 --episodes 8 \
  --eval-roots 16 --out outputs/option_rl_v45/run1
```
输出：`history.json`、`result.json`、`policy.pt`（含 history）。
