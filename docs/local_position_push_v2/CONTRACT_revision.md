# CONTRACT 修订（round 2，范围收紧版）

## 1. 正式协议：**未改**

按本轮范围收紧要求，正式的控制与执行协议与 v1 完全一致：

| 项 | 值 | 状态 |
|---|---|---|
| `settle_surface`（正式判稳门槛） | **0.02 m/s** | **未改** |
| 判稳估计量 | 相邻两帧（0.05 s）世界系法向速度，逐点绝对值的中位数 | **未改**（正式路径） |
| `brake_settle_limit` | 4.0 s | **未改** |
| BRAKE/SETTLE 力 | `clip_norm(Kv·(v_ref − v), 15 N)`，`v_ref` 归零 | **未改** |
| APPROACH/READY/PUSH 计时、力上限、白名单、物理 | 同 v1 | **未改** |

`outputs/` 下所有 v1 日志仍然由这段协议产生，因此 A1/B 的分析对象与 v1 交付一致。

## 2. 代码新增（全部**默认关闭**，不影响正式协议）

为做 A1 的因果诊断，`OptionConfig` 新增三个字段，默认值等于 v1 行为：

| 字段 | 默认 | 作用 |
|---|---:|---|
| `settle_window` | `0.0` | `0` = 用 v1 的相邻帧估计量。>0 时改用**窗口化**估计量：把匹配基线从相邻两帧放宽到该秒数（样本时间差），同一个"逐点绝对值中位数"统计量；重复时间戳不再清空窗口。**本轮不采纳为正式协议**，仅用于证明"失败在测量分辨力而非门槛"。 |
| `settle_window_min_baseline` | `0.10` | 窗口至少要跨这么多样本时间才允许下结论。 |
| `brake_hold_force` | `0.0` | `0` = v1 制动。>0 时在制动阶段叠加沿局部法向的有界预压（主动制动）。**本轮不采纳**（范围收紧明确要求暂不实现主动制动）。 |

新增函数：

- `control.update_windowed_surface_estimate(...)`：窗口化估计量（返回结构与 v1 相同）。
- `control.SurfaceSample`：窗口缓冲区条目。
- `control.brake_force_with_hold(...)`：`hold_force=0` 时与 `brake_force` **逐位相同**；>0 时有界，
  只读局部法向与自身状态，不写物理量。

`option.py` 的 SETTLE 分支按 `settle_window` 选择估计量；BRAKE/SETTLE 力经
`brake_force_with_hold(..., config.brake_hold_force)`（默认 0，等价于原调用）。
`LocalControlState.surface_history` 只在窗口化路径使用，`start()` 里清空。

## 3. 新增实验与分析脚本（全部**只读**，不重跑控制器）

`experiments/local_position_push_v2/`：

| 文件 | 作用 |
|---|---|
| `audit_a1.py` | A1 分类审计 + 估计量坐标系核对；写 `audit_a1.{json,md}` |
| `analyse_a2_counterfactual.py` | A2 仅日志门槛/预算反事实，**分开**判据决定集与中止集；写 `a2_counterfactual.{json,md}` |
| `analyse_b.py` | B 可观测终态转移（留一 root）；写 `analyse_b_<arm>.{json,md}` |
| `run_pair.py` | 带 `--settle-*` / `--brake-*` 参数的 V/X 配对实验入口（诊断用，**正式结果未使用**） |

## 4. 新增测试

`tests/test_local_position_push_v2.py`（10 项）：

1. 估计量是世界系：物体静止 + 机器人移动 → 0；
2. 估计量看得见真实平移（0.004 m/0.05 s → 0.08 m/s）；
3. 窗口化估计量在 2 mm 逐射线噪声下把虚警地板降到门槛以下（相邻帧中位 > 0.02，窗口 < 0.02）；
4. 重复时间戳不清空窗口；
5. 缺失观测**永远不算**静止证据；
6. `brake_force_with_hold(...,0)` 与 `brake_force` 相同；
7. 主动制动力有界、方向朝表面、无法向时退化为普通制动；
8. **B 的目标不含物体全局位姿分量**（全局刚体变换下逐位不变）；
9. **留一 root 不拆 root**（同 root 的 8 个变体不跨折）；
10. 三个分析脚本不 import `run_split` / `Continuous2DEnv`（保证只读）。

## 5. 未做（按范围收紧）

- 不实现/不采纳主动制动；
- 不修改正式判稳门槛，不以 90% 为目标；
- C 节楔形接触实验**未开始**（无部分实现需要保留）；
- 未训练世界模型，未接规划器，未做不确定性感知规划；
- 未重新生成动画。
