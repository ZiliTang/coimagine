# D008 执行回报：端点、anchor 与参考点离线审计（复现缺口）

- 状态：**incomplete**。全量测量/跟踪/参考账本与 probe 分解已出；但历史 EndpointTracker 的 R0 重放只覆盖 **38/100 pilot**（要求 ≥99%），因此按任务书**停止 R1/R2**，不据此改控制器，也不声称“只删一项”可行。
- 新增物理运行 **0**、训练 **0**、生产控制器/估计器改动 **0**；仅离线读取 D007 第二轮 34 条记录。
- 输出：`outputs/agent_exchange/D008_20261030_003/`（`summary.json`、`side_ledger.csv`、`measurement_ledger.csv`、`missing.json`、`manifest.json`）；离线测试 `tests/test_d008_endpoint_audit.py`（5 项通过）。
- 任务书：`tasks/D008_Endpoint_Reference_Audit.md`；配置：`tasks/D008_config.json`（66 个输入哈希全部匹配）。

## 先回答

**误差在测量还是跟踪？** 这次不能给出最终归因。测量区间本身**覆盖真端点 100%**（10,689 条 measurement，区间外 0），但区间宽度中位数 **4.28 cm**、P95 **21.1 cm**，已经大于 4 cm 纠偏；这说明存在可观测性限制。跟踪 anchor 的误差 median **8.5 mm**、P95 **47 mm**；reference 的 P95 达 **24 cm**。但 R0 历史重放未通过，所以不能把 reference 误差归因到预测/校正，也不能据此删任何一项。

**为什么会抵消偏移？** 已有 probe 分解支持“offset 写入正常、reference 变化抵消 net target”，与 GPT 复核一致；direct offset 归一化响应中位数 **1.000**，reference 项中位数约 **−0.329**。但本轮的 reference 恒等式尚未闭合到任务要求的 1e-7，因此不能进一步把 reference 变化归因到 endpoint 算法错误。

**能否只删一项？** 不能。R1/R2 未运行；本轮没有可支持的唯一改动。

## 1. 分母与覆盖

| 项目 | 数量 |
|---|---:|
| runs | 34 |
| control ticks | 3,934 |
| side-ticks | 7,868 |
| endpoint rows | 7,568 |
| side-ticks without endpoint | 300（保留，不静默删除） |
| measurements | 10,689 |
| face binding ambiguous | 0 |
| anchor switch events | 14 |

测量层：

| 指标 | median | P95 | max |
|---|---:|---:|---:|
| interval width (m) | 0.04283 | 0.21120 | 0.32950 |
| true endpoint in interval | **100%**（10,689/10,689） | — | — |
| outside >1 cm / >2 cm | 0 | 0 | — |

区间覆盖不等于中点准确；本轮没有把 `shadow_censored` 当测量错误，也没有用中点误差替代覆盖。

跟踪/参考层（7,568 行）：

| 指标 | median | P95 | max |
|---|---:|---:|---:|
| anchor error vs true endpoint (m) | 0.008455 | 0.04720 | 0.14050 |
| reference midpoint error (m) | 0.002562 | 0.24000 | 0.26805 |
| tangent reference error (m) | −0.000052 | 0.24442 | 0.28541 |
| normal reference error (m) | 0.0 | ~1e-15 | ~2e-15 |
| L_hat (m) | 0.52 | 1.00 | 1.08 |

- 参考恒等式残差（`M_hat` vs `face_midpoint` / target reconstruction）中位数 2.6 mm，但 P95 **24 cm**；未达到任务要求的逐 tick ≤1e-7。原因尚未定位（anchor side/L_hat/参考公式三者之一或记录语义），因此**不把该 P95 当最终 reference 误差**。
- 宽区间与 anchor 误差提示“测量区间可观测性”是候选解释；但 R0 未复现，不能宣布。

## 2. Probe 分解（使用 GPT 复核的物体坐标系结果）

`gpt_review/corrected_probes.json`，16 条 probe：

- 有效 14/16；B/C 共同有效 6/8（GPT 复核）。
- direct offset 归一化：median **1.000**，min 1.000，max 1.000。
- reference_without_offset：median **−0.3295**，min −1.5469，max +0.0348。
- G_robot：median **0.78835**；G_target：median **0.67052**。
- 典型 adjacent/B/robot1/−4 cm：direct ≈ +1.000，reference ≈ −1.547，net target ≈ −0.547；C 对应 +1.000−1.163≈−0.163（GPT 复核）。
- visibility/reservation 裁剪：0；与 GPT 复核一致。
- 这些是闭环轨迹分解，不是同一状态的开环导数。

## 3. R0 历史估计器重放

- 用记录扫描重建 LidarScan + scan_patches，匹配记录 `face_points`（最大点距 0），重启 EndpointTracker（tau=.10/max_age=.25/max_speed=.30/residual_limit=.15）。
- **pilot：`013_probe` side0，38/100 行匹配 ≤1e-7；最大误差 4.18e-5 m。**
- 未达到 R0 门槛（≥99%）；因此：
  - 不运行 R1（去速度预测）；
  - 不运行 R2（当前区间中点）；
  - 不用 R0/R1/R2 支持任何调参或控制器改动。
- 复现缺口的最可能来源：记录中被记录的 endpoint 是**融合后区间**，而离线重建的 patch/length/anchor 语义还不完全一致；具体差异尚未定位。需要先修重放，而不是放宽 1e-7 或从真值补状态。

## 4. 测试

- `tests/test_d008_endpoint_audit.py`：5 项离线纯几何/schema 测试通过；不实例化 env/backend/scan_lidar。
- 没有新增物理实例，没有训练，没有 D009。

## 5. 未完成/偏差

1. **R0 重放覆盖 38%**，低于 99%；R1/R2 未运行。这是本轮最先要修的复现问题。
2. 参考层恒等式未闭合到 1e-7；当前 P95 24 cm 不能作为最终误差归因。
3. 任务书要求的分场景/profile/面/kind 分组统计、连续事件表（≤12）、12 张图、可检索 index.html 未完整交付；目前只有全量账本/摘要。
4. 未补改动前基线源码快照；未新增物理运行，也没有补采。
5. 测量覆盖 100% 但区间宽（P95 21 cm）；可观测性限制是候选解释，不是已证因果。

## 6. 交回 GPT

建议下一步**只做 D008 的复现修复**：

- 先定位 R0 的 62/100 行为什么不匹配（融合后区间 vs 原始 measurement、length/anchor side、patch 选择、reset/去重时刻）；
- 修到 ≥99% 后再跑 R1/R2；在此之前不修改偏移入口、不放开 reservation、不调 gain、不改估计器；
- 若修复后仍显示宽区间主导，再讨论可观测性/测量校准；若跟踪误差主导，再讨论预测/校正。

## 7. 产物

- `outputs/agent_exchange/D008_20261030_003/summary.json`
- `side_ledger.csv`（7,568 endpoint 行）、`measurement_ledger.csv`（10,689 measurement）
- `missing.json`（300 无 endpoint + 0 ambiguous）
- `manifest.json`（HEAD、输入/脚本 hash）
- `tests/test_d008_endpoint_audit.py`
- 旧 D007 记录：`outputs/agent_exchange/D007_20261030_002/runs/`（只读）
