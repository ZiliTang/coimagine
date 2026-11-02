# D002 执行回报：持续执行验证与衰减切换成本短对照

- 状态：`complete`（A 阶段完成；**A 门槛未通过，B 按任务书不运行**）。
- 任务书：D002 v1（`tasks/D002_Switch_Cost_and_Persistence.md`）。
- 开始/结束：A 阶段 2026-10-29 17:57–17:58（Asia/Shanghai）。
- 实际预算：A 正式 24 条分支/12 对，**7.4 s**，4 worker；0 errors；无新增物理异常；A ≤120 s。另有同日 1 个源、2 条分支的实现探针（见“偏离与限制”），未进入正式 24 条。
- 输入：parent `dac637e6...`、panel `a60bdcc1...`、D001 输出完整（92/92，0 异常），SHA256 与 D002_config 一致。
- 输出：`outputs/agent_exchange/D002_20261029_1757/A/`。

## 先回答

**“多坚持是否有用？”** 在这 12 个定向状态上，**没有足够证据**。把原同推动作连续执行 3.2 s 后：

- 只有 **3/12** 的 progress 改善 ≥1 cm（门槛 ≥6/12）；
- 每根改善数 **dual_2 = 1/6、dual_0 = 2/6**（门槛每根 ≥2/6）；
- progress 改善中位数 **+2.46 mm**（门槛 ≥10 mm）；
- 严重退化（≤−2 cm）**3/12**（门槛 ≤2/12）；
- engaged_seconds 改善 ≥0.4 s 的只有 **0/12**（门槛 ≥4/12）。

因此**不进入 B**：没有足够依据用 0.08/2.4 s 的切换成本去鼓励坚持。未修改 c_max、衰减时间、样本或阈值；没有启动训练；没有改 reward/控制器/物理/观测。

**“惩罚是否转化为稳定成功？”** 未测试。B 没有运行，960 维 ledger、0.08/2.4 s 成本和四次短训都未实现/未启动。

**“有没有妨碍停止或退出失败动作？”** A 只覆盖 3.2 s 窗口，且 hold/recorded 都没有 success；没有新增异常、没有截断窗口。不能据此回答长期停止/固执问题；该问题需要 B，而 B 未获准运行。

## A 方法与复现

- 12 个源边界来自 D001，已写死在 `D002_config.json`；每根 6 个，每 episode 最多 1 个。
- 每个边界从 reset 重放 D001 的 5 Hz 命令前缀；到 source t_start 检查 raw_obs、mask、物体/机器人位姿、总 period。
- 两条分支各执行 3.2 s（64 control tick）：`recorded` 重放原 5 Hz 后续动作；`hold` 固定 source `(a0,a1)`。窗口内原安全/失败/恢复/终止全部生效。
- 20 Hz 记录请求/提交动作、stage、ready、contact_ready、joint_active、pending、gap、局部失败/等待、真实接触、力、物体状态/速度/距离；真值只做离线评估，不进入策略。
- 12 对全部完成：raw_obs 最大误差 **0.0**、mask 误差 **0**、位姿最大误差 **1.19e-7**、总 period 全部一致；24/24 source 检查通过。
- `recorded` 与 D001 旧轨迹重叠段最大误差 **1.19e-7**；24 条分支 0 异常、0 success、0 truncation。
- 门槛按任务书原值判定，未改样本、阈值或统计口径。

## A 逐对结果

`delta = hold progress − recorded progress`；progress 为 3.2 s 窗口内物体到目标的距离减少（米）。engaged 采用双方提交同向平移 PUSH、双方 `contacting_object/contact_ready/joint_active` 均为真、按 0.05 s 累加的真实秒数。

| 源 | t_start | delta progress | recorded / hold progress | delta engaged | hold 双方 push 阶段 | hold wait / failure | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| dual_0 seed1001 d17 | 13.6 | −0.0467 | +0.047 / 0.000 | 0.000 | 0.35 s | 0.25 / 0.00 | 退化 |
| dual_0 seed1002 d103 | 82.4 | +0.0725 | −0.183 / −0.110 | 0.000 | 3.05 s | 0.10 / 0.00 | 改善 |
| dual_0 seed1004 d21 | 16.8 | +0.0062 | +0.062 / +0.068 | 0.000 | 0.00 s | 0.15 / 0.00 | 小 |
| dual_0 seed1006 d3 | 2.4 | +0.0335 | +0.073 / +0.106 | 0.000 | 0.00 s | 0.10 / 0.15 | 改善 |
| dual_0 seed1010 d43 | 34.4 | +0.0077 | −0.009 / −0.001 | 0.000 | 0.35 s | 0.45 / 0.00 | 小 |
| dual_0 seed1011 d3 | 2.4 | −0.1950 | +0.227 / +0.032 | 0.000 | 0.75 s | 0.40 / 0.00 | **严重退化** |
| dual_2 seed1006 d3 | 2.4 | −0.0519 | +0.158 / +0.106 | 0.000 | 0.00 s | 0.10 / 0.15 | 严重退化 |
| dual_2 seed1008 d76 | 60.2 | −0.0104 | −0.108 / −0.118 | 0.000 | 1.75 s | 0.15 / 0.10 | 小 |
| dual_2 seed1010 d62 | 49.6 | −0.0144 | −0.102 / −0.116 | 0.000 | 0.35 s | 0.25 / 0.00 | 小 |
| dual_2 seed1011 d2 | 1.6 | +0.0084 | +0.020 / +0.028 | 0.000 | 0.55 s | 0.40 / 0.00 | 小 |
| dual_2 seed1012 d104 | 83.2 | −0.0012 | −0.009 / −0.010 | 0.000 | 1.55 s | 0.20 / 0.00 | 小 |
| dual_2 seed1013 d10 | 8.0 | +0.0228 | +0.009 / +0.032 | 0.000 | 0.30 s | 0.40 / 0.00 | 改善 |

关键观察：

- 3 个改善是 +0.0725、+0.0335、+0.0228 m；3 个严重退化是 −0.195、−0.052、−0.047 m；中位数只有 +2.46 mm。
- d103 的 +0.0725 m 是 recorded −0.183 / hold −0.110：两条分支都在**远离**目标，hold 少退了一些；不能称为建立有效推进。hold 本身有 5/12 的 progress<0。
- **所有 12 对的 engaged_seconds 都是 0**，但 `engaged=0` **不等于**握手从未激活：逐 tick 复核为双方接触 2 个 tick、双方 `contact_ready` 2 个 tick、双方 `joint_active` 7 个 tick、四条件合取 0；准确的结论是双方接触十分稀少，且没有与激活状态形成持续重合。详见下方“GPT 复核修正”。
- hold 有时把“双方 push 阶段”从 0 提到 0.75–3.05 s，但 `push` 阶段标签不能当作有效双机推进。
- hold 的 wait 大多 0.10–0.45 s、failure 0–0.15 s；没有新增异常，也没有 success。3.2 s 的持续执行没有稳定形成持续双方接触/激活重合，也没有稳定带来推进。
- 这个结果与 D001 的“同推请求 80 次、45 段仅一次决策”一致：问题更像是双方接触稀少/未与激活持续重合，而不是“再坚持一下就会好”；但不能直接说握手逻辑坏了。A 只覆盖 12 个定向状态，不能据此证明所有状态都无益。


## GPT 复核修正（不是原回报结论）

复核报告：[D002_Switch_Cost_and_Persistence.md](../reviews/D002_Switch_Cost_and_Persistence.md)。复核逐项复算了 12 条 hold 的 768 个 20 Hz 记录，并核对了控制器时序；执行侧独立复算与之一致：

| 条件 | 20 Hz tick | 记录时长 |
|---|---:|---:|
| 双方 `contacting_object` 都为真 | 2 | 0.10 s |
| 双方 `contact_ready` 都为真 | 2 | 0.10 s |
| 双方 `joint_active` 都为真 | 7 | 0.35 s |
| 双方 `ready_flag` 都为真 | 191 | 9.55 s |
| 四条件同时成立（engaged） | 0 | 0 s |

- 10/12 条 hold 没有记录到双方同时接触；另两条各只有 1 个 tick。
- `dual_0/seed1002/d103`：85.25 s 双方接触/ready 为真、joint_active 为假；85.30 s joint_active 变真、接触/ready 变假；该分支 joint_active 累计 7 个 tick。握手激活确实出现，只是没有与持续双方接触重合。
- `dual_2/seed1012/d104`：86.40 s 双方接触/ready 为真、joint_active 为假；这是窗口最后一拍，没有后续 tick 判断下一拍是否激活。
- 源码时序：每 tick 先由 `_coordinate` 读取此前的 `contact_ready`，再由 `skill.step` 根据本 tick 观测更新，最后推进物理并写 row。因此 row 里的接触来自 tick 开始观测，skill flags 来自该次控制计算；engaged 合取为 0 不能单独解释为“握手从未建立”或“握手逻辑故障”。
- 更准确的结论是：**双方接触十分稀少，且未形成与联合激活持续重合的执行条件。** 需要区分“所选状态/方向不适合”“接触建立后很快丢失”“接触具备但协调/执行没有利用”三种情况。
- 这不改变 A 门控：改善数、中位数、严重退化三项独立失败，停止 B 不依赖 engaged 一项。
- 边界：B 未运行，不能宣布切换惩罚无效；A 固定原方向/速度持续 3.2 s，与 B 允许同族内调整速度/方向并不等价，也不能推广为“任何持续机制都没用”。

## A 门槛

| 门槛 | 实测 | 结果 |
|---|---|---|
| 12 对完整/可复现、hold 无新增异常 | 12/12；24/24 source；0 异常；0 截断 | 通过 |
| ≥6/12 delta≥0.01 m，每根≥2/6 | 3/12；dual_2 1/6，dual_0 2/6 | **失败** |
| delta 中位数≥0.01 m | +0.00246 m | **失败** |
| delta≤−0.02 m 不超过 2/12 | 3/12 | **失败** |
| ≥4/12 engaged 改善≥0.40 s | 0/12 | **失败** |
| A 通过 | — | **未通过** |

## B、测试与偏差

- B 未运行：没有 `experiments/option_rl_v17` 的 960 维/ledger/cost 训练实现，没有 A0/A1 四次短训、没有 640 训练 + 400 评估 episode。A 失败即停止，不实现 B 的训练扩展。
- 聚焦测试：`tests/test_option_rl_v14.py`、`v15`、`v16` 共 **20 项**全部通过；这些是冻结环境/接口/PPO 的回归，不覆盖未实现的 ledger/cost，也不覆盖 A 新增 20 Hz 诊断字段的语义。A 自身的复现检查（raw/mask/pose/period、recorded 重叠、24/24 source）通过。
- 实现探针：正式 A 前，用 `dual_2 seed1006 d3` 的 recorded/hold 两条分支验证了记录路径（source raw 0、pose 7e-8、recorded 重叠 1.18e-7）；这 2 条不进入正式 24 条，也未单独计时（约数秒），不计入 7.4 s 的正式 A 时长；正式 A 仍为 12 对/24 分支。探针 + 正式 A 合计仍远低于 120 s。
- A `manifest.json` 初版 `head` 字段为空：诊断脚本误用了 `git HEAD` 而非 `git rev-parse HEAD`。已在输出 manifest 中补上 `head=22c1333...` 并加 `head_note`；没有改动任何 branch/pair 结果。
- `head` 已修正后工作区 148 个未提交项；完整源码快照在 `A/source_snapshot/`，哈希在 `A/source_hashes.json`。
- 任务书、config 已复制到 `A/task.md`、`A/config.json`；parent/panel/D001 manifest 哈希在 `A/manifest.json`。
- 任务总预算 1200 s；本轮只用了 A 正式 7.4 s，B 未启动；没有为凑 B 减 episode 或改参数。
- 7.4 s 是正式 branch 阶段（source 重放 + 两个窗口）的时长，不含 pair/replay/snapshot 写入、分析或 2 条实现探针；这些没有单独精确计时，不能把 7.4 s 当作全部交付开销。后续应把探针、分析、重试分别计账。
- A 的 deadline 在超时后 cancel future，但 ProcessPoolExecutor 退出仍可能等待已运行 worker；本轮 7.4 s 未触发，不影响结果。后续复用前应补进程级硬截止验证。

## 交回 GPT

A 是事先确定的保守启动门槛；A 的负结果支持按门槛停止 B，不自动加惩罚、不调 c_max/2.4 s、不开始长训、不选 D003。复核接受这个负结果，但收紧了接触/握手解释：不能说握手从未建立，也不能说切换惩罚无效。

若要继续，按复核建议做一次范围很小的**接触建立/保持审计**：加入已有成功双推参考序列作正对照；对比“始终没有双侧接触”和“短暂双方接触后丢失”的代表窗口；在统一 tick 前后时序下记录间隙、方向、接近速度、接触、ready/release/active、恢复原因；先判定是状态/方向不适合、接触保持问题，还是协调/执行没有利用，再选唯一改动。independent 到达后停止/判稳仍是另一条独立候选，本次不并入、不开工。这需要新的任务书，本轮不自行扩展。

## 原始产物

- [summary.json](../../../outputs/agent_exchange/D002_20261029_1757/A/summary.json)
- [gate.json](../../../outputs/agent_exchange/D002_20261029_1757/A/gate.json)
- [pairs.csv](../../../outputs/agent_exchange/D002_20261029_1757/A/pairs.csv)
- [branches.csv](../../../outputs/agent_exchange/D002_20261029_1757/A/branches.csv)
- [manifest.json](../../../outputs/agent_exchange/D002_20261029_1757/A/manifest.json)
- [replay.html](../../../outputs/agent_exchange/D002_20261029_1757/A/replay.html)
- [source_hashes.json](../../../outputs/agent_exchange/D002_20261029_1757/A/source_hashes.json)
- [branches/](../../../outputs/agent_exchange/D002_20261029_1757/A/branches/)（24 个 branch JSON；20 Hz 行）

## 下一步状态

D002 A 完成、门槛未通过；B 未运行。`CURRENT.md` 为 `reviewed`；GPT 复核已写入 `reviews/D002_Switch_Cost_and_Persistence.md`。没有自动启动下一轮或 D003。
