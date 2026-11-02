# D010 m3b 独立八次验证报告：complete / not-supported

- 小任务状态：**complete / not-supported**。固定 8 个 attempt 全部执行，未跑第 9 次；D010 总体验收仍 **incomplete**。
- 新预算独立记账：最多 8 attempt、20min、2 worker；实际 **8 attempt / 12s**（不含本报告整理）。旧 unknown 账本未改写。
- 没有算法/参数/EKF/FF/物理/脚本改动；没有训练、没有默认替换、没有 D011。
- 原始日志的 `tick/time` 字段有记录 bug（按行数而非控制 tick），已用行序离线纠正为 `row//2`；不影响控制输出，报告中明确标注。

## 先回答

**候选是否改变首触？** F 的首次接触位置相对 B 只移动了约 **2–3 mm**（reposition_0/2），没有形成新的稳定接触点。

**角偏/推进是否改善？** F 的相对角峰值明显降低（reposition_0：B .718 → F .139 rad；reposition_2：B .702 → F .154 rad），但指定方向净推进反而下降（B .116/.109 m → F .029/.031 m），最终距离更差（B .358/.360 → F .423/.421 m）。两条 reposition 均 **fail**。

**dual 是否保住？** **没有。** B 两条 dual 均 success（距离 .054）；F 两条 dual 均 fail（距离 .436）。F 的首次真实接触从 B 的 tick 45 推迟到 94，且首触时 candidate 为 `None`；这是明确的 dual 回归，不能把候选闭包视为可接受。

## 逐 case

| case | B | F |
|---|---|---|
| T_reposition_0_p0 | fail, dist .358, progress .116, angle .718 | fail, dist .423, progress .029, angle .139 |
| T_reposition_2_p0 | fail, dist .360, progress .109, angle .702 | fail, dist .421, progress .031, angle .154 |
| T_dual_0_p0 | **success**, .054 | **fail**, .436 |
| T_dual_2_p0 | **success**, .054 | **fail**, .436 |

- 首触：F reposition_0/2 在 tick 313；B 在 311。
- F reposition 的 candidate 初始 q 约 `(2.625,1.738)` / `(2.238,2.125)`，有多次 `support_invalid/support_extension`；但 **actual shift path = 0.0**（由 truth 机器人位置计算），也就是记录了 SHIFT 命令但没有实际位移，不能声称“移到了新接触点”。
- 首触后 .5s 角变化：F reposition_0 .059 rad / reposition_2 .053 rad；B .081/.081；F 角变化较小，但没有推进。
- F dual 在首触时 candidate 为 `None`，phase 在 approach/None 间反复，物体没有推进；B 同刻已成功。
- 原始真实接触 `contacting_object`、cp、own、patch 均来自 pre_obs；B/F 逐 tick 行在 `attempts/`。

## 日志与产物

- `attempts/01..08_*.json`：8 个固定 attempt 原件（含 B/F）。
- `metrics_corrected.json`、`summary.json`、`attempts_ledger.jsonl`。
- `charts/`：四张 B/F 对照图（路径、首触/候选、角偏；phase 线）。
- 源码 hash：
  - B `29b05254e26b7567e42158d6af82495f018fc8f497049d9d8dcc6543a38e4df2`
  - F `c3f515b65bc745c0e832af953f0b209bcb57f5378d6b4d1a1e3658763332728b`
- 输入 D009 acceptance hash 在 `summary.json`。

## 限制

- 只有 4 个固定工程 case，不能说总体成功率/统计显著性；不宣称候选闭环有效。
- F dual 回归和 actual shift=0 说明 m3b 不能接正式 RL/默认；需要先修候选/dual 路径。本轮只验证，不调参、不扩 40/new/contract。
- 旧 D010 attempt 仍 unknown；本小任务账本独立。

## 结论

8 次验证完成，但结果 **not-supported**：候选没有产生新的稳定接触，reposition 仍失败，且 dual 从 2/2 退回 0/2。D010 总体验收仍 incomplete。
