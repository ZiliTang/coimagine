# 利用 A、B 的共同失败开发第三位专家 C

新 200 个同房间对照中，A、B、C 各自执行成功 93、121、98 间。纯专家成功并集从 A/B 的 **134/200** 到 A/B/C 的 **142/200**。C 从初态净新增 **8** 间，允许在既有 A/B 前缀接手又提供可单独计量的机会。

| 路线 | 全部成功 / 200 | A/B 纯执行都失败中的成功 |
|---|---:|---:|
| 全 A | 93 | 0 / 66 |
| 全 B | 121 | 0 / 66 |
| 全 C | 98 | 8 / 66 |
| A 首次触发后交给 C | 99 | 9 / 66 |
| B 首次触发后交给 C | 97 | 7 / 66 |

把两条固定接手与三条纯路线一起事后去重，覆盖 143/200。每条路线都有真实执行，其中 1 个新增房间只靠固定接手路线才解出。实际部署的路由仍需在线决定何时接手。

| 同一前缀对照 | 触发房间 | 救回 | 丢失 | 全部房间百分点差、配对 95% 区间 |
|---|---:|---:|---:|---:|
| A→C 对全 A | 184 | 29 | 23 | +3.00 [-3.51, +10.00] |
| B→C 对全 B | 182 | 9 | 33 | -12.00 [-18.00, -6.00] |

## 开发和选择

冻结 A、B 后扫描原有 200 间开发房间，得到 62 间双败。触发家族为 `stall`。固定 24 个房间中的 16 间为双败，另 8 间会由首发专家自行恢复。每层按 A、B 前缀等额取样。作者使用 gpt-6.2-sol，可读公开前缀和两份代码。共完成 48 局试跑，76,133 ticks，包括重放前缀。冻结第 2 版。

| 版本 | 救回 / 16 | 原成功丢失 / 8 |
|---|---:|---:|
| 1 | 2 | 2 |
| 2 | 2 | 2 |

作者选版理由。Revisions 1 and 2 each rescued 2 of 16 joint failures and preserved 6 of 8 parent successes. Revision 2 completed the same 24 cases in fewer total ticks (37,689 versus 38,444) and directly addresses repeated attempts to push a stationary target from the same direction. Revision 3 exceeded the native source-size limit and had no probe result.

独立面板在开发前从自然房间生成器冻结，按内容哈希核对已知面板。A/B/C 运行均使用固定 72 束和局部中心估计，接手时重放真实公开历史，保持原剩余预算。区间按房间配对重抽。

[协议](plan.md) · [逐房间结果](rooms.csv) · [配对指标](results.json) · [作者开发笔记](author_notes.md)
