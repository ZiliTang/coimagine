# B 的开发与冻结

作者为 gpt-6.2-sol。参照 A 为 usual s003。本轮真实 A 前缀为 16 个最终失败和 8 个最终自行恢复，B 在同一时刻接手并执行到底。预算取最多三版，每版测试同一批 24 个房间。作者选定第 2 版，选择理由如下。

V2 rescued 6 of 16 failures and preserved 6 of 8 A successes on identical paired development prefixes, for a net gain of four. V3 tied both counts while trading one rescue for another, so V2 is the simpler tested source.

| 版本 | 救回原失败 | 丢失原成功 |
|---|---|---|
| 1 | 5/16 | 2/8 |
| 2 | 6/16 | 2/8 |
| 3 | 6/16 | 2/8 |

B 源码标识 `3fd06a5c8d566941fdece843250016236f3ebf01422ad59bb375624e54bf1875`，authoring_v2 清单标识 `7c828a58bf301668f71a8e2fc1e306561460ed15825e0cf90d8a0278a86a7aca`。重复查询这批开发案例会带来选择偏差。泛化由独立房间检验。

## 策略做了什么

B 在通常情况下沿用 A 的决策。遇到公开历史显示的持续停滞后，它在当前合法的单机器人推动与重新站位之间选择。站位用扫描、历史可见静态点及同伴里程计判断可达性。重复尝试同一推手和方向而没有可见进展时，暂时降低该选择的优先级。第 2 版在靠近目标时按可见面的法向估计剩余误差，区分是否已接触，缩短推动时长，减少再次脱离适合站位的风险。这些是设计意图，不能单凭开发案例断言其分别造成了收益。

本轮三版作者试跑共 72 次真实重放，合计 105,057 个执行 tick（包括从初态重放 A 前缀）。开发模型为 gpt-6.2-sol high，作者访问范围由工作区指令限定。
