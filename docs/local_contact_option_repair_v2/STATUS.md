# STATUS — local_contact_option_repair_v2

run_id `local-contact-option-repair-v2`。A–D 全部执行完毕，代码可执行，无 BLOCKED。
v1（`local_contact_option_v1/`）文件与结果全部保留，未覆盖。

## A 修复与定向测试 — 完成

| 项 | 修复 | 测试 |
|---|---|---|
| A1 表面速度单位与时间戳 | 除以采样时间戳差、RMS 聚合、0.20 s 新鲜度、重复/乱序拒绝 | 6 项 |
| A2 固定延迟队列 | 投递已到达最新帧、保留在途帧、自身+队友、启动 unavailable | 2 项 |
| A3 制动渐变进入执行器 | 限幅参考→力；BRAKE 入口初始化；单次更新；4 s 从 BRAKE 起算 | 3 项 |
| A4 握手与退出一致性 | 逐机器人 READY、0.01 m/s 预接触、5 s 上限、三标签、被动制动、区间计时 | 3 项 |
| A5 分母与 bootstrap | 主分母 40 root、角度门槛、按 root 重采样并断言、覆盖率 960 | 见 metrics |
| A6 接续终态 | attempted/push_completed/normal、入口资格、每步位移、终态快照 | 2 项 |

`pytest tests/test_local_contact_option_repair.py tests/test_local_contact_option.py` → 32 passed。

## B 同批实验 — 完成

3840 条测试执行 + 768 条开发执行，12 root × 3 技能 × 2 臂接续。root/扰动文件与 v1 SHA256 相同，
Kv 未重选。产出 `comparison.md`（修复前后配对、四臂主/次分母、0.26 子集、分阶段位移、位移分箱）。

门槛：G2 通过（主 0.925 / 次 0.9375）、G3 通过（0%）、**G4 不通过**（0.532）、
**G5 不通过**（位置 −60.2% 通过，角度 +12.2% 不通过）。

## C 动画 — 完成

8 案例 × 4 臂 = **32 个单臂 MP4** + **8 个四宫格 MP4** + **8 个预览 GIF**，1× 实时（50 ms 采样、
20 fps 导出），离线 `index.html` 与 `ANIMATION_GUIDE.md`。选例规则、初态 hash、退出、位移、转角、
耗时、是否正常完成记录在 `animation_index.csv/json`。O/F 终帧冻结并标 `ENDED at t=...`。

## D 交付

`docs/local_contact_option_repair_v2/`、`outputs/local_contact_option_repair_v2/local-contact-option-repair-v2/`、
`local_contact_option_repair_v2_review.zip`（含完整 3840 条 JSONL）与
`local_contact_option_repair_v2_animations.zip`（单臂+四宫格+GIF+索引，不依赖重跑）。

## 下一步（只一项）

做一轮参考速度跟踪校准（把 0.26 m/s 同命令下的位移比从 0.67 拉回 ≥0.80），再考虑小规模 option 数据。
