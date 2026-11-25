# 一个故障簇的 CAP 模块开发

作者已交卷。[实时进度](http://localhost:8775/research-journal#single-cluster-cap)

母程序为固定 72 束、局部中心估计的 s003。开发池固定对角线起终点，只改变一个静态障碍和一个可动障碍的位置。本轮选择跨失败房间重复最多的 staged/fixed 停滞。它表示机器人已经接近目标物后方可用推位，物体朝目标直进会受固定障碍或墙阻挡。在至少 120 tick 的观察窗两端处于该局面，目标距离缩短不足 3.5 cm。记录每间房的第一次合格窗口。开发池 160 间里，18 间出现过它且最终失败，5 间出现过它且自行恢复。这个定义基于母程序，改变策略后仍使用相同房间分组。

6.2-Sol 只使用 12 个簇内失败、3 个簇内自行恢复和 7 个簇外原成功房间的公开案例、公共技能指南、原程序与开发试跑反馈。作者可自行决定何时交卷，最长 30 分钟，无实验 token 配额。代码冻结后才评估新模块在全新 200 间上的表现。预先约定严格判据：新面板簇内母程序失败房间救回率的 95% Wilson 下界高于 50%，同时原程序的成功房间实测零丢失；另行报告未达判据时的部分救回。

三分支从同一个已重现的公开历史、相同剩余预算开始：母程序继续、请求模块立即接手、由作者入口决定。强制分支仍需有可执行动作，模块首次放弃后由原程序执行到底。部署测试从初态运行，允许多次接手。每个房间是一个统计单位。


[打开七组同步物理回放](http://localhost:8775/journal-file/single-cluster-cap/replays.html)。在簇内原失败的 21 间中，启用模块后执行历史改变的 7 间已全部收录。左右两侧从同一初态运行；可分别定位接手、首次指令分歧、交还和终局。房间 136 的模块接手期间物体净移动 42.3 cm，整局救回；110 净移动 49.6 cm，最终仍失败；084 的目标物在该时段基本未动。

## 冻结验收

新房间簇内原失败救回 2/21，簇内原成功丢失 0/5；簇外原成功丢失 2/128。整局 133 → 134/200。未达到预先登记的系统性解决标准。

作者选定 trial09，提交用时 15.5 分钟。Enters after persistent ineffective contact PUSH, with the current target contact clear, a nearby rear station, and a static obstacle limiting straight transport. It positions a single pusher and executes a lateral bypass or follows the open side of the obstacle until observed progress and transverse clearance support handback. It releases on bounded elapsed time, stalled or off-path motion, loss of feasible support, or completed bypass. Development full starts improved from 10/22 to 14/22 with all prior successes preserved. Both fixed-anchor probes had no outcome changes.

| 母程序上的分组 | 房间 | 母程序成功 | 模块程序成功 | 救回 | 丢失 |
|---|---:|---:|---:|---:|---:|
| 簇内 | 26 | 5 | 7 | 2 | 0 |
| 簇外 | 174 | 128 | 127 | 1 | 2 |
| 全部 | 200 | 133 | 134 | 3 | 2 |

簇内原失败救回率为 9.5%（95% Wilson 区间 2.7% 至 28.9%）。簇外原成功的丢失率为 1.6%（95% Wilson 区间 0.4% 至 5.5%）。这些区间描述该固定布局生成规则下的房间差异。仅运行一次独立作者开发。

簇外原成功的 128 间中，实际执行历史发生改变的有 2 间。

## 同一时刻的三条后续路线

| 从停滞窗口末尾起步 | 模块实际给出动作的房间 | 整局成功 | 相对原程序救回 | 相对原程序丢失 |
|---|---:|---:|---:|---:|
| 原程序继续 | 0 | 5/26 | 0 | 0 |
| 请求直接接手 | 9 | 6/26 | 2 | 1 |
| 按作者入口接手 | 4 | 5/26 | 1 | 1 |

每间房只取第一次合格窗口。三条路线到达相同的公开历史哈希，原程序重放的整局终点历史也逐项一致。请求直接接手时仍须通过作者的可执行条件。此表衡量这一处机会；从初态运行的程序还可以更早介入或反复尝试。逐例表中的局部推进是在起点与至少 120 tick 后首个命令边界之间，目标物到终点距离的减少量（米）；不足一个窗口留空。

原程序发生目标故障并最终失败的 21 间里，整局启用模块后有 7 间的终点公开历史发生变化。在救回的 diag_confirmation_026 中，模块首次给出动作的时刻为 539 tick，母程序停滞窗口末尾为 1063 tick。在救回的 diag_confirmation_136 中，模块首次给出动作的时刻为 828 tick，母程序停滞窗口末尾为 934 tick。

## 作者开发记录

| 提交 | 已测开发房间 | 簇内原失败救回 | 原成功丢失 |
|---|---:|---:|---:|
| trial01 | 22 | 1 | 2 |
| trial02 | 22 | 0 | 1 |
| trial03 | 22 | 1 | 1 |
| trial04 | 22 | 1 | 0 |
| trial05 | 22 | 1 | 0 |
| trial06 | 22 | 2 | 0 |
| trial07 | 22 | 2 | 0 |
| trial08 | 22 | 2 | 0 |
| trial09 | 22 | 4 | 0 |
| trial10 | 22 | 2 | 0 |

截至写入所选版本，作者累计使用 7,116,145 token，其中缓存输入 6,935,296。选版之后的试作与交卷开销另计入工作记录的作者总量。

全会话作者用量为 8,412,497 token，其中输入 8,375,917（含缓存 8,219,776），输出 36,580。实验以墙钟时间限制作者，未设 token 上限。

[作者工作记录与逐版用量](worklog.html) · [模块源码](module.py) · [全部新房间](rooms.csv) · [同起点三分支](branches.csv)
