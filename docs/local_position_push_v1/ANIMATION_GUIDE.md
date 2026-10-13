# 动画观看指南（V/X）

全部视频 **1× 仿真时间**（日志 50 ms 采样，20 fps 导出）。结束较早的一侧冻结终帧并标 `ENDED`。

每例包含：`figures/pairs/<case>__VX.mp4`（左右并排）、`figures/singles/<case>__V.mp4` 与 `__X.mp4`、
`figures/gif/<case>__VX.gif`（10 fps 轻量预览）。离线索引 `index.html` 不依赖网络。

**X 的叠加标记**：`q` 红方块＝观测到的表面点；`p_contact` 红色空心圆＝无压入的接触中心参考；
`p_ref` 绿色空心圆＝推进参考（**虚拟参考，位于物体内部，不是物理机器人**），与 p_contact 之间
用绿色点线相连；机器人中心到 p_ref 的细蓝线是位置误差。V 没有位置目标，标 `p_ref N/A`。
箭头比例固定：法向红、参考速度绿（0.26 m/s 尺度）、实际施力黑（15 N 尺度）。

建议先看这 3 例：

1. `1_solo_central__VX.mp4` —— 中央单推。两臂几乎重合（零间隙等价），X 侧可看到 β 在前 0.2 s
   把推进参考从 p_contact 逐渐推进到 p_ref。
2. `7_max_difference__VX.mp4` —— |X−V| 位移差最大的 root（结果驱动选例，不代表总体）。
   看差异是出现在 β 爬升期还是出现在接触丢失之后。
3. `8_failure__VX.mp4` —— X 中止而 V 正常的 root。看 X 是 `contact_lost` 还是
   `patch_support_lost`，以及中止前 gap_signed 与位置误差的形状。

其余：`2_solo_offset`（偏心推）、`3_dual_same_side`（同侧双推）、`4_dual_rotate`（转动）、
`5_incompatible`（不兼容输入）、`6_constrained`（受限场景）。
三个时序诊断案例（中央单推/同侧双推/偏心转动）另有逐周期曲线：
`timing/<case>__diagnostic.png` 与配套 CSV。
