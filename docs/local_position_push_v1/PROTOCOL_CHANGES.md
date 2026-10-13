# 公共协议改动（本轮）

`src/coimagine/local_contact_option/` 整模块相对 git HEAD 是新增目录，因此这里给的是
**逐处改动对照**而不是对未跟踪基线的 diff。两处改动都作用于 V 与 X 共用的传感，
属于协议修复，不归功于新控制器。

## 改动 1：射线分组的距离连续性基准

文件 `sensing.py::scan_surface`。

```python
# 改动前：与本次 run 的第一条命中射线比较
end = index
while end + 1 < scan_rays:
    nxt = hits[end + 1]
    if nxt is None or abs(nxt[0] - hit[0]) > gap_tolerance:   # hit = hits[index]
        break
    end += 1

# 改动后：与相邻射线比较
while end + 1 < scan_rays:
    nxt = hits[end + 1]
    if nxt is None:
        break
    if abs(nxt[0] - hits[end][0]) > gap_tolerance:            # 相邻
        break
    end += 1
```

实测影响（dev/constrained/000 同一帧）：改前目标物体被切成 7/3/5 条射线三段（跨度 0.068 m），
改后 11/10 条射线两段。

## 改动 2：360° 扫描改为环形分组

文件 `sensing.py::scan_surface`。改前从 index 0 线性扫描分组，跨 0/2π 边界的片段被拆开；
改后先找一处真实断点作为起点，再按环形索引扫描，`_linked(first, second)` 用
`(index-1) % scan_rays` 判相邻。

实测影响（dev/solo_central/000 同一帧）：机器人正对物体 0.25 m 端面时实际 14 条射线命中，
改前以「8 条 + 6 条」出现（跨度 0.088 m），改后合并为一段。

**为什么必须改**：未改前 X 的 `patch_support_lost` 在 dev 六类中的四类于 PUSH 开始 0.15 s 误触发
（垂足被判为离开观测线段，而实际是分组人为截断了可见跨度）。

## 改动 3：日志与指标字段（只增不改）

- 行内 `observed[*]` 新增：`beta`、`signed_distance`、`gap_signed`、`q`、`p_contact`、`p_ref`、
  `error`、`error_bounded`、`force_raw`、`support`、`support_loss`、`tangential_offset`、
  `position_active`、`segment_world`、`contact_point_robot`、`contact_impulse`。
- 行内新增 `phase`（由逐机器人阶段派生的联合相位，供分阶段位移分解使用）。
- `metrics` 新增：`terminal_robot_speeds`、`terminal_robot_omegas`、`approach_seconds`。
- 新增退出原因 `patch_support_lost`。
- 既有字段与既有退出原因语义未变；上一轮的执行记录仍可被同一指标脚本读取。

## 未改动

物理参数、力上限、控制周期、信息白名单、APPROACH/READY/BRAKE/SETTLE 语义、判稳实现、
延迟队列、`V` 臂的速度式控制律、root 与扰动定义。
