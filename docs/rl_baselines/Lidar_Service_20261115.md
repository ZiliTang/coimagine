# 雷达访问服务

运行时入口为 `src/coimagine/random_start/lidar_service.py` 的 `LidarService`。每个环境持有 `env.lidar`，控制器、公开观测、测台与低层诊断通过环境适配器调用这个服务，不在消费者中再编写射线查询循环。

## 分层与观测约定

| 层 | 职责 |
|---|---|
| `lidar_service.py` | 采集入口、不可变 `RayFrame`、缓存生命周期、目标类别视图 |
| `lidar.py` | 历史解析几何 scalar/numpy_batch 内核，`LidarScan` 与表面拟合 |
| `ray_query.py` | Pymunk 最近命中内核、原生 ABI 加速与参考回退 |
| 环境适配器 | 给传感器提供几何、将命中投影为合法观测、构建接触/动作菜单 |

实验目录的 `relational_search/ray_query.py` 仅保留旧导入兼容，不再持有一套实现。`verify.py` 与测试中的直接内核调用是数值参考，不是新增运行时访问入口。

三个历史扫描约定保持显式：

- 单物体旧控制器：解析几何雷达，使用既有 scalar/numpy_batch 配置及完整几何缓存键。
- 多物体控制器和公开包：Pymunk、72 条射线、`R.LIDAR_RANGE`（当前 5 m）、机器人航向。公开视图保留所有命中的可动物体 ID；控制器视图只把当前操作目标标为目标，其他可动物体标为障碍。两个视图共用距离/有效位，不重新投射射线。
- `PhysicsProbe`：保留 72 条、1.2 m、世界固定方向和 Chipmunk 原始命中坐标；不能默默替换成上面的协议。

**单物体的控制器扫描与公开包扫描仍是不同后端。** 本轮没有强制将两者改为同一数值结果。初始样例中距离已有约 1e-15 的差别；特殊几何的命中和 tie 规则也不能假定完全相同。要统一这两个观测约定，需要另行版本化和行为验证。

## 调用与生命周期

环境/测台可取 `MultiObjectOptionEnv.ray_frame(robot_id, tick, time)`，返回：

- `scan`：原有不可变 `LidarScan`；
- `entity_ids`：逐射线命中的不透明 ID 或 None；不包含未命中实体；
- `hit_points`：仅显式需要原始交点的诊断协议填充。

控制器仍使用 `scan_for`，部署策略仍使用 `public()` 投影，不接受传感器服务、真实物理对象或标签全表。

在批量读取同一瞬间的多种视图时：

```python
with env.lidar.read_scope(enabled=True):
    frame = env.ray_frame(robot_id, tick, time)
    target_scan = env.lidar.target_view(frame, target_id)
```

多物体 `public()` 和 `joint_step()` 已管理此作用域，启用条件为既有 `cache_public_sensors`。同一公开包每机器人只采集一次物理扫描，即使需要多个目标菜单。作用域可嵌套，最外层退出或异常都会清空物理扫描/视图/传感器缓存。

物理步进前后、目标切换、分支恢复调用 `invalidate()`。编辑器在两个调用之间修改位姿不受 tick 限制，不会沿用陈旧缓存。**不允许在读取作用域内手动改变位姿、形状、过滤器或标签而不调用 `invalidate()`**。解析扫描的跨调用缓存仍以完整几何与时间为键。

服务对象只存在于环境进程内，不进入策略输入、轨迹或 checkpoint。默认训练仍只保留短历史和 PPO buffer；需要回放的少量 trial 显式启用详细记录，保存耗时继续输出至 `[storage]` 日志。

## 解析雷达编译融合后端（2026-11-15）

`lidar_native.py` / `lidar_kernel.c` 将线段、墙面胶囊、圆形求交、裁剪及最近命中选择融合到一个 C 调用。仅加速原 `numpy_batch` 解析扫描；scalar 数值参考和 Pymunk 查询保持原样。`lidar.py:scan_lidar_batch` 本身仍是独立 NumPy 参考。

先显式构建一次：

```bash
PYTHONPATH=. .venv/bin/python -m coimagine.random_start.lidar_native build
```

构建只需 gcc/libm，无新增 Python 包。关闭 fast-math 和浮点乘加融合；原有 CPython 三角函数、hypot 及射线角度顺序保留。默认产物位于 `~/.cache/coimagine/lidar-native-v1/<key>/`，可用 `COIMAGINE_LIDAR_CACHE` 指定目录。源码、平台、编译选项决定缓存路径；manifest 记录编译器和二进制哈希，加载前核验。并发构建有文件锁并原子发布。

在进程启动前设置 `COIMAGINE_LIDAR_KERNEL`：

- `auto`（默认）：已有匹配二进制时加载；否则输出一次回退日志并使用 NumPy。
- `numpy`：显式使用原 NumPy 参考后端。
- `native`：要求已有编译产物；缺失时明确报错。

运行中不自动安装工具或启动编译器。损坏的二进制/manifest 不静默回退。选择在进程内缓存，修改环境变量后应重启采样进程。已有测台若要使用新实现也需要正常重启，不能假定运行中服务已热更新。

日志形如 `[lidar] analytic_kernel=native source=...`；新语义 RL rollout 摘要增加 `analytic_lidar_backend`，记录选定解析后端及实际构建信息（不表示每一种场景都调用了解析扫描）。新 campaign 源码快照和包分发包含 `lidar_kernel.c`。旧冻结实验不修改；严格重放旧运行时可使用其源码快照，或在当前实现下显式选择 NumPy 做兼容检查。

C 内核没有共享可变场景缓存；每次调用的输入/输出数组独立，CDLL 调用可释放 GIL。它只替代传感器内核，不改变观测字段、射线数、采样时机、可见性、动作或物理步进。部署策略仍不能直接访问这个宿主传感器模块。
