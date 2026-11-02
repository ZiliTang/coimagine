# 接近阶段东推：option 与测台耗时 profile

2026-11-03，GPT。用户明确：option 行为合理性与计算开销是调优目标，task complete 仅作附带标签。本轮没有以 task success 作为优化门槛。

**主要长耗时在历史运行列表，不在本实例的 option 执行。已修复并重启8765。**

## 1. 固定输入与方法

- preset：`进度预算 · 接近阶段东推`，id `20261102_183931_e5fe2a`。
- 冻结用户最新 run：`20261103_015411_f2e7d3`，完整初态/goal/overrides/sequence 存入 `request.json`；91 control tick=4.55 s仿真。
- 同一 preset、同一控制器 f30f7f72…；不改物理、目标、动作/控制/观测频率、预算和记录字段。
- 两类测量分开：真实8765 API wall time；独立进程 cProfile+嵌套分段计时。cProfile 有开销，不能把其数值与无 profiler wall time 直接当同条件加速比。
- 输出目录：`outputs/testbed/profiles/20261103_approach_east/`。含 before_source、输入/源码hash、完整 `.prof`、按函数 self/cumulative 的 JSON/文本、分段计时、baseline/after 全轨迹及API验证。

## 2. 端到端发现

| 测量 | 优化前 | 优化后 |
|---|---:|---:|
| 同实例 POST `/api/run` | 3.267 s，一次现场测量 | .243/.254/.300 s，三次；中位数 **.254 s** |
| GET `/api/runs` | **44.960 s**，现场测量 | .00536/.00562/.00669 s；中位数 **.00562 s** |
| 独立进程旧历史列表，cProfile | **22.993 s** | 持久索引冷进程读取 .00855 s；内存热读取 .00255 s，无profiler |
| 无profiler session执行和逐帧记录 | 原API未记录该分项 | .171–.228 s |

旧接口返回约101 KB摘要，却每次加载504条轨迹、合计 **2,366,517,247 bytes**。独立 cProfile 中：

- JSON decode：**19.120 s**（约83%）。
- 文件读入/文本解码：**2.302 s**。
- 列表循环及其他：约1.57 s。

页面初始化和每次运行后都会调用历史列表接口。大量JSON解码与仿真请求在同一Python进程中竞争执行时间，会拖累用户感受到的等待。现场44.96 s与独立22.99 s并非同负载/同缓存条件，不能报告为严格的速度倍数；也不能将用户先前每次“>10s”精确倒推出同一分项比例。

该证据支持先解决历史加载。没有证据要求当前为了速度降低传感频率、削弱估计器或删掉可视化所需的记录。

## 3. option 内部的完整分解

同一91tick，优化前独立 cProfile 运行总session **.316 s**；优化后 **.305 s**，属于小幅测量波动，算法没有改动。

下表为优化前的**包含子调用计时**。父子不能相加。

| 环节 | 调用数 | 累计秒 | 语义 |
|---|---:|---:|---|
| decision loop | 23 | .310 | 包含观测、记录、协调、slot和物理 |
| 观测构建 | 182 | .130 | 含扫描、patch、旧anchor及contact等 |
| 雷达扫描 | 182 | .070 | 观测的子项；batch实现 |
| patch拟合 `scan_patches` | 182 | .037 | 观测的子项 |
| 旧anchor tracker | 182 | .0097 | 观测的子项 |
| 协调/分配 | 91 | .0435 | 含共享估计和memory |
| 共享运动估计 | 91 | .0276 | 协调的子项 |
| support memory update/calibrate | 91/23 | .00277/.00322 | 协调的子项 |
| R0 slot.step | 91 | .00657 | PUSH/切换控制调用 |
| R1 slot.step | 91 | .00081 | BRAKE调用 |
| physics+gate | 91 | .0265 | 物理/门控，不含外层记录全部耗时 |
| prepare记录路径 | 182 | .0344 | 其中诊断mask约.0269；不是PUSH控制成本 |
| control-step记录包装总计 | 91 | .0460 | 含上面的physics+gate，勿重复相加 |
| JSON格式化（profiler关闭后） | 1 | .0478 | 约1.67MB、indent=1，保留原记录内容 |
| 文件写入 | 1 | .00131 | 同机写入 |

完整函数表见 `baseline_pstats.txt`、`baseline_functions.json`，原始profile见 `baseline.prof`；优化后对应 `after_*`。旧profile行号对应 `before_source/` 的源码。

在这条实例上，option执行、估计、物理均非秒级瓶颈。若下一步要进一步压核心CPU，观测/雷达/patch是最大分项，其次是估计和记录mask；目前优先级低于已消除的整库解析。不能由一个实例外推所有option耗时相同。

## 4. 已落地优化

### 历史列表只读摘要索引

`testbed_server.py` 新增 `outputs/testbed/run_index.json` 持久化摘要索引，保存文件名/mtime_ns/size和原列表所需摘要。

- 初次为历史文件一次性建索引；之后新进程也直接加载摘要。
- `_run` 记录完成后从已有结果对象更新摘要，不为摘要再解析一次轨迹。
- `_run_list` 只检查文件stat；仅新增/修改的文件需要解码，删除时剪除索引项。
- index和run采用原子替换，索引访问加线程锁，避免并发新run丢摘要。
- 历史原件与排序/摘要语义保留；无索引/损坏索引可以重建。**如果索引被删除，首次重建仍会解析历史文件一次**，不宣称该路径也恒为毫秒级。

一次性bootstrap记录在 `profile_history.py`；测试没有依赖该bootstrap才能通过，普通缺索引路径可自行构建。

### 可见的分项计时

`session.py` 记录执行/逐帧记录、finalize、回放frame整理、session总计及仿真时长。

前端显示：

- 仿真/记录秒数、回放整理秒数；
- POST接口+传输+前端JSON解析总秒数；
- JS同步绘制调用毫秒数（**不是浏览器实际呈现完成时间**）；
- 历史列表获取秒数与DOM整理毫秒数。

记录本身仍完整，不通过删日志或降采样制造加速。Run按钮执行中禁用，避免重复提交叠加负载；状态请求返回后也复查锁，处理快速双击。

## 5. 验证与运行状态

- 新增 `tests/test_testbed_run_index.py` **5项通过**：未改轨迹不重读、跨进程持久索引复用、修改/删除失效、写入直接更新、损坏索引/无效run、并发摘要完整性（部分在同一测试中包含多个场景）。
- JS通过 `node --check`。真实浏览器性能尚未采样；上面的前端计时会在用户下一次运行时显示，不将服务器数据冒充浏览器实测。
- 新服务 **PID1346288，8765**。`outputs/testbed/server.json` 指向当前进程，日志在本轮profile目录。
- 新服务三次同实例验证：`20261103_020406_832f4f`、`20261103_020406_5da2db`、`20261103_020407_fbe9e7`。
- **三次全部逐帧frame与优化前profile记录完全相同**，位姿/速度最大误差0；每次仍91tick、182次扫描。控制器/估计器源码hash不变。
- API smoke、baseline profile、after profile、3次API复测，共 **6条完整同实例运行**；另一次profile脚本在初始化包装dict tracker时中止，0 control tick，修正纯profile脚本后重试。历史列表profile不运行物理。无其他实例/全矩阵/训练。

验证元数据 `api_after.json`、`baseline_summary.json`、`after_summary.json`、`history_*`，保留所有原始证据。task标签照常保存，不以其为性能优化验收条件。

## 6. 当前结论

这次明显长等待已经定位到测台历史加载，优化后当前实例接口约.25s。页面刷新即可加载新的耗时显示；继续围绕用户指出的具体option/profile，而不为追求task complete改控制行为。
