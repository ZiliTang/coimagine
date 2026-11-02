# 仿真核心组成与无行为变化提速

2026-11-03，GPT。用户关心未来RL采样效率，要求仿真能提速就提速。只运行固定命令的环境基准，没有策略训练、参数更新或正式RL rollout。

## 1. 将三种耗时分开

固定“进度预算 · 接近阶段东推”原输入，91个20Hz控制tick=4.55s仿真。

1. **测台**：含完整逐tick审计、mask诊断、回放整理、JSON、HTTP。前轮其历史列表重复解析2.37GB已解决；不把那些时间算进物理或option。
2. **无记录核心执行**：同一命令、同一预算逻辑，`record=False`，无测台Instrumentation。
3. **RL式环境采样**：在2的基础上，按v16调用模式加入.8s actor观测、.2s dense观测，以及实际的向量编码和动作mask。固定动作仍来自同一脚本，**不运行神经网络、normalizer、PPO、worker通信**。

v16当前默认factory仍是旧控制器接口，D013没有因此自动接入正式训练。本轮在D013环境模拟相同观测消费节奏，不能将结果叫作“完整v16/PPO训练吞吐”。

产物：`outputs/testbed/profiles/20261103_core_speed/`，含 before/after源码、输入、每模式5次计时、独立capture、cProfile及按外层调用归并的 `breakdown.json`。复现脚本 `bench.py`。

## 2. 当前主要计算部分

优化后的RL式采样 cProfile，按互不重叠的外层调用归并，约.201s（profiler有额外开销）；比例不用于精确预测其他任务：

| 部分 | profile占比 | 内容 |
|---|---:|---|
| 感知/观测构建 | **50.3%** | 雷达、patch、旧anchor、状态/contact读入、观测对象；含缓存命中路径 |
| 协调与运动估计 | **18.1%** | 共享估计、direct、memory、q关联/分配、coordinator |
| 物理与力门控 | **13.1%** | 物理推进、碰撞/安全检查、门控 |
| 动作mask | **4.5%** | actor/dense采样侧合法性计算 |
| 观测向量编码 | **3.3%** | 基础、控制上下文和执行反馈编码 |
| 两侧slot控制 | **3.7%** | PUSH/BRAKE/切换调用 |
| 其他 | **7.0%** | 构造、prepare、reward/终止、预算和循环等 |

感知内部仍以批量雷达为最大项：本次cProfile约.056s，patch约.019s；两者已包含在50.3%里，不能再次加到总量。完整profile包含182次真实扫描，240次观测消费。

旧测台profile的“观测.13s、雷达.07s”等数值来自另一种带记录执行方式；本表专门去掉测台记录，不混用两份profile的分母。

## 3. 已落地的三项等价优化

### 3.1 同一时刻的传感几何只算一次

`RandomStartEnv.scan_for` 新增每侧单条缓存。key包含请求tick/time、实际投射所需物体位姿、双方完整机器人状态/运动学heading、arena、障碍、多边形和backend。

`observe` 仍每次重建当前状态、contact等反馈；只复用同一个scan及对应own位置的不可变patch几何。`restore_branch_state` 清缓存。

这避免“actor观测→dense观测→同tick控制观测”重复投射/拟合。任何下一tick、时间变化、heading/物体/机器人几何改变都重新计算，未沿用上一物理时刻传感；不改变20Hz、72射线或5m量程。

### 3.2 nearest查询只计算最终命中点坐标

`LidarScan.nearest` 先找最近距离及索引，再对最终点算sin/cos。保留原来严格小于比较和相等时取首个命中的顺序，不再为所有同类命中先计算一次世界坐标。

### 3.3 跳过必然被丢弃的无效单点run

`scan_patches` 中无效射线只会形成长度1的run，默认min_points=3时进入fitter也立即丢弃。现在不为这些run构造零点列表和拟合调用。min_points≤1的旧路径保留，未修改拟合标准或分割阈值。

## 4. 本机微基准结果

均为同一实例、每组5次的中位数，计时段不含环境初始化；source及完整重复数据保存，不能外推所有几何或worker数。

| 模式 | 修改前 | 修改后 | 耗时减少 |
|---|---:|---:|---:|
| 无记录核心执行 | .11935s | **.11343s** | 约5.0% |
| RL式观测采样 | .15040s | **.12275s** | 约18.4% |

无记录模式仍182次扫描/拟合；RL式模式从 **240→182次**，省去58次重复几何计算，但仍240次观测消费、29次向量/mask采样、91次物理tick。

当前RL式基准约 **741 control tick/s、37.1倍仿真时间**。按.8s hold折算actor节奏只是理论频率，不是已测策略transition吞吐，因为没有网络推理/PPO/进程通信。

Headless约5%的小收益可能受机器负载影响，未声称统计显著；RL式5次计时从.1498–.1526s变为.1225–.1235s，与减少重复扫描的调用计数一致。

## 5. 等价性与测试

- before/after两个模式逐control tick的object/robots/forces/headings/candidate trace **完全相同**。
- 全部采样的观测向量、mask、终态、reward、termination **完全相同**。
- 上线后完整测台API run `20261103_022125_5e8280` 与前轮带记录原轨迹的全部frames **完全相同**，91tick不变。
- 新增 `tests/test_perception_reuse.py` 11项，覆盖重复时刻复用、tracker不重复更新、tick/time/heading/物体/自身/peer变动失效、contact保持新鲜、分支恢复清缓存、nearest相等顺序、无效单点分组。
- 与D013 q/selector/nullspace、历史索引测试合计 **32项通过**；另7项scalar/batch射线、边界、反事实和branch配置检查通过。合计 **39项聚焦测试**。不运行全任务矩阵。
- 基准每版：两模式各5次计时+1次capture，再1次RL式profile=13次；两版26次完整固定脚本。上线另1次；传感/几何单测不当作任务评估。未启动正式训练。

## 6. 测台状态与版本

8765已重启，PID **1352102**，q控制器仍 f30f7f72…；运行时版本 **a3e0ead1…**。

为避免只改env/lidar后header仍显示旧q hash而误认为已加载新仿真，server现在对12个关键控制/仿真Python文件保存启动时hash，状态返回runtime摘要并检查是否需要重启。run记录runtime摘要。页面显示 `q / sim` 两个版本。

证据/当前状态：`deployment_check.json`；主进程指针 `outputs/testbed/server.json`。原测台历史索引优化保留。

## 7. 后续提速优先级

按当前数据优先检查雷达的几何/数组准备复用、两侧批量投射和patch构建；其次是共享估计器的小矩阵分配。每项都应以未改观测/控制/轨迹为门槛，再测真实收益。PUSH状态机只有约4%，不是当前主要CPU消耗。

正式接RL时另测：真实策略推理、训练数据搬运、PPO更新、多worker通信与线程配置。当前环境微基准不等于完整训练速度，也不授权启动长训。
