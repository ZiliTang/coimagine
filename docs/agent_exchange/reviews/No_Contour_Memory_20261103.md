# 无轮廓 memory：简化实现与测台更新

2026-11-03，GPT。用户明确要求移除reposition阶段轮廓memory，初始q只用当前观测；option无明显行为异常即可更新测台，供用户查看。本轮按此执行，不以task complete为验收。

## 改动范围

- 删除 `SupportMemory` 实现、env中的实例与每tick更新/预测/calibrate/faces转换。
- 删除 `_merge_memory_faces` 以及selector/选面函数的 `extra_faces` 历史几何注入参数。
- 单机初选只使用该机器人当前 `visible_faces(obs)`；双机联合分配仍可读取双方当前观测，但各自候选来自各自当前可见支持。
- 活跃分配路径移除旧union allocator兜底调用，避免绕过上述约束。其纯函数仍保留给旧诊断/测试，不是当前PUSH调用路径。
- 删除session的memory快照维护，新frame不再包含`memory`字段。run/status显式记录 `support_memory_enabled=false`、`q_acquisition_mode=current_visible_support`。
- EKF/direct的运动历史、锁定q与边段关联状态、同一时刻传感缓存保留。这些是控制连续性状态，不是此次移除的历史轮廓地图。
- 旧run原件保留。前端仅为旧run兼容显示已有memory数据，并标记“历史记录：含轮廓memory”；新run仅显示当前拟合面、q/q′等，不产生新地图。

没有修改物理、观测频率/量程/射线数、reward、成功判据、q′增益或脚本预算。没有训练。

## 单例问题的回归测试

在真实allocator的无物理fixture里构造：R0当前支持y∈[1,1.4]，peer看到[1.2,2]。

- 旧路径会用共享memory扩展支持，初选q_y=1.5，随后自身tracker无法接受。
- 新路径初选q_y=1.2，立即匹配自身当前支持，不进入等待。
- 另覆盖“先观察宽面、再切到窄支持PUSH”，历史宽面不能扩展新q。
- 双机测试检查每个q都落在其所属机器人自己的当前支持内。

相关q/selector/感知缓存/运行索引测试 **35项通过**；JS语法检查通过。新要求改变了memory接口，原来测试已删除的merge helper的断言改为直接验证不相连的当前支持不会被合并。

## 六个典型测台实例的前后对照

先验证上述明确失败条件，再检查少量典型组合；未运行全体40-case矩阵或种子扩展。输入初态/sequence/overrides/goal冻结在产物目录。

| 实例 | 控制tick 前→后 | R0 push阶段tick 前→后 | 状态对照 |
|---|---:|---:|---|
| 接近阶段东推 | 91→91 | 46→46 | object/robots全部状态逐tick完全一致 |
| 绕行后南推1.9m | 538→538 | 49→49 | 完全一致 |
| N+E双侧协推 | 99→99 | 47→47 | 完全一致 |
| 双机北推 | 87→87 | 42→42 | 状态分量最大差约6.9e-15，数值精度级 |
| 单机东推 | 87→87 | 56→56 | 完全一致 |
| 八边形东推 | 800→800 | 769→769 | 完全一致 |

六例均有限、施力不超原限值；没有新增同一candidate/同action连续帧超过5cm的q跳变，推进/等待阶段计数保持。未以task标签筛选或要求全部task成功。该小检查支持“未见明显option行为异常”，不证明任意遮挡/形状均无退化。

本轮不是严格性能实验，未以单次wall time声明移除memory的提速比例。

## 已更新测台

- 主入口：http://127.0.0.1:8765/ ，PID **1360366**。
- controller SHA：`41ec39f4476af1036a55127b09f6a58694f6dfde9460355738ad27c9ef06ada3`。
- runtime SHA：`88ce64315ddfc0d0a331eb07ecf993782b355e98329c78863625f26b8ea92c56`。
- 页面header显示“无轮廓memory”；status返回false且无需重启。
- 六条候选结果已放入Runs，名称前缀“无轮廓memory · …”，原presets和旧run没有覆盖。
- 可直接查看绕行后首次q：[run/tick393](http://127.0.0.1:8765/?run=20261103_024743_0193b4&tick=393)。
- 另做1条正式HTTP路径验证：`20261103_025249_3a1957`，全部frame与离线检查过的无memory候选一致。

## 证据与停止点

`outputs/testbed/experiments/20261103_no_contour_memory/`：冻结输入、before/after源码与diff、6×2条完整轨迹、逐case摘要、比较、attempts账本、published_runs、上线验证与日志。12条离线同实例对照运行（6种preset各前后一次）+1条上线验证；无训练或其他实验扩展。

本轮到此交给用户在测台查看。后续不为task标签自动恢复轮廓memory或追加补偿机制；是否保留简化版、继续调整由实际option行为反馈决定。
