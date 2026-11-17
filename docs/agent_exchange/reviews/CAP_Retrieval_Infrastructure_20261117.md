# CAP检索基线基础设施与小规模验收

2026-11-17，完成。对应[实现与设置](../../research/CAP_Retrieval_Baselines_v1_20261117.md)及[研究计划v5](../../research/CAP_Integration_Mechanisms_Research_Plan_v5_20261117.md)。

新增独立模块，实现R0初态检索锁定一个闭环CAP、Rt逐option边界检索切换CAP；复用原runner、合法观测和控制器。运行中的GDA扩数据任务源码未改，parent/collection实时源码校验通过。未启动正式检索基准或新的LLM/训练。

## 验收证据

- `tests/test_program_retrieval.py`：6项通过，覆盖未知/失败标签、回退、成功前缀投票、缺失值、索引篡改、测试数据拒绝、采样密度、R0锁定与Rt实际历史记忆更新。
- 旧封印开发采集按物体数交错选8个初态（每组2例），80次已有CAP执行，构建8行初态结果索引和442行成功边界索引；k=1，最多16边界/成功轨迹。构建0新物理ticks。
- 使用另外两个开发初态，R0/Rt各执行一遍，共4执行/7070ticks；每例最多1800ticks/300决策/180秒，全部正常完成，没有接口错误。
- 两个R0结果与其选中CAP此前从相同初态独立执行的**全部公开历史和物理回放精确一致**。
- 两个Rt结果逐边界在独立CAP运行时重算，所选动作及公开历史哈希一致；未选CAP同样按实际执行动作更新记忆。

| 开发场景 | R0选中CAP | R0结果 | Rt结果 | Rt切换次数 |
|---|---|---|---|---:|
| newdev_2_002 | CAP-RR-002 | 90秒预算耗尽，失败 | 90秒预算耗尽，失败 | 39 |
| newdev_3_002 | CAP-RR-001 | 90秒预算耗尽，失败 | 83.5秒成功 | 13 |

这仅证明管线可运行与接管信息流正确。两个开发案例、小索引、固定默认距离不支持任何泛化优劣结论；未据此调参或挑换案例。39次切换是可供后续诊断的现象，不证明切换导致失败。

## 已交付与待做

已交付：开发索引构建与完整来源追踪；snapshot/history两种输入配置；确定性k近邻/未知标签/回退；冻结索引校验；共享执行入口；路由、近邻和切换日志；耗时字段；原格式物理回放；可替换的Router接口。

snapshot是当前紧凑状态估计，保留G编码的实体缓存、上一动作与初始相对坐标框架，不应称严格无记忆图像。默认精确检索，尚未优化近邻或只运行选中CAP。本版R0首次索引查询成本计入prepare_seconds，后续retrieval_seconds主要是取缓存，不能直接与Rt检索耗时对比。

待做：同经验池/同预算的正式矩阵、距离与k的开发验证、学习式路由LR的训练器与接管标签、F的建议并集约束对照、前端路由专用图表。LR目前只有统一接口，不称已训练或已比较。

完整验收：[ACCEPTANCE.json](../../../outputs/research/program_retrieval_20261117_01/ACCEPTANCE.json)。
索引记录：[BUILD.json](../../../outputs/research/program_retrieval_20261117_01/index/BUILD.json)。
可重跑脚本：[smoke.py](../../../outputs/research/program_retrieval_20261117_01/smoke.py)。
回放与路由记录位于`outputs/research/program_retrieval_20261117_01/runs/{initial,boundary}/<case>/result.json.gz`。
