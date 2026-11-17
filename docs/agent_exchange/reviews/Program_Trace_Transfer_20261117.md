# 程序执行特征方案与增量移交验收

状态：方案落盘、增量包和验收完成；新插桩与CAP建议池模型尚未实现，没有启动新的正式训练、PPO或LLM调用。

## 交付

- [研究计划v6](../../research/CAP_Program_Trace_Selection_Plan_v6_20261117.md)：完整CAP建议池、稳定程序特征schema、候选评分、真实历史与续接标签、实施顺序。
- [目标PC说明](../../development/Program_Trace_Transfer_20261117.md)：任意仓库根目录、依赖旧包、校验与下一步。
- [迁移工具](../../../tools/program_trace_transfer.py)：export / verify / pack，相对引用，禁止导出到原证据目录，重复引用逐项校验，压缩包逐成员读回验证。
- 本轮D/A实验的驱动、表示、训练等价改进、接管修复、统计工具、测试与结果报告一并提交；均为当前研究证据的复现依赖。没有提交模型、数组、压缩包、缓存或外部论文。

## 最终增量包

`outputs/transfers/cap_program_trace_v1.tar.gz`

- 大小：**1,408,892,416字节**，约1.41GB / 1.31GiB。
- 文件：3769个负载文件＋1个包manifest，共3770个压缩包文件。
- SHA256：`1e01d6bcfc75da482fa03c1d3b76306d800349e47056649918a6d0f95bf97e8c`。
- 校验文件：`outputs/transfers/cap_program_trace_v1.tar.gz.sha256`。
- 必需依赖：此前已交付的完整`priority_teacher_v1`，不重复传输。

新包提供12个本轮模型、1个历史诊断donor、1696次正式执行的回放、10条修复后缀索引、24接管点、源码快照、全部结果与图表。原270示范按哈希复用旧包；1979条去重数据引用均可解析。约10GB派生特征数组未携带，原始源机冻结产物未修改。

包内`report/plan.md`、`report/results.md`和README使用可解析的相对链接；研究计划、结果及图片可以直接在包内阅读。原报告和源机绝对路径作为来源保留；运行只使用新`index.json`，不改写原JSON、不建立旧用户名软链接。

## 验收

1. 源机：既有16项表示/接管测试及新增6项迁移测试通过。12模型和历史donor的源码输入协议一致；真实示范边界的四级特征、目标、历史哈希逐值相同。
2. 不同根路径：复制当前Git文件到`/tmp/.../home/tzl/repo/coimagine`，只提供两份移交包；使用相同宿主依赖，在新outputs冷编译CAP。禁止Python读取源机outputs后仍验证通过。最终导出工具增加防写原证据、重复引用校验及便携文档后，再次验证最终源码和包。
3. 隔离目录：20项通过，2项明确skip，原因分别为原冻结库目录和原donor目录不存在；这两项在源机已通过。新包的相对读取另验证了完整特征、12模型及历史donor，不把skip说成通过。
4. 所有3770个压缩包文件逐成员读回SHA核对通过，并生成整包校验文件。不同根路径是同一宿主上的迁移验收，不是已在第二台PC硬件上执行；目标机仍需verify。
5. 仅包含小型工程检查（包括既有训练器3更新等价测试），没有新正式策略训练或大规模评估。

日志：`outputs/transfers/program_trace_acceptance/`；导出/校验/压缩统计：`outputs/transfers/program_trace_export.log`、`program_trace_verify.log`、`program_trace_pack.json`。最终提交号随Git历史和外部`PROGRAM_TRACE_HANDOFF.md`提供；包内父提交号不是运行版本的唯一依据，源码字节清单才是验收依据。

## 下一位接手者

第二台PC先保存现有修改并更新Git，校验并解包新包，再运行说明中的verify。保留该机已完成的priority结果。之后按v6阶段顺序实现插桩、建议池和小规模拟合验收，再冻结正式计算预算；本机本次没有自动开启下一轮。
