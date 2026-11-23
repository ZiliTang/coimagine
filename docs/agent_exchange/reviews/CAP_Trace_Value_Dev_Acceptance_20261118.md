# CAP trace价值学习开发包：目标机数据验收

日期：2026-11-18，Asia/Shanghai。状态：`complete`，仅完成用户授权的已有数据检查。没有执行CAP、编译语义、重算程序trace、运行仿真或训练；v7实验仍为`planned`。

## 结论

`cap_trace_value_dev_v1.tar.gz`在目标机全量验收通过，5320条开发运行齐全，1759成功、3561未成功，缺失/异常0。此前4510条原运行轨迹的缺口已补齐，新包不依赖旧priority包即可验收。结果来自本机重新检查，不只是引用源机报告。

- 压缩包：4,831,484,107 bytes；SHA256：`afcba4390868aa894e44e89e6017c4de4ba8d6942ad112311507bd7c8b8bb1df`，与随包SHA256文件一致。
- 解包到新目录：`outputs/transfers/cap_trace_value_dev_v1`；6226个归档成员路径、类型和唯一性检查通过；manifest的6149个负载文件哈希全部通过。
- 全量检查760个场景、10CAP×760摘要、7CAP×760运行的唯一键与相互对应关系，逐条解码gzip、重算历史哈希链、核对实际动作、事件连续时间、观测/事件/决策数量和回放末尾tick。
- 369726次执行决策，375046个含初态的观测边界；index与inventory完全一致，gaps为空，重新汇总与源audit_summary一致。
- 包内809份运行源码与本机文件逐字节一致。原本只缺`tools/cap_trace_value_transfer.py`，已按用户交接说明从已核验快照复制；两个已有迁移工具未改动。
- 760个场景fixture、全部7600条成败及ticks、7个CAP程序和语义分组均与旧priority包一致；810条共有gzip的本地旧文件SHA再次通过，新包另外提供4510条原运行。

## 逐CAP计数

每个CAP均为760个相同开发初态；决策数不等于轨迹数。

| CAP | 成功 | 未成功 | 执行决策 | 含初态观测边界 | decision_budget |
|---|---:|---:|---:|---:|---:|
| 009 | 311 | 449 | 47837 | 48597 | 5 |
| 003 | 309 | 451 | 50698 | 51458 | 17 |
| 001 | 280 | 480 | 49336 | 50096 | 15 |
| 010 | 269 | 491 | 57411 | 58171 | 4 |
| 006 | 126 | 634 | 55886 | 56646 | 1 |
| 007 | 203 | 557 | 56897 | 57657 | 5 |
| 002 | 261 | 499 | 51661 | 52421 | 17 |
| 合计 | 1759 | 3561 | 369726 | 375046 | 64 |

## 终止和输入边界

5256条为环境终止：1759条`success`，3497条`task_budget_exhausted`。另64条为`decision_budget`，逐条额外解码核实恰好300决策、301观测边界，末尾tick在1588–1799之间，宿主error为空、环境termination为空。原summary的`status=incomplete`保持原样；inventory的`completion=complete`指记录完整，不能据此声称环境已经终止。这64条原`task_cost_seconds`为空，不能用其替代执行时间。

实际时长读取`event.periods`或`end_tick-start_tick`，不使用动作的命令ticks。新价值目标尚未构造；应按冻结后的任务预算和bootstrap约定处理这些截断。若维持v7提议的1800ticks/300决策有限任务、成功1/其余0，则决策预算耗尽是该有限任务的结束，同时仍保留独立的截断标记，不能混同宿主错误。

场景身份清单1940条来源记录对应1180个不同fixture/content哈希，包含760开发记录、1100条此前使用面板记录、80条DA记录。开发760个content哈希全部从fixture重算通过，旧200评估面板全部能在清单中找到；清单不含成败字段。它不声称覆盖已删除的早期其他任务族。

本次验证了历史记录的完整性与一致性，未验证插桩行为等价，也未执行CAP记忆重建。程序局部变量仍需依据合法观测前缀离线重算；场景和replay真值不可作为部署策略输入。此前`cap_program_trace_v1`严格特征验收的72个数值差异仍未解决，本次数据验收不覆盖该问题。

## 命令与证据

使用已有Python3.11.18环境（源机3.11.4），不新增依赖。全量verify采用8个进程，514.60秒，退出0，完成于2026-11-18 01:36:37 +08:00。

```bash
sha256sum -c cap_trace_value_dev_v1.tar.gz.sha256
outputs/transfers/venv_py311/bin/python -u -m tools.cap_trace_value_transfer verify \
  --bundle outputs/transfers/cap_trace_value_dev_v1 --workers 8
```

- [本机验收总结果](../assets/cap_trace_value_acceptance_20261118/CHECK_RESULT.json)
- [verify完整日志](../assets/cap_trace_value_acceptance_20261118/verify.log)、[退出和耗时](../assets/cap_trace_value_acceptance_20261118/verify.exit.json)
- [补充审计及64条截断明细](../assets/cap_trace_value_acceptance_20261118/supplemental_audit.json)、[审计脚本](../assets/cap_trace_value_acceptance_20261118/supplemental_audit.py)
- [新旧包交叉核对](../assets/cap_trace_value_acceptance_20261118/cross_bundle_check.json)、[源码与工作区记录](../assets/cap_trace_value_acceptance_20261118/preflight.json)
- 原始本机审计目录：`outputs/transfers/cap_trace_value_target_acceptance_20261118/`。

下一步仍按[v7计划](../tasks/CAP_Trace_Value_RL_20261118.md)处理数值迁移、trace插桩及行为同值验收；本次不自动启动P1/P2或后续实验。
