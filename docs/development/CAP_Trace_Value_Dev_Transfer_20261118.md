# CAP trace价值学习：完整开发运行移交

仅盘点、校验与复制已有数据，不运行CAP、训练或仿真。目标7 CAP：009、003、001、010、006、007、002；760初态×7=5320运行。未完成/异常不冒充完整失败，发现问题时先交inventory.json与gaps.json，不生成完整包。

包内 scenes.json为760冻结场景；development_summaries.json为10CAP×760完整开发摘要；index.json为5320条运行的相对路径、SHA256、完成状态、终止结果及字段盘点；library.json与policies为7程序及语义分组；sources与runtime_sources.json为运行/协议源码。没有模型、训练数组、机器编译缓存。独立manifest核验负载，不依赖旧priority包。

## 历史可用性

public_rows保存初态及每次option执行后的公开观测、实际动作和历史哈希；观测robots包含扫描、接触、执行反馈、时间。events和decisions保留请求/实际动作、开始/结束tick、执行periods、切换和终止信息。summary保存原成败、成本、终止原因；replay保留物理回放。每条记录逐项检查数量、哈希链、动作对应、事件时间连续及末尾tick。

注意：动作中的ticks是选择的执行上限；实际执行时长必须读取event.periods或end_tick−start_tick，末段可能提前结束。不能把命令时长直接当实际时长。原task_cost_seconds也可能含失败成本，不等同于物理推进秒数。

历史足以向原CAP重播决策边界输入，重建其由这些输入确定的记忆；不等同于保存了所有内部变量。程序局部表达式/分支trace需要离线重算，并验收插桩不改变原行为。未保存每个物理tick的完整合法传感输入，不能凭回放真值伪造更密的策略观测。后缀成功标签、剩余执行时间可由终止结果和tick重建；折扣收益需要先确定奖励/折扣和截断bootstrap约定，不能假装文件已保存新价值目标。decision_budget与环境终止区分；宿主错误/中断不算完整失败。

场景及replay物理帧只用于环境初始化/离线审计，不能作为部署策略的特权输入。原gzip逐字节复制，不改写其中的绝对路径；这些路径仅作来源证据，运行查找只读index.path与library.path。

used_scene_identities.json提供已有随机房间开发/评估面板及DA80的ID、完整fixture哈希与去掉顶层scene_id的content_sha256内容哈希，不带评估成败标签；不声称覆盖已删除的早期其他任务族。未来测试集同时按ID与content_sha256去重（同时保留fixture_sha256核验原文件），不能只比较随机seed。

## 验收与解包

将压缩包及.sha256放在同目录，先运行`sha256sum -c cap_trace_value_dev_v1.tar.gz.sha256`，再解到新目录，不能覆盖其他实验。代码工具为`tools/cap_trace_value_transfer.py`；包内亦保存相同工具源码。

```bash
.venv/bin/python -m tools.cap_trace_value_transfer verify --bundle /path/to/cap_trace_value_dev_v1 --workers 8
```

需要现有仓库Python依赖，不需要API密钥。verify仅解码、重算历史哈希与核对清单，不执行CAP、仿真或训练。结果数量及异常清单以audit_summary.json、inventory.json、gaps.json为准。

若目标仓库尚未同步新验收工具，可将包内 `sources/tools/cap_trace_value_transfer.py`、`priority_transfer.py`、`program_trace_transfer.py` 复制到仓库的 `tools/` 后运行上述命令。`sources/` 是依赖源码快照；不要将包根目录当作可直接训练的工作区。`environment.json` 记录源机依赖版本；实际重新计算编译语义时需核对 Cython 工具链，验收本身无需编译。

本次验收：5320条完整，1759成功、3561失败，异常/缺失0；369726次执行决策、375046个含初态的观测边界。逐CAP决策数和终止类型见audit_summary.json；缺失字段清单为空指本协议要求的保存字段齐全，不代表局部程序变量已经保存。64条decision_budget是完整预算截断，价值学习时应单独处理其终止掩码。

与旧priority_teacher_v1负载SHA256交叉核对：共有810条原gzip重合，新包另含4510条；本包直接提供完整5320条，不需要合并旧包。
