# CAP 实验研究日志

从已有开发面板进入[研究日志](http://localhost:8775/research-journal)。侧栏按研究方向展开，右侧先介绍方向，再按 [entries.json](entries.json) 的原始顺序展示该方向的尝试。记录每五秒刷新。本机服务与原面板共用 8775 端口和 tmux 会话 cap_regional5_ui。

新记录用 `direction` 指定所属方向。当前默认入口是[CAP 的静态验证与仿真校准](http://localhost:8775/research-journal#cap-verification)，标识为 `cap-verification`。该方向从[可执行规则的最小验证](http://localhost:8775/research-journal#guarded-options)开始。以整局救回和丢失为目标的历史母策略及作者检查实验仍归[从失败案例逐步改进 CAP](http://localhost:8775/research-journal#cap-recovery)，标识为 `cap-recovery`。更早的[CAP 递进开发与混合](http://localhost:8775/research-journal#cap-mixture)使用 `cap-mixture`，也接收未填分组的历史记录。新增方向时在[日志页面](../../../tools/cap_research_journal.html)的 `directions` 中登记名称、目标和最小闭环，首项为默认打开的方向。单次尝试的原链接继续使用该条记录的 id。

每次新尝试追加一条记录，填写稳定的 id、日期、标题、动机 motivation、目标 goal 和 key_findings 下的结果目录 folder。可用 assets 登记有标注的报告、图、回放或逐例数据。结束时在同一条补上 result 与 analysis，旧记录保留。也可先写入 status 为“待启动”的记录。

运行中的记录可以设置 run 为 outputs/semantic_rl 下的运行目录，并在 progress 数组中按顺序登记阶段。每个阶段填写 label、total 和一种数据来源。glob 统计逐例落盘文件，rows 统计一个 JSON 数组或对象的条数，counter 加 key 读取 JSON 的整型计数，file 检查运行目录中的结束标记，finding 检查报告目录中的交付件。全阶段达到目标后才转为已完成；运行目录的 ERROR.json 优先显示异常。进度是已写出的记录数，批量写出的阶段会整批更新。补充记录和更新文字不需要重启服务。

### 从失败案例逐步改进 CAP 的作者入口

后续开放式开发轮使用[作者提示词](../../../experiments/planar_tasks/authoring_v2/recovery_round.md)。宿主将当轮已授权的开发材料、可重放的前缀和实际可用的接口说明传给下列入口，保存生成的完整提示词交给作者。时间线可列失败发生前后的各个可干预时刻，按当时可见的状态匹配相似的自行恢复对照。输入材料须在宿主侧排除验收房间。入口负责组装提示词，试跑和冻结由相应测台与 authoring_v2 宿主承担。此前只接受模块片段的实验保留原协议。

```bash
.venv/bin/python -m tools.cap_recovery_round_prompt \
  --parent policy.py --timeline dev_timeline.json \
  --controls dev_controls.json --fork-tools round_interface.md > author_prompt.md
```
