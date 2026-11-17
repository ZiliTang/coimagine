# 轨迹修复与语义学习管线

本模块将“成功种子 → 扰动初态 → 原序列复放 → 失配定位 → 局部修复 →
同前缀比较 → 语义评分拟合 → 评分驱动执行”串成可追溯的开发流程。
独立实验入口是 `experiments.planar_tasks.authoring_v2.relational`；旧作者
协议与冻结实验保持原样。完整作者契约见 [pipeline_guide.md](pipeline_guide.md)。

## 查看过程

- [管线测台：修复与拟合](http://127.0.0.1:8767/?campaign=engineering_acceptance_20261113_02&round=r001)
- [评分选择动作后的执行](http://127.0.0.1:8767/?campaign=engineering_acceptance_20261113_02&round=r000)
- [场景编辑器](http://127.0.0.1:8765/?preset=relational_clear_transport_demo_v1)

页面显示每轮初态变化、继承的种子、作者简短决策依据与提交、全部尝试、
同步双轨回放、失配/修复跳转、合法观测分页、比较样本、语义源码、S/E/M
损失曲线、逐对评分、残差和所选候选。窄屏改为上下排列，回放画布一起缩放。
测台只读；刷新不会触发仿真、训练或模型请求。

```bash
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.relational_search.pipeline_server --port 8767
```

默认数据目录 `outputs/relational_pipeline/`，也可用 `--root` 指定。
这是人工检查界面，包含完整几何，**不能作为 LLM/策略的观测接口**。

## 新建与调用

以下创建只建立宿主配置/源代码快照；不会调用模型或执行动作：

```bash
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.authoring_v2.relational outputs/relational_pipeline/NEW_RUN create
```

宿主侧也可以在创建时传入 `fixture`、`sequence`、预算和显式模型元数据。
fixture 是 `pipeline_execution.seed_fixture()` 的五字段字典；从 Workbench
导入时取 object、robots、movable_objects、overrides，goal 取预设的
`scene_meta.goal`。初态始终留在宿主；作者收到实际局部扫描。

```python
from experiments.planar_tasks.authoring_v2.relational import RelationalSession
s = RelationalSession('outputs/relational_pipeline/NEW_RUN')
knowledge = s.tool('knowledge', {})
start = s.tool('new_round', {'delta': {}})
summary = s.tool('replay', {'rid': start['round_id']})
# 先看 summary，再按需 trace/context；具体请求模式见 pipeline_guide.md。
```

CLI 单次请求用 JSON 文件，避免 shell 转义：

```json
{"name":"context","arguments":{"rid":"r000"}}
```

```bash
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.authoring_v2.relational outputs/relational_pipeline/NEW_RUN tool --request /tmp/request.json
```

`RelationalSession` 是供外部作者使用的工具适配器，尚未接入自动 provider
驱动。没有后台付费循环。LLM 的模型身份、思考挡位、简短决策解释、提交与
外部用量应通过 `brief` 显式写入；宿主不会伪造模型记录或把预留额度当成账单。
原生 agent 若能随意读整个工作目录，仍不能声称其信息边界已由该工具强制保证。

## 证据与拟合

- `repair` 必须引用已有执行、作者记录和一个已观察到的边界。前缀命令和
  合法历史哈希一致才认定为匹配修复；记录介入点和检测点，不要求两点相同。
- 原序列预声明预期只按原索引检查。修复改变序列长度后，不把旧预期强套到
  同数字索引的新动作上。
- `compare` 从同一初态分别重放相同前缀，执行一个候选，再执行完全相同的
  后缀。失败按原任务代价计；非法动作/工具错误不产生物理负例。
- 所有比较保留，包括平局；当前成对 logistic 只拟合明确偏好，不约束平局的
  分数必须相同。因此未被监督的候选分数差不能解释成动作最优性结论。
- Python 语义通过 bubblewrap 隔离；每进程仅一条合法历史和一个候选，不给
  标签、未来观测、其他样本、仓库或仿真对象。缺少隔离能力时明确失败。
- S/E/M 使用相同数据、训练集归一化和固定浅层线性排序。E 是公开的几何/
  历史固定摘要，尚非神经历史编码器。M 冻结 S 后拟合带正则的残差。
- 修订前后可传相同 `comparison_refs`，用数据哈希验证同数据再拟合。
  `apply_ranker` 展示评分到动作执行的连接，尚非完整搜索策略评测。

## 恢复与冻结

每次执行先原子预留 ticks/reset/attempt，可信完成后结算实际 ticks；前缀
重放全部计费。异常中断保留预留，不自动退款或重试。重新打开同目录即可
查看已有结果并开展剩余预算内工作，`audit` 将遗留 running 标为 interrupted。
仍有未结算预留时不允许冻结；需人工审阅，不能将缺失记录当作零成本。

源码/知识变更后拒绝继续同一开发运行，应创建新版本。完成后：

```bash
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.authoring_v2.relational outputs/relational_pipeline/NEW_RUN freeze
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.authoring_v2.relational outputs/relational_pipeline/NEW_RUN verify
```

冻结包含源代码副本、配置、预算、原始/失败尝试、合法历史、独立人工回放、
作者记录、比较集和拟合检查点。历史验证使用已封存源码副本，不要求当前
工作树仍停留在旧版本。严禁把已冻结运行继续补写成另一次模型实验。

## 验收与范围

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_relational_pipeline.py
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.relational_search.pipeline_acceptance outputs/relational_pipeline/NEW_ACCEPTANCE
```

第二条会真正执行有界物理实验、拟合并冻结结果；作者与修复是预先人工编写，
没有 LLM 调用。现有 `_02` 验收：2 个初态、11 次执行、7033 ticks、13 次
reset、4 个偏好对与 2 个平局、两版语义同数据拟合、一次评分驱动执行成功。
只涉及一个开发母组，没有独立验证/test；不能作为研究方法有效性的结论。

后续需要真实受约束 LLM 作者运行、历史混淆/反例任务、跨母组留出、完整
规划器与学习式历史编码器对照。旧 authoring_v2 回归目前会因之前 Workbench
修改导致旧 baseline 封印拒绝；本 profile 单独版本化，不篡改旧封印解决此问题。
