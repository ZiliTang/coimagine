# CoImagine — CAP行为与语义的监督整合

当前研究：积累相互重叠的局部CAP程序及其决策语义，通过监督整合形成可复用的操作策略，并以几何/语义消融和检索基线分析覆盖、泛化与样本效率。

## 当前入口

- [研究计划 v5](docs/research/CAP_Integration_Mechanisms_Research_Plan_v5_20261117.md)
- [当前实验状态](docs/agent_exchange/CURRENT.md)
- [16方法覆盖报告](docs/research/reports/CAP_Coverage_Comparison_20261117/index.html)
- [导师汇报（含历史汇总）](docs/research/reports/CoImagine_Progress_20261117/index.html)
- [检索基线协议](docs/research/CAP_Retrieval_Baselines_v1_20261117.md)
- [关系迁移任务](docs/research/Relational_Transfer_Task_Suite_v1_20261117.md)

## 实现与运行

- [平面任务与测台](experiments/planar_tasks/README.md)
- [统一CAP作者接口v2](experiments/planar_tasks/authoring_v2/README.md)
- [公开Option知识](experiments/planar_tasks/authoring/option_guide.md)
- [语义评分、监督及RL实现](experiments/planar_tasks/semantic_rl/README.md)
- [工程协作规则](AGENTS.md)

```bash
source .venv/bin/activate
pytest tests/test_gd_attribution.py tests/test_coverage_report.py tests/test_program_retrieval.py
```

当前研究数据位于`outputs/semantic_rl/`，任务日志位于`outputs/semantic_rl_authors/`。启动新实验前读取CURRENT及对应任务书；已有源码、控制器和基础模型未随清理修改。

## 数据保留

2026-11-17按用户授权永久清理旧CAP原始输出、旧RL/早期消融数据及被替代文档。保留当前14个实验目录、相关报告与运行依赖；历史汇报中的汇总数字仍保留，但已删除阶段的原始数据不再可复现。详见[清理记录](docs/agent_exchange/reviews/Workspace_Cleanup_20261117.md)。

`docs/agent_exchange/state/`是测台/控制器运行输入，不是可删除历史；`_old/`及旧代码保留为实现历史，不代表当前研究目标。
