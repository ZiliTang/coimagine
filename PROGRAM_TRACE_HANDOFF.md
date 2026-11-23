# 第二台PC接续

代码已推送 origin/main：`de27a7c9bb7496a3286989beb208867764ee2eef`。

需要手动复制的文件只有：

- `cap_program_trace_v1.tar.gz`（约1.41GB）
- `cap_program_trace_v1.tar.gz.sha256`

依赖旧 `priority_teacher_v1` 解包目录，另一台PC已有，不必重传。新包是增量证据，不恢复源机全部 outputs。

先保存目标机代码/文档修改，正常更新Git；保留已跑完的priority结果。若本地提交导致不能快进，正常合并，不强制reset。

```bash
cd /home/tzl/repo/coimagine
git status --short
git pull --ff-only
```

在压缩包与校验文件所在目录执行 `sha256sum -c cap_program_trace_v1.tar.gz.sha256`。确认目标目录尚不存在，再在仓库中执行：

```bash
mkdir -p outputs/transfers
tar -xzf /path/to/cap_program_trace_v1.tar.gz -C outputs/transfers
.venv/bin/python -m tools.priority_transfer bootstrap --bundle outputs/transfers/priority_teacher_v1
.venv/bin/python -m tools.program_trace_transfer verify --bundle outputs/transfers/cap_program_trace_v1 --dependency outputs/transfers/priority_teacher_v1
```

研究计划：`docs/research/CAP_Program_Trace_Selection_Plan_v6_20261117.md`。
详细说明：`docs/development/Program_Trace_Transfer_20261117.md`。
验收记录：`docs/agent_exchange/reviews/Program_Trace_Transfer_20261117.md`。
包内可直接阅读 `README.md`、`report/plan.md`、`report/results.md`。

下一步是实现稳定程序插桩和CAP完整建议池；新模型尚未实现，尚未启动新正式训练。先通过小规模接口/拟合验收，再冻结新两臂实验和预算。已有200和80评估面板都已查看，不称盲测。

包SHA256：`1e01d6bcfc75da482fa03c1d3b76306d800349e47056649918a6d0f95bf97e8c`。
