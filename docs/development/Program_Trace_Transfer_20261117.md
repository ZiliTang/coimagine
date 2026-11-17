# 程序执行特征方案：第二台PC增量移交

方案：[研究计划v6](../research/CAP_Program_Trace_Selection_Plan_v6_20261117.md)。本次只移交代码、计划及已完成证据；自动程序插桩、CAP建议池评分器和下一轮正式训练尚未实现/启动。

## 1. 需要带什么

- GitHub上的本次提交：D/A实验代码、测试与报告，新方案和`tools/program_trace_transfer.py`。
- 新包：`cap_program_trace_v1.tar.gz`及同目录`.sha256`文件。
- 依赖：另一台PC此前已经使用的`priority_teacher_v1`完整解包目录。若只保留旧压缩包，重新解包即可；不需要重复传输该包。

新包是派生证据子集，不是原campaign目录的完整镜像：包含本轮12模型、历史诊断donor checkpoint、48 donor/112分叉/1536模型评估回放、10条修复后缀的可移植索引、24接管点、结果、源码快照和验收特征。省略约10GB可重新生成的训练特征数组。原270条示范从旧priority包的904条去重轨迹中按哈希引用，逐项核对与本轮原数据一致。

旧包还提供10 CAP、760开发场景的覆盖矩阵、fastest/priority各570条选择与编译工具链。远端已完成的priority模型和评估保留原处，本包不覆盖它们。本轮新80初态的结果已经查看，随包的评估回放只作诊断，不再用作盲测。

## 2. 更新代码并安装数据

目标仓库可以是`/home/tzl/repo/coimagine`或其他目录。先保存目标机未提交修改，再正常合并远端提交，不使用强制覆盖：

```bash
cd /home/tzl/repo/coimagine
git status --short
git pull --ff-only
```

如果另一台PC存在本地提交导致不能快进，先正常合并并解决冲突；保留其priority执行报告，不reset或清除输出。建议沿用已跑通priority实验的环境，不为本次移交升级依赖。

将新包与校验文件放在同一目录，在该目录执行：

```bash
sha256sum -c cap_program_trace_v1.tar.gz.sha256
```

回到仓库，确认目标`outputs/transfers/cap_program_trace_v1`不存在，再解包（把`/path/to`替换成实际位置）：

```bash
mkdir -p outputs/transfers
tar -xzf /path/to/cap_program_trace_v1.tar.gz -C outputs/transfers
.venv/bin/python -m tools.priority_transfer bootstrap --bundle outputs/transfers/priority_teacher_v1
.venv/bin/python -m tools.program_trace_transfer verify \
  --bundle outputs/transfers/cap_program_trace_v1 \
  --dependency outputs/transfers/priority_teacher_v1
```

若旧包在别处，`--dependency`填写其实际目录。新包不能取代旧包；校验会检查依赖manifest、负载、当前运行源码、真实边界特征及12个模型的输入协议。不启动物理仿真、训练、PPO或LLM。

```bash
.venv/bin/python -m pytest -q tests/test_program_trace_transfer.py tests/test_joint_evidence.py tests/test_da_joint_study.py tests/test_semantic_rl_handoff.py tests/test_library_features.py
```

完整验收要求源码与包内哈希一致。后续开发改变了旧运行模块时，可用`verify --data-only`仅检查数据，但它不等价于运行兼容性通过；新方法应版本化，不能绕过哈希后继续声称复现旧模型。

## 3. 从哪里读取数据

唯一可执行的跨机器路径入口是新包`index.json`。每条引用格式如下：

```json
{"root":"priority", "path":"trajectories/<sha>.json.gz", "sha256":"..."}
```

`root`只允许`priority`（旧包）或`bundle`（新包），均由命令行指定目录解析。可直接用：

```python
from tools.program_trace_transfer import resolve
from tools.priority_transfer import read

bundle = "outputs/transfers/cap_program_trace_v1"
dependency = "outputs/transfers/priority_teacher_v1"
index = read(f"{bundle}/index.json")
row = index["repair_suffixes"][0]
trajectory_path = resolve(row["trajectory"], bundle, dependency)
start_boundary = row["start_boundary"]
```

- `original270`：本轮原示范，引用旧包，不重复复制。
- `repair_suffixes`：10条补充成功轨迹及监督开始边界。读取整个真实前缀恢复记忆，只从`start_boundary`开始监督，不能删除前缀再假装同一接管状态。
- `handoff_states`：24个真实历史引用、边界及16训练/8诊断留出划分。
- `models`：12模型引用及表示种类，可用旧Scorer读取；完整源码/模型兼容性由verify检查。
- `diagnostic_donor`：历史42000更新的原270 GDA模型；与12个本轮28000更新模型区分，不是同预算对照。
- `replays`：1696次正式执行的结果、回放引用和用途标签；评估成败不能偷偷进入训练。
- `development_cases`、`viewed80_cases`：本轮场景配置；环境初始化使用，不能直接作为部署策略的特权输入。

历史JSON中的`/home/zilitang/...`路径原样保留作为来源证据，不应直接执行。不要批量替换旧JSON或创建旧用户名软链接。`da_joint_run`等旧实验入口仍依赖源机的完整原目录，不能在这个精简包上直接重新运行；下一轮应以本包相对索引和旧priority库构建新driver。

`report/results.md`是适配新包相对图片链接的报告副本，可直接打开；原报告在Git中。`evidence/original_full_inventory.json`是源机完整目录的审计清单，其中的数组没有全部携带，不能把它当成本包manifest。

两项旧产物专用测试在精简目录会明确skip（源机完整库路径和旧donor路径未恢复）；源机已通过。移交验收另从相对索引检查真实特征、全部12模型及历史donor，不通过伪造旧路径消除skip。

## 4. 接手后先做什么

1. 完成上述verify和小测试，记录本机环境；不用再跑完整D/A实验。
2. 阅读v6计划，先实现稳定程序特征schema与无行为改变的插桩。第一版记录标量、比较、分支、返回与有界循环摘要，掩码区分未执行和零。
3. 构建合法完整CAP建议并集、动作去重与多来源标记；全部CAP根据真实执行历史更新。
4. 小规模验收示范动作是否在建议池、拟合与同场景闭环复现，之后才冻结G+A和G+T+A两臂的大计算协议。

本次没有提供自动启动新训练的命令，因为新的插桩和建议池模型尚未实现。预算、场景划分与新盲测应在实现验收后另行冻结。

## 5. 导出方命令

```bash
.venv/bin/python -m tools.program_trace_transfer export \
  --bundle outputs/transfers/cap_program_trace_v1 \
  --dependency outputs/transfers/priority_teacher_v1
.venv/bin/python -m tools.program_trace_transfer verify \
  --bundle outputs/transfers/cap_program_trace_v1 \
  --dependency outputs/transfers/priority_teacher_v1
.venv/bin/python -m tools.program_trace_transfer pack \
  --bundle outputs/transfers/cap_program_trace_v1 \
  --archive outputs/transfers/cap_program_trace_v1.tar.gz
```

导出与pack要求新路径；不覆盖旧包。压缩包生成后逐成员读回校验，另写整包SHA256。数据包、模型和数组不提交Git。包的`source_git_parent_commit`只记录导出时父提交，运行字节以`runtime_sources.json`为准；最终Git提交号见本次移交回报。
