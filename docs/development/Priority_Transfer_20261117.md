# CAP教师优先级对照：另一台PC的迁移与运行

本机原路径为`/home/zilitang/repo/coimagine`；目标机可以是`/home/tzl/repo/coimagine`，不需要旧用户名、软链接或修改冻结JSON。

## 实验定义

对160＋200＋400＝760个已有开发初态的10 CAP完整成败矩阵，按成功数降序排序，名称打破平局；绝不使用评估成功率排序。570个可解初态每个选一条完整成功轨迹：

- fastest：每例最快成功CAP，与当前扩570示范的选择逐项校验一致。
- priority：每例在成功CAP中选择全局开发排名最前者。这次对全部570例重选，包括原270例，不只重选新增300例。

两组都从头训练GDA，seed17/29/43，各42000次更新。沿用当前library_learning的按决策边界抽样、网络、归一化和优化器，不额外改变为按轨迹均衡抽样。轨迹长度不同会改变边界数/有效样本权重，需报告各组边界数与拟合率；不能称纯粹的标签交换实验。模型仍在完整合法动作树上评分，不是强制CAP路由。

评估使用当前已查看结果的200个新一轮场景，属于诊断面板，不再称独立盲测。两组×三种子×200＝1200执行。三个种子共用200初态，不是1200个独立场景。后续正式确认应另冻结测试集。

## 迁移包内容与边界

包内只有两种选择所需轨迹的并集、10个CAP程序/语义分组、完整7600条开发成败摘要、场景、协议、哈希与跨机器特征参考。不复制旧训练数组和重复源码快照；目标机从轨迹重新计算特征。不带评估成败标签，不需要API密钥。

原轨迹gzip逐字节复制，旧路径只留在provenance审计字段；运行只读取新的相对路径selections。原campaign清单仅记录来源摘要，本包是有独立manifest的派生子集，不冒充原campaign的完整副本。

## 在目标PC安装

先从GitHub更新对应提交；若目标工作区有修改，先保存，勿强制覆盖。

```bash
cd /home/tzl/repo/coimagine
git pull --ff-only
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install 'numpy==2.4.6' 'pymunk==6.9.0' 'matplotlib==3.11.1' 'PyYAML==6.0.3' 'pytest==9.1.1'
```

这些版本来自源机器实际环境，不保证目标镜像均提供；如无法取得，记录差异并先通过下面的特征验收，不要默默使用不同版本。优先Linux/WSL、CPU PyTorch，需要C编译器供现有雷达内核构建。无需CUDA。

将`priority_teacher_v1.tar.gz`和其SHA256文件放在同一目录，校验后解包：

```bash
sha256sum -c priority_teacher_v1.tar.gz.sha256
mkdir -p outputs/transfers
tar -xzf priority_teacher_v1.tar.gz -C outputs/transfers
.venv/bin/python -m tools.priority_transfer verify --bundle outputs/transfers/priority_teacher_v1
.venv/bin/python -m pytest tests/test_priority_transfer.py tests/test_library_learning.py tests/test_library_features.py tests/test_random_room_tasks.py
```

`verify`检查所有负载哈希、运行源码哈希，以及从一条真实示范重算四级动作面板后的特征/目标/历史哈希逐值一致性；它不启动仿真。若失败先排查，不开始正式训练。仓库目录可以不同，但运行源码必须一致。

## 运行

完整实验（建议持久终端tmux中执行）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -m tools.priority_transfer run \
  --bundle outputs/transfers/priority_teacher_v1 \
  --out outputs/semantic_rl/priority_teacher_20261117_01 \
  --workers 8 > outputs/transfers/priority_study.log 2>&1
```

先至多4进程抽取特征，共享同一轨迹的特征分片；然后6份模型逐个训练，每份7200秒训练上限；最后8进程评估，单例最多1800ticks/300决策/1800秒。这样避免同时6份拟合挤占内存。不会调用LLM、采集新开发场景或执行PPO。

也可以分步使用`extract`、`fit --arm fastest|priority --seed 17|29|43`、`evaluate --arm ... --seed ...`。每个命令均需`--bundle`和相同`--out`。完整run默认重用已完成分片/模型/评估；未完成的拟合会从该seed重新开始，不支持恢复中途优化器。不要在输出目录迁移或修改数据后直接续跑，派生目录含本机路径；此时用新out重新提取。

进度：`extract_progress.json`、`fit/<arm>_s<seed>/progress.json`、`evaluation/<arm>_s<seed>/progress.json`。
结果：模型accuracy.json、每组evaluation/result.json及最终FINISHED.json。错误保留，不把未完成当失败率或伪造完整结果。

## 本机导出命令（另一台PC不需要执行）

```bash
.venv/bin/python -m tools.priority_transfer export --bundle outputs/transfers/priority_teacher_v1
```

新输出目录必须不存在。导出先验证原冻结数据，确认fastest与原570示范完全相同，再复制两个选集的轨迹并集；不启动新训练/仿真。
