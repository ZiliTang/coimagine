# GD补齐与统一覆盖比较（2026-11-17）

用户授权：补齐GD，在现有规模对各CAP、G/GA/GD/GDA、R0/Rt做一致横向比较。

唯一新增训练：GD，固定同270完整成功示范/9292边界，seed17、28000次Adam更新；沿用library_learning的网络、归一化、采样顺序、损失和学习率。从头训练；仅保留冻结特征分组G、D、common，A值和缺失标记均屏蔽。动作候选合法支持不变，D仍需执行原CAP程序计算，不是完全脱离CAP的蒸馏。

唯一新增执行：固定旧100已查看场景，4 worker，每例至多1800 ticks/300决策/1800秒；共100例/180000 ticks。训练墙钟上限7200秒，完整持久任务上限14400秒。不增加LLM/CAP/RL、不修改成功标准、旧模型/源码/控制器或其他正在运行实验。新GDA扩数据的200新测试不纳入此次面板。

冻结输入：random_room_coverage_20261117_02的data_expanded、fit_expanded、evaluation封印；program_retrieval_comparison_20261117_01/evaluation封印。启动时验证，parent记录全部封印和协议，训练完成再次冻结后评估。仅比较expanded G/GA/GDA（同270/28000/seed17），不混入repaired或原124示范版本。复用已完成方法逐场景结果，不为绘图重复仿真。

输出：16×100成功矩阵、16×16共同成功数/Jaccard/有方向补救覆盖/包含率矩阵，交互HTML、PNG/SVG/PDF、CSV/JSON。Jaccard=交集/并集，不把共同失败当成相似；空并集未定义。补救矩阵(i,j)=j成功且i失败的场景数。待完成方法记缺失，不记失败。每个格子可定位案例；视觉排序不改变统计。

局限：旧100是固定已见诊断，单seed；成功集合相近不代表决策行为相同。CAP、快照检索、历史学习器的信息与计算不完全一致，不能把所有差异单独归因架构。CAP事后并集不是可部署路由器。

启动：`.venv/bin/python -u -B -m experiments.planar_tasks.authoring_v2.gd_coverage_comparison`。
产物：`outputs/semantic_rl/gd_coverage_comparison_20261117_01/`；宿主日志/进度/退出状态：`outputs/semantic_rl_authors/gd_coverage_comparison_20261117_01/`。
报告：`docs/research/reports/CAP_Coverage_Comparison_20261117/index.html`，完整评估成功后自动补齐；异常保留原始记录并停止，不静默补跑。
