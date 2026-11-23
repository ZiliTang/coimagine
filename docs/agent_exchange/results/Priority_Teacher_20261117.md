# CAP教师优先级对照：目标PC执行回报

状态：complete；六份模型均42000次更新，1200/1200评估完成，各组200例，评估错误记录为0，宿主退出码0。
执行时间：2026-11-17 16:54:39至20:01:38（Asia/Shanghai），总墙钟11219.62秒，约3小时7分钟。未追加实验或预算。

## 结论

本轮已查看的200例诊断面板上，开发成功数优先教师的跨种子平均成功率为42%，最快成功教师为41%，净增1个百分点。三个种子方向不一致，当前证据不支持优先级选择具有稳定优势；也不能据此证明两者等效。

| 训练seed | fastest成功/200 | priority成功/200 | priority差值 | fastest独有成功 | priority独有成功 |
|---|---:|---:|---:|---:|---:|
| 17 | 89（44.5%） | 87（43.5%） | -1个百分点 | 35 | 33 |
| 29 | 82（41.0%） | 82（41.0%） | 0个百分点 | 34 | 34 |
| 43 | 75（37.5%） | 83（41.5%） | +4个百分点 | 32 | 40 |
| 跨seed平均 | 82（41.0%） | 84（42.0%） | +1个百分点 | — | — |

三个种子分别有68、68、72个初态的成败发生变化，即34%、34%、36%。总体成功率接近，但成功覆盖并不相同。priority并非简单保留fastest成功后再增加成功，存在明显的成功丢失和补回。

## 示范与拟合

排名由760个开发初态×10 CAP的完整7600条摘要计算，成功数降序、名称破平，未读取评估成败标签。两组均覆盖同一570个可解开发初态；priority重选全部570例，334例更换教师。两组选集共904条去重轨迹。

| 组 | 决策边界数 | seed17完整动作拟合 | seed29完整动作拟合 | seed43完整动作拟合 |
|---|---:|---:|---:|---:|
| fastest | 20007 | 19823/20007（99.08%） | 19988/20007（99.91%） | 19999/20007（99.96%） |
| priority | 24114 | 24095/24114（99.92%） | 24111/24114（99.99%） | 24087/24114（99.89%） |

priority边界数增加4107（20.53%）。沿用按决策边界抽样和相同网络、归一化、优化器；轨迹长度和有效样本权重改变，因此不能称为纯标签交换。相同42000更新也不等于相同epoch数。部署仍在完整合法动作树上评分，不强制CAP路由。

两组离线拟合率均很高，但闭环成功率只有37.5%至44.5%。这说明高训练拟合率不足以保证诊断面板闭环成功；本次未进一步定位失败机制，不能直接归因于某一表示或控制问题。

## 验收、预算与迁移边界

- 启动HEAD：`c55afed3a4d573058a10d6345e2b7685dfeedb20`；启动前工作区干净。本次未修改冻结输入或运行源码，仅补充执行状态与回报。
- 压缩包SHA256：`f9630d51f3bf247d82f54b307a92f9901ea47431282020fa9b555bee4cf35d83`。
- 派生包manifest SHA256：`ce5a2b13d74cecc01e1c791b247d2eb826d65282fb15bff0330464efd86f65c7`。
- 1244负载文件和运行源码哈希验收通过；从开发矩阵重算排名及两组选集一致；真实示范四级面板特征、目标索引、历史哈希逐值验收通过。运行后再次验证包/源码哈希通过，六份模型SHA256及曲线末次更新数复核通过。
- 原270/扩570的源机产物未整套迁入；fastest与原570一致的来源断言来自源机封印/导出记录，本机直接重算的是本包7600条摘要对应的fastest选择。
- 固定Cython3.0.12的320个工具链文件经核验安装，CAP语义模块在本机隔离冷编译；未关闭bubblewrap。
- 指定四组测试共11 passed；未运行全套测试。
- 目标环境Ubuntu22.04/WSL，独立`outputs/transfers/venv_py311`，Python3.11.18（源机3.11.4）。torch2.14.1+cpu、numpy2.4.6、pymunk6.9.0、matplotlib3.11.1、PyYAML6.0.3、pytest9.1.1均匹配交接版本。
- 4进程特征提取；6份模型顺序训练，每份训练上限7200秒，实际各约807至828秒；8进程评估，单例1800ticks/300决策/1800秒。OMP/MKL各1线程。评估动作seed沿用运行器17，训练seed为17/29/43。
- 日志显示本机雷达使用`analytic_kernel=numpy reason=native_not_built`，两组相同；另有只读NumPy数组转PyTorch张量警告，未出现运行异常。语义模块的Cython冷编译验收不等于雷达C内核已构建。未在实验中途切换后端。
- 本机fastest成功数89/82/75与历史源机扩570报告74/77/90不同。单条参考特征验收不证明跨机器整段训练/闭环轨迹逐位复现；差异原因尚未诊断，不归因于某一个环境差异。教师选择的效果只用本轮同机两组比较，不能把旧fastest与本轮priority拼成对照。

实际入口（由持久tmux监督器运行）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 outputs/transfers/venv_py311/bin/python -u -m tools.priority_transfer run \
  --bundle outputs/transfers/priority_teacher_v1 \
  --out outputs/semantic_rl/priority_teacher_20261117_01 --workers 8
```

## 产物与解释范围

- 已纳入Git的[审计汇总](../assets/priority_teacher_20261117/AUDITED_SUMMARY.json)、[1200条逐场景结果](../assets/priority_teacher_20261117/per_case_results.csv)、[环境与验收记录](../assets/priority_teacher_20261117/preflight.json)及[退出记录](../assets/priority_teacher_20261117/study.exit.json)。模型与完整轨迹保留在本机outputs目录。
- [完成及退出记录](../../../outputs/semantic_rl/priority_teacher_20261117_01/study.exit.json)
- [审计汇总及配对计数](../../../outputs/semantic_rl/priority_teacher_20261117_01/AUDITED_SUMMARY.json)
- [逐场景结果](../../../outputs/semantic_rl/priority_teacher_20261117_01/per_case_results.csv)
- [模型、拟合率及原始评估轨迹目录](../../../outputs/semantic_rl/priority_teacher_20261117_01/)
- [环境与启动验收](../../../outputs/transfers/priority_teacher_preflight_20261117/preflight.json)
- [运行日志](../../../outputs/transfers/priority_study.log)

1200是执行次数；三个训练种子共享同一200初态，不能当作1200个独立场景。面板已经查看过，属于诊断，不是独立盲测。本轮无剩余必需执行项，未调用LLM、增加CAP、采集新开发场景或执行PPO。

建议暂不将priority确立为更优教师规则。若后续需要正式确认，另冻结测试集并保留同机配对对照；本次不自动启动下一轮。
