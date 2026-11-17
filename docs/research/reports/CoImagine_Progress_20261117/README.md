# CoImagine 导师与合作者汇报

2026-11-17。入口：[交互汇报 index.html](index.html)，[静态PDF](CoImagine_Progress_20261117.pdf)。

`index.html`当前约1.6MB，图表、12个场景视图和10条真实回放全部内嵌，可单文件离线转发，不依赖本机测台或外部资源。原始证据链接需保留仓库目录；离线转发后，汇报内容和回放不受影响。PDF同步更新，为静态阅读版，不能播放轨迹，打印回放窗口显示选定时刻。

本次更新增加：随机房间10 CAP库与首次整合结果；270条完整示范与25条真实接管后缀；九模型等更新预算拟合对照；单CAP失败/边界整合成功、失败前缀/CAP接管成功的真实回放。页面明确区分训练拟合、开发机制验收与最终留出；新100例的完整评估尚未完成时不填成功率。

最终评估与额外125次机制验收均成功退出后，持久刷新任务将自动重新读取结果、更新HTML/PDF并再次运行浏览器检查。入口为 `refresh_after_evaluation.py`，状态与日志在 `outputs/semantic_rl_authors/random_room_coverage_20261117_02/report_refresh.*`。它只读取和展示结果，不启动模型、训练或物理执行。本页是静态快照，不会实时拉取后台进度；快照时间见“新增证据”部分。

内容保留研究问题、六个早期阶段的思路迭代、监督数据覆盖、完整动作/有限参数RL、输入归因和关系迁移负结果，并加入已执行的覆盖驱动多CAP能力库及修复监督闭环。事实、机制假设与尚未运行的计划分开标明。不同实验的成功率不视作统一排行榜。

回放包括清障开发成功、贴墙释放保留、同初态CAP失败/监督成功，以及同跨布局初态CAP失败/几何组成功。案例按结果选取作说明；总体统计包含所有预定案例。可视化使用已有仿真真值存档，仅供人检查，不是策略输入。图像位置为显示压缩到1e-6，约10Hz取样并保留决策边界，未重新仿真。

- `summary.json`：嵌入汇总数字及来源。
- `evidence_manifest.json`：所读原始文件SHA256及HTML哈希。
- `results_overview.png/.svg`：可单独用于幻灯片的统计图。
- 场景与回放初态SVG：由存档几何绘制。
- `build_report.py`、`latest_evidence.py`、`template.html`、`report.css`、`report.js`：可再生源文件。

在仓库根运行：

```bash
PYTHONPATH=. .venv/bin/python docs/research/reports/CoImagine_Progress_20261117/build_report.py
```

该命令只读取冻结产物并重建汇报，不调用模型、不训练、不运行物理仿真。生成PDF可用浏览器的“打印 / PDF”。

真实Chromium离线检查覆盖图表切换、6组/10条回放、播放/暂停、拖动、决策跳转、接管时刻跳转、复位及390px/1440px布局；零页面脚本错误、零外部网络请求、无横向溢出。最新检查、截图和PDF导出记录见`outputs/research/supervisor_report_20261117/latest/`。新增最终接管回放时检查数量会自动更新。

更新前的整份报告保留为 `outputs/research/supervisor_report_20261117/before_random_room_update.tar.gz`；原始实验、历史报告和之前的浏览器检查不变。
