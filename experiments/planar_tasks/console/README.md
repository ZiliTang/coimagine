# CoImagine 研究测台（最小只读版）

```bash
.venv/bin/python -m experiments.planar_tasks.console.server --port 8766
```

打开 http://127.0.0.1:8766/ 。默认读取 `outputs/agentic/`，可用 `--root` 指定其他实验产物目录。只绑定 localhost。模型、仿真器、传输和优化驱动不会被导入或启动；没有创建实验、修改额度、重试、删除或测试反馈按钮。运行中的实验可手动刷新。可用 `/?run=<实验 ID>` 直接打开并分享指定记录。

- 按 LLM 后端筛选记录，显示精确模型名、版本、冻结配置、停止原因和提交阶段。
- 单独显示保守预留额度和供应商实际返回的 token usage；未知用量明确标示，金额不推算。自主会话尚未使用的累计资源显示 0；单次/存储上限没有累计用量时不伪造余额。最终提交预留单独标示，并明确属于总额度的一部分。
- 自主事件按实际顺序分页；只投影模型公开内容、工具名、工具参数和执行摘要，不下发完整模型请求或 `reasoning_content`。配置中的凭据字段和常见密钥格式会遮盖。
- 自由 CAP 查看 `policy.py`；Option-search 查看作者的 `heuristic.py`（固定执行器不作为作者策略版本）。显示它们在脚本实验、策略测试和最终提交时的真实快照、SHA256 和文本差异。不是自动识别/虚构“第 N 轮优化”。大于 256 KiB 的代码明确截断。
- 开发 episode 与冻结后独立评估分区；评估批次的 `summary.json` 未落盘前不展示逐题结果。旧 pilot 明确标为 Legacy，transfer 仍属于 development。
- 回放直接读取 episode 的 `replay.json`，复用现有 Option Testbed 的画布、执行反馈、时间轴和播放控制。编辑功能被隐藏，服务端所有变更方法返回 405。不依赖 8765 服务、不复制回放、不运行物理。

记录发现以 `session.json` 为入口，跳过源码快照和符号链接；冻结评估以 `manifest.json.kind=frozen_evaluation` 为入口。评估目录需要位于选定的 `--root` 下。源产物可读取并不意味着重新验证了其历史哈希；测台展示的是记录，正式校验仍由基线工具完成。

只读 API：`GET /api/experiments`、`/api/experiments/{id}`、`/{id}/events?start=0&count=100`、`/{id}/policy/{version}`、`/{id}/diff?before=...&after=...`。ID 是服务端生成的记录标识，不接收任意文件路径。`/replay?run={id}` 是只读回放页。

验证：

```bash
.venv/bin/python -m pytest tests/test_research_console.py -q
```

最小版不包含并行调度、余额充值、自动费用换算、图形化启动实验或模型输出的隐式修复。统一 token 计数没有数据时不把 Codex 会话用量伪装成已测量值。
