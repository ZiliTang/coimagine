# CAP 实验研究日志

从已有开发面板进入[研究日志](http://localhost:8775/research-journal)。日志按 [entries.json](entries.json) 的先后顺序展示，每五秒刷新。本机服务与原面板共用 8775 端口和 tmux 会话 cap_regional5_ui。

每次新尝试追加一条记录，填写稳定的 id、日期、标题、动机 motivation、目标 goal 和 key_findings 下的结果目录 folder。可用 assets 登记有标注的报告、图、回放或逐例数据。结束时在同一条补上 result 与 analysis，旧记录保留。也可先写入 status 为“待启动”的记录。

运行中的记录可以设置 run 为 outputs/semantic_rl 下的运行目录，并在 progress 数组中按顺序登记阶段。每个阶段填写 label、total 和一种数据来源。glob 统计逐例落盘文件，rows 统计一个 JSON 数组或对象的条数，counter 加 key 读取 JSON 的整型计数，file 检查运行目录中的结束标记，finding 检查报告目录中的交付件。全阶段达到目标后才转为已完成；运行目录的 ERROR.json 优先显示异常。进度是已写出的记录数，批量写出的阶段会整批更新。补充记录和更新文字不需要重启服务。
