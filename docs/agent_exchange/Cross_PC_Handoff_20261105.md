# 2026-11-05：V2训练结束后的跨PC交接

当前结果先读 [V2最终总结](reviews/V2_Training_Final_20261105.md) 和 [CURRENT.md](CURRENT.md)。V2已按12h预算结束，无训练进程需要迁移；best@3900验证29/32，最近完整@4160为28/32，final@4170未额外验证，新test未运行。不要在新机器自动续训或运行test。

## 一个文件上传网盘

本机交付文件：

```text
outputs/handoff/20261105_completed_rl/coimagine_handoff_20261105.tar.gz
outputs/handoff/20261105_completed_rl/coimagine_handoff_20261105.tar.gz.sha256
```

压缩包独立于Git，**不会随push上传**。上传tar.gz即可；建议同时上传小的sha256文件，下载后先核验。精确字节数、SHA和打包清单在同目录`PACKAGE_INFO.json`，以实际文件为准。

包内顶层目录为`coimagine_handoff_20261105/`，包含：

- `README_请先阅读.md`：中文结果、版本边界、恢复和检查步骤。
- `code/coimagine.bundle`：本次提交后的main完整Git历史，可离线clone；不依赖GitHub已成功push。
- `payload/outputs/option_rl_v21/`：本轮正式训练与预检查，含best/final/checkpoint、现存周期快照、完整history、保留的逐根验证、manifest/source_snapshot、监督退出记录。没有补采未保留轨迹。
- `payload/outputs/rl_baselines/V1_event_progress_20261103/`：V1冻结模型、历史结果、面板和匹配的686项frozen_code。禁止用当前V2控制器冒充V1复现。
- 两次V2中期监测与最终总结的离线输入/脚本，以及启动和STOP预检查证据。
- 当前testbed的全部presets/overrides，以及V2八个审查实例、11月4日最新十条回放、与preset同ID的现存回放、V1已有六条RL展示回放。精确清单见包内MANIFEST；没有携带整个历史3.2GB回放库，也没有复制旧server PID/run_index缓存。
- `MANIFEST.json`、`SHA256SUMS`、`verify.py`、`restore.py`：文件清单、完整性校验与恢复工具。

代码由bundle携带；大模型和原始输出由payload携带。没有虚拟环境、登录凭据、进程、全部历史outputs或全量训练轨迹。旧报告链接若指向未打包的历史实验，在新机器上仍可能缺失，不能用报告代替原数据。

## 在新PC恢复

```bash
sha256sum -c coimagine_handoff_20261105.tar.gz.sha256
tar -xzf coimagine_handoff_20261105.tar.gz
cd coimagine_handoff_20261105
python3 verify.py
python3 restore.py /path/to/new/coimagine
```

`restore.py`只接受不存在的目标路径，先校验包，再从bundle clone完整代码并复制payload；不会覆盖已有仓库，不安装依赖，不启动训练或服务。恢复后origin设为`https://github.com/ZiliTang/coimagine`，但不会访问网络或推送。

依赖环境在恢复后自行创建；版本信息保留在V1冻结目录与V2 manifest。Python工具默认使用新PC的`.venv/bin/python`，不要照搬原机器绝对路径或PID。需要在已有clone继续时，先备份已有outputs，再参考包内说明按路径合并；自动恢复工具有意不做覆盖式合并。

Git远端是否已同步，以本次提交/推送执行结果为准；bundle是完整离线后备。不要将旧的12:50中期迁移包或V2开训前bundle当作此次最终结果。
