# scripts

`regenerate_assets.sh` 用一条命令重建全部素材。它只读取已保存的序列、token 数组、检查点与分析 JSON，**不重训、不重新采集、不运行新的物理仿真**。

同目录下的 `*.py` 是渲染脚本的副本，使素材包自包含；权威版本在 `experiments/night_sequence_v1/`。
