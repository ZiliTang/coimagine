# STATUS — coverage_scale_v1 256k 续跑

run_id `coverage-scale-256k`（数据追加在原 `coverage-scale-20261013` 目录）。
起点 08:12，交付约 10:00，墙钟约 1.9 h，4 h 上限内；未扩到 512k。

## 数据（只增不改）

| split | 去重窗口 | 独立根 | 说明 |
|---|---:|---:|---|
| train 128k 档 | 128006 | 4591 | 原样保留，`transition_id` 前缀逐条未变 |
| train 256k 档 | **256004** | **9176** | 达到目标的完整根前缀，574 个连续 shard |
| val | 3584 | 128 | 原样保留 |
| test | 3497 | 128 | 原样保留，选模期间未读 |
| narrow 对照 | 16019 | 626 回合 / 6 初态 | 原样保留 |

被污染的重复 shard 300–434 已移入 `outputs/coverage_scale_v1/coverage-scale-20261013/superseded_duplicate_shards/`，
进度另存 `collection_progress_superseded_256k.jsonl`，不计入任何规模统计（见 [errata_256k.md](errata_256k.md) E10）。

## 训练（6/6 完成）

`256000_{256,512}_seed{0,1,2}`，每个 120060 次更新、60.0 遍数据、单 run 墙钟 258–455 s；
评估点 42 个（每 3000 次 + update 0/100 + 最终）；保存 `best_new_val_total`、`final`、`update60000`。
不变量：同 seed 跨宽度样本顺序流相同；同宽度同 seed 跨档初始参数 hash 相同（已断言）。

## 评估（清单冻结后执行）

- `model_list.json`：30 个 checkpoint（128k 12 个 + 256k 18 个），含文件 SHA256 与参数 hash，
  整体 SHA256 冻结；`holdout_256k.json` 是一次性新 test 评估，不参与选模。
- `groups.json` / `groups_table.json`：6 个分组的逐刚体误差与 persistence 基线。
- `candidates/`：主选点、H=1 与 H=3 分表、共同 `(start_id, candidate_seed)` 配对，真值复用原缓存。

## 交付物

- `docs/coverage_scale_v1/report_256k.md`（结论）、`CONTRACT_256k_addendum.md`（增补契约）、
  `errata_256k.md`（更正）、本文件。
- `outputs/coverage_scale_v1/coverage-scale-256k/`：`runs/`、`groups*.json`、`holdout_256k.json`、
  `candidates/`、`comparison.md`、`figures/`。
- `coverage_scale_v1_256k_review.zip`。

## 恢复命令

```bash
python experiments/coverage_scale/collect.py --run-id coverage-scale-20261013 --workers 6 --train-shards 600
python experiments/coverage_scale/run_matrix_256k.py --parallel 2
python experiments/coverage_scale/run_groups.py --kinds best_new_val_total,final,update60000
python experiments/coverage_scale/run_eval_256k.py
python experiments/coverage_scale/make_report_256k.py
python experiments/coverage_scale/make_bundle_256k.py
```

## 下一步（只一项）

在同一冻结协议下把上限提高到 512k 跑一档，用来判断 256k→512k 是否仍像 128k→256k 那样
把 P 往下压（尤其是 512 宽度首次出现的稳定优势是否继续）。
