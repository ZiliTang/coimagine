# 复现命令（round 3）

`PY=.venv/bin/python`，仓库根 `/home/tzl/repo/coimagine`。环境见 `ENVIRONMENT.md`。

## A. 判稳测量与记账

```bash
# dev 选型：old / 0.20 / 0.30
for spec in "0 w0" "0.20 w020" "0.30 w030"; do set -- $spec
  $PY experiments/option_prediction_mechanism_v1/phase_a/run_a.py --split dev --window $1 --tag $2 --workers 12
done
$PY experiments/option_prediction_mechanism_v1/phase_a/analyse_a3.py --split dev --tags w0 w020 w030 \
    --out outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/a3_dev.json

# test 960 复核（X 不重跑），并与既有回归集配对
$PY experiments/option_prediction_mechanism_v1/phase_a/run_a.py --split test --window 0    --tag w0_test
$PY experiments/option_prediction_mechanism_v1/phase_a/run_a.py --split test --window 0.20 --tag w020_test
$PY experiments/option_prediction_mechanism_v1/phase_a/analyse_a3.py --split test \
    --tags w0_test w020_test --pair-with w0_test \
    --out outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/a3_test_compare.json
```

## B. 终态预测数据与训练

```bash
B=outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1

# 32-root 管道检查
$PY experiments/option_prediction_mechanism_v1/phase_b/collect_b.py --out-dir $B/check32 \
    --train-roots 32 --val-roots 64 --test-roots 64
$PY experiments/option_prediction_mechanism_v1/phase_b/verify_b.py --data-dir $B/check32

# 第一档 256/64/64（≤3072 条）
$PY experiments/option_prediction_mechanism_v1/phase_b/collect_b.py --out-dir $B/phase_b \
    --train-roots 256 --val-roots 64 --test-roots 64 --workers 12
$PY experiments/option_prediction_mechanism_v1/phase_b/verify_b.py --data-dir $B/phase_b
cat $B/phase_b/executions_train.jsonl $B/phase_b/executions_val.jsonl \
    $B/phase_b/executions_test.jsonl > $B/phase_b/records_tier1.jsonl
$PY experiments/option_prediction_mechanism_v1/phase_b/train_b.py \
    --data $B/phase_b/records_tier1.jsonl --out $B/phase_b --tier 1 --epochs 200

# 第二档（仅当第一档验证集上至少一项任务相关目标比持久化好 20% 以上）
$PY experiments/option_prediction_mechanism_v1/phase_b/collect_b.py --out-dir $B/phase_b_tier2 \
    --train-roots 512 --val-roots 64 --test-roots 64 --workers 12
cat $B/phase_b_tier2/executions_{train,val,test}.jsonl > $B/phase_b_tier2/records_tier2.jsonl
$PY experiments/option_prediction_mechanism_v1/phase_b/train_b.py \
    --data $B/phase_b_tier2/records_tier2.jsonl --out $B/phase_b_tier2 --tier 2 --epochs 200

# 评审后补做：动作差值误差与排序一致性（含置换对照）
$PY experiments/option_prediction_mechanism_v1/phase_b/action_sensitivity.py \
    --data $B/phase_b_tier2/records_tier2.jsonl \
    --checkpoints $B/phase_b_tier2/checkpoints --out $B/phase_b_tier2
```

B 的两档合计约 100 s 采集 + 约 4 min 训练（6 个网络 × 3 种子 + 2 个基线）。

## C. 信息歧义机制实验

```bash
$PY experiments/option_prediction_mechanism_v1/phase_c/run_phase_c.py            # 64 对搜索
$PY experiments/option_prediction_mechanism_v1/phase_c/run_phase_c.py --repeat-check
$PY experiments/option_prediction_mechanism_v1/phase_c/belief_reference.py       # C4 参照
$PY experiments/option_prediction_mechanism_v1/verify_c_positive.py              # 独立复核
```
详细入口见 `experiments/option_prediction_mechanism_v1/phase_c/README.md` 与 `C_ambiguity.md`。

**注意**：`phase_c/` 是第一版，其正例结论**已撤回**（特权控制器通路 + 越权协议改动）。
修正后的冻结协议重做在 `phase_c_frozen/`：

```bash
$PY experiments/option_prediction_mechanism_v1/phase_c_frozen/run_frozen_c.py
# -> outputs/option_prediction_mechanism_v1/phase_c_frozen/frozen_c_result.json（负结果）
```
说明见 `C_frozen_negative.md` 与 `errata_review.md`。

## 动画

```bash
$PY experiments/option_prediction_mechanism_v1/make_animation_a.py \
    --root "test/solo_central/007" --variant noise --old-tag w0_test --new-tag w020_test
# -> docs/option_prediction_mechanism_v1/A_settle_fix_old_vs_new.mp4

$PY experiments/option_prediction_mechanism_v1/make_animation_c.py --pair T1-12-symmetric
# -> docs/option_prediction_mechanism_v1/C_ambiguous_pair.mp4
```

## 审查包

```bash
$PY experiments/option_prediction_mechanism_v1/make_bundle.py
# -> option_prediction_mechanism_v1_review.zip（约 1.5 MB；大文件按 MANIFEST.md 列路径与哈希）
```

## 测试

```bash
$PY -m pytest tests/test_option_prediction_mechanism_a.py tests/test_option_prediction_mechanism_b.py -q
$PY -m pytest -m "not slow"          # 全套回归
```

注意：全套回归里有 9 项长期失败，全部因为引用**旧轮次、从未入库**的
`outputs/grid/world_model/*`（6 图 + `belief.npz`）。这是既有环境缺口，与本轮无关，
按合同"既有缺失输出引发的测试失败单独登记，不为补旧图片重训旧模型"处理。
