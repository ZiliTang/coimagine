# 复现命令（round 2）

基线提交 `6c7c177`。设 `PY=.venv/bin/python`。

## 0. 环境（新机器）

```bash
python3 -m venv .venv
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/python -m pip install -e '.[dev,video]'
.venv/bin/python -m pip install -e '.[deterministic-grid]' \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

细节与已知缺口见 `ENVIRONMENT.md`。

## 1. 定向测试

```bash
$PY -m pytest tests/test_local_position_push_v2.py -q      # 10 passed
$PY -m pytest tests/test_local_position_push.py -q         # 18 passed
```

## 2. 重跑基线（`outputs/` 不在 git 里）

```bash
$PY experiments/local_position_push/run_pair.py --split dev  --workers 12   # 384
$PY experiments/local_position_push/run_pair.py --split test --workers 12   # 1920
$PY experiments/local_position_push/analyse_settle.py                       # v1 对照（B/C/D 节为旧版）
$PY experiments/local_position_push/analyse_predictability.py               # v1（目标设错，仅作对照）
```

期望：dev 基准 48 子集 V 23/24、X 21/24；test 判稳 V 0.674 / X 0.617。

## 3. 本轮分析（全部只读，不重跑控制器）

```bash
$PY experiments/local_position_push_v2/audit_a1.py --split test
$PY experiments/local_position_push_v2/analyse_a2_counterfactual.py --split test
$PY experiments/local_position_push_v2/analyse_b.py --split test --arm V
$PY experiments/local_position_push_v2/analyse_b.py --split test --arm X
```

产物写在 `outputs/local_position_push_v1/local-position-push-v1/`：
`audit_a1.{json,md}`、`a2_counterfactual.{json,md}`、`analyse_b_{V,X}.{json,md}`。
README 里引用的核心表已归档到 `docs/local_position_push_v2/`：
`A1_tables.*`、`A1_classification.md`、`A2_counterfactual.*`、`B_transfer_V.*`、`B_transfer_X.*`。

## 4. 诊断用（**默认关闭，非本轮正式协议**）

```bash
# 窗口化估计量：dev 的 noise+delay 子集
$PY experiments/local_position_push_v2/run_pair.py --split dev --workers 12 \
    --variants noise,delay --settle-window 0.30 --run-id smoke-window
# 六个候选条件的 dev 对照（含默认关闭的主动制动路径）
for cfg in "v1-window0 0.02 0 0 4.0" "c1-thresh003 0.03 0 0 4.0" "c2-hold 0.02 0 2.0 4.0" \
           "c3-budget 0.02 0 0 6.0" "c4-window 0.02 0.30 0 4.0" "c5-window-hold 0.02 0.30 2.0 4.0"; do
  set -- $cfg
  $PY experiments/local_position_push_v2/run_pair.py --split dev --workers 12 \
      --settle-surface $2 --settle-window $3 --brake-hold-force $4 --brake-settle-limit $5 \
      --run-id dev-$1
done
```

## 5. 规定回归测试

```bash
$PY -m pytest -m "not slow"
```

期望：**426 collected, 417 passed, 9 failed**。9 项失败都需要
`outputs/grid/world_model/*`（旧轮次、从未入库），见 `ENVIRONMENT.md` 第 5 节。
