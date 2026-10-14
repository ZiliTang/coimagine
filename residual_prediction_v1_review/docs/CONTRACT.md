# CONTRACT — 目标变换与评价接口（v1，冻结）

版本号：`residual-prediction-v1/targets-1`。所有含义以 `experiments/residual_prediction_v1/targets.py`
为准；本文件是它的文字描述，两者不一致时以代码为准并视为缺陷。

## 1. 坐标系、单位、列名

- 坐标系：每台机器人自己的 **option 起始坐标系** `R_i0`。不使用世界系，不做跨 option 对齐。
- 单位：中点与位移 **米**；角度 **弧度**；速度 **m/s**、**rad/s**。
- 几何向量 8 列，顺序固定：

  `0 r0_mid_x, 1 r0_mid_y, 2 r0_normal_sin, 3 r0_normal_cos,`
  `4 r1_mid_x, 5 r1_mid_y, 6 r1_normal_sin, 7 r1_normal_cos`

- **中点列的数值定义**：拟合扫描片段的**两个端点连线的中点**，不是扫描点的均值。
- 本轮只有 `0,1,4,5` 四列的中点参数化发生变化。

## 2. 目标变换

对中点列 `y`：

| 参数化 | 编码 | 解码 |
|---|---|---|
| `direct` | `z = (y - mu) / sigma` | `y_hat = sigma * z + mu` |
| `residual` | `delta = y - reference`；`z = (delta - mu) / sigma` | `y_hat = reference + sigma * z + mu` |

- `mu`、`sigma` **只在可见训练标签上拟合**；不可见列的占位数值不进入任何统计。
- `sigma` 有下限 `MIN_COLUMN_SIGMA = 1e-3`；某列完全无有效标签时记入 `missing_columns`，
  `sigma` 取 `DEFAULT_COLUMN_SIGMA = 1.0`，并记为**该列从未被监督**，不得当作成功。
- **模型 `forward` 只输出标准化数值**，绝不在内部添加米制参考值。
- 解码恰好加一次 `reference`。

### 归一化模式

- `per_group`：四个中点列共用**一个标量** `mean/std`（`np.atleast_1d(...)` 形状为 `(1,)`）。
- `per_column`：四列各有自己的 `mean/std`。

### 零输出与保持预测的关系（必须分清）

`mu != 0` 时，`z = 0` 对应的是**均值位移**，**不是**"保持当前位置"。保持预测是
`z = -mu / sigma`。本实现保证：

- `decode_prediction(hold_output(spec), reference, spec) == reference`（误差 0.0）；
- 若把输出层偏置设为 `hold_output(spec)`，则 `z = 0` 也精确等于保持预测（残差配置默认启用）。

报告中禁止把"零物理位移"与"零标准化输出"混为一谈。

## 3. 参考值（reference）

采用不扩大模型的默认策略：

- 当前片段**可见** → `reference = 当前中点`；
- 当前片段**不可见** → `reference = R_i0 固定原点`（数值零）。

因此"不可见 → 可见"的样本退化为**相对固定原点的位置回归**，明确**不是**复制无效中点。
参考值只使用当前或更早的**可见**观测，**不得**读取未来可见性或真值。

## 4. 可见性与 mask

- 目标**可见**：计入几何损失与几何指标（含"当前不可见 → 目标可见"）。
- 目标**不可见**：不计入几何误差，但**保留**有效性、退出等监督。
- 四种转移分别计数并上报：`sustained_visible / became_visible / became_invisible /
  sustained_invisible`。**不删除困难样本。**
- 保持基线使用**同一 reference 规则**；若另有历史实现，单独留一列，避免悄悄改变基线定义。

## 5. 评价面板

| 面板 | 来源 split | 用途 | 窗口/root |
|---|---|---|---|
| `train` | d0 + d1ext | 优化 | 2048 / 1536 |
| `train_diagnostic` | d0 | 未参与优化的训练分布诊断 | 2048 / 512 |
| `val` | val | 训练期 checkpoint 选择 | 2048 / 256 |
| `eval_val` | val | 报告（含 25% t=0 配额） | 2048 / 256 |
| `eval_test` | test | 冻结后测试 | 2048 / 512 |

- 每个面板保存内容哈希；`eval_*` 与 `train` 的**共享记录数必须为 0**。
- 优化面板的定义**不得**为了让新实现好看而改动；改动会使既有检查点失效，必须重训。
- 报告时 `t=0` 与 `t>0`、下一步与终态、三个 split **不得混算**。

## 6. 必报项目

两机器人分别报 p50 / p90 / RMSE；每个格子给出**窗口数**与**独立 root 数**；
按机器人**平移速度**分组（近静止 <0.01、慢 0.01–0.05、运动 ≥0.05 m/s）；
两个基线（P0 保持、P1 命令与时间分组统计）；root 级配对 bootstrap；参数量与延迟。

P1 的命令签名必须包含**两台机器人的 active 标志**（本轮修正，见 `REVIEW_FIXES.md`）。

## 7. 禁止事项

不得用未来真值决定参考位置；不得把不可见标签计入几何回归；不得改评价口径制造成功；
不得用测试集选配置。本轮只接规划器之外的预测器，不做自由 rollout。
