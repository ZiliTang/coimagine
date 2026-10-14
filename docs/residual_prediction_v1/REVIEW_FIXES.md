# REVIEW_FIXES — 审查发现 3.1–3.6 的逐条修正

审查对象：`night_sequence_v1_review(2).zip`（SHA256 `ca35ab41…`）。
下列每一条都先在旧代码上复现，再修正，并给出可检验的证据。

## 3.1 残差诊断再次引入左补齐索引错误 —— 成立，已修

**复现**：`fit_diagnostic.py` 的 `MidpointHead.forward` 用 `valid.sum() - 1` 读 current。
窗口是左补齐，只有 1 个有效 token 时它在第 32 号槽，旧代码读第 0 号槽；17–32 个有效 token 时也
可能读到较早的真实观测。

**修正**：本轮不再复制该公式。`data.build_batch` 通过 `night_sequence_v1.models.last_valid_index`
取当前中点，`MidpointHead.forward` **完全不接触参考值**（参考值只在 decode 阶段加入）。

**证据**：`tests/test_residual_prediction_v1.py::test_first_and_last_valid_token_positions_for_every_window_length`
对 1/6/16/17/32/33 个有效 token 逐一断言首末位置，并显式断言旧公式在 count<33 时给出不同结果。

## 3.2 归一化残差与米制位置混加 —— 成立，已重构

**复现**：旧 `forward` 内先把米制 current 加进网络输出，外层又反归一化并加一次 current，
结果多出 `sigma_delta * current_wrong` 项。

**修正**：`encode_target` / `decode_prediction` 是唯一入口；`forward` 只返回标准化数值；
解码恰好加一次参考值。旧结果因此**只作为方向性线索**，本轮收益全部来自新实现。

**证据**：`test_encode_decode_round_trips_and_adds_the_displacement_once` 断言往返误差 <1e-12；
`test_synthetic_constant_position_and_known_displacement_decode_exactly` 用常数目标与已知位移验证到机器精度。

## 3.3 诊断 A/B/C 的 mask 口径不符合正式评价 —— 成立，已修

**修正**：单头损失是 `sum((pred-target)^2 * mask) / sum(mask)`；归一化统计只在可见标签上拟合；
所有上报误差都按目标可见性筛选。多任务损失沿用第 6 轮的逐组 mask。

**证据**：`test_invisible_placeholders_do_not_enter_statistics_or_errors` 把一半标签填成 999.0
并断言统计量与掩码误差完全不变。

## 3.4 小集合实验没有证明整条序列拟合成功 —— 成立，已修

**复现**：旧版评价时每条序列只随机取**一个**时刻；且"前 N 个 root 再取前 N 条序列"使日志里的
roots 计数不可靠（标为 4 的子集实际只来自 1 个 root，标为 16 的来自 2 个）。

**修正**：`tiny_fit.select_sequences` 明确选 **N 条序列**（默认每条来自不同 root，并记录
`distinct_roots`）；`all_adjacent_frames` 让评价覆盖每条序列的**全部合法相邻帧**
（1 条 → 68 窗口，16 条 → 1215，64 条 → 4955）。

**证据**：`outputs/residual_prediction_v1/tiny_fit_d1.json` 中每条目同时给出
`sequences / distinct_roots / evaluation_windows / sequence_ids`。

## 3.5 抽样、初始化与统计结论不完整 —— 成立，已修

- **抽样**：诊断训练样本改为从固定的 `train` 面板抽取（D1 下含 d0 与 d1ext），
  不再写死 d0；(train, val) 记录交集为 0。
- **初始化与批次顺序**：`set_all_seeds` 在**构造模型之前**固定 Python/NumPy/Torch 三个随机源；
  每个 seed 只造**一份** `state_dict`，四个格子全部加载它；批次顺序用同 seed 的生成器，
  四个格子逐批次相同。日志记录 `initial_state_hash` 以便核对。
- **per_column 对照**：已实现，并在 2×2 中作为独立格子跑完 3 个 seed。
- **尾部与局部分组**：除中位数外同时报 p90、RMSE，并按运动状态、可见性转移、两台机器人分别报；
  root 级配对 bootstrap 单独给出，不与边际中位数混用。

**保留的局限**：2×2 对照的是**工程配置**。残差与直接的目标振幅相差约 31 倍，
因此该对照**不能**把残差的纯结构效应与目标振幅效应分开；报告如实标注，不越界表述。

## 3.6 结果与复现链路需统一 —— 成立，已修

- 旧的 `single_d1.json`（1.9402339）与 `single_next_d1.json`（1.9680709）不一致：
  两份文件来自前后两次不同代码状态的运行。本轮所有结果写入**唯一**的
  `single_head_diagnostic_d1.json`，内含 12 个格子（4 配置 × 3 seed）。
- 终态实验的指标此前仍叫 `next_step`：本轮区分 `next` 与 `terminal`，各自的基线也分开计算
  （`hold_next` / `hold_terminal`）。
- `all_d1.json` 不是汇总：本轮 `--experiment all` 统一写出，且不再作为交付依据。
- MANIFEST 与交付消息的提交号不一致：本轮 `MANIFEST.json` 只记录**包内文件**的大小与哈希，
  并明确声明"这不验证任何实验结论"；提交关系在交付消息中单独给出。
- `verify_from_zip.py` 只读回保存的指标：本轮 `verify_package.py` 在**干净目录**里
  加载包内窗口张量 → 重建变换 → 推理 → 解码 → **用真实 mask 重算**保持基线与模型的 p50/p90，
  与包内参考值逐项比较后才打印成功。它明确声明这只是**包内子集复现**，
  不等于重跑完整测试面板。

## 7.6 P1 命令签名缺少第二台机器人 active 标志 —— 成立，已修

`night_sequence_v1/evaluate.py` 的签名原为 `alpha0|alpha1|speed0|speed1|active0`，
**遗漏 active1**，使仅机器人 1 参与情况不同的两条命令落进同一查表组。现为
`…|active0|active1`。
