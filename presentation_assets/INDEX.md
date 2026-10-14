# INDEX — 导师汇报素材索引

> **⚠ 实现审查修正轮**：`F06`/`F07` 与报告中的“历史窗口优于单帧”“Transformer 不如历史 MLP”“数据规模饱和”三条结论**已撤回**。
> 原因是 MLP-current 与 Transformer-history 训练时未读到有效输入（左补齐下索引取错，t=0 读到全零补齐区），另有退出时刻 T、几何中点定义、P1 可见性 mask、验证集抽样四处问题。详见 `docs/night_sequence_v1/IMPLEMENTATION_FIX.md`。
> **本轮素材已全部按修正后的重训模型重新渲染。**

所有主图与数据均已生成；本文件只做索引与图注，不是完整报告。

统一约定：模型颜色固定（MLP-current 蓝、MLP-history 绿、Transformer 红），机器人颜色固定（robot0 蓝、robot1 橙），真值用灰/黑，预测用模型色。

证据分级：**历史已核对结果** / **本轮结果** / **研究假设**，每图注明。

## F01 研究动机与当前主线

- 图片：`presentation_assets/figures/F01_motivation_and_mainline.png`（矢量版 `presentation_assets/figures/F01_motivation_and_mainline.pdf`）
- 数据：`p`, `r`, `e`, `s`, `e`, `n`, `t`, `a`, `t`, `i`, `o`, `n`, `_`, `a`, `s`, `s`, `e`, `t`, `s`, `/`, `d`, `a`, `t`, `a`, `/`, `F`, `0`, `1`, `_`, `p`, `r`, `o`, `g`, `r`, `e`, `s`, `s`, `_`, `f`, `a`, `c`, `t`, `s`, `.`, `c`, `s`, `v`
- 建议用途：左：Gridworld 已观察到的选择偏差（历史讨论结论，非本轮结果）；中：转向连续 2D 后接触、局部观测与变时长技能带来的问题；右：固定模型预算下组织预测/反馈/搜索的候选方向，并配已完成/本轮/待验证状态表。
- 能说明什么：物理乐观是已有实验动机，信息乐观与观测条件分支仍是待验证机制。
- 不能说明什么：不能读成“已获得算法成功”。Gridworld 数字来自历史讨论，未在本轮复现。

## F02 连续 2D 环境与场景设计

- 图片：`presentation_assets/figures/F02_environment_and_scenes.png`（矢量版 `presentation_assets/figures/F02_environment_and_scenes.pdf`）
- 数据：`presentation_assets/data/F02_scene_configurations.csv`, `presentation_assets/data/F02_config.json`
- 建议用途：四种实际构型（物体轮廓、两机器人朝向、米制坐标），全部由仓库内的实际生成器产出。
- 能说明什么：当前主采集的物理与传感配置固定，且与含不确定参数的旧 Gridworld 版本不同。
- 不能说明什么：未画不存在的墙/通道，也不主张已有避障任务。

## F03 option 与真实执行过程

- 图片：`presentation_assets/figures/F03_option_execution.png`（矢量版 `presentation_assets/figures/F03_option_execution.pdf`）
- 数据：`p`, `r`, `e`, `s`, `e`, `n`, `t`, `a`, `t`, `i`, `o`, `n`, `_`, `a`, `s`, `s`, `e`, `t`, `s`, `/`, `d`, `a`, `t`, `a`, `/`, `F`, `0`, `3`, `_`, `e`, `x`, `e`, `c`, `u`, `t`, `i`, `o`, `n`, `_`, `t`, `i`, `m`, `e`, `l`, `i`, `n`, `e`, `.`, `c`, `s`, `v`
- 建议用途：实际阶段机流程、一条真实执行的阶段时间轴、速度与可见性曲线，并标出实际退出时刻 T。
- 能说明什么：退出不等于物体整体静止；区分正常退出、中止与观测不足。
- 不能说明什么：不使用终态复制帧冒充真实过程。

## F04 数据覆盖与分布

- 图片：`presentation_assets/figures/F04_data_coverage.png`（矢量版 `presentation_assets/figures/F04_data_coverage.pdf`）
- 数据：`p`, `r`, `e`, `s`, `e`, `n`, `t`, `a`, `t`, `i`, `o`, `n`, `_`, `a`, `s`, `s`, `e`, `t`, `s`, `/`, `d`, `a`, `t`, `a`, `/`, `F`, `0`, `4`, `_`, `d`, `a`, `t`, `a`, `_`, `s`, `c`, `a`, `l`, `e`, `.`, `c`, `s`, `v`
- 建议用途：以物体初始坐标系对齐的机器人轨迹中心线叠加（预先固定抽样、按 root 均衡），另给出root/序列/真实帧规模、退出类别与终态可见性、序列长度分布。
- 能说明什么：显示 D0/D1 的相对规模与训练/验证/测试的独立 root 统计。
- 不能说明什么：密集轨迹不等于独立样本多；窗口数不计入 root 数；不声称已覆盖新物理机制。

## F05 模型与学习任务

- 图片：`presentation_assets/figures/F05_models_and_task.png`（矢量版 `presentation_assets/figures/F05_models_and_task.pdf`）
- 数据：`p`, `r`, `e`, `s`, `e`, `n`, `t`, `a`, `t`, `i`, `o`, `n`, `_`, `a`, `s`, `s`, `e`, `t`, `s`, `/`, `d`, `a`, `t`, `a`, `/`, `F`, `0`, `5`, `_`, `m`, `o`, `d`, `e`, `l`, `_`, `p`, `a`, `r`, `a`, `m`, `e`, `t`, `e`, `r`, `s`, `.`, `c`, `s`, `v`
- 建议用途：共同数据流：真实观测前缀（O0 + 最近 32 个真实网格时刻，≤33 token）→ 三种编码器 → 两组预测头；另给出实际参数量。
- 能说明什么：真实前缀与预测未来分色；t=0 没有尚未发生的历史。
- 不能说明什么：本轮不含自由 latent 递推或完整规划器。

## F06 学习过程、规模效应与拟合诊断

- 图片：`presentation_assets/figures/F06_learning_curves.png`（矢量版 `presentation_assets/figures/F06_learning_curves.pdf`）
- 数据：`presentation_assets/data/F06_run_summary.csv`, `presentation_assets/data/F06_data_scale_to_error.csv`, `presentation_assets/data/F06_inference_cost.csv`, `presentation_assets/data/F06_fit_diagnostic.csv`
- 建议用途：四格：(a) 同一更新轴上的验证曲线；(b) 同口径训练/验证误差与「保持当前」基线；(c) 独立 root 数与误差的关系；(d) 直接预测位置 vs 当前位置＋预测位移。
- 能说明什么：训练与验证几乎重合说明没有过拟合；训练本身就远差于平凡基线说明欠拟合；残差参数化把下一步误差降 19× 并首次超过基线，把瓶颈定位到训练方式。
- 不能说明什么：未完成或预算不同的 run 明确标注；单 seed 不画置信区间；数据规模一栏只在修正后的实现下成立，**不能**升级为“数据收益已饱和”的普遍结论；残差结论来自单头小规模对照，尚未在完整六头模型上验证。

## F07 拟合与预测性能（修正后重训）

- 图片：`presentation_assets/figures/F07_prediction_performance.png`（矢量版 `presentation_assets/figures/F07_prediction_performance.pdf`）
- 数据：`presentation_assets/data/F07_metrics_t0.csv`, `presentation_assets/data/F07_process_by_motion.csv`
- 建议用途：P0/P1 与新模型在 t=0 终态与执行中下一步/终态的主指标比较，含两机器人中点与法向角、可见/退出类表现，以及 t=0 动作差值/排序表。
- 能说明什么：执行前终态、执行中下一步、执行中终态分开报告；固定 Δt 头另按真实运动状态拆分。
- 不能说明什么：几何排序不等于任务收益排序；真实前缀条件下的预测不是自由 rollout。固定 Δt 头在近静止子集上远差于「保持当前」基线，**尚不能说已学好短时动力学**。

## F08 对闭环规划的意义与未决问题

- 图片：`presentation_assets/figures/F08_closed_loop_implications.png`（矢量版 `presentation_assets/figures/F08_closed_loop_implications.pdf`）
- 数据：`p`, `r`, `e`, `s`, `e`, `n`, `t`, `a`, `t`, `i`, `o`, `n`, `_`, `a`, `s`, `s`, `e`, `t`, `s`, `/`, `d`, `a`, `t`, `a`, `/`, `F`, `0`, `8`, `_`, `e`, `v`, `i`, `d`, `e`, `n`, `c`, `e`, `_`, `g`, `a`, `p`, `s`, `.`, `c`, `s`, `v`
- 建议用途：“已有证据 → 可能用途 → 尚缺验证”表，并附已有 C 负结果的小表。
- 能说明什么：本轮没有闭环实验，因此本轮未评估闭环收益。
- 不能说明什么：不能画成功率提升或算法胜负；C 的零收益差仅限原场景/技能/评分。

## videos

- `presentation_assets/videos/execution_success.mp4`：phase_b/train/solo_central/0000#cmd0，真实时长 3.85s，1x (fps=20, dt=0.05)，由已保存轨迹渲染，未重新运行仿真
- `presentation_assets/videos/execution_failure.mp4`：night/d1ext/dual_same_side/01196#cmd0，真实时长 2.85s，1x (fps=20, dt=0.05)，由已保存轨迹渲染，未重新运行仿真

## 可复现命令

```bash
bash presentation_assets/scripts/regenerate_assets.sh
```

素材包自带渲染脚本副本（`scripts/`），因此解压后即可重建，不依赖工作区其余部分。

该命令只读取已保存的序列、token 数组、检查点与分析 JSON，不重训、不重新采集、不运行新的物理仿真。
