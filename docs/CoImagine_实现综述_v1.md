# CoImagine 实现综述（第 20 轮诊断的代码事实）

目的：让读者能按代码手工追一次两步预测，并据此决定下一个干预。只写实现事实，不写方案史。
所有数字来自 `outputs/grid/diagnosis/review.json`（本次只读运行，未训练新配置、未改方法）；
`scripts/review_round20.py` 是生成它的只读脚本。

一句话结论：**学习模型的一步预测接近精确模型（cell top-1 0.900 对 0.905，反馈 0.572 对 0.613，
代价头 argmin 100/100），差距出现在第 2 步及以后；但现有"模块替换"实验没有把转移网络与状态估计头
`dyn_head` 分开，也没有把"多步推演方式"与"转移精度"分开。** 这两件事必须先补对照，否则补数据的
方向无法判定。

---

## 1. 当前主链路

有两条独立的线，不共用模型，也不共用规划器；它们只共享 `grid.dynamics.execute` 与
`observation_vector`。

| | 主线 A：factored 信念空间 | 历史线 B：latent/direct 前瞻 |
|---|---|---|
| 模型 | `BeliefWorldModel`（`learning/belief_model.py:77`） | `LatentWorldModel` / `DirectWorldModel`（`learning/world_model.py:101,197`） |
| 检查点 | `belief.npz` 等 4 个（全部可加载） | 只有 `latent_released.npz` 可加载；`latent.npz`、`direct.npz` 与当前结构不兼容 |
| 规划器 | `belief_probe.plan_with_search:959`，入口 `plan_with_belief:352` | `world_model_probe.plan_with_model:212` |
| 回合驱动 | `belief_probe.run_search_episode:1037`（搜索）/ `run_belief_episode:674`（两层） | `world_model_probe.run_closed_loop:294` |
| 状态语义 | 41 维动态向量 + 16 维物理后验 | 188 维观测 / 32 维潜向量 + 解码头 |

`plan_with_model` 与 `plan_with_search` **不是同一个规划器的两种深度**：前者对每个首动作预测一个
点状态、再选一个续步（不分支、不更新信念）；后者在首步按私有反馈符号分支、并沿分支更新 16 维后验。
本轮的深度扫描、种子实验、模块替换都在 A 线上做；B 线只做了"精确替换"对照。
A 线是产出历史头条数字与"越深越差"的那条；B 线是更早的编码器/潜转移链。

## 2. 一次决策的数据流（主线 A）

1. `ray_cast(scene, 0, radius=5)` → `OccupancyScan`（该协议下无噪声）。
2. `observation_vector(scan, own_pose, history_a, public_history)` → **188 维**：扫描平面 6×6×5=180
   （每格 one-hot：未知/自由/墙/物体/机器人）、自身坐标 2（÷5）、自身反馈直方图 4、公共事件直方图 2。
   没有特权输入。
3. `model.dynamic_state(observation)` → **(1,41)** = [已解开概率(sigmoid), 36 个物体格概率(softmax),
   4 个机器人坐标(线性, 训练目标为坐标÷5)]。`fuse_hidden_state:659` 用上一步携带的已解开概率覆盖第 0 维。
   **注意**：这 4 个坐标含队友位置，而 188 维观测里只有自身坐标 2 维，队友位置要靠扫描里的
   `KIND_ROBOT` 格推断。
4. 信念 `(1,16)` 由驱动器显式传入，`model.update_belief(belief, dynamic, action, symbol)` 更新。
5. `plan_with_search`：
   `first_states = model.step_actions(dynamic, belief)` → (6,41)；`first_costs = cost_actions` → (6,)；
   对每个动作 `predict_outcome(state_a, a, belief)` → (1,6) 反馈符号概率；
   对概率 ≥ `BRANCH_THRESHOLD=0.02` 的符号：`updated=update_belief(...)`(1,16)、
   `state_after=step(dynamic, a, updated)`(1,41)、`value=_beam_value(state_after, updated, steps=depth-1, beam)`；
   `score_a = cost_a + Σ_s p_s·value_s + (1-Σp_s)·cut(state_a)`。取 `min((score, 动作名))`。
6. 执行 `execute(scene, _joint_for(scene, ACTION_IDS[chosen]))`，把 A 的反馈与公共事件追加进历史，
   再 `belief=update_belief(...)`、`carried=step(dynamic, chosen, belief)`。
7. 下一个决策点：**物体格分布由新扫描的 `dyn_head` 重新给出**，只有"已解开"概率是携带的
   （`fuse_hidden_state`）。所以多步物体预测误差不会跨真实决策点累积，但会在前瞻内部累积。

**分支只在私有反馈上。** `predict_outcome` 名义上输出 6 个符号（4 私有 + 2 公共），但学习观测头只用
私有反馈当标签、公共符号概率≈0，精确适配器也只给私有符号质量；`0.02` 阈值把公共符号丢掉。
所以"公共 moved/stalled"永远不会被当成分支来估值，只在发生之后进入下一观测直方图。

## 3. 状态与不确定性的表示

- 表示：`released` 是一个概率；位置是 36 维单纯形（`_cut_value` 直接读 `state[1:37]` 做期望距离）；
  物理参数是 16 维后验。**没有 (位置 × 参数) 的联合表示。**
- 学习模型：`step`/`step_actions` 把同一个**边缘**动态向量复制 16 份
  （`np.repeat(dynamic, N_PHYSICS)`）喂给转移头，再用 `einsum("nt,atd->ad", belief, states)` 按 b(θ) 混合；
  `update_belief` 里的 `step_per_theta:216` 会给出每个 θ 各自的下一个状态，但**输入仍是那个边缘状态**。
  结论：**"每种物理假设有自己的状态"只在计算似然时临时成立，向前携带的始终是一个混合状态。**
- `plan_with_model`（B 线）不推分布：`predict_next` 给点预测，续步用 `continue_from`——潜模型在潜空间
  续推（`world_model.py:169`），direct 模型把预测的 188 维观测再喂回去（`:238`）。位置分布只作为输出
  解码，不进入下一步输入。
- 精确适配器 `ExactFactoredModel` 维护 `(origin, robots, θ, released)` 的显式联合并枚举；它的
  `update_belief` 与 `predict_outcome` 精确。但在 `_beam_value` 的更深层，它只能从自己的混合向量里
  重新解码点状态（`_sparse_cells`/`_decode_poses`/`_project_poses`，`exact_adapter.py:719,712,729`），
  **因此精确适配器的"续步"也带同样的接口限制**；它精确的地方是首步与信念更新。

## 4. 精确替换到底替换了什么（主线 A）

| 接口方法 | 学习模型 | `ExactFactoredModel` |
|---|---|---|
| `dynamic_state` | `dyn_head(observation)` 网络头 | 从扫描直接解出物体格与位姿（精确） |
| `step` / `step_actions` | 转移 MLP + 按 b(θ) 混合 | 表格动力学，按 (θ, released) 枚举后混合 |
| `cost_actions` | `cost_head` 网络 | 常数 0 / 0.3 / 1.0 |
| `predict_outcome` | `observation_model` softmax | 由 `execute` 得到的确定性符号 |
| `update_belief` | 学到的似然做贝叶斯更新 | 精确似然做贝叶斯更新 |
| `parameter_belief` | `theta_head` | 均匀（实际未被使用，驱动器显式传 belief） |

**没有被替换**：`plan_with_search`、`_beam_value`、`_cut_value`、`run_search_episode`、观测向量、
候选集、初始信念、终止与平局规则。**因此"精确替换"同时替换了状态估计、转移、观测似然与信念更新，
不是只换转移。** 这也意味着：`HybridFactoredModel:769` 的"精确转移"一侧同时拿到了精确的
`dynamic_state`，"学习转移"一侧同时带着学到的状态估计头——**现有模块替换不能区分转移网络与
`dyn_head`。** 这是下一轮必须先拆开的地方。

另外，学习模型不存在"离散化/解码"这一步可供精确适配器绕开：`step` 直接输出连续的 41 维向量，
规划器直接消费；精确适配器输出硬混合，两者都满足 `_cut_value` 对 `state[1:37]` 与 `state[0]` 的读取。
特权信息方面，精确适配器只读观测向量、静态地图、当前公共位姿与规划器传入的信念，从不读
`Scene`/`HiddenPhysics`。

## 5. 搜索与评分

```
# plan_with_search(depth d, beam b)
first = step_actions(state, belief); cost = cost_actions(first, belief)
for a in 6:
    p = predict_outcome(first[a], a, belief)          # 混合反馈分布
    exp = Σ_{s: p[s]>=0.02} p[s] * beam_value(step(state,a,update_belief(belief,state,a,s)), steps=d-1, beam=b)
    exp += (1-Σp) * cut(first[a])                      # 被丢掉的尾部按先验状态估值
    score[a] = cost[a] + exp
choose min(score, action_name)                          # 平局按动作名字典序

# beam_value(dynamic, belief, steps, beam):
states=[dynamic]; costs=[0]
repeat steps:  expand all 6 actions from every kept state;
               score_i = cost_so_far_i + cut(next_state_i);
               sort by (score_i, 枚举序号); keep first `beam`;        # beam=1 => 每层只留一个偏计划
return min(cost_so_far + cut(state))

# cut(state) = 3 * E[物体最坏格到目标的距离] + 20 * P(物体不在目标区)
```

终端没有"成功价值"，只有 `cut`：物体在目标区时 `cut=0`，否则每个中间状态都被记一次 20。
`beam=1` 时每层只保留一个偏计划。**此前写"20 摊在每个中间状态上，所以 `wait` 总是最优"是不准确的**：
`cut` 是每个候选各自加上去的，若几个候选都没完成，相同的 20 相互抵消；真正要看的是 **beam=1 会不会
在收益出现之前，就把暂时更贵的动作链剪掉**。场景 6 的跟踪只说明"那一层保留的是 `wait`"，不构成一般结论。

**"两步等于一步"的适用范围（修正此前的一般化说法）**：只对 B 线的 `plan_with_model` 成立——
若 `wait` 的转移是恒等且代价 0，则 `step_value(wait∘a) = step_value(a)`，于是
`S_2(a) = cost_a + w·dist_a + terminal = S_1(a)`，两者选出同一个首动作。实测：精确适配器在整批
100 个评估场景上 depth1=深度2=深度3（100/100、100/100，`review.json.depth_equivalence_scan`）。
学习 B 线模型的 `wait` **不是**恒等（格分布总变差均值 0.0047、最大 0.032），所以它 depth1/depth2
只有 13/30 相同——差异来自学到的 `wait` 本身，不是"看得更远"。对 A 线，depth 一般不是空操作；
但在场景 6 这类"信念分散 + 20 摊在 cut 上"的局面里，depth 3 仍然每层保留 `wait`，选择不变。

**场景 6 的分数拆解**（`r=1, free, A=slip, B=grip`，物体 (1,0)，A(0,0) B(0,1)，精确可达 3 步；
热身 `probe:0:E` 后 A 得到 `slipped`，后验均匀落在 8 个"自身打滑"的假设上，真值 idx=2 占 0.125）：

| 首动作 | first_cost | Σ 分支贡献 | 尾部 | 总分 | 续步选中 |
|---|---|---|---|---|---|
| `wait` | 0.000 | 1.000×29.000 | 0 | **29.000** | `wait` |
| `probe:0:E` | 0.300 | 1.000×29.000 | 0 | 29.300 | `wait` |
| `push:1:E` | 1.000 | 1.000×28.625（A 只会收到 `idle`，无信息） | 0 | **29.625** | `wait` |
| `co_push:E` | 1.000 | 1.000×28.625 | 0 | 29.625 | `wait` |
| `wait`（真实物理已知） | 0.000 | 1.000×27.000 | 0 | 27.000 | `push:1:E` |
| `push:1:E`（真实物理已知） | 1.000 | 1.000×24.000 | 0 | **25.000** | `push:1:E` |

读法：在这个后验下 B 单独推只有 1/8 的假设能推动，**期望前进 0.125 格 = 0.375 < 推进代价 1.0**，
所以"等"并不荒唐。真实物理已知时同一规划器立刻推（25 < 27）。但**这不能单独解释场景 6**：还有一条
独立的结构限制——**公共 `moved/stalled` 不参与想象分支**（见 §2），所以规划器无法预先给"执行
`push:1:E` 后可能收到的公共证据"定价。该限制**单独登记，不与状态传播修复同时改**，否则原因无法区分。
此前把场景 6 归因于"信念分散 + cut 平摊失败罚"的后半句已收回。

## 6. 训练与关键失败

- **数据**：`world_model_data.collect_trajectories:42`；`generate_situations(lead_in=2)` 之后最多 8 步，
  动作从 `ACTION_MIXTURE`（9 项：push:0 ×2、push:1 ×1、co_push ×3、probe ×1、release ×1、wait ×1）
  随机抽；从 **A 的视角**记录 `observation`(188) 与 `next_observation`(188)，无扫描噪声。
  标签：`physics`(恒定)、`object_cell`、`next_object_cell`、`released`、`next_released`、
  `feedback_symbol`(A 的私有反馈)、`moved`(公共)、`cost`、双方机器人坐标。
- **损失**（`train_belief_model:320`）：t=0 从真实观测解 `theta`(CE 对真物理)、`released`(BCE)、
  `cell`(CE)、`pose`(MSE)；随后 `rollout_length` 步
  `dynamic = transition([dynamic, action, theta_onehot])`，损失为 `released_next`/`cell_next`/`pose_next`，
  再把 `dynamic` 过 softmax/sigmoid 后送 `observation_model([action, dynamic, theta_true])`(CE 对
  `feedback_symbol`) 与 `cost_head`(MSE 对 cost)。**θ 每一步都是真值 one-hot（teacher forcing）；
  动态输入从第 1 步起就是模型自己的预测，梯度穿过整个展开。**
- **训练/推断不一致（已收紧表述）**：训练时三个头永远看到 one-hot θ 与接近真值的状态；推断时
  `step`/`step_actions`/`observe`/`cost_actions` 看到的是复制 16 份的边缘状态再按 b(θ) 混合。
  但**不能把"反馈 argmax 0.973→0.572"全部归因于这个不一致**：同等"不知道真值"的条件下精确适配器
  也从 0.953 降到 0.613——相当一部分下降是**信息变少后反馈本来就难以唯一预测**。同等信息下的差距是
  0.572 对 0.613（argmax）与 0.576 对 0.584（真实符号上的概率质量），比 40 个点小得多。
  另外 **top-1 相近不等于概率够准**：信念更新用的是整条分布，top-1 看不到那些概率值。
  这条推断的重做见 `CoImagine_状态关联验证_v1.md`。
  **代价头不受影响**（MAE 0.00047，argmin 100/100）。
- **关键失败**：一步层面学习模型与精确模型接近；闭环差距出现在 depth ≥ 2——
  同一批 43 个精确可达回合，精确模型 0.349 / 0.535 / 0.721（depth 1/2/3），
  5 个训练种子均值 0.307 / 0.098 / 0.135，同批随机基线 **16/43 = 0.372**。
  也就是说：depth 1 时三种方法都在随机附近，学习模型在 depth ≥ 2 时反而降到 0.10–0.14，
  而精确模型升到 0.54–0.72。**退化发生在"递归使用模型"这一步，而不是单步头。**

---

## 附件

1. 调用关系与函数清单：本文 §1–§6 的 `file:line`；生成脚本 `scripts/review_round20.py`。
2. `outputs/grid/diagnosis/review.json`：本节全部原始数字，含
   `seed_episodes.per_seed[*].success[1|2|3].episodes`（5 种子逐回合）、
   `seed_episodes.random_on_same_batch`（同批 43 回合随机基线）、
   `seed_episodes.exact_model_episodes`（同批精确模型逐回合）、
   `module_replacement_episodes`（含检查点路径与逐回合 scene_id）、
   `scene6`（场景 6 的完整分支拆解）、`state_and_one_step`、
   `cost_and_likelihood_heads`、`depth_equivalence_scan`。
3. `outputs/grid/diagnosis/diagnosis.json`：第 20 轮汇总原始数据。
4. 模块替换所用检查点：`outputs/grid/world_model/belief.npz`；场景集为
   `generate_situations(n=100, lead_in=0, seed=55000)` + 公共热身 `probe:0:E`，可达场景号见
   `seed_episodes.reachable_scene_ids`。
