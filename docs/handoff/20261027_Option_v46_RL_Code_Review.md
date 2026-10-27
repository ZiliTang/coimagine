# Option v4.6 / RL 进度核查与代码审查

日期：2026-10-27（Asia/Shanghai）。审查代码：`a4da755b77c1c82666a64eeb22f08c41ba5eaf44`。
对象：交接报告、`experiments/option_rl_v45/`、`src/coimagine/random_start/` 及相关测试。
这是审查结果，不是训练修复验收；没有修改训练器、控制器、奖励或运行中的 run9，没有新增长训练。

## 1. 判断

低层 option 已有可重复的局部能力，但当前 RL 实验仍处于训练链路与任务可学性的诊断阶段。
现有数据不足以证明“平坦编码失败的原因就是探索”“因子化已经解决探索”或“远距失败主要是缺少换面规划”。
应先修复训练目标与环境不一致、非法初始几何、异常终止的回报处理，再在相同条件下比较编码。
当前 run9 可以保留为原实现的诊断记录，不能在上述问题修复后与新实现直接混为同一基线。

本次确认的进展：

- v4.6 短面接触、双推 ready/release 的聚焦测试通过；交接报告中的同侧 after-engagement contact
  100% 是特定测台构型的结果，不代表任意随机构型的协同成功率。
- run7/8 的历史文件确实存在最高 `3/16`，run8 最后为 `1/16`。它们处于课程训练背景，不是 run9
  的同分布对照；也不能由一个峰值推断稳定泛化。
- 2026-10-27 10:10:05 取证快照：run9 已写到 iteration 49，累计约 1655 s；最近评估为 iteration 45，
  success `0/16`、mean progress `-0.424915`，截至该快照历史最好评估也为 0。6 h 训练尚未结束。
- 另保留了 iteration 40 的 checkpoint 快照并复评，结果 `0/16`、mean progress `-0.308374`；
  终止原因是 13 个时间预算耗尽、3 个穿透异常。这与相应训练评估点一致。

## 2. P1：训练目标不符合冻结的无折扣任务

位置：`train.py:54`、`train.py:180-183`、`train.py:203-215`；
对照：`spec.py:203`、`env.py:113-115`、`env.py:651-678`。

环境明确固定 `GAMMA=1.0`，并在 `EnvConfig` 中拒绝其他 gamma；返回的是每个 option 内
**未折扣**的控制周期奖励和。训练器却另设 `GAMMA_STEP = 0.99 ** CONTROL_DT`，
在 option 边界乘 `GAMMA_STEP ** periods`。

这同时产生两个问题：训练目标与报告的任务回报不同；同一物理奖励序列仅改变 option 分段，
就会改变 value target。它也不是完整的逐控制周期折扣回报，因为 option 内部并未折扣。

真实环境复现（`simple_test_0000`，均不移动物体、均 1800 周期 / 90 s）：

| 固定联合动作 | option 数 | 环境总回报 | 训练器首步 return target |
|---|---:|---:|---:|
| WAIT 2 s / WAIT 2 s | 45 | -6.516956 | -3.828467 |
| WAIT 24 s / WAIT 24 s | 4 | -6.516956 | -4.302547 |

因此当前策略不仅在比较物理效果，也受到人为的分段偏好影响。应恢复冻结任务的 gamma=1，
而不是据此调大成功奖励或修改测试阈值；修复后需要重新建立基线。

## 3. P1：当前 simple scene 生成器不能保证初始几何合法

位置：`scenes.py:37-59`。run9 确实使用该生成器：进程参数包含
`--simple-scene --hours 6.0 --episodes 8 --eval-roots 16 --eval-every 5 --entropy 0.02 --bc-steps 300`。

根因：

1. 采样物体中心时使用未旋转的半宽/半高，随后才随机姿态，没有按旋转后的四角检查墙界。
2. 机器人沿射线到矩形边界后增加径向距离；这不能保证机器人圆盘到矩形的最短距离大于半径。
3. 把机器人中心夹到墙内后，没有重新检查物体重叠、双机重叠和朝向。

按 run9 的原始种子完整复核，几何判定只计真实重叠或越界，不把“安全余量不足”算入非法：

| 原始固定面板 | 根数 | 至少一种初始几何非法 | WAIT 2 s 首周期穿透异常 |
|---|---:|---:|---:|
| train，`sample_scenes(1,64,'train')` | 64 | 12 | 4 |
| test，`sample_scenes(2,16,'test')` | 16 | 3 | 3 |

训练面板中有 2 根物体越墙、1 根双机重叠；这些类别可与机器人—物体重叠重合。
评测非法根为 `simple_test_0005/0011/0012`，机器人—物体初始穿透分别约 14.15、7.66、11.78 cm。
iteration 40 的真实策略复评也在这三根的第一个控制周期（0.05 s）终止。

这三根初始距离都大于 1.2 m，当前远距组一共只有 7 根。直接把它们算作长程规划失败会混入
严重的场景错误；这也不意味着其余远距失败已得到解释。

修复时复用现有 `layout.py` 的几何检查原语，并单独声明 simple 场景的合法余量，按旋转矩形和圆盘
检查后重采样；不要把旧评测失败根静默删除。旧面板和失败记录应保留，新生成器应有新版本、
冻结的新面板，所有方法一起重跑。

## 4. P1：异常终止仍可廉价结束训练 episode

位置：`env.py:598-611`、`env.py:664-670`，以及 `train.py:179-195`。

环境只有时间预算耗尽才向 `_reward_terms` 传 `budget=True`。穿透、越界、数值异常会设置
`termination`，但失败惩罚为 0。训练器看到任意 termination 就认为 episode 完结，不会进入
补失败惩罚的 skill-cap 分支。

例如 `simple_test_0005` 仅执行 WAIT 即在 0.05 s 穿透终止：reward 约 `-0.001654`、
failure penalty 为 0；正常 90 s 失败则还要累计时间、距离代价。这是可核对的回报漏洞，
并非已经证明策略主动利用了它。训练根 24/33/34/43 也存在首周期异常且无失败惩罚。

如果物理异常应排除出有效 rollout，必须显式标记、记录并按预先确定的规则补足采样；
如果它们属于任务失败，则必须按协议统一收费。不能现在这样默认为低代价的有效终局。
此修复应作为记账协议修正记录，不应作为奖励调参试验。

另外，`MAX_SKILLS=120` 的说明假定最短 option 是 2 s；实际 safety abort 可以远短于 2 s。
iteration 40 的 root 15 已执行 110 个 option，207 次机器人级 `safety_abort`。因此 cap 不是
必然不可达的纯保险；若触发，只补 -2 仍会跳过剩余物理时间代价。当前复评未触发 cap，
这项是应补测的剩余风险，不是本次已观察到的利用行为。

## 5. 信息边界：当前是集中式观测 PPO 基线

位置：`train.py:89-108`、`train.py:133-143`；`env.py:348-394`。

两个 actor 的共享 trunk 同时读取双方完整雷达、接触和局部 patch；代码还明确写着双方可使用对方视图。
没有本侧历史编码、消息码本、B-bit 预算和消息后的公共提交过程。mask 也分别使用本侧私有 patch，
函数注释中的“Public feasibility”不能使这些数据自动成为公共信息。

本次没有发现物体真值坐标被直接拼进 760 维策略输入；物体距离用于奖励和离线评测是合理的。
但“没有读物体真值”与“满足分散式信息边界”是两件事。当前实现适合标注为
**完整共享局部观测的集中式 PPO 可学性诊断**，不能作为 B=0 或有限通信的部署策略，
也不是 exact `O_central`。

如果此轮仅验证 option 组合能否学习，可以保留该集中式基线；若用于研究主方法，则必须另行落实
本侧输入与消息边界。当前文件中也没有世界模型预测或基于模型的未来分支搜索，不能据此声称
世界模型与有限消息规划框架已完成验证。

## 6. 动作编码：实现可用，因果结论需要降级

位置：`train.py:39-51`、`train.py:162-171`、`train.py:347-357`。

当前确实恢复了每机器人全部 28 个动作；采样、PPO log-prob、训练和贪心评估的索引
`type*4+scale` 相互一致。当前 mask 对一个 type 的四个尺度相同，因此 type-only mask 在现有
菜单上没有发现错选被禁尺度的问题；将来若各尺度可行性不同，必须重做条件 mask。

不过“7+4 探索”只是减少输出 logits、共享统计结构，不是把实际动作数从 28 降成 11。
无 mask 且均匀时，每个动作仍是 `1/7 × 1/4 = 1/28`，联合动作仍为 `1/784`；
最大熵也仍是 `log(7)+log(4)=log(28)`。因子化是否有样本效率收益必须实测。

此外当前使用 `P(type|o)P(scale|o)`，scale 不以已选 type 为条件。同一观测下不能独立控制
“长推”和“短等待”的随机概率而不引入交叉组合。全部确定性动作仍可表达，受限的是随机策略分布。
条件尺度头可作为后续单变量对照，不是本次已证明必须修改的缺陷。

报告中“28 路不能学是因为探索太贵”只能算假说：编码、BC、熵系数、场景和课程同时变过。
“如果远距还失败，说明因子化解决探索但未解决规划”同样不能成立；需要分别验证探索覆盖和技能序列。

历史措辞也应更精确：本机 run1 的 history 在部分评估点有 `1/16`，run3 前三个评估点也有
`1/16`。run1 曾有 wrapper 有效性问题，不能拿它约 515 轮末值为 0 作为干净的平坦头证据。
这些零星成功也不能反过来证明平坦头有效。

## 7. 基线、分布和评测复现

### 7.1 固定启发式入口与 run9 原先不在同一面板

`fixed_policies.py:294` 用 `layout.build_split('L0','test')`，arena 为 3.2×2.2；
run9 用 `sample_scenes(2,16,'test')`，arena 为 4×3。
因此旧固定策略的 return 和成功例不能直接解释当前 RL 的奖励尺度及可达性。

本次已在 **run9 相同的 16 根** 上重跑合法启发式和 WAIT，未改奖励或阈值：

| 策略 | success | mean return | mean progress | 时间耗尽 / 穿透异常 |
|---|---:|---:|---:|---:|
| WAIT 2 s | 0/16 | -5.847746 | +0.000085 | 13 / 3 |
| 合法目标启发式 | 0/16 | -6.386136 | -0.260062 | 13 / 3 |
| iteration 40 PPO 快照 | 0/16 | 见逐根记录 | -0.308374 | 13 / 3 |

所以“启发式同根 0/16”现在得到本次复核支持，但它不证明示范覆盖充分，也不能证明奖励没有
信用分配问题。正回报成功路径存在只说明成功值得做，不说明采样能到达或梯度能找到。

### 7.2 “uniform 全随机”不是“距离分层等概率”

生成器独立均匀采样中心坐标，欧氏目标距离不服从均匀分布。默认训练也不是每次新采一个场景：
`train.py:265,327` 先固定 64 根，再从中有放回采样；退火关闭后不刷新面板。

本次按预先写明的描述区间统计：

| 面板 | 近距 <0.6 m | 中距 0.6–1.2 m | 远距 >1.2 m |
|---|---:|---:|---:|
| 64 train roots | 8 | 27 | 29 |
| 16 test roots | 4 | 5 | 7 |

训练面板还有一根 `simple_train_0063` 初始距离约 0.0897 m，已在成功距离阈值内，WAIT 也可满足
速度与保持条件。这不是偷偷改口径，但需要单报，不能把训练 success 的每次出现都解释为学会推动。

持续查看这 16 根、据此调整方法后，它们承担的是验证集职责。独立最终留出集尚未由此建立；
峰值 3/16 不适合当作稳定的最终测试成绩。可以保留相同 uniform 任务分布，增加独立根和种子，
同时报告近中远分层，不需要因此引入课程。

### 7.3 checkpoint 与元信息不足

`train.py:283-287` 只保存权重、obs mean/var、少量结构与 seed：没有 optimizer、obs count、
Python/NumPy/Torch RNG、完整 CLI、git revision、场景清单/哈希和明确行为版本；CLI 没有 resume。
这支持推理加载，不支持精确恢复训练。

`checkpoint.pt` 每次评估覆盖，`policy.pt` 是最后权重；没有持久保存 best 或逐评估 checkpoint。
`save()` 在 `history.append(entry)` 之前，所以 iteration 40 权重内部 history 只到 39。
仅凭 history 不能重放峰值时的策略或核查当时成功根。`ENV_VERSION` 还停留在 v4.1，
不能充当 v4.6 行为兼容性的依据。

## 8. 逐段记录已经揭示的失败形态

以下仅属于保存的 iteration 40 快照，不外推为 run9 最终结论：

- root 1（初始 0.667 m）：连续 5 次双方快推 1 m，物体不动；一侧反复 `safety_abort`，
  另一侧 `option_budget` / 最后 `joint_window_end`。这里不是远距缺少多段换面的问题。
- root 3（初始 1.254 m）：第一段推 1 m + 队友换面后距离降到 0.342 m，第二段降到 0.295 m；
  随后出现 `patch_lost`、`blocked_no_progress`，约 10.65 s 后到 90 s 没有进一步接近。
  这是“有长推但末端恢复失败”的明确样本，不能概括为“没有远距推进”。
- root 6（初始 1.910 m）：首次推后更远，之后换面与长 WAIT 反复出现，最终 1.953 m。
- root 15（初始 0.711 m）：110 个 option、207 次机器人级 `safety_abort`，显示存在反复无效请求。

下一步最有价值的是系统记录 option 动作、实际距离变化、耗时、退出原因、接触和合法观测下的面变化，
区分初始非法、接触/协同不匹配、推错方向、换面失败、末端纠偏和正常预算耗尽。
物体真值、姿态变化可以进入离线审查记录，不能回填 actor 输入。

## 9. 测试核查与清理后的实际状态

本次实际执行：

```bash
.venv/bin/python -m pytest tests/test_push_v46.py tests/test_option_testbench.py \
  tests/test_random_start.py tests/test_random_start_anchor.py tests/test_random_start_rl.py -q
.venv/bin/python -m pytest -m 'not slow' -q
```

结果：聚焦集 40 passed；全量非 slow 为 **508 passed、10 failed、37 skipped**。
全量失败全部是缺失文件或失效引用：

- 8 项文档检查：世界模型设计文档的断图链、6 张 grid 图，以及缺失的
  `docs/CoImagine_2D_最终结论_v1.md`；该结论文档在本次清理的提交范围内被删除。
- 1 项 grid 信息边界测试：缺 `outputs/grid/world_model/belief.npz`，未运行到边界断言。
- 1 项 observation-terminal 链路测试：缺 `terminal_observations_train.jsonl`。

因此当前 checkout 不能沿用“全量只有 1 个已知失败”的说明；不能把这些缺文件说成算法断言失败，
也不能以工作树干净代替验证通过。应恢复仍被维护的文档依赖，或给历史产物定义明确的生成/归档契约。
不要通过简单跳过信息边界测试来声称已覆盖。

特别是 `tests/test_random_start_rl.py:14` 导入的是
`experiments/random_start_reposition_v1/rl.py`，不是当前 `option_rl_v45/train.py`；
全 tests 也没有引用当前训练器的 `returns_and_advantages` / `RunningMeanStd` / `type_mask`。
40 passed 因而不能覆盖本次找到的训练核心问题。

本次排除的初查疑点：RunningMeanStd 方差合并公式正确；不同大小批次与直接合并计算核对，
均值、方差最大误差分别约 `1.3e-15`、`1.7e-14`。不应将其列为训练失败原因。
终止处 returns accumulator 的重置也已在当前实现中存在，不能继续沿用旧“跨 episode 泄漏”的诊断。

## 10. 给下一轮执行的优先顺序

1. 修复 gamma 与统一记账协议；覆盖当前训练器的终止、不同时长、分段等价性和异常终局测试。
2. 修复 simple 场景合法性，冻结新版本面板；保存旧结果和非法根审计，不静默筛掉失败。
3. 同一入口加载固定面板，重跑 WAIT、合法启发式和 PPO；逐根保存上述分段指标及真实耗时。
4. 在相同 BC、熵系数、种子、物理交互预算和 uniform 场景下，比较 flat28 与 factorised28。
   6 动作策略作为动作受限的诊断，不作为证明因子化优势的唯一对照。
5. 保留训练/验证/最终留出职责，保存 best 与 last checkpoint、完整配置和版本信息。
6. 基础问题排除后，再决定选面监督、历史编码、条件尺度头或世界模型。当前数据不支持直接
   把失败交给世界模型，也不支持把多段任务规划无限塞入单个 PushSkill。

对报告六个开放问题的当前意见：先做上述失败分类，再讨论远距层级；选面监督目前不是必选项；
信用分配先纠正目标一致性，不增加分段 success 或其他新奖励；立即按距离/异常类型分层评测；
同侧协同可作为已验证能力范围，但保留相邻侧负例；世界模型接口可保留合法观测历史、联合动作及尺度、
实际时长、接触/技能退出反馈和不确定的局部面线索，隐藏模式/真实 face ID 只作离线标签。

## 11. 原始证据与最小复现

本机审查产物：`outputs/option_rl_v45/audit_20261027/`（被 gitignore，不随本报告提交）：

- `review_snapshot.json`：10:10:05 的 live history 与测试计数快照。
- `minimal_checks.json`：64+16 原始根的几何检查、首个 WAIT 和合成分段例子。
- `real_wait_segmentation.json`：真实 90 s 两种 WAIT 分段的回报/target。
- `checkpoint_snapshot.pt`、`checkpoint_eval.json`：iteration 40 权重与全部 16 根逐段轨迹。
- `fixed_policy_same_roots.json`：同面板 WAIT 与合法启发式结果。
- `pytest_full.txt`：完整失败堆栈。

checkpoint 快照 SHA256：`712beac16be3c76701888583b6492483a3030a8b1e31fad2f3ad292ccd139083`。
该快照的内置 history 最后轮次为 39，原因是前述保存顺序；对应 iteration 40 的评估。

在原代码版本上，可用以下短诊断复现主要的训练记账和初始异常问题，不启动训练：

```bash
.venv/bin/python - <<'PY'
import numpy as np
from experiments.option_rl_v45 import scenes, train as T

for action in (20, 23):
    env = T.E.RandomStartEnv(scenes.sample_scenes(2, 1, 'test')[0])
    steps = []
    while env.termination is None:
        b = env.joint_step(action, action, record=False)
        steps.append(T.Step(
            obs=np.zeros(1), a0=action, a1=action, logp=0, value=0,
            reward=b['reward']['total'], discount=T.GAMMA_STEP ** b['periods'],
            mask0=np.ones(7, bool), mask1=np.ones(7, bool),
            terminal=b['termination'] is not None))
    print(T.spec.ROBOT_ACTIONS[action].label, env.reward_total,
          float(T.returns_and_advantages(steps)[0][0]))

for index in (5, 11, 12):
    instance = scenes.sample_scenes(2, 16, 'test')[index]
    env = T.E.RandomStartEnv(instance)
    b = env.joint_step(20, 20, record=False)
    print(instance.name, b['periods'], b['termination'], b['reward'])
PY
```

若检查新修复版本，应保留这些反例并用新的预期验证，而不能继续预期旧数值。
