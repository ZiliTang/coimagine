# CoImagine 策略命名与谱系登记

更新：2026-11-16。适用于研究实验、工程诊断和后续v4；诊断策略也有稳定身份。机器可读登记见[Policy_Registry.json](Policy_Registry.json)。

P＋R 研究线的组合评测见[母策略与恢复分支的组合命名](CAP_组合命名.md)。

## 命名规则

- 标准显示形式：`CAP-002.v2｜接近布局CAP·绕行修复版`。编号是身份，中文名是用途说明，不是能力保证。
- `CAP`：可执行程序教师；`SEM`：语义及建议接口集合；`SUP`：监督拟合评分器；`RL`：强化学习后的评分器。不能把监督或RL模型继续称为某次CAP。
- 家族编号保持稳定；同一局部行为开发线的源码修改使用新版本。新增可独立积累的局部行为建立新CAP家族，并登记来源。失败版本也保留；版本较大不表示表现更好。
- 当前登记选取主要快照，版本号不枚举所有调试提交，也不是整个仓库所有历史CAP的总次数。较早、不同研究线的策略以后单独补登记。
- 只有源码相同且执行协议等身份条件相同的复制才作为同一身份的别名；改变执行/观测协议必须重新登记。checkpoint哈希固定，继续训练生成新版本。
- 原实验目录和局部`s000/s002/f000`编号不改名、不重写封印。这些是定位出处的编号，不能脱离campaign单独使用。
- 中途模型使用例如 `RL-004.v2 / round_0544` 并附原checkpoint路径与哈希；它不等于该版本固定最终模型。
- 评估成绩绑定到“身份＋面板版本＋执行配置”；不能把某次34/48写进策略名称，也不能用同名掩盖跨面板比较。

## 已登记主要策略

| 稳定身份 | 显示名称 | 类型/协议 | 父项 | 已知边界 |
|---|---|---|---|---|
| CAP-001.v1 | 三场景基础CAP·初稿 | cap/progress | 本登记起点 | 自由/清障根成功；窄边贴墙未解 |
| CAP-001.v2 | 三场景基础CAP·贴墙续修 | cap/progress | CAP-001.v1 | 自由/清障根成功；原贴墙任务仍未完成 |
| CAP-001.v3 | 三场景基础CAP·可解贴墙课程 | cap/progress | CAP-001.v2 | 三个开发根成功；贴墙课程初态已变更 |
| CAP-001.v4 | 三场景基础CAP·时长协议版 | cap/duration | CAP-001.v3 | typed_stream迁移及清障停车修订后的基础教师；7训练初态+3根成功 |
| CAP-002.v1 | 接近布局CAP·西侧接近版 | cap/duration | CAP-001.v4 | 局部成功开发版，不能替代基础CAP |
| CAP-002.v2 | 接近布局CAP·绕行修复版 | cap/duration | CAP-002.v1 | 新增open03/clear01/clear02成功；并非通用过冲恢复 |
| CAP-002.v3 | 接近布局CAP·旋转探索版 | cap/duration | CAP-002.v2 | open02修复未成功；保留的失败探索版本，不是当前推荐教师 |
| SEM-001.v1 | 基础与接近布局联合语义 | semantic_bundle/duration | CAP-001.v4, CAP-002.v2 | 41候选/27状态通道；含建议接口，不单列为新的独立CAP教师 |
| SUP-001.v1 | 基础CAP监督评分器 | supervised/duration | CAP-001.v4 | 三个开发根示范初始化；非13轨迹整合模型 |
| SUP-002.v1 | 双CAP监督整合评分器 | supervised/duration | CAP-001.v4, CAP-002.v2, SEM-001.v1 | 13轨迹/407边界整合，13/13闭环恢复 |
| RL-001.v1 | 全合法动作PPO·短训 | rl/duration | SUP-001.v1 | 2048训练episode；此身份仅指固定最终模型 |
| RL-001.v2 | 全合法动作PPO·长训 | rl/duration | RL-001.v1 | 累计16384episode；固定训练10/12、既有test6/12 |
| RL-002.v1 | 时长参数PPO | rl/duration | SUP-002.v1 | 512episode；固定25例贪心行为与监督一致 |
| RL-003.v1 | 时长速度参数PPO | rl/duration | SUP-002.v1 | 512episode；本轮48test31成功，配对监督31成功 |
| RL-004.v1 | 时长速度邻向PPO·短训 | rl/duration | SUP-002.v1 | 512episode；身份不随评估面板结果变化 |
| RL-004.v2 | 时长速度邻向PPO·长训 | rl/duration | RL-004.v1 | 累计16384episode；新48test34成功，配对监督28成功 |
| SUP-002.v2 | 双CAP监督整合·旧数据等更新对照 | supervised/duration | SUP-002.v1, SEM-001.v1 | 原13轨迹；3000更新；锚点13/13、开发9/24、新留出16/24 |
| SUP-002.v3 | 双CAP监督整合·覆盖补数据版 | supervised/duration | SUP-002.v1, SEM-001.v1 | 20轨迹676边界；3000更新；锚点13/13、开发14/24、新留出14/24；不替代原基线 |
| SUP-002.v4 | 双CAP监督整合·大预算旧数据对照 | supervised/duration | SUP-002.v1, SEM-001.v1 | 13条407边界；16800更新；锚点13/13、训练95/216、新测试105/216；单seed |
| SUP-002.v5 | 双CAP监督整合·随机覆盖150示范 | supervised/duration | SUP-002.v1, SEM-001.v1 | 150条5351边界；16800更新；锚点11/13、训练134/216、新测试136/216；单seed |

## 当前常用称呼的准确映射

- “旧CAP / 基础CAP”：**CAP-001.v4**。
- “已有修复CAP”：**CAP-002.v2**；以后不再只用“修复CAP”指代。
- “监督整合模型”：**SUP-002.v1**。
- “短预算邻向RL”：**RL-004.v1**。
- “本轮长预算RL”：**RL-004.v2**。
- “前三类任务的CAP”：自由输运、贴墙释放、清障输运在当时是**一份三场景CAP的分支**，不是三份独立策略。若指11月14日三轮作者阶段，则分别对应CAP-001.v1/v2/v3。

## 接管实验写法

`handoff_entry_20261116_02` 中 s000 → s002 应写为：

**CAP-001.v4 → CAP-002.v2，切点为前缀第16个决策边界。**

策略身份与接管实验身份分开：切点、初态、前缀轨迹、预算、接收模型都属于实验配置，不因此制造新策略版本。此次只是诊断，也必须使用上述统一身份。

## 后续登记要求

新增或修订策略时记录：稳定ID、名称、父项、作者/训练方式、源码或checkpoint路径、文件SHA256、执行协议、示范来源、证据报告及能力边界。中间失败保留局部提交编号；用于比较、整合或后续接续的快照必须登记。

v4后续局部CAP开发：若修订现有接近布局程序，登记CAP-002的新版本；若形成独立行为，登记新的CAP家族。依据产物职责判断，不预先宣称新技能已经形成，也不由命名决定下一轮实验对象。

## 产物与证据索引

- **CAP-001.v1**：[s010.py](../../outputs/semantic_rl/sol_stage1_20261114_01/semantics/s010.py)；[证据](../agent_exchange/reviews/Sol_Semantic_RL_Stage1_20261114.md)。文件SHA256：`517a8fcd9a88e00e01b626e538d6f25273cbec0d67a5a38a7949fed176ab6e36`。
- **CAP-001.v2**：[s025.py](../../outputs/semantic_rl/sol_stage1_20261114_resume01/semantics/s025.py)；[证据](../agent_exchange/reviews/Sol_Semantic_RL_Stage1_Continuation_20261114.md)。文件SHA256：`9ea6d6d98ad204aa37b4dcab38ea43cdf8b472b4234fb7a55d08d237c7efce06`。
- **CAP-001.v3**：[s000.py](../../outputs/semantic_rl/sol_stage1_wall_anchor_v2_20261114_01/semantics/s000.py)；[证据](../agent_exchange/reviews/Sol_Semantic_RL_Wall_Anchor_v2_20261114.md)。文件SHA256：`b70170856d6890944f4bb74a19f7c0f8f04700b019369d62ba832a98aaf61e3d`。
- **CAP-001.v4**：[source.py](../../outputs/semantic_rl/duration_stream_continuation_parent_20261115_01/inputs/source.py)；[证据](../agent_exchange/reviews/Duration_Stream_PPO_Start_20261115.md)。文件SHA256：`be39a890540f24b731b652562ab235305dd49a5df41b11a9333d1b3ead3efe8b`。
- **CAP-002.v1**：[s001.py](../../outputs/semantic_rl/local_cap_loop_20261116_01/semantics/s001.py)；[证据](../agent_exchange/reviews/Local_CAP_Parameter_Loop_20261116.md)。文件SHA256：`fa4d201dc4321e1490ec3967e11c0fa708ae04cceff0b74c68c8b6c9e2168b27`。
- **CAP-002.v2**：[s002.py](../../outputs/semantic_rl/local_cap_loop_20261116_01/semantics/s002.py)；[证据](../agent_exchange/reviews/Local_CAP_Parameter_Loop_20261116.md)。文件SHA256：`85879b791842352add3e129ebb6f5fdaad235615c639cffa41f023ea2ca86237`。
- **CAP-002.v3**：[s003.py](../../outputs/semantic_rl/local_cap_loop_20261116_01/semantics/s003.py)；[证据](../agent_exchange/reviews/Local_CAP_Parameter_Loop_20261116.md)。文件SHA256：`bbfaf7bd58bd6850724eeb108a63bd668da8065c3917d2e5ef979334fefb1211`。
- **SEM-001.v1**：[s000.py](../../outputs/semantic_rl/local_cap_integration_20261116_01/semantics/s000.py)；[证据](../agent_exchange/reviews/Local_CAP_Parameter_Loop_20261116.md)。文件SHA256：`928559bb592d8f2ab35c6b262dfbccd90e08895f6c54789c7fda50025cdb637a`。
- **SUP-001.v1**：[initial.pt](../../outputs/semantic_rl/duration_stream_parent_20261115_01/inputs/initial.pt)；[证据](../agent_exchange/reviews/Duration_Stream_PPO_Start_20261115.md)。文件SHA256：`489a2c86633431e45c723252941e578e8426fb07d2ab937bba9dc28603967984`。
- **SUP-002.v1**：[policy.pt](../../outputs/semantic_rl/local_cap_integration_20261116_01/semantic_fits/f000/policy.pt)；[证据](../agent_exchange/reviews/Local_CAP_Parameter_Loop_20261116.md)。文件SHA256：`5bc597be25fd17572fe34431840bfb93eb31c051e648943ccadfb747c63d19e6`。
- **RL-001.v1**：[final.pt](../../outputs/semantic_rl/duration_stream_training_20261115_01/models/final.pt)；[证据](../agent_exchange/reviews/Duration_Stream_PPO_Final_20261115.md)。文件SHA256：`cae7f25a0dac772e88130482663414a307e2857b17e56668a2a1a3d9f4108a64`。
- **RL-001.v2**：[final.pt](../../outputs/semantic_rl/duration_stream_continuation_20261115_01/models/final.pt)；[证据](../agent_exchange/reviews/Duration_Stream_PPO_Final_20261115.md)。文件SHA256：`0f678f78a2f4cf05fcabb3924ffa519fd0207b698921a1438e14a9f552461233`。
- **RL-002.v1**：[final.pt](../../outputs/semantic_rl/local_cap_parameter_rl_20261116_01/models/final.pt)；[证据](../agent_exchange/reviews/Local_CAP_Parameter_Loop_20261116.md)。文件SHA256：`8b92d65dce5364605e357bd04eaccc388e408855094d38fc317d6367219f11ca`。
- **RL-003.v1**：[final.pt](../../outputs/semantic_rl/random_parameter_speed_20261116_01/models/final.pt)；[证据](../agent_exchange/reviews/Random_Parameter_RL_Result_20261116.md)。文件SHA256：`43f539c906226fb24e52caeab8a340d1aca9cb257bc59b15244afd8a8b09b3d9`。
- **RL-004.v1**：[final.pt](../../outputs/semantic_rl/random_parameter_direction_20261116_01/models/final.pt)；[证据](../agent_exchange/reviews/Random_Parameter_RL_Result_20261116.md)。文件SHA256：`977103a68295cb8ff30df370635c7cfb891f39c9a4d171a90507a6cbf48be6f1`。
- **RL-004.v2**：[final.pt](../../outputs/semantic_rl/random_parameter_direction_long_20261116_01/models/final.pt)；[证据](../agent_exchange/reviews/Bounded_Direction_Long_Final_20261116.md)。文件SHA256：`b467d62f33c8adeb683cccb5f4e0e1b79bf2cb17a5f1f4bab696f35abaa72ee7`。
- **SUP-002.v2**：[control.pt](../../outputs/semantic_rl/coverage_distillation_fit_20261116_01/models/control.pt)；[证据](../agent_exchange/reviews/CAP_Coverage_Distillation_20261116.md)。文件SHA256：`7a780c167a413dbd53238d3524ec933cdb46609140d312810a18b1bef4c6addf`。
- **SUP-002.v3**：[expanded.pt](../../outputs/semantic_rl/coverage_distillation_fit_20261116_01/models/expanded.pt)；[证据](../agent_exchange/reviews/CAP_Coverage_Distillation_20261116.md)。文件SHA256：`6d987db67c14de2f2a185fd965c5632d4dcea4853dae580598b3c24adaba77bb`。

2026-11-16覆盖补数据先导：SUP-002.v2/v3为从头拟合的等预算对照/扩充数据版本，父项表示数据与语义来源。编号增加不代表更优；当前保留SUP-002.v1为原监督基线。

- **SUP-002.v4**：[checkpoint](../../outputs/semantic_rl/large_coverage_fit_20261116_01/models/control.pt)；[证据](../agent_exchange/reviews/CAP_Large_Coverage_20261116.md)。文件SHA256：`fb4daee3c63b1dd9a5a1509011a75fba94988184d030c6abed698eb3b4303c28`。

- **SUP-002.v5**：[checkpoint](../../outputs/semantic_rl/large_coverage_fit_20261116_01/models/expanded.pt)；[证据](../agent_exchange/reviews/CAP_Large_Coverage_20261116.md)。文件SHA256：`25abc8d14b76aea4de8af503c8a7032b9df63663879b1aab10de9db1476b8421`。

随机覆盖两模型均从头拟合；v5新测试整体提高但丢失2个原锚点，不自动替换原基线SUP-002.v1。
