# D003：成功参考与失败窗口的接触建立/保持时序审计

2026-10-29，GPT规划v1；ready_for_handoff。用户已要求编写本任务书。
> **执行侧补记（2026-10-29 18:43）**：GPT 完成正文后额度耗尽。用户在本次会话临时授权 DeepSeek 补齐 `D003_config.json` 并执行。正文的 8 个固定样本、阈值、16 次重放与 300 s/120 s 预算未改；config 只补机器可读输入、哈希、样本路径和预算。

先读[用户说明](../briefs/D003_Why_This_Audit.md)和[固定配置](D003_config.json)。
本轮只定位断点，不训练、不改控制器/奖励/944维观测/46动作/掩码/0.8秒actor周期。
D002 B继续冻结，不实现ledger或切换成本，不并入independent停车，不自动实施修复。

## 1. 要回答的问题

同推动作被选中后，是起始状态/方向不适合、接近未形成接触、接触建立后丢失，
还是已有信息未被协调逻辑利用？必须区分这些情况，不能用engaged=0代替机制解释。
先用已知成功双推参考验证记录器，再检查固定失败窗口。
审计阈值仅标记候选断点，不是控制器新阈值，也不把相关性冒充干预因果。

## 2. 冻结样本、重放与数量

固定8个case，每个运行baseline（原环境）与instrumented（只读记录包装）各一次，共16次重放。
所有输入和SHA256见config，禁止自行换例、增加探针或按新结果挑样本。

| ID | 来源 | 固定执行 | 角色 |
|---|---|---|---|
| W_N | v14 witness dual_2 | 原commands完整重放 | 成功控制器正对照 |
| W_E | v14 witness dual_0 | 原commands完整重放 | 成功控制器正对照 |
| N_E_gap | D002 dual_0/1001/d17，t=13.6，动作5/6 | 同动作保持4.8 s | 原3.2 s未双接触、双侧接近未完成 |
| N_E_one | D002 dual_0/1011/d3，t=2.4，动作3/2 | 同动作保持4.8 s | 原3.2 s仅单侧接触 |
| N_N_far | D002 dual_2/1006/d3，t=2.4，动作10/9 | 同动作保持4.8 s | 一侧近、一侧远的准备状态 |
| N_N_direction | D002 dual_2/1010/d62，t=49.6，动作11/10 | 同动作保持4.8 s | 原3.2 s未双接触、推进方向需核查 |
| L_E | D002 dual_0/1002/d103，t=82.4，动作5/4 | 同动作保持4.8 s | 原窗口短接触后丢失，active曾触发 |
| C_N | D002 dual_2/1012/d104，t=83.2，动作25/25 | 同动作保持4.8 s | 原窗口末拍才接触，右删失 |

上述角色仅描述D002证据，不能预设D003结果。C_N必须单列，不先算作接触丢失。
后6例统一从源边界观察4.8秒=96control tick，前3.2秒复现D002，新增1.6秒专门消除部分右删失。
不得再延长；4.8秒末才发生事件仍记censored。新后缀不得用于重算D002门槛或启动B。

W_N/W_E读取 `outputs/option_rl_v14/learnability_002/logs/dual_2.json` 和 `dual_0.json`：
分别原33条5 Hz命令、26次(10,10)/(2,2)再7次(44,44)，原成功时间6.45秒。
直接使用文件的instance和commands，不重新运行按真值距离选制动的witness决策器。
这些命令原本由特权离线规则产生，显式标为控制器可解性参考，非RL成功/训练示范。
原日志各有连续64个20 Hz双接触记录；这是记录器的正对照，不是新结果。

后6例从panel原root reset，逐0.2秒重放D001 execution_samples到t_start，
再维持source动作24次joint_step。原安全、恢复、局部失败锁存与任务终止保持生效。
从完整前缀重建内部状态，不只复制位置速度，不恢复未验证的pickle快照。
局部失败照原规则处理，不重置重试；任务success/anomaly/budget立即停止，保留分母与退出原因。
6个case×2重放=12次前缀/短窗口；2个witness×2=4条参考完整重放。
不增加随机rollout，不调用actor抽样，不产生新训练数据。

## 3. 记录器实现方式已定：原方法只调用一次

新文件放 `experiments/option_rl_v17/contact_timing_audit.py`，版本 `d003-contact-timing-v1`。
禁止修改v13–v16源码和D002 stage_a.py，不复制/重写控制循环以插入新行为。
使用进程内可撤销的wrapper包裹原env/Slot方法；原函数和参数原样透传，恰调用一次。
baseline完全不装这些wrapper。每个job独立创建env，wrapper在finally恢复，无跨job状态。

按真实调用顺序记录以下阶段，每条使用相同integer control_tick及t_start/t_end：

| 阶段 | 包装位置 | 记录含义 |
|---|---|---|
| P0 | 每侧Slot.prepare(obs)入口/出口 | 已生成的本tick观测、prepare前后提交/pending及skill flags |
| P1 | env._coordinate(now, observations)入口 | 两侧prepare结束后的旧flags，协调器实际会读取的状态 |
| P2 | 同一次_coordinate出口 | ready/release/active/恢复/暂停的变化 |
| P3 | 每侧Slot.step(obs)入口/出口 | 使用哪份观测、flags变化、参考速度与返回力 |
| P4 | 底层env.env.step_control入口/出口 | 最终施加力/heading、物理推进前后状态及后接触查询 |

另包裹clearance.step的原调用，记录其是否接管及返回的两侧命令。
若clearance接管导致Slot.step未调用，P3标skipped_by_clearance，不能当作漏记录或未执行。
在P4入口补记最终skill flags/ref，可覆盖这条旁路。调用树中任何未预期的阶段缺失须标记而非填假值。

不得为了记录再次调用observe、_coordinate、skill.step、_observe_face、tracker.update或滤波器。
P0/P1/P3直接复制收到的原obs必要数值；只读field生成新dict，不能保存随后会变的对象引用。
接触后的物理状态用object_state/robot_states/backend.contacts只读查询，禁止重新生成传感器观测。
对比下一tick P0接触与上一tick P4后接触；有差异就记录来源差异，不能把两者混成同一口径。
调用_source验证所需的observe仅在baseline/instrumented同一边界各调用一次，沿用D002方式。

可包装ReferenceVelocity.update以记录输入raw、dt、更新前后值，必须原方法恰调用一次，
使用所属对象标签(skill/brake/pause_brake/clearance)关联，禁止改变限幅。
这里raw是该update实际收到的输入（可能已被上游限幅），不能称未限幅的原始目标。

## 4. 必须记录的字段

每侧：requested/committed action ID、direction/speed、pending、skill类别/稳定实例编号、
stage、ready、contact_ready、release、joint_active、joint_pair/cooperative、done/exit、pause_reason、
recovery_requested/recovery_ready、skill_time；P1/P2另记pair identity是否改变、recovery状态及次数。
实例编号用本进程按创建次序分配的整数，跨baseline比较语义，不能比较Python内存地址。

观测侧：obs的period/time、own pose/velocity、contacting_object/contact point、patch有效性/法向/距离。
控制计算侧：原info中的gap及有效性、preparation_gap_target、tangential_error、contact_target、
heading误差、切向速度、reference/raw/desired velocity、施加力、力饱和、progress_guard。
字段未产生就null并标来源/有效性，不能填0假装已经对齐。配置中的准备/接触/恢复阈值原样导出。

离线真值侧：物体/机器人pose、linear/angular velocity、后接触列表、穿透、距目标及该tick距离变化。
如果记录接触冲量需backend已有只读字段；没有就标unavailable，不修改120 Hz物理引擎去新增传感。
机器人驱动力不能叫物体接触力；力指令正常不等于力已传给物体。
法向闭合速度可离线计算：机器人中心速度减物体在所选face.point处的刚体速度，再点乘该face.normal。
注明是中心相对表面法向速度、正值表示朝面接近；face缺失/时刻不一致时null，真值不回馈控制。

## 5. 必须先通过的非干扰/正对照检查

顺序固定：先W_N、W_E各baseline/instrumented，通过后才运行后6个case各两次；最多4worker。
每case两次的命令序列、control period、termination、success必须一致；
每tick物体/机器人pose/velocity和施加力最大绝对误差≤1e-9（角度包裹差）；
旧快照为float32时与旧日志比较容许≤1e-5，不能把新baseline与instrumented也放宽为1e-5。
两次task reward误差≤1e-9。baseline也开原record=True，仅额外wrapper不同。

后6例source raw_obs≤1e-5、mask完全一致、物体/机器人pose≤1e-5、总period一致；
前64tick的轨迹与对应D002 hold的20 Hz位置/角度记录≤1e-5，窗口动作一致。
各前缀两次均record=False；instrumented wrapper只在目标审计窗口启用，避免大量无关记录。

W_N/W_E必须：复现success且终止129tick/6.45秒，末距及逐tick旧轨迹误差≤1e-5；
仪器能记录连续≥0.5秒的双侧接触，并记录至少一次joint_active激活（不要要求所有flags同拍合取）。
原轨迹与instrumented一致但无法识别激活，也停止为instrumentation_or_contract_unresolved，
不能继续把失败窗口全零解释为物理失败。若旧witness无法复现，不调整制动时刻、速度或阈值来修它。

## 6. 预先定义事件，避免同拍合取误判

- contact_onset_i：P0 observation.contacting_object从false→true；初始即true记left_censored。
- simultaneous_contact：同一个P0时刻两侧接触为真。单独报告总记录时长、最长连续段、首次时刻。
- ready_update_i：P3后contact_ready false→true；若clearance旁路则按P4入口标其阶段。
- activation：P1至P2发生joint_active false→true，双方分开记；初始化已true记left_censored。
- contact_loss_i：P0接触true→false，记录与激活的先后及相隔tick；末拍才接触记right_censored。
- stable_contact：P0连续≥10tick（0.5秒）双侧接触。只是诊断标签，不改变success_hold。
- activation_delay：从本次有效ready证据的更新tick到协调激活tick；正常一拍延迟不计故障。
  pair更换、failure或recovery使证据失效时，重建匹配关系，不拿过期ready计算延迟。

物理contact/ready更新/active激活/目标位移画成独立时间线，不继续使用单个engaged值作因果判据。
保留D002旧engaged仅用于核对前64tick数值，明确它是混合阶段合取。
分别报告前3.2秒与新增1.6秒，不把新后缀并回旧实验结论。

## 7. 判定规则：多标签、允许不确定，不强行归为一个原因

先找最早有证据的断点，然后列后续现象。责任标签必须带case/tick/字段值/源码条件。

1. `goal_opposed_command`：离线push方向与物体到目标单位向量点积≤−0.25持续≥10tick。
   只说明该段不是直接朝目标推；准备动作也可能暂时远离目标，不能独自证明全局策略错误。
   同时报告两侧局部mask、face法向匹配，区分“局部可执行”与“对目标有用”。
2. `preparation_not_ready`：存在≥20tick（1秒）的未接触区段，ready条件持续不满足。
   将原代码条件逐项重算为gap/tangent/alignment/lateral_settle/face_valid/peer_clearance阻塞表；
   原阈值来自冻结配置（例如tangent 0.025），不修改。若字段不足，记unknown，不凭肉眼判几何不可行。
3. `approach_response_suspect`：连续1秒中≥80% tick没有pending/pause/clearance/失败接管，
   inward reference≥0.02 m/s、有效间隙≥0.02 m，但间隙累计减少<0.005 m。
   同时要求face关联稳定（法向累计变化≤10度），否则优先标geometry_unstable。
   这是执行/估计问题线索，不直接判控制器bug；用真实相对运动/施加力和正对照解释。
4. `contact_not_maintained`：接触建立后<0.5秒丢失，且没有新命令、BRAKE或已知安全退出解释。
   记录单侧/双侧接触、激活前后参考/实际闭合速度、gap、恢复触发。速度突变与丢失相关不等于因果。
5. `coordination_contract_violation_candidate`：P1满足原_coordinate的激活条件且identity稳定，
   无done、换对清零、恢复路径覆盖，却在P2没有按原分支设置active；或ready更新后下一tick在
   对应条件持续成立时仍不激活。必须逐项展示原代码条件，不能只看P0接触或P3的当前ready。
   不再次调用_coordinate验证，使用记录离线核对；原有恢复/暂停行为本身不能叫bug。
6. `censored_or_unresolved`：窗口不够、信号不全、只有相关性、正对照不匹配时明确保留。

标签1可以与2/4并存，不能因方向背离就省略接触过程；标签4不意味着接触损失由握手造成。
不要求每例都得出根因，不以“类似参考但失败”单独证明执行缺陷。
最终只能提出一个最小后续改动候选（或明确目前不足以选改动），本轮不实施。

## 8. 预算、硬截止与测试

只读审计正式执行总墙钟上限300秒，含16次重放、前缀、分析、回放和写盘；实现时间另列。
聚焦测试总上限120秒，不跑全仓库pytest，不调用GPU/训练/API。
不允许额外物理探针、失败替换样本或自动重跑；探针需求用上述正式正对照解决。

必须先补独立外层supervisor：subprocess.Popen(..., start_new_session=True)启动整个审计，
300秒发SIGTERM到进程组，2秒后仍存活则SIGKILL，监督进程最后写incomplete及实际耗时。
进程退出清理最多额外2秒，不拿future.cancel()当作硬截止。
所有worker归同进程组，不自行setsid；supervisor不与worker在同一组，不误杀其他训练进程。
失败时保留原始partial JSONL/状态，不等所有future成功才写数据；完成后atomic生成汇总。

聚焦验证固定为：

- synthetic两tick：tick k接触更新ready，k+1激活且接触已丢失；必须识别激活，不能输出never_activated；
- 最后一拍才接触标right_censored；初始已接触标left_censored；新pair/recovery使旧ready失效；
- 20Hz时钟P0/P1/P2/P3/P4顺序、每原方法一次、clearance旁路不误记漏调用；
- 外层截止测试：无物理的dummy进程组含sleep worker，1秒截止、2秒强杀宽限，验证子孙进程清理。
  该测试计入120秒，不占16次物理重放。
- 实际8组baseline/instrumented非干扰验证利用正式数据，不额外采样。

任何hash/来源/非干扰失败立即停止为incomplete，交回后由GPT决定；不要现场修复并重跑正式任务。
物理异常同样保留并停止后续新job；不能自动改控制器以继续。

## 9. 交付

产物：`outputs/agent_exchange/D003_<run>/`；manifest记录所有输入/源码hash、完整命令、
实际墙钟、16次运行完成数、异常/右删失/未完成数。成功参考和历史窗口分开标注来源。
保存raw_ticks.jsonl（带P0–P4）、events.csv、case_metrics.csv、blocker_intervals.csv、
reproduction.json、8个case的逐阶段时间线和回放；baseline轨迹也保留。
图表将接触、ready更新、active、法向参考/实际速度、gap、目标位移对齐，不画混合快照。

写 `results/D003_Contact_Timing_Audit.md`，开头用普通语言回答：
“正常成功过程是什么？失败最早在哪一步分叉？现在最该改策略选择还是执行过程，证据是否够？”
注明诊断记录字段不是新的策略输入；本轮没有RL提升结果。
更新CURRENT到ready_for_review并停止，不发布D004、不解冻D002 B、不实施候选修复。
