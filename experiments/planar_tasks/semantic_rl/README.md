# CAP 引导的语义 Option RL：工程管线 v1

## 2026-11-15：新作者默认增量语义窗口

默认 `feature_runtime='typed_stream'` / `SEMANTIC_API='typed-stream-v1'`。
每条新边界只调用一次update_memory和prepare_context，缓存H个状态语义向量；
每个候选仍由批量原生内核求动作相关特征。固定容量的语义历史与候选语义一起
进入浅层评分器；几何残差使用紧凑实体摘要。运行时只留一条不可变原始事件，
完整公开轨迹继续落盘，供新语义版本离线重提取。

支持显式状态快照/恢复、broker重启、episode重置、漏步拒绝、语义协议与PPO批次
版本校验。新状态槽预留32，候选槽默认64；H默认8。新协议改变表示与记忆范围，
需要重新初始化/拟合，不能把旧checkpoint修改SHA后继续用。
见 [STREAM_SEMANTICS.md](STREAM_SEMANTICS.md) 与中性 [stream_example.py](stream_example.py)。
旧typed_batch/cached_fork/legacy继续按原配置复现；执行尺度协议独立、不变。

## 2026-11-15：此前的带类型批量语义

此前版本默认 `feature_runtime='typed_batch'`，现保留供显式复现。作者提交一份
Cython纯Python风格源码：历史记忆更新、一次性 `prepare_context`，再由带类型的
`evaluate_candidates` 输出候选特征矩阵。源码标记 `SEMANTIC_API='typed-batch-v1'`
也可由独立Features/rollout及并行采集入口识别。历史campaign保留原配置；旧见证
入口显式cached_fork。执行时长协议仍独立选择，不随源码格式改变。

接口、类型、工具链安装及隔离边界见 [TYPED_SEMANTICS.md](TYPED_SEMANTICS.md)。
默认知识包包含此指南与中性 [typed_example.py](typed_example.py)。宿主负责隔离
编译、类型声明降低与原生参数校验；作者不写C/编译脚本。

固定16任务完整采集20.86→18.74s，语义链路88.2→70.4ms/决策；32任务完整同值，
115项回归通过。首次编译另计，原模型只在等价迁移诊断中改源版本标签，未重新训练。
详见[验收记录](../../../docs/agent_exchange/reviews/Typed_Batch_Semantics_20261115.md)。
尚未启动新的LLM作者实验，原PPO保持停止，新duration清障示范仍待补齐。

## 2026-11-15：全链路第二轮优化

快速语义运行时升级为 `forked-window-feature-worker-v3`，仍保留每窗口记忆重建、
每查询隔离与独立输入。新增有界子进程GC优化、内部内容检查加速、采样分布复用、
数组输入转换、同步scope接触缓存及Pymunk 6.9轻量射线结果；不改扫描/物理/动作协议。
未知Pymunk ABI回退公开查询。并行采集保留全量记录，gzip使用level 1，磁盘约增33%。

固定16任务含落盘26.09→20.88s；完整32任务35064ticks/32.99s，约1063tick/s。
32条完整特征/概率/动作/观测/回放同值；99项回归及duration复现通过。只是采集吞吐，
不含PPO更新；原训练保持停止。清障的新协议成功示范仍未补齐。
详见[验收报告](../../../docs/agent_exchange/reviews/Semantic_RL_Chain_Optimization_20261115.md)。

## 2026-11-15：特征吞吐与动作尺度协议

新并行驱动默认 `feature_runtime=cached_fork`：合法历史按内容缓存，隔离的
上下文进程每个64条窗口重建记忆，四级候选在独立fork中评分；保留bubblewrap
与输入副本，不把作者代码放入仿真进程。时间/文件/反射等运行时依赖不属于
缓存函数契约；不兼容源码须显式选择legacy。详见作者指南。`Features`的低层
默认仍为legacy，以保留历史调用行为；`Features.close()`关闭持有的隔离进程。

同tick公共传感器复用、事件检测轻量反馈、原格式历史hash缓存、C级数据复制
也已接入快速采样。16路32条旧任务的完整对照约965tick/s；另外直接调用正式
collect()、含完整历史/回放/PPO批次gzip落盘与spawn，实测915tick/s（35064ticks/38.30s）。
32条特征/概率/动作/观测/逐tick回放均相同。历史V2约5000tick/s，尚未追平；
硬件/任务不同，且当前数字不含PPO与后续审计，不能承诺长训始终同速。

尺度契约见 [ACTION_SCALE.md](ACTION_SCALE.md)。新实验推荐显式
`execution_mode='duration'`；旧默认progress保留兼容，不静默改旧模型含义。

```python
from experiments.planar_tasks.authoring_v2.semantic_rl import SemanticRLCampaign
campaign = SemanticRLCampaign.create(
    NEW_PATH, fixture_profile='stage1-wall-anchor-v2', execution_mode='duration')
```

生成的 `stage1_execution_protocol.json` 和尺度指南默认提供给作者。
`duration` 是选择动作及1..40tick时间上限、正常阶段内部继续、异常返回。
候选尺度仍是时间，尚未增加可选择的位移或角度。模型、特征、示范与PPO
携带并核对同一协议；旧未标注权重不能直接进入新duration执行器。
扰动管线继承父实验的事件协议，并行驱动的 `--execution-mode` 必须与初始化匹配。

本次原CAP在新协议下自由/贴墙成功、清障失败，不能跳过三根恢复门槛直接长训。
两个成功根的39历史完成200epoch监督及保存加载回放验证；它只是接口验证，
不构成完整课程或泛化结论。原长训已按用户要求停止，本次没有重启正式PPO。

以下为v1及各历史增量说明；其中progress特指legacy-time-cap-events-v1。


这是 [v3.1 方法](../../../docs/research/Semantic_Bootstrapped_Option_RL_v3_1_20261114.md)的独立实现入口。旧动作语义实验和冻结文件不修改。不调用 LLM/API，不默认启动后台训练。

冻结语义的扰动先导入口为 `authoring_v2.semantic_rl_perturbation`。它预先生成互不重合的
12个训练/12个测试初态，执行24轮训练，再封存最后模型并对CAP、原始监督模型和RL模型进行配对评估，
另外保留3个标准根回归。训练温度固定为2，trace保存温度，PPO按相同条件分布重算logp；
默认温度1以及旧checkpoint/trace兼容。评估采用确定性选择，不按测试表现选模型。
参数、样本规模和截止规则见该模块的protocol；仅显式运行命令才会启动实验。

## 开发场景版本

`authoring_v2.semantic_rl.SemanticRLCampaign.create(path, fixture_profile='stage1-wall-anchor-v2')`
显式选择已知成功几何的课程版本：自由输运、清障输运不变，贴墙任务采用
`planar_wall_anchor` 的横放1.0×0.25m矩形、内部墙和1cm间隙。初态、布局和源码随新campaign封存；
`scene_id` 只由宿主解析，部署策略仍只看合法观测。省略参数保留历史 `stage1-v1` 的窄边贴外墙布局，
不把已有实验静默改成新课程。

可用 `python -m experiments.planar_tasks.authoring_v2.wall_anchor_check --out NEW_DIRECTORY --witness WITNESS_JSON`
验证所提供的旧物理见证：一次固定时长重放、一次按公共tick维持相同时序的进度模式重放，
每次最多400ticks，合计最多800ticks。该命令不调用作者、不拟合或训练；它验证技能/场景兼容性，
不是新的CAP策略发现或泛化实验。测台按0.2s粒度执行的回放若另行生成，必须单独计费。

## 动作与网络

策略按 `target_id → a0 → a1 → ticks` 条件分解；选完后共同执行。每个前缀保留全部可合法补全的选择。动作全集与现有 `contract.action_space()` 相同：当前可见实体的公共配对、目标无关 MOVE/BRAKE 配对、1..40 ticks。没有前48候选截断。公共菜单本身仍需提供配对关系，分解减少的是神经评分数量，不消除菜单构建成本。

现有执行接口只支持两个机器人共享操作目标；本实现没有新增分别操作两个物体的能力。全支持仅相对于这个接口。时长是进度模式下的执行上限；允许1tick，没有通过硬掩码制造长动作偏好。每一级贪心不等于联合全局最优。

共享两层32单元 tanh MLP 对每个前缀候选评分，输入为数值语义、建议匹配、已选择字段标记和动作编码。原程序分支不作为硬路由。M 条件增加单独几何残差 MLP，约束条件候选间的中心化残差；S/E/M 都可运行。当前几何通道是已有固定历史/几何摘要，不是学习式历史编码器。critic 独立，使用合法语义与几何摘要。标准化只在监督训练样本上拟合，之后冻结，缺失标记保留。

联合 log-prob 是四个条件 log-prob 之和；PPO 更新前检查与采样时重算误差。监督目标为完整示范动作的负对数概率。暂未支持同一历史多个正动作的集合损失，也未自动合并多个作者模块：第一版输入一份已组织好的扁平语义源码，可在其中包含多个 CAP 的信号。

## 语义作者接口

复用 `action_semantics` 的 `NAMES / initial_memory / update_memory / propose_actions / semantic_features` 和进程隔离。推荐额外实现：

```python
def prefix_semantic_features(memory, observation, prefix, proposals):
    # prefix 只含已经选择的字段；返回与 NAMES 对应的数值/None。
    # 对完整动作也必须有定义。
    ...
```

不提供此函数时，使用确定性的合法完整动作作为前缀代表，调用旧 `semantic_features`。这只是兼容代理，不代表对所有后续动作求期望或最优值。具体代表保存在每个 panel，不能把这种近似归因于条件分解本身。新作者应直接定义前缀语义，尤其说明尚未选择的协同动作如何表达未知。

所有特征仍只见合法观测和最多64个实际边界的历史；没有完整仿真状态。几何真值仅在回放中供人查看。实体绑定信息保留，不使用实体编号作为固定类别。前缀更新不推进专家记忆；语义记忆按实际历史重建。

## 执行、奖励与更新

`Environment(mode='time')` 调用原始命令适配器，保证旧回放不变。`progress` 在控制器逐tick执行过程中检查：

- 新目标切换时先制动，返回明确 `target_handoff` 边界；后续请求仍需完成原绑定协议。
- 稳定接触、接触丢失、公开受阻反馈、控制器结束。
- PUSH 在接触且方向有效时累计机器人沿指令方向的位移；MOVE 使用自身沿指令方向的位移。不是物体真值位移。
- 旋转和换位暂用控制器结束/执行上限，不声称已接入完整的旋转进度协议。
- 最多执行请求的1..40tick，采样块耗尽可提前终止本次收集。相同动作连续请求不会反复产生同一个稳定接触事件。

参考 v16/v21 的 gamma=1 与按真实时间衰减的 GAE：`lambda=exp(-periods*0.05/10)`。新训练目标采用现有 planar 任务成本：成功为实际秒数，失败为 `2*time_budget`；不搬用旧单物体 RL 的距离 shaping，也不给“使用/偏离专家”奖励。每tick成本0.05秒，失败补齐到固定失败成本。仅收集预算结束时标记 `incomplete` 并 bootstrap，不伪装成终止或成功。物理异常保留记录并排除 PPO 更新。

残差系数默认0.05；非推荐动作 penalty 为0，尚未实现。条件分解下仅在末级加惩罚不等价于完整联合动作的全局偏置，所以没有悄悄采用这种近似。动作通过随机分类采样探索，未实现熵奖励和额外专家探索混合。

固定语义版本内支持监督热启动和多轮短收集/PPO。checkpoint 保存网络、标准化及配置，另存优化器和 RNG；提供网络加载，但没有自动恢复训练驱动。结构变化须重新准备数据/热启动，不能混用旧 log-prob。LLM 自动修订、并行版本切换和配对反事实执行尚未接通。

## 有界命令

使用仓库 `.venv`，避免系统 Pymunk 版本不匹配。输出目录必须全新。

```bash
# 先在目标事件协议下收集 CAP 成功见证；失败不得进入监督初始化。
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.semantic_rl.runner collect \
  --source /path/to/source.py --fixture /path/to/fixture.json \
  --out outputs/semantic_rl/collect_new --mode progress \
  --max-ticks 1800 --max-decisions 600

# 默认一轮；预算、模式和源文件应在正式实验前冻结。
PYTHONPATH=. .venv/bin/python -m experiments.planar_tasks.semantic_rl.runner train \
  --source /path/to/source.py --fixture /path/to/fixture.json \
  --demonstration outputs/semantic_rl/collect_new/demonstration.json \
  --out outputs/semantic_rl/train_new --mode progress \
  --episodes 1 --max-ticks 1800 --max-decisions 600
```

收集目录包含成功状态、公开历史和特权回放。训练目录包含输入源码、配置/源码哈希、示范来源哈希、监督曲线、初始与每轮 checkpoint、每轮轨迹、逐条件特征批次及概率、PPO 指标和累计执行预算。`Policy.load()` 可以独立恢复评分。当前为 JSON/命令行入口，尚未接入网页测台。

每次偏离保存完整联合推荐和实际动作，状态为 `not_a_preference_label`。它是待取证事件，不是“新动作更好”的自动标签。

`acceptance.py` 是有界工程验收：公开种子序列按时钟组织成可执行源码，在新事件协议下重新收集成功见证；只选前4个成功历史做30epoch初始化，再收集最多6次决策/80ticks做PPO。它不属于干净作者实验，也不构成泛化或RL优越性结果。

## 16路同步PPO与命名语义槽位（2026-11-14）

`authoring_v2.semantic_rl_parallel` 使用16个spawn进程（每进程单计算线程）。每批采样同一
checkpoint，收齐后逐episode计算GAE，随后跨episode混合mini-batch更新；不将episode尾部
与另一个episode开头相连。默认128批/2048episode、4epochs、mini-batch64，实际optimizer
steps和KL提前停止分别记录。原奖励、动作支持、控制器和事件定义不变。

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m experiments.planar_tasks.authoring_v2.semantic_rl_parallel \
  --parent outputs/semantic_rl/perturbation_pilot_20261114_01 \
  --out outputs/semantic_rl/parallel_NEW
```

这条命令会显式启动长训；默认最多6小时，每episode600秒墙钟/1800ticks/600决策。
父实验只提供train面板与原监督初始化。worker逐轨迹保存请求、原始采样概率/特征、公开历史和
人工回放；中央进程保存实际/预留计费、每批与每次optimizer step指标、权重/优化器。
异常worker的预留不退款、不重试；不对部分基础设施失败批次做PPO。
`completion`区分完整执行与全局墙钟提前结束；源码在运行中不可变更。无自动断点续训。

`Features(..., fast_embedding=True)` 在单次前缀评分内复用同实体的几何历史摘要；只重算
动作相关分量。原`embedding`及默认旧路径保留，候选特征逐值一致。每次特征求值的sandbox与
embedding墙钟分别记录；没有移除隔离或让CAP代码与仿真器同进程运行。

新模型经`schema.migrate`从旧监督模型迁移，默认64个命名语义槽位；当前24通道用40个空位。
每通道仍有数值和缺失标记，空槽编码为数值0（不是未知None）。固定动作/前缀编码位于槽位之后。
首次迁移必须使用旧source及其原NAMES；新源码加入通道时调用`migrate(model,new_names,new_sha)`，
已占用名称和位置保留、新输入权重置零、旧归一化不变。新语义可以使用train-only校准尺度，
默认mean0/std1并经原Scale裁剪。新版本必须重新创建optimizer和采样，禁止使用旧logp批次。
这些操作保留旧网络的计算映射，但**不能保证被LLM改写的同名函数仍有相同含义**；语义修订后仍须回归。
普通旧作者接口继续限制24通道；只有显式Schema路径放宽到其声明容量，最多128。不会静默改旧协议。

研究测台 `http://127.0.0.1:8767/rl` 提供训练曲线、worker状态、槽位和轨迹回放。
损失/KL/熵/裁剪/梯度等按optimizer step展示，成功率/吞吐/墙钟截断按采样批展示；
熵为采样前缀上的条件熵之和，非完整联合动作树的精确熵。历史未记录指标显示缺失。

## 内存采样与可选详细记录（parallel-v3，2026-11-15）

新 `authoring_v2.semantic_rl_parallel` 默认通过 worker 返回值直接传递 PPO 数值批次，
不再写 gzip batch 再由主进程读回。当前使用 NumPy float32 数组和进程间 pickle 传输，
尚非零拷贝共享内存。每轮保留所有条件候选及其顺序、选中索引、旧 logp/value、奖励、
实际 periods、终止和 bootstrap，PPO 多 epoch 完成后释放；不同 episode 独立计算 GAE。
每个批次仍校验源码/技能尺度/语义版本，且更新前强制重算旧概率。

默认 `--detail-trials 0 --detail-every 0`：仅保存 request、进度、summary、曲线、
模型/优化器 checkpoint 与源码/配置，**不生成 result.json.gz 或 batch.json.gz**，
也不在内存构造逐 tick 回放和整段原始观测历史。详细记录可显式开启：

- `--detail-trials 2`：仅前两个 trial 保存完整回放、公开历史、事件与详细特征批次。
- `--detail-every 100`：全局编号0、100、200……保存详细记录。
- 两项可组合，按并集选择；`--detail-every 1`表示全部，非默认。

记录选择在执行前确定，不按成功/失败事后筛选。详细记录仅作人工审计，不参与PPO输入。
测台只在回放列表列出已记录的 trial，普通 trial 显示“未记录”。训练开始、每个worker
启动/完成、每轮结束会打印 `[storage]` 日志，包含详细记录状态与 `save_seconds`。
该计时包括序列化、压缩、文件写入；不是仅统计磁盘syscall。worker计时之和不是并行批次
的额外墙钟。宿主指标为累计写入成本；一次性初始化（含源码快照/模型设置）、最终封存
（含hash核验）分别报告，避免混入热采样文件计时。summary文件自身写入成本在worker完成
日志和主进程campaign/metrics中完整计入，summary文件内的save_seconds为最后一次写入前值。
曲线保留所有更新点，文件最多约两秒刷新一次并在轮末强制刷新。

typed_stream训练采用MemoryHistory：只保留当前合法观测、增量显式memory和既有H个语义
向量；以episode内游标进行边界顺序检查，不生成完整JSON哈希链。对作者的输入仍为独立
副本，公开白名单和执行合法性检查保留。游标字段明确标记为非证据哈希，不允许导出为
旧StreamHistory快照；作者/历史复现默认仍使用原canonical历史。详细回放文件会说明历史
身份种类。少量作者JSON边界验证、候选键仍存在；不是整个系统禁用JSON。

当前无自动中途续训：未更新的内存批次在进程退出后不保留，周期模型/优化器文件照常保存。
原始collect请求缺少新batch_transport字段时继续走旧disk路径，便于显式复现；新driver只
生成memory请求。普通runner的默认仍保留详细记录；本次优化默认作用于parallel训练入口。

### 决策边界复用与可信数据复制（2026-11-15）

内存训练历史接受Environment私有入口移交的已验证公开包，不再重复投影；任意外部调用
`MemoryHistory.append()`仍执行完整白名单校验并取得独立副本。每次提交新边界都重建
`BoundaryActions`，候选验证复用只含公开信息的菜单集合，校验规则仍调用原`validate_action`。
索引绑定具体包身份、tick与active_target，不跨包复用。执行前仍使用原始公开包检查动作。
索引仅供宿主同步只读借用；对外history.events和作者函数输入均为独立副本，不允许宿主
在索引生命周期内原地修改菜单。ActionTree预建各层有序选择表，动作支持和排序不变。
全局MOVE/BRAKE菜单模板为静态不可变tuple，调用者获得独立list。

trusted_inprocess的纯数据副本使用marshal C层复制，保留update_memory、prepare_context
以及宿主之间的所有权隔离；非普通数据类型退回deepcopy。没有修改进程全局copy模块，
作者默认隔离后端也不变。仍保留JSON结果规范化和显式memory限制；这不是取消输入输出验证。
