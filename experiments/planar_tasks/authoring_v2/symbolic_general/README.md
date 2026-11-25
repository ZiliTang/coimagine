# 可抽取 CAP 策略与效果预测闭环

这个目录实现可检查的规则接口。作者提供字面量 `SYMBOLIC_POLICY`，其选择条件、评分、派生表达式和效果承诺都能直接抽取。宿主提供基础量、单位、命令、执行条件与写入/保持集合。运行器和检查器共享表达式及选择顺序。

依赖：仓库虚拟环境，加 `pip install -e '.[symbolic]'`。符号求解器版本为 `z3-solver==5.4.1.0`。新作者同时读取上级 `README.md`、`experiment_api.md`、`checked_policy_guide.md` / 示例、公共 option guide，以及本目录 `author_guide.md`。

## 接口

```python
from experiments.planar_tasks.authoring_v2.symbolic_general.core import Policy

policy = Policy(program, host_schema)
chosen = policy.select(current_primitive_values)
report = policy.check(
    current_intervals,
    chosen['option'],
    [dict(updates=joint_post_intervals, path_safe=True)],
    quality='empirical',
)
affected = policy.dependencies(['changed_primitive_or_derived_symbol'])
```

`check` 使用预测的下一边界元表达式重新计算实际胜出的规则，然后检查该规则的执行条件。它不以另一个可执行动作的存在代替真实选择。评分相同按规则声明次序决定。模式之间用并集保留关联；模式内部是区间包络。被动作写入但缺少预测的基础量变为未知，不会沿用原值。

依赖分析支持基础量变化与派生定义修改。`rules` 是直接或传递的数据依赖；`selection_competitors` 提醒调用方：修改一个规则的启用条件或评分，也可能改变其他竞争规则是否胜出。程序和宿主结构均有独立摘要，不能把旧程序的证明用于新程序。

`proved` 只表示给定宿主模型内的结论；经验集合通过检查为 `conditional`。求解超时、无模型支持、过程安全依据不足保留为 `unknown`。模型覆盖真实后果的概率与获准子集的实测违约另行检验。

## 当前平面实例

- `frontend.py` 从公开历史的 72 条机体坐标射线、反馈和已观察静态点生成语义摘要。箱体拟合在可见点的局部坐标系完成。
- `scaffold.py` 生成规范 CAP 源码，并核对作者仅修改字面量程序。PUSH 为 6/12 ticks，独立绕行为 12 ticks。作者最终使用哪些档位由规则决定。
- `prediction.py` 预测公开摘要的联合后果：连续增量和离散后值；拟合与校准按房间分离。房间 ID 只用于均衡和溯源。
- `online.py` 从公开观测构造多个名义几何假设，并检查与当前测距/自由空间是否相容。对象速度、控制器冷启动和未观测空间的完备性没有硬保证。
- `simulation.py` 执行真实命令，另用测台真值逐物理 tick 核对效果。真值不进入运行期选择器。

正式研究使用 `tools.cap_symbolic_general_runtime.Controller`。已有模型反例尚未排除时，不能用少数名义成功的在线分支覆盖它。模型支持不足但没有保留的反例时，在线分支与原有预测模式合并检查。停止保留为动作，不能算作有效覆盖或整局成功。

`permit:*` 是宿主证据获取的摘要，不是物理状态。每次真实边界先重置，再按模型和在线试算更新；最后仍由同一份 CAP 规则选择动作。宿主另证明任意非停止权限组合下，实际选中规则满足启动条件。该性质不承诺动作能完成。

## 当前研究命令与证据

入口 `tools.cap_symbolic_general` 提供作者的 `build/test/trace/probe/fit/diagnose`，使用独立时钟和物理预算。作者交卷经过 `tools.cap_policy_check submit/verify`，并冻结 `authoring_v2.RoomCampaign`。

```bash
.venv/bin/python -m tools.cap_symbolic_general_study freeze-models
.venv/bin/python -m tools.cap_symbolic_general_evaluate freeze --policy <checked_submission/policy.py>
.venv/bin/python -m tools.cap_symbolic_general_evaluate prepare --workers 4
.venv/bin/python -m tools.cap_symbolic_general_evaluate paired --workers 8
.venv/bin/python -m tools.cap_symbolic_general_evaluate tasks --workers 12
.venv/bin/python -m tools.cap_symbolic_general_evaluate summarize
```

默认研究目录是 `outputs/semantic_rl/cap_symbolic_general_v1`；已有冻结结果不会被默默覆盖。`freeze-models` 从初始模型开始，用上一版实际运行新的开发房，在不确定/停止边界分叉，加入数据并独立校准，再冻结下一版。作者规则不变。

确认集只在全部冻结后执行。固定前缀和候选逐版本配对；**后继也经过对应版本的真实放行器**，随后实际执行选中的后继来测量完成能力。K 同时计入根边界和这些实际后继决策的在线分支及决策边界。完整整局运行在每一边界重新决策，另报成功、接触、停止、成本及在线开销。

候选实验包含宿主提供的替代档位；它衡量候选放行与接续，并不表示冻结策略在整局中实际选过每个候选。整局实际选择见逐房间轨迹。当前样本属于固定对角起点布局族，不能据此声称任意新房间的泛化或普遍物理安全。

源快照、模型摘要、逐例结果、原始回放、程序修改依赖和仿真账目均保留；作者时期源码快照与正式评测宿主修订分开保存。框架针对性测试位于 `tests/test_symbolic_general*.py`。
