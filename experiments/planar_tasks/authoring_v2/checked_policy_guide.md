# 可检查的 Python 规划策略 v1

适用于新的 P / 恢复研究线。作者用原有 typed-stream Python 接口编写正常策略，在同一份源码中公开推动判断。每次普通决策的正式动作取自 `prepare_context` 的第一条提案。宿主独立重建当前目标物体合法的平移 PUSH 动作对，离线逐项复算；整局部署维持原有的搜索预算。单位、动作编号、时序与公开观测沿用该轮固定的 v2 指南。

## 作者须提供的接口

1. **`POLICY_CHECK_RULES`** 是顶层 Python 字面量。列出本程序能用哪些宽度证据作出哪些结论，以及对每种同伴指令允许使用哪些运动假设。静态阶段只解析源码。候选生成使用的宽度证据也在申报范围内。
2. **`review_push(memory, observation, pair)`** 返回对这个动作对的判断。正常出招与离线检查调用同一个函数。对所有合法推动都要能回答，包括生成器跳过的动作。用 `pass`、`reject`、`unknown` 分别表示通过程序的标准、被程序否决、依据不足。用 `peer_robots` 列出判断依赖其未来位置的机器人，按需给出 `peer_predictions` 和 `widths`。多个动作时长、力度或控制变体共享动作对时，应在作者说明中写明本函数检查的时长规则；v1 的全菜单范围是两台机器人的动作编号对。
3. **`audit_policy(memory, observation, menu)`** 返回当时实际搜索过的候选 `proposed`、判断 `claim`、最终选择 `selected` 及生成候选时用到的证据。完整的 `reviews` 列表由宿主逐项调用 `review_push` 建立。作者可在普通决策中记录候选与剪枝原因，或从同样的输入重算。宿主在另一个隔离调用里用相同源码和记忆执行正常出招，与申报选择比对。

[完整可运行的 Python 示例](checked_policy_example.py) · [边界数据格式和检查实现](policy_checks.py)

一个只用可见箱面宽度，并在评估让行时将同伴静止作为一种假设的声明如下。无此类证据或预测的策略可以使用空字典。

```python
POLICY_CHECK_RULES = {
    "version": 1,
    "widths": {"visible_overlap": ["sufficient", "unknown"]},
    "peer_models": {"YIELD": [["stationary", "hypothesis"]]},
}
```

### 如何报告证据

| 字段 | 含义 |
|---|---|
| `claim` | `selected` 已选中目标平移推动；`abstain` 主动改做其他动作；`no_proposed_push` 报告考虑过的候选无通过项；`no_passing_push` 声称整个菜单都经过检查且没有通过项。最后一种在 v1 中需要实际覆盖整个菜单。 |
| `selected` | 目标推动用 `[a0, a1]`；其他行为填 `None`。时间上限、目标等从宿主实际提案中读取。 |
| `widths` | 每条证据含 `source, lower_m, upper_m, required_m, conclusion`，决策级证据另含 `pair`。不可用的端点填 `None`。共同可见宽度或观测下界足够时可以判断 `sufficient`，较小时返回 `unknown`。可信上界低于要求才支持 `insufficient`。`pruned=True` 表示该条判断实际造成剪枝。 |
| `peer_predictions` | 每条含 `robot, model, scope`，决策级记录另含 `pair`。可用模型为 `stationary, co_moving, trajectory_set, unknown`。v1 以 `scope="hypothesis"` 表示对物理后果的假设；`used_to_veto=True` 表示这项预测实际参与否决一个已考虑的推动。 |

当同一条判断用于多个地方时，让数据从实际的判断函数流出，避免另写一套专给审查用的政策。检查器能发现申报与真实第一条命令不符，也能发现搜索遗漏和不成立的区间推理。它能核对申报值的逻辑用途，证据值本身、候选生成器是否完整披露内部计算，以及对动力学的近似仍需独立观测或分叉核验。

## 作者试查及交卷

以下命令从仓库根目录执行，`boundary.json` 由宿主从开发集提供。初态可用 `{"observation": ...}`；同一历史供不同作者复用时，从 tick 0 提供普通边界序列 `{"events": [{"observation": ..., "action": null}, ...]}`，后续每项用到达该边界的上一条实际动作。对中途的已提交记忆，可用 `{"observation": ..., "memory": ..., "source_sha256": "..."}`；只能用于与该记忆匹配的程序。可额外附上宿主记录的完整 `action` 作逐项核对。含中途独立回调或待执行命令的决策，应另行验证对应执行器协议。

```bash
.venv/bin/python -m tools.cap_policy_check lint --policy policy.py
.venv/bin/python -m tools.cap_policy_check run --policy policy.py --boundary boundary.json
.venv/bin/python -m tools.cap_policy_check submit --policy policy.py --boundary boundary.json --out outputs/checked_policies/candidate01
.venv/bin/python -m tools.cap_policy_check verify --submission outputs/checked_policies/candidate01
```

可重复给出 `--boundary` 以检查多个宿主指定的输入。请用新目录收卷。在对应 campaign 执行原有 `verify_frozen` 后，整局评测使用已验收的 `policy.py`，组合登记引用同一源码哈希。涉及未确定宽度而剪枝、静止让行假设或漏掉通过项的告警随程序一起保存，供下一步决定在哪个前缀做短分叉。证据与硬结论矛盾、接口缺失和执行错误会阻止收卷。程序运行必须使用项目已有的 Cython 工具链及隔离环境。
