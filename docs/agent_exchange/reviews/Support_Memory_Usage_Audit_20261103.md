# SupportMemory 在 PUSH 控制中的实际用途

2026-11-03，GPT。用户要求先核对memory用途；只读代码和纯观测夹具探针，不修改控制/感知算法，不新增物理运行。已确认8765加载的控制器 f30f7f72…与当前源码一致，restart_required=false。

**结论：有使用，但只接入提交时的初选q/锚点；提交后的持续控制没有直接使用SupportMemory。当前也不是完整的跨reposition历史轮廓积累。**

## 实际数据流

- env级 `support_memory` 跨option存在，每control tick更新。
- 任一侧最近patch的gap≤.02时，先用上一拍EKF twist预测memory，再freeze观测融合；freeze不是停止预测。
- 无该接触估计时，`calibrate(observations)` 每拍只从双方当前patch建cluster、PCA拟合，并 `self.lines=fitted` 替换旧线。旧的、双方当前均不可见的面不会可靠保留。该函数旧docstring提到prior endpoints，实际实现没有将它们加入cluster。
- PUSH提交时，`mem_faces`传入single/pair selector和初始`_face_for_q`，可以改变支持长度、初选q和锚点。
- `_merge_memory_faces`只扩展当前已经可见且匹配/连接的线，不引入一条当前自身完全看不到的旧面。
- 提交有效期间，`track_contact_q`只循环 `visible_faces(obs)`，不接收memory；速度由`velocity_at_contact`提供，n来自匹配的当前patch。SharedMotionEstimator也没有读取SupportMemory，它有自己独立的线运动历史。
- 当前任何一侧skill/action变化导致commit重置时，会再次走初选，因此不能说“进入PUSH后memory永远不可能再读”。持续q中保留的初值也可能间接受memory影响。
- session另外读取memory.snapshot用于回放，这条显示链不代表控制链。

## 纯观测探针证明它会影响选择，但未接通跟踪

构造同一面x=2：R0当前看到y∈[1,1.4]，R1当前看到y∈[1.2,2]，均为合法拟合patch输入，不实例化env/backend。

1. 不带memory，simple selector得到q=(2,1.2)。
2. memory融合两侧当前支持后，selector得到q=(2,1.5)。说明memory不是纯显示或死参数。
3. 将这个q传给当前R0 tracker，其自身支持上界1.4，超出.05容差，返回 `support_unobserved_hold` / matched=false。
4. R1视野消失后再次calibrate，memory支持从[1,2]缩回[1,1.4]，selector回到q=(2,1.2)。

这暴露了**选点可使用的共享支持，与持续跟踪可接受的自身支持不一致**。可能出现初选利用memory得到一个点，紧接着跟踪因自身看不到该点而等待；不是此次探针宣称某条实际轨迹一定如此。

## 对下一步的意义

不能直接按“memory没有用”删掉它，它会改变初选；也不能将当前实现视为已实现绕行观测的长期价值。

在进一步优化这部分计算前，需要先明确/补齐：不可见面的保留与身份、静态校准如何合并历史而非覆盖、PUSH关联是否允许经过验证的memory支持，以及失配时的控制语义。当前仅核对，不自动实施这些行为变更。

主代码：`experiments/option_rl_v19/unknown_shape.py` 中 SupportMemory、`_merge_memory_faces`、`track_contact_q`、UnknownShapeEnvMixin.coordinate。
