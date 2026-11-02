# D006 控制算法审查：先减法，再校准

2026-10-30，GPT。范围：当前源码、DS 两份公式/侧移报告及现存诊断 JSON 的只读审查；未运行物理实验、未修改控制实现、未完成 D005/D006 全面验收。

## 1. 结论

不批准把 `rotation_feedback_gain=-0.15/-0.18` 直接升级为默认阻尼。它们是单个 preset 的终点偏转较小的参数，不是闭环稳定性或普适符号的证明。

当前最值得解决的是：外层同时用 heading、角速度、横向漂移移动 target，内层又受到位置速度截断、额外绝对速度阻尼、参考加速度、合力限幅、接触和投影约束。只有终点角度和世界 x 位置，无法判断是哪一段控制了响应。

下一轮唯一研究问题：**去掉外层 target 修正后，接触目标的小幅侧移能否被实际跟踪；再去掉额外切向速度阻尼，传递是否改善且无明显退化？**

方案是三个可选 profile 的有界对照，不改全局默认。保留正常接近、端点感知、同面安全投影、避障、力/加速度限制、接触门限、物理和任务判据。先不加新的补偿项，不重写 endpoint tracker，不允许推动中换边抢位。

## 2. 符号链：代码有一致之处，DS 的符号结论过强

来源：`random_start/lidar.py::_build_patch`、`effect_push.py::tangent/_update_effect/_choose_contact`。

令 `J(x,y)=(-y,x)`，法向 `n` 指向机器人所看到的物体表面，切向 `t=Jn`。因此 `cross(n,t)=+1`，`cross(t,n)=-1`。`t` 是法向逆时针 90°，只有机器人朝向与 n 对齐时才适合简称“机器人左侧”；不是固定世界左侧。

`heading_error=atan2(cross(d,n), dot(d,n))` 是“从请求方向 d 转到当前法向 n”的有符号角。连续跟踪同一面且估计可靠时，`angular_rate` 的正号对应面法向 CCW。跳面、拟合误差和暂停后的稀疏观测均可能破坏它与物体真实角速度的对应，不能无条件等同。

理想局部力矩推导（只用于审查，不往控制器输入真值）：取真实接触点相对物体质心

```
r = -a n + q t, a > 0
F_on_object = F_n n + F_t t, F_n > 0
tau_z = cross(r,F) = -q F_n - a F_t
```

如果 target 沿 +t 移动确实使接触位置 q 增加，且暂时忽略切向力变化，那么 `d tau_z/dq=-F_n`。令 `q=k*omega, k>0`，增量力矩为 `-F_n*k*omega`，是阻尼方向。此推导在物体不同面上仍成立，不需要根据世界左/右再翻一次符号。现有 ROTATE 的 `q=-rotation_sign*offset` 也与该理想关系一致。

因此不能仅凭 `gain<0` 的某条轨迹最终更接近零角度，判定 `+gain` 数学上必然是正反馈。实际系统还有切向摩擦力、目标裁剪、机器人跟踪滞后、两机耦合、旧外环和拟合法向误差；增益变化也会改变整个接触历史。DS 的 sweep 支持“此 preset 的终点偏转对 gain 敏感”，不支持“已经识别出阻尼符号”。不得将驱动机器人的命令力冒充施加在物体上的接触力。

## 3. anchor / offset / target 的含义

跟踪端点为 anchor，低端 side=0 时 `s=+1`、高端 side=1 时 `s=-1`：

```
anchor_distance = L/2 + s*offset
target = anchor + t*s*anchor_distance
       = anchor + t*(s*L/2 + offset)
```

两种 anchor 下 offset 的正号都沿 +t；这里没有额外的符号错误。等价端点切换在无噪声几何下应给出同一 target。要分别检查真实端点关联错误、部分遮挡、anchor 切换和 rate 估计；不能因为 target 移动就断言 tracker 错了。

target 是**期望接触位置/切向位置参考**，不是物理引擎指定的施力点。控制器用 `e_t=(target-own)·t`；平面接触下机器人中心到表面的半径偏移沿 n，投影到 t 为零，因此这里用中心计算切向误差本身是合理的。真实接触点由机器人位置和碰撞几何决定。

DS 所称“target 不是被控量”宜改为“target 是参考输入，实际被调节的是机器人切向位置；物体力矩是更下游的间接效果”。世界 x 差不是统一切向误差；跨 gain 的终点位置差也不能冒充单条轨迹中的侧向位移。

`SameFaceReservation.project` 主要是把 preferred target 裁到当前可见、安全且保持机器人顺序的区间，不是每次强行换成一个固定点。区间内导数为 1，触边后沿受限方向导数为 0；这可以吃掉反馈，但也可能正是在避免相撞。side 锁存不能直接视为错误，本轮保留。只有日志证实长期没有可行空间后，才另议“退出推→换位→重新分配”，不在 PUSH 中解锁穿越。

## 4. 确认存在的公式/记录差异

| 项目 | 当前源码事实 | 审查决定 |
|---|---|---|
| 控制时间 | `CONTROL_DT=0.05 s`；session 每个 0.2 s 提交执行四个控制 tick | 报告的“dt=0.2 s 控制”错误；不改变频率 |
| 两条角速度通路 | `angular_rate_gain=.10` 进入 point_offset；新 `rotation_feedback_gain` 再直接加一次 | 通路不完全等价，后者绕过 point_offset 的 rate limit；无必要证据支持同时保留 |
| 点变化率限制 | `.16 m/s` 只限制 point_offset，不限制 anchor 运动、旋转后的 t、role 或最终投影 | 不能把所有 target 运动都归因于 point_rate |
| 切向计算 | `sat(sat(k_t e_t)-c v_t)`，不是 `sat(k_t e_t-c v_t)` | 必须记录两个中间量，保留 legacy 作对照 |
| 参数 | adjacent sweep 的 preset 是 `k_t=8.8, v_t_max=.5`，默认是 `4.8,.213333333333` | 对照用同一 preset 的实际值；不混用默认，不再扫 gain |
| 额外总速限制 | ContinuousPush 在 reference.update 前把 raw 向量模长限制到 `.5 m/s` | 法向+切向同时大时会互相挤占；日志补齐，不放宽 |
| pair gate | READY/ENGAGE/APPROACH/PUSH 实际放行；RECOVER/EXIT 返回零力 | 不能按 `normal_disabled` 文本或 `force_enable=False` 推断实际被清零；只修记录，不改 gate 行为 |
| 接触 | ready 接受 gap≤.05；恢复进入有效值 .07 | 沿用，物理 contact 和 near-ready 分开；本轮不调门限 |

源码入口：`experiments/option_rl_v13/push_controller.py:101` 附近的两次截断；`__init__` 的 .5 模长限制；`experiments/option_rl_v19/system.py::gate` 与 `_row/update` 的记录不一致。

例如 `k_t=8.8,e_t=.1,v_t=.2,c=.7,vmax=.5`，当前结果是 `.36 m/s`，单次截断结果是 `.5 m/s`。这是数学上的不等价，不足以单独证明当前设计错误。

未饱和、静止 target、理想速度跟踪条件下，额外阻尼给出 `v_t=k_t e_t/(1+c)`；大误差且首级饱和时约为 `vmax/(1+c)`。它会减慢跟踪，但也可能抑制振荡。内层 `F=Kv(v_ref-v)` 已有速度反馈，所以本轮将“删除额外 c 项”作为明确可证伪的减法候选，而非声称所有串级阻尼都冗余。

对移动 target，缺少切向速度前馈的 P 控制自然有跟踪滞后。不能把非零 e_t 都当 bug；也不能立即加端点速度前馈，因为它可能放大感知跳变。先校准，再决定是否需要该项。

## 5. 具体方案及其边界

三组以同一代码、同一输入执行，仅配置不同：

- A：当前控制器，`rotation_feedback_gain=0`；保留旧 heading/rate/lateral 与 c=.7（各场景采用冻结配置）。
- B：PUSH 的 heading/rate/lateral/direct-rotation 四个 target 修正全部置零，保留 role、anchor、限速和投影；保留 c=.7。隔离“外环把参考点不停移动”的影响。
- C：B 再把 PUSH 的 `contact_tangent_damping=0`；接触阶段的侧向参考只剩 `sat(k_t e_t)`。这是推荐实现的最小校准候选。

B/C 不等于永久取消方向控制；它们是可选择的校准基线。PUSH 的目标方向能力、不同场景推进及稳定性必须一起报告。现阶段不为 C 宣称完整直线推控制器，不自动替换 RL 的默认环境。

在 B/C 中分别只给一侧一个 ±.04 m、持续 1 s 的参考偏移，经过原来的 point_rate、端点/同面投影、速度/力限制；对照自身零偏移轨迹。看请求→投影后目标→机器人切向位置→真实接触点→物体角速度各层的响应。保持左右镜像对照，避免把一个方向偶然抵消终点偏转当成稳定控制。

如果请求被投影拒绝，结果是几何不可行；若目标通过而参考速度被截断，先定位执行限幅；若速度/力已输出而相对侧移很小，才进一步考虑接触摩擦/物体携带运动；若拟合法向和物体角速度不一致，先处理估计信号，不能调阻尼符号。各项允许同时发生，不强行给所有失败贴单一原因。

物体真值、实际接触点、真实角速度只用于离线画图和误差评价，不进入 controller、mask、动态参数选择。现有 backend 几何接触记录没有真实接触冲量，本轮不以命令力代算实际力矩，也不改物理求解器。

执行细节、固定输入与停止条件见 [D007 任务书](../tasks/D007_Lateral_Transfer_Calibration.md)。本审查不替代尚欠的 D005 全 8-case 当前版本验收。
