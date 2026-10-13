# local_contact_option_v1 更正与偏离记录

## E1 A.1 审计结论：未发现与任务描述不一致之处

逐条核对了窗口时长、速度继承、旧 push 的接触判定/接近/推送、方向所在坐标系、完成与制动
语义、控制周期与力上限，全部与任务书一致（见 [A_implementation_audit.md](A_implementation_audit.md)）。
旧实现、旧数据、旧 checkpoint 均未改动。

## E2 快照无法表达「引擎外的位姿/速度写入」（实现期发现并修复）

**现象。** 变体的位置扰动与残余速度最初在 `reset` 之后直接写 body，随后各臂 `restore_exact`
报 `SnapshotRestoreError: position=2.0e-03 > 1.0e-09`。

**原因。** `restore_exact` 只能重建「由自身场景出发、按 prefix 重放」得到的世界；引擎外写入的
位姿与速度不在 prefix 里，无法复现。

**修复。** 位置扰动写进 `SceneSpec.robot_poses`（属于初态），残余速度改为快照导出**之后**的
显式初态函数 `apply_initial_velocities`，四臂在同一快照上重放后以同一函数写入。既恢复了精确
重放语义，又保证各臂起始状态逐位相同（有测试）。

## E3 中止原因曾被 BRAKE/SETTLE 的判稳覆盖（实现期发现并修复）

**现象。** 把 `push_wall_limit` 设成 0.05 s 做超时测试时，退出原因仍是 `settled_observed`。

**原因。** 中止后进入 BRAKE→SETTLE，判稳成功时无条件写入 `settled_observed`，覆盖了中止标签。

**修复。** 引入 `abort_exit`，中止原因在 BRAKE/SETTLE 全程保留；此后若仍判稳只增加
`settled_after_abort` 标志。修正后测试集终止分布变为 N：897 settled / 60 contact_lost /
3 face_switch（修正前被误记为 957 settled）。**所有修正后的数字来自重跑**，未沿用旧结果。

## E4 `push_wall_limit` 配置项未被使用（实现期发现并修复）

`push_abort_trigger` 原先读模块常量而不是 `OptionConfig.push_wall_limit`，使该配置无效。
已改为读取配置，默认值不变（3.0 s）。

## E5 偏差：O 臂的记录 schema 统一（不改控制路径）

O 臂仍通过 `env.execute_joint_skill` 执行旧代码路径（有测试与直接调用逐位比对），但为了统一
指标脚本，其日志行改成与其它臂相同的字段结构（`true`/`force`/`observed`/`active_contact`/
`push_wall`）。计算方式未变：`active_contact` 是该 2 s 窗口内有接触的累计时间，`push_wall`
是窗口已过时间。

## E6 说明：P 臂在 READY 阶段可能已经推动

P 保留旧力公式原样，协议只接管阶段计时、中止与制动。旧公式进入接触后仍按其自身间隙条件切到
push 分支，因此 P 在 READY 的双 ready 保持窗口内就可能推动物体；任务书要求记录这一现象，本
轮通过 `stage_motion` 的 `ready` 分量报告，未修改旧公式来掩盖它。

## E7 未做

未训练世界模型、未收集 512k 数据、未改旧模型结构/权重、未接入学习规划器、未实现 SLAM 或物体
全局定位、未新增转矩执行器或 yaw 控制。这些按任务书属于本轮外。
