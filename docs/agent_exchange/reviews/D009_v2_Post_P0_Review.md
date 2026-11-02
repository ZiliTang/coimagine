# D009 v2 P0修复后复核：30/40，仍incomplete

2026-10-30，GPT。读取修复源码、before/after结果表及4条保存的dual轨迹；只做离线复核，不修改运行算法、不新增物理运行、不发布D010。

## 已独立核对的结果

- `dev_task_eval_after_p0_fixes.json`：30/40；contact/approach/dual六根各5/5，reposition两根各0/5。
- `before_p0_fixes.json`：22/40。旧的原型30/40与本次30/40是不同版本，不能混用源码/证据。
- 隔离当前类重做合成探针：同刻两个视角法向相差1e-6rad，首拍不产生rate；下一拍每条线只用一份rate；交换observer遍历顺序不再产生之前的伪角速度；同tick重复调用x/P不变。
- 当前observer历史修复在上述测试范围内有效，不等于所有关联/噪声/可观测性问题已验收。

复核脚本/结果/当下源码及数据快照：`outputs/agent_exchange/D009_v2_20261030_001/gpt_review_post_p0/`。
命令：`.venv/bin/python outputs/agent_exchange/D009_v2_20261030_001/gpt_review_post_p0/check_post_p0.py`。

## dual轨迹给出的新证据

两条p0：dec2记录clearance启动，dec3–5处于RECOVER且非零gate_pre==gate_post，dec11首次在记录中看到实际接触，最终success。
两条p4：记录中无clearance，dec7首次见接触，最终success。
两条p0在dec6仍有一次gate_pre非零、gate_post零，但此时clearance=[started=True,finished=True,failed=False]，下一记录进入READY，后来完成推进。它不再表现为持续的恢复力闭锁。

这支持恢复路径已能在实际新轨迹中工作；由于两个P0一并修复、缺旧版逐tick配对轨迹，不能分解各修复对八条回归恢复的独立贡献。

**数据精度边界**：该JSON每条33行，按dec记录，是.2s决策末采样，不是完整20Hz control tick日志。first_contact/first_gate_zero均为首次“记录到”的决策索引，不是精确物理首拍。
目前没有line关联/innovation/NIS、接触点、机器人位姿、选面/目标/FF、force owner信息，不能用它完成估计精度验收，也不能诊断reposition。

## 恢复门控还需补的契约检查

源码按clearance/recovery全局状态放行当前forces，EXIT也直接放行；尚未显式校验每侧力的来源。
在当前clearance调用链中，分离模块确实提供对应控制，因此上面的成功轨迹可接受为功能恢复证据。
后续至少补：普通PUSH在RECOVER不得推进、分离恢复可执行、接触恢复可执行、EXIT只执行制动/安全控制、完成交接时不丢必要制动力。不能单凭phase字符串声称完整安全验收。
此项不要求继续扫参数，也不据一个人工不可达状态武断宣布新P0。

## 下一步只定位换面后的接触链

冻结当前30/40版本，估计器/限幅/奖励/物理保持不动。先取`T_reposition_0_p0`、`T_reposition_2_p0`，逐20Hz记录从换面结束前1s到8s PUSH结束。

最少字段：

- 请求/执行action、phase、pause/done/exit、clearance、gate前后力与控制来源；
- own位姿/速度、当前全部合法patch的法向/支持/span/RMS、选中patch及选择理由；
- approach目标、目标切向误差、raw法向/切向、ref与FF；
- 实际接触点与径向法向、该法向与选面法向夹角、EKF x/P及接受/拒绝因子。

独立scorer可追加物体真运动/角偏，只用于复核，不回流控制。日志明确pre-control与post-physics状态，避免混合快照当同一时刻。

按首个断点分流：

1. 首次接触前就选错方向/无匹配面却fallback：先改面选择/不可用语义。
2. 面方向正确，但目标随视角漂移或接触位于支撑边缘：先改APPROACH的固定局部接触候选与安全余量；不要求知道物体中心。
3. 接触后才出现持续角偏：再设计基于实际转动响应的轻推/减速/重定位，不直接恢复追质心外环。
4. gate_pre正常但gate_post持续零：先修对应的控制权/退出链。

只改第一个得到证据支持的断点；局部修复后先检查两个reposition根，并回归原六根，再跑40条。不要同时更换滤波结构、ICP、接触策略和多个gain。
当前不预先断言必须知道中心或必须ICP，也不能把没有实现contact策略说成未知形态不可行。

## 状态

D009仍incomplete：task总门槛与每根门槛失败；contract/184/估计/信息边界/完整先验剥离缺口保留。没有替换默认、没有训练、没有D010。
本复核完成，下一位用户决定是否移交上述接触链诊断。任务/算法实现未自动启动。
