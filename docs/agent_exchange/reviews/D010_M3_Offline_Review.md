# D010 M3离线复核：候选基础接入有效，控制器集成仍有缺口

2026-10-30，GPT。读取当前源码、M3报告/快照/manifest和12项测试；重跑12项全部通过。另用真实factory、controller.step、OptionSlot配合合成观测做无物理fixture；未实例化env/backend，新增物理运行0、运行算法未修改。

## 1. 已确认的进展

- 当前源码/unknown_shape_m3.py/manifest的SHA256一致。
- 候选对象、初始选择、误差接线已进入step；负误差可得到相应方向，不再是此前一直None的成员。
- ContactCandidate可跨tick保存，支持变化有显式重选分支；12项辅助函数/数据对象测试通过。
- 实际OptionSlot在SHIFT中收到BRAKE请求，合成观测序列得到switching→brake→brake，最终hold、pending清空。该抢占fixture通过。
- 没有M3物理运行，符合当前次数上界unknown的停止条件。此前30/40属于前M3源码，**不能自动绑定M3工作区**。

原件：`outputs/agent_exchange/D010_20261030_m3/gpt_review/`，含真实step/Slot fixture脚本、checks.json与源码/测试快照。
复核命令：`.venv/bin/python outputs/agent_exchange/D010_20261030_m3/gpt_review/check_controller.py`。

## 2. P1：无候选仍能沿上一次方向侧移

448–456行：et=None时先设approach，随后又无条件设retreat；旧shift_sign也未清零。
真实step的合成反例：candidate=None、shift_sign=-1（上一周期残留）、触发退让；退让后进入SHIFT，desired_velocity=(+.08,0)。
因此shift_sign_for_error(None)==0的单函数测试不能证明controller在无候选时不会盲移。

修正合同：候选无效时必须显式清除侧移命令/旧sign，不能进入候选驱动SHIFT；准备等待/观察/安全撤退与有目标的重定位应区分。实际step覆盖此分支。

## 3. P1：日志字段未按回报接入

当前_info_unknown没有candidate/q/id/误差/更新原因、own/own_v、raw contact_point_robot字段。真实step返回的info/effect中也不存在candidate与own。
恢复早返回前刷新了contact_ready，但cp坐标变换仍在早返回之后；恢复阶段contact_target因此可为None，即使原始接触存在。
所以“已增加candidate/own/contact_point日志”未在冻结源码中成立。应从同一时刻obs构造公共反馈，然后由所有分支复用，不在各状态零散补字段。

## 4. P1：接触时长双计

317–320与357–359在常规接触分支各累计一次effective_contact。
真实step单步dt=.05、contact=True，累计增量=.10s。此错误会污染进展/接触时间指标，不能拿该字段做契约评分。
每tick只从raw contact累计一次；独立scorer也必须从原始观测重算。

## 5. 候选有效性与闭环合同尚未闭合

- 目前valid/verified_time/support/margin/updates/view_ticks多是初始化字段，没有完整的逐tick再验证和更新账本。
- 保持候选只看切向投影是否在支持范围±.02m，没有验证候选到当前线的法向距离、法向关联/时间新鲜度。
  合成观测将线从y=0移至y=.02，q仍为(0,0)、valid=True、verified_time仍停在.05，而skill_time已.15。
  这不直接证明物理失败；但该q不能继续未经解释地称为“已验证的接触表面位置”。可按当前线投影/可信运动搬运，或失效后重新获取，必须明确空间候选语义。
- short span的margin_limited分支是新降级选择，应明确安全策略；辅助函数如实返回margin不等于controller已确认足够的接触边距。
- RETRY没有显式提出新q，目前仅朝同一q侧移。若意图是补足未完成对齐，应这样标注；不能当成“换了接触候选”。
- align仍.60s超时转touch，没有完成误差/相对速度稳定门槛；此项原报告已列未完成，保持未通过。

## 6. 建议处置：一次完成当前M3控制器集成，不新增M4

离线预算内集中完成：无候选分支与旧sign清理、raw obs/候选共同日志、接触时长单计、候选再验证、align稳定判据及重定位目标语义。
至少增加真实factory/step层面的事件fixture：缺候选且旧sign非零、恢复阶段仍有接触点、计时单计、观测面移动/旋转、候选重选记录、align未达标超时不强触碰。
保留本次已通过的BRAKE抢占fixture，并继续验证原模式制动握手。无需为这些纯控制逻辑测试恢复物理预算。

不要求继续为每个helper另开一轮；原M3本就包含整个候选闭环。修完再统一冻结源码/manifest/日志schema，物理结果仍标未运行。
旧attempt上界无法恢复时，不能伪造剩余预算；未来若另获明确的新物理预算，应单独从零记账、历史unknown保留，不能用新预算改写旧账。本复核不授予新次数/时间。

D010仍incomplete。M3可以记“基础对象与辅助函数已接入”，不能记“完整离线接触闭环已验证”或“30/40已保持”。
