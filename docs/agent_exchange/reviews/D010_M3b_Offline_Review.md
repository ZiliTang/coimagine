# D010 m3b离线复核：可以准备小物理验证，不等于任务验收

2026-10-30，GPT。读取m3b源码/快照/manifest，重跑21项测试全部通过。
另用实际factory/controller.step/Slot＋合成合法观测复核：

- e_t=0、相对vt=.1m/s时保持ALIGN，settle=0、法向desired=0，旧反例不再出现。
- 两条件连续达标3个.05s tick，阶段为align→align→touch。
- 两个达标tick后速度超限，settle清零；再达标两个tick仍ALIGN。
- SHIFT收到BRAKE仍由Slot切换为hold，pending清空。
- 工作区/unknown_shape_m3b.py/manifest均为c3f515b65bc745c0e832af953f0b209bcb57f5378d6b4d1a1e3658763332728b。

原件：`outputs/agent_exchange/D010_20261030_m3b/gpt_review/`；真实控制器fixture、checks.json与源码/测试快照。新增物理0，运行代码未修改。

处置：此前阻止本轮离线集成确认的ALIGN反例和交付hash不一致已修复。当前可记**m3b scoped offline checks passed / physical unverified**，不称contact-acquisition accepted。
完整未知形态先验剥离、部署安全/估计门槛、新40/contract/260与真实闭环效果仍未完成；旧30/40不绑定m3b。

旧D010 attempt账目unknown继续保留，不能倒算旧剩余。为避免永久依赖无法重建的旧账，另准备**待用户移交生效**的独立8次/20min小验证方案。
该方案既不重算旧12h/1600，也不自动授权DS运行；用户实际移交后单独记账。只比同一四个case的冻结B与m3b F，不调算法/扩矩阵/训练。
