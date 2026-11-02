# D010 执行回报：接触获取长周期诊断（incomplete）

- 状态：**incomplete**。当前 40 task **30/40**，reposition_0/2 各 **0/5**；未达到 36/40 与换面每根≥3/5。40 new、contract、最终 184 比较、20Hz 正式日志/replay 未跑齐。
- 基线：冻结 `D010_planning_20261030/baseline_source.zip`，B 复现 **30/40**。当前工作区保留接触获取实验代码，但性能未超过 B。
- 没有训练、没有默认替换、没有 D011。

## 一句结论

接触获取的阶段化/候选/撤退重试没有把 reposition 从 0/5 提升；普通 contact/approach/dual 能保持 30/40，但换面后的首次偏心接触仍没有可靠解决。D010 不能写 accepted。

## 结果

| family | 成功/总数 |
|---|---:|
| contact_0/2 | 5/5 each |
| approach_0/2 | 5/5 each |
| dual_0/2 | 5/5 each |
| reposition_0/2 | **0/5 each** |
| **total** | **30/40** |

- 当前验证：`outputs/agent_exchange/D010_20261030_phase1/eval_reverted_30_again.json`
- 旧 30/40 基线、22/40、28/40 等历史结果分别保留，不混写。

## 尝试与否定

1. 完整 `approach→standoff→align→touch→push` 阶段机：
   - 双机同时推时停在 align，dual 回归；已回退。
2. 仅非双机 PUSH 启用 align/touch：
   - contact/dual 恢复，reposition 仍失败。
3. rotation retreat：
   - 检测冻结 EKF 的持续 |omega|>.12 后撤退；reposition 仍未成功。
4. 有界切向 shift retry：
   - 撤退后沿面 shift .18m/s×.45s，最多 3 次；reposition 仍未成功。
5. support-inset candidate：
   - approach_0/2 各掉到 4/5，40 task 28/40；已回退。
6. 首次接触偏心即撤退：
   - contact_0/2、approach_0/2 掉到 3/5，40 task 22/40；已回退。

## 失败机制

- 首次接触点仍偏离可见支撑中心/物体中心；接触后立即产生转动。
- 使用可见支撑质心/内缩点并不能保证稳定接触；支撑会随视角/遮挡变化。
- 再加重试/撤退会消耗固定 PUSH 窗口，甚至破坏普通 contact/approach。
- 不是 EKF、FF 或 gate 的单点故障：dual 已恢复，contact/approach 保持。
- 当前没有证据支持“必须换 ICP”；也没有证据支持“已经找到无先验稳定接触选择”。

## 未完成/缺口

- 40 new validation（D010_validation.json）未跑；候选不满足冻结前禁止查看新集的规则，也未冻结最终版本。
- 40 contract 未评分；没有独立 scorer。
- 20Hz 正式日志 schema、pre/post 分离、独立 scorer、8 条回放、6 图：未完整交付。
- prior_removal、信息边界测试、消融、final 260 比较：未完成。
- 没有最终源码哈希绑定；当前是开发实验版。

## 产物

- `outputs/agent_exchange/D010_20261030_phase1/`
  - `progress.md`、`eval_reverted_30_again.json`、`eval_first_contact_retreat.json`、`eval_inset_single.json`、`eval_single_phase_retreat.json` 等
  - `unknown_shape_phase1.py` 历史快照
- 基线：`outputs/agent_exchange/D010_planning_20261030/baseline_source.zip`
- 当前代码：`experiments/option_rl_v19/unknown_shape.py`（单机 phase+retreat+shift 实验版；30/40，未验收）

## 建议

不要再在双机/普通 contact 上同时改接触阶段。下一步若要继续，应把 reposition 的首次接触候选作为唯一变量，做**同源码、同命令、局部实验**：记录首触点在可见支撑上的位置/边距、实际接触法向、首触后 .5s 角变化，再决定是候选评分、轻推观察还是沿面重定位。当前版本不能接正式 RL。
