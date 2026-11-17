# RL 基线登记

2026-11-09 本机已从最新完整移交包恢复 V1 冻结目录与 V2 结束产物，模型读取检查通过。包内已含 V1，无需另找下文提到的旧独立 V1 压缩包。当前模型路径、版本边界、清理记录与本机命令见 [接续记录](../agent_exchange/Local_Restore_20261109.md)。

## V1：事件进度集中式 PPO（2026-11-03）

**上一轮正式 RL 训练现明确登记为第一版基线 V1；状态 frozen。** 新的 YIELD/方向约束属于 V2，不追溯改变 V1 的动作、指标或源码。

| 项目 | V1 冻结结果 |
|---|---|
| 原运行 | `outputs/option_rl_v20/event_12h_20261103_v2/`（目录中的 v2 指第二次启动，不是研究基线版本） |
| 结束状态 | `wall_budget_reached`，monotonic 运行43202.07s，5331轮 |
| 有效训练 episode / 事件 transition | 170,592 / 6,888,474 |
| 训练异常 | 0 |
| 验证选择的 best | iteration4640，32/32，平均相对距离改善0.83358 |
| 最终策略最近完整验证 | iteration5330，29/32，改善0.72708；final权重保存于5331 |
| 已有独立 test | **旧 best@2580：26/32**；未对 final/best4640新增test |
| 信息/模型 | 集中共享、958维、分离actor/critic、双256Tanh；单训练seed |
| 适用范围 | 固定矩形分布；mask保留尺寸先验；无公平固定时间消融 |

完整登记、权重哈希与配置：[V1_event_progress_20261103.json](V1_event_progress_20261103.json)。所有成功率带有其场景和 checkpoint，不能将旧test结果绑定新best，也不能把验证峰值解释为跨形状泛化。

## 保留内容与复现方式

本地冻结目录：`outputs/rl_baselines/V1_event_progress_20261103/`，包含：

- `best.pt`（4640）、`policy.pt`（5331），均保留normalizer、优化器/RNG和训练配置；
- 完整history、manifest、result、validation/test面板与既有test；
- 第一轮、1000/2000、2580、3500、best4640、最终验证5330的逐根评估；
- **686个Python/协议表等源码文件**的`frozen_code/`，覆盖导入依赖，不仅是36项训练顶层hash；
- `SHA256.json`、实测依赖版本、被删除pilot/首次启动的少量配置元信息。

原始配置中的36项源码在冻结前逐项核对，无漂移。之后当前工作区中的v19共享控制器已被V2改动，**即使V1权重仍可读取，也不能用新工作区控制器冒充V1复现**。

在仓库根目录，可以只读验证冻结模型能够读取；不采样、不训练：

```bash
cd outputs/rl_baselines/V1_event_progress_20261103/frozen_code
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" ../../../../.venv/bin/python - <<'PY'
from pathlib import Path
from experiments.option_rl_v20.long_run import load_state
from experiments.option_rl_v19 import unknown_shape
print('controller source:', unknown_shape.__file__)
for name in ('best.pt', 'policy.pt'):
    policy, normalizer, optimizers, generator, state = load_state(Path('..') / name, lr=3e-4)
    print(name, state['iteration'], policy.obs_dim, state['best'])
PY
```

以上命令从仓库根目录进入 frozen_code 后，使用本仓库的虚拟环境。未来需要重新评测时，从冻结源码加载manifest中的固定面板并另写输出；此次没有新增V1策略评测或训练授权。

最终V1迁移包：`outputs/rl_baselines/V1_event_progress_20261103.tar.gz`，大小与SHA见[V1_archive.json](V1_archive.json)。包在本机，受`.gitignore`排除，**不随Git同步**。它替代12:50的中期模型快照用于最终V1接续；旧迁移包仍保留，但不是最终基线。

## 清理范围

校验冻结目录后，删除已结束v20的周期checkpoint、非关键逐轮原始评估、worker临时快照、重复checkpoint，以及被替代的短pilot/首次中止启动。保留完整聚合history、关键逐根记录、best/final、报告和已发布RL回放；不清除v14–v16历史诊断、用户presets或其他历史测台轨迹。

另删除已校验迁移压缩包的重复解包暂存目录。总计释放约**2.30GB（十进制）**；逐文件删除账本在`outputs/rl_baselines/V1_cleanup_20261103.json`，摘要见[cleanup.json](../agent_exchange/assets/option_v2_20261103/cleanup.json)。Git跟踪的历史代码和报告未删除。

现存未跟踪文件`tests/test_d010_contact_state.py`依赖已经不存在的历史函数，已保留到`outputs/legacy_tests/`。其他D010/D005旧断言仍保留；此次发现的7个失败已在修改前V1冻结源码中复现，不计为V2新增回归或通过项。

## V2 当前状态

V2已完成一轮12h正式训练：4170轮/133440episode，训练异常0；best@3900验证29/32，最近完整@4160为28/32，final权重@4170尚未额外验证，独立test未运行。[完整总结与证据](../agent_exchange/reviews/V2_Training_Final_20261105.md)。此结果不替换V1冻结基线，也不证明V2优于V1或已解决挡路/对抗。V1不自动扩展出YIELD，不与V2权重/观测契约互换；手工改动历史见[V2改动与测台入口](../agent_exchange/reviews/Option_V2_Yield_20261103.md)。
