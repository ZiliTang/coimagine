# 连续平台收尾（closeout A）

run_id `closeout-20261012-193644`。只做必要修正，不扩建平台。

## A1 初态合法性

- 新增 `src/coimagine/continuous2d/legality.py`：按真实几何（边界 capsule 半径 0.05、
  障碍箱、物体多边形、机器人圆）逐对计算有符号间距；非接触对要求统一裕量 0.005 m。
  不用“引擎 contacts 为空”判断。
- 三个 S0 场景原机器人初态与边界重叠：S0_dev −0.010、S0_v2 −0.060、S0_v4 −0.040 m。
  按 `minimal_robot_fix`（沿违规约束内法向迭代投影，与成功率无关）做最小修复：
  - S0_dev / S0_v4：仅 x → 0.235001；
  - S0_v2：两台机器人 x → 0.235001，第二台另需 +0.038670 的 y 位移（仅 x 会触犯物体裕量）。
- 修复后 10/10 场景合法，min clearance ≥ 0.005 m。物体、目标、障碍、质量、摩擦、阻尼、
  时间步、力限幅、成功阈值、控制器与候选均未改动。
- 被修改场景标记 `scene_version=closeout_v2` 并有内容哈希；其余 7 个几何未变，保留 `v1`。
  完整 before/after 与逐对间距见 `docs/closeout_v1/scene_legality.md` 与运行目录
  `scene_legality.json`。

## A2 定向复跑与汇总

只复跑被修改的 S0_dev / S0_v2 / S0_v4（random seeds 0/1/2、depth1、depth3 beam=3、12 窗口）；
未变场景沿用上一轮 `outputs/protocol_fix_v1/20261012-160943/continuous2d`。汇总表每行标注
`source`（本轮/沿用）、`scene_version`、`scene_content_hash`、`code_version`，明确是混合表。

| 条件 | 成功（10 场景） |
|---|---:|
| random（3 种子） | 1/30 |
| mpc depth1 | 5/10 |
| mpc depth3 beam3 | 9/10 |

（修复后 depth3 在 S0_v2 成功；S0_dev/S0_v2 的 depth1 失败，属场景几何变化后的真实结果，
不作为要修的目标。）

## A3 交付

- `continuous_closeout_review.zip`：本轮与上一轮协议修复的相关源码、测试、配置、依赖、
  场景合法性表、受影响场景结果、回归证据与运行命令、源码 SHA256 与缺失列表。
- 状态表述：**已实现、已有测试支持**；未代替外部审查宣布无 bug。

## 复现命令

```bash
pytest tests/test_protocol_fix.py tests/test_continuous2d.py -q
python experiments/continuous2d/scene_legality.py --out outputs/closeout_v1/<run_id>
python experiments/continuous2d/run_baseline.py --scenes S0_dev,S0_v2,S0_v4 \
  --run-id <run_id>/targeted --out-root outputs/closeout_v1 --no-render
python experiments/continuous2d/combine_closeout.py \
  --previous outputs/protocol_fix_v1/20261012-160943/continuous2d \
  --targeted outputs/closeout_v1/<run_id>/targeted/continuous2d \
  --out      outputs/closeout_v1/<run_id>/combined
python experiments/review_bundle/make_closeout_bundle.py --run-id <run_id>
```
