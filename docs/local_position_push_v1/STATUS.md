# STATUS — local_position_push_v1

run_id `local-position-push-v1`。A–D 全部执行完毕，代码可执行，无 BLOCKED。
上一轮目录（`local_contact_option_v1/`、`local_contact_option_repair_v2/`、
`local_contact_option_push_v3/`）全部保留，未覆盖。

## A 基线与回归 — 完成

`A_baseline_facts.md`：Kv=25、`a_ref_max`=3.0、接近 0.30/0.06/4.0/0.02、力上限 15 N、
控制周期 0.05 s、速度式公式、四个阶段语义、延迟队列、判稳实现（采样时间戳差 + 逐点 RMS +
0.20 s 新鲜度）。与实际代码一致，无版本差异。六项既有问题逐条给出回归测试名。

## B 新控制器 — 完成

- `src/coimagine/local_contact_option/position.py`：`surface_reference`（d/q/p_contact/gap_signed/
  support）、`segment_support`、`PositionPushState`（β 爬升与重置规则）、`position_push_force`、
  `velocity_law_force`（供等价性测试比较两套实现）。
- 接入 `option.py`：ARM `V`（=冻结 v3 速度式）与 `X`（位置反馈）；两臂共用
  APPROACH/READY/PUSH 计时/BRAKE/SETTLE 与同一传感；新增退出 `patch_support_lost`。
- `tests/test_local_position_push.py` 18 项：白名单/真值隔离、垂足与符号、穿透不隐藏、
  支撑检查、旋转平移等变、**零间隙等价（1e-9）**、非零间隙差异、误差与力上限、β 启用/撤销、
  缓存重置、执行级满有效推动、全程不写物理状态、同快照配对与副本隔离、日志一致性、
  δ 映射、支撑丢失退出、两臂共用接近/制动。
- 全量测试：**414 passed**。

## C 配对实验 — 完成

| 项 | 数值 |
|---|---|
| dev 执行验收 | 24 root × 8 变体 × 2 臂 = 384（基准 48 次子集：V 23/24 判稳、X 21/24） |
| test | 120 root × 8 变体 × 2 臂 = **1920** |
| 接续 | 12 root × 3 技能 × 2 臂 = 24 条链 |
| 时序诊断 | 中央单推 / 同侧双推 / 偏心转动，各 V/X 并列（PNG + CSV） |

六类指标见 `tables.md`。结论方向：X 的有效推动 0.900→0.800、contact_lost 0.081→0.153、
位移中位 0.265→0.209、接续 9/12→7/12；假稳定两者 0；扰动敏感度无实质变化；切向滑移改善 4%。

## D 动画与交付 — 完成

8 案例 × (V + X) = **16 个单臂 MP4**、**8 个 V/X 并排 MP4**、**8 个预览 GIF**、
离线 `index.html`（含三个时序诊断直链）与 `ANIMATION_GUIDE.md`。
1× 实时（50 ms 采样 / 20 fps），X 叠加 q、p_contact、p_ref 与位置误差细线，p_ref 为空心点并注明
虚拟参考；V 标 `p_ref N/A`；箭头比例固定。全部文件已核验存在、非空、与执行记录一致、
同案例初态 hash 唯一。

## 交付物

`docs/local_position_push_v1/`、`experiments/local_position_push/`、
`src/coimagine/local_contact_option/position.py`（及 `option.py`/`sensing.py`/`metrics.py` 改动）、
`tests/test_local_position_push.py`、
`outputs/local_position_push_v1/local-position-push-v1/`、
`local_position_push_v1_review.zip`、`local_position_push_v1_animations.zip`。

## 下一步（任务书 5 的答案）

不要开始技能终态学习。先补三件事：把判稳覆盖率从 0.62–0.67 提到 ≥0.90；
用同一份数据做一次 V/X 表示对照；确定教师臂。理由见 `report.md` 第 5 节。
