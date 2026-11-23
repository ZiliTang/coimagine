"""Summarize interventions on naturally flipping room / CAP / transform triples."""
import csv
from tools import cap_symmetry_mechanism as m
from tools import cap_union_trend as t

e=m.e


def publish():
    data=[r for path in sorted((m.OUT/'rooms').glob('*.json')) for r in e.read(path)]
    assert len(data)==58
    def key(r):return r['room'],r['arm'],r['name']
    def files(name):
        return {key(r):r for p in (m.OUT/name).glob('*.json') for r in [e.read(p)]}
    factors=files('ablations')
    once=files('once')
    order=files('order_full')
    assert len(factors)==37 and len(once)==len(order)==58
    cols=['room','arm','name','original_success','first_boundary','first_tick','drift','max_position_m','max_angle_deg','pair_order','center','stall_turn','one_step_restored','full_order_restored','open_loop_status','open_loop_success']
    rows=[]
    for r in data:
        f=r['first'];k=key(r);a=factors.get(k)
        assert once[k]['status']=='completed'
        def corrected(field):
            return None if a is None or field not in a else int(a[field]==f['mapped'])
        rows.append(dict(**{i:r[i] for i in ('room','arm','name','original_success')},first_boundary=f['i'],first_tick=f['tick'],drift=f['drift_before_or_now'],
                         max_position_m=f['peak']['m'],max_angle_deg=f['peak']['deg'],pair_order=corrected('pair_order'),center=corrected('corrected'),
                         stall_turn=corrected('stall_turn'),one_step_restored=once[k]['restored'],full_order_restored=order[k]['restored'],
                         open_loop_status=r['status'],open_loop_success=r['replay_success']))
    with (t.DOC/'mechanism_cases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    restores=sum(r['full_order_restored'] for r in rows)
    ties=[r for r in rows if r['pair_order']]
    tie2=sum(r['full_order_restored'] for r in ties)
    text=f"""# 房间对称变换后为什么改选、为什么翻转

对 32 个开发母房间，取 3 个变换与 2 位专家中出现自然终局翻转的全部 58 个配对。多个配对可来自同一房间；本表用于故障诊断。每一例重放原始完整命令，再把每条命令映射到新房间。同步核对决策时刻、两台机器人和所有可动物体的位姿、可见射线和 CAP 的自然提案。特权位姿只用于离线测量。

| 变换 | 自然成败翻转 | 首次选不同动作时仍几何对齐 | 同样的前缀命令下已出现位姿或时间偏差 |
|---|---:|---:|---:|
| 180° | 15 | 10 | 5 |
| 左右镜像 | 18 | 11 | 7 |
| 上下镜像 | 25 | 16 | 9 |

对齐要求从开始直到首次提案不同，每个选项边界均同 tick、各刚体中心偏差至多 1 mm、朝向偏差至多 0.1°。更严格的 1 微米和 0.0001° 阈值下仍有 32/58 对齐。7 个配对在 tick 0 即选择不同动作。采样粒度是决策边界，未证明物理、接触求解或所有中间传感字段逐 tick 相同。

## 在同一公开输入上干预程序

对仍几何对齐的 37 例，用原程序源码的 Python 诊断副本逐例复现编译版的实际提案。单独改变候选列表顺序，恢复 22/37 个应有的对称动作。测试保持候选集合、观测数值及 CAP 记忆相同。仅把当前估计的目标中心位置换成原场景估值的变换结果，恢复 5/37。仅在镜像中反转停滞恢复的左右转向表，恢复 3/27。这三组案例互不重叠；另 7/37 的首次分歧仍未由这些改动解释。单独改 MOVE 枚举顺序、恢复目标路线点或交换同面双机分工，在适用样本中均未恢复完整提案。

- 房间 1_dev_4_05 的左右镜像，CAP A 在 tick 0 面对两组联合 PUSH。动作编号 (7,15) 和 (11,19) 的原始分数同为 1.8171449153847314。默认保留先遍历到的前者，按原场景的对应顺序遍历就选后者。PUSH 评分用接触面法向；不同的离散命令可以具有相同的几何评分。候选列表按世界坐标方向对应的动作编号遍历。
- 房间 1_dev_3_13 的上下镜像，真实初始位姿精确对齐，目标点云的对称距离约 3×10⁻¹⁵ 米，CAP A 在移动前的目标中心估计已偏 16.3 cm。向中心估计函数输入数学精确镜像的原始点和面也偏 16.9 cm。只替换中心估计，第一条双机命令从 (77,74) 恢复成 (74,74)。该实现将近似平行的面法向视为同轴，同时使用相对世界原点的平面投影；对本例去除平移项、单测绕坐标原点反射，中心误差为 0。
- 另一个可复现的镜像偏置来自停滞恢复。相同停滞计数选择相同符号的转向系数；镜像要求符号相反。

## 首次分歧与终局分别检验

在新房间仅强制纠正首次不同的完整命令、随后让原 CAP 自主执行，14/58 次恢复原场景的终局成败。另一次干预在每个决策都仅把传给提案函数的合法动作对按原方向坐标系的编号顺序排列，{restores}/58 次恢复原结局。在上述 22 个首次明确受排序影响的配对中，此项全程干预恢复 {tie2}/22。恢复原结局含成功变回成功及失败变回失败，不表示控制器改进成功率。

单独盲放整条原命令的映射序列时，33/58 遇到映射命令因当前观测而不可用，7/58 耗尽原命令时尚未终止，18/58 得到完整终局。因而不把盲放未完成的结果计作失败。21 个出现较早执行偏差的案例提示低层控制、接触状态、反馈或执行时长也会破坏对称；实验没有将这 21 个进一步归因于浮点误差。

[逐例数据](mechanism_cases.csv) · [对称成败检查](symmetry.md)
"""
    (t.DOC/'mechanism.md').write_text(text)
    return restores,tie2


if __name__=='__main__':print(publish())
