"""Evidence-only summary after the random-room study completes."""
from pathlib import Path
import json
from .random_room_study import HOST,BASE,FIT,DATA,AUTHOR,ROOT,ARMS
from experiments.planar_tasks.relational_search.pipeline import read,save

def run():
    r=read(HOST/'RESULT.json');s=r['summary'];best=r['best_single_cap_selected_by_dev'];state=read(AUTHOR/'campaign.json');ex=read(DATA/'extraction.json')
    lines=['# 随机房间 CAP 程序库与监督整合：完成记录','',
      '目标：在同一批随机任务上，先由多个局部程序逐步覆盖开发失败，再以成功示范整合为每个 option 边界工作的扁平候选评分器。此轮没有 PPO。','',
      '200个几何合法初态预先固定：160开发、20验证、20最终留出。固定6×4.8m房间、2–5个可动物体（含目标）、0–3个静态障碍；目标距离至少1.8m。所有失败留在分母中；几何检查并未证明每个任务动态可解。','',
      '作者使用 GPT-6.1 Sol High，干净原生上下文，仅看开发集。离线可查看自有场景真值和实现；部署程序仅读合法因果观测。原生调用token计量平台未提供，不能据此报告精确LLM费用。','',
      '## 程序库的开发覆盖','', '| 程序 | 单程序成功/160 | 新增成功 | 累计并集 |','|---|---:|---:|---:|']
    union=set()
    for p in r['library']:
        union.update(p['success_tasks']);lines.append(f"| {p['name']} ({p['sid']}) | {len(p['success_tasks'])} | {len(p['new_success_tasks'])} | {len(union)} |")
    lines += ['',f"正式库成员{len(r['library'])}个；内部提交{len(state['submissions'])}版；作者实际完成测试{sum('success' in x for x in state['runs'])}次，实际推进{state['used']['execution_ticks']} ticks。源码、未成功修订及回放均保留。",'',
      '## 监督整合','',f"每个开发初态选择已提交教师中成功且仿真耗时最低的一条完整轨迹，共{ex['demonstrations']}条、{ex['boundaries']}个决策边界。没有使用验证或最终留出轨迹训练。各程序沿同一条真实历史更新；拼接描述语义、程序建议和公开几何，评分器在完整合法动作树上条件选择实体、两个机器人动作和时长。并非整段选择一个专家。",'',
      'G：公开几何历史；GA：几何加程序建议/控制信号；GDA：再加入描述语义。三组同示范、同架构、同更新预算，使用固定最终checkpoint。已保守把“程序所选方向的可推性”归为A，不作为独立描述语义。目标矩形已知尺寸是CAP使用的公共形状先验。','',
      '| 条件 | 开发/160 | 验证/20 | 最终留出/20 |','|---|---:|---:|---:|']
    names=[p['name'] for p in r['library']]+list(ARMS)+['teacher_union']
    for name in names:lines.append('| '+('教师成功并集（离线参考）' if name=='teacher_union' else name)+' | '+' | '.join(str(s[split][name]['success']) for split in ['dev','validation','test'])+' |')
    lines += ['',f'最佳单CAP按开发集预先选为 **{best}**。教师并集依赖事后知道哪一程序会成功，是覆盖参考，不能作为一个可直接部署或最优的策略。','',
      '| 模型 | 训练完整动作匹配 | 更新数 |','|---|---:|---:|']
    for arm in ARMS:
        a=read(FIT/arm/'complete.json');acc=a['accuracy'];lines.append(f"| {arm} | {acc['exact_actions']}/{acc['boundaries']} | {a['updates']} |")
    audit={k:sum(x['action_audit'][k]>0 for x in ex['rows']) for k in ['non_target_requests','rotation_requests','paired_push_requests','move_requests']}
    lines += ['',f"成功示范中：请求操作非目标实体{audit['non_target_requests']}条、请求旋转{audit['rotation_requests']}条、请求双推{audit['paired_push_requests']}条、包含MOVE{audit['move_requests']}条（可重叠）。这些是操作请求统计，不能单独证明清障/旋转产生了必要效果。",'','## 留出中的整合保留与丢失','']
    for arm in ARMS:
        q=s['test'][arm];lines.append(f"- {arm}：保留教师并集成功{q['teacher_union_retained']}例；丢失{len(q['teacher_union_lost'])}例；教师均失败而本模型成功{len(q['outside_teacher_union'])}例。")
    paired={}
    for left,right in [('GDA','GA'),('GA','G'),('GDA',best)]:
        a={x['case_id']:x['success'] for x in r['rows'] if x['split']=='test' and x['condition']==left};b={x['case_id']:x['success'] for x in r['rows'] if x['split']=='test' and x['condition']==right}
        paired[left+' vs '+right]=dict(gained=[k for k in a if a[k] and not b[k]],lost=[k for k in a if b[k] and not a[k]])
    save(HOST/'PAIRED_TEST.json',paired)
    lines += ['','## 结论边界','',
      '- 程序库并集相对单CAP的增量，只证明这些冻结程序在本任务分布上互补；监督是否保留该增量以闭环结果为准。',
      '- GDA相对GA才是加入本批描述语义的增量；GA相对G混合了程序建议与程序控制状态的作用。',
      '- 最终留出仅20例、训练单seed，不能把几个案例的差异说成稳定优势。任务未证明全部动态可解，失败也受90秒执行预算和技能能力约束。',
      '- 尚未与相同LLM开发预算、持续维护一个通用CAP的流程做对照，因此不能据此声称增量开发成本已更低。',
      '- 本轮仍在线求值各CAP及其语义；没有证明把昂贵程序完整蒸馏为廉价网络。G部署路径已数值核验后跳过CAP求值；GA/GDA仍需在线运行程序库，此轮并未验证独立廉价语义提取器。',
      '- 验证/测试结果在程序库和模型冻结后获取，没有反馈作者补策略。没有继续RL，也没有将负结果隐藏或剔除。','',
      '## 产物','',f'- 作者：`{AUTHOR.relative_to(ROOT)}`',f'- 整合：`{BASE.relative_to(ROOT)}`',f'- 最终机器可读结果：`{(HOST/"RESULT.json").relative_to(ROOT)}`',
      '- 场景及作者回放：http://127.0.0.1:8773/outputs/research/random_room_20261117_01/','']
    out=ROOT/'docs/agent_exchange/reviews/Random_Room_CAP_Integration_20261117.md';out.write_text('\n'.join(lines))
    current=ROOT/'docs/agent_exchange/CURRENT.md';old=current.read_text();i=old.index('\n')+1
    note=f"\n2026-11-17 `random_room_cap_integration_complete`：[完成记录](reviews/Random_Room_CAP_Integration_20261117.md)。{len(r['library'])}个冻结CAP，开发覆盖并集{len(union)}/160；{ex['demonstrations']}条成功示范、{ex['boundaries']}边界完成G/GA/GDA监督与闭环。最终留出20例G/GA/GDA={s['test']['G']['success']}/{s['test']['GA']['success']}/{s['test']['GDA']['success']}，按开发选最佳CAP {best}={s['test'][best]['success']}，教师并集{s['test']['teacher_union']['success']}；所有失败保留，详见报告边界与封印。无PPO。\n"
    current.write_text(old[:i]+note+old[i:]);print(out)
if __name__=='__main__':run()
