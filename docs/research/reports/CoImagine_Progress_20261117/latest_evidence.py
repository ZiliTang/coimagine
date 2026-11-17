"""Read existing random-room artifacts; never launch experiments or train models."""
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
HOST = 'outputs/semantic_rl_authors/random_room_coverage_20261117_02'
BASE = 'outputs/semantic_rl/random_room_coverage_20261117_02'
OLD = 'outputs/semantic_rl_authors/random_room_20261117_01'


def table(headers, rows, caption=''):
    return '<div class="table-wrap"><table><caption>'+escape(caption)+'</caption><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def enrich(data, read, compact):
    old = read(OLD+'/RESULT.json')
    collection = read(HOST+'/COLLECTION_ANALYSIS.json')
    diagnosis = read(HOST+'/DIAGNOSTIC_ANALYSIS.json')
    repair = read(HOST+'/REPAIR_ANALYSIS.json')
    baseline = read(HOST+'/REPAIR_COHORT_BASELINE.json')
    fits = read(HOST+'/FIT_BY_SOURCE.json')
    progress = read(HOST+'/progress.json')
    training = read(BASE+'/expanded_training_rollouts/result.json')
    selected = read(BASE+'/data_repaired/selected.json')
    final = read(HOST+'/RESULT.json') if (ROOT/HOST/'RESULT.json').exists() else None
    acceptance = read(HOST+'/REPAIR_ACCEPTANCE.json') if (ROOT/HOST/'REPAIR_ACCEPTANCE.json').exists() else None
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M CST')
    added=[]
    def replay(path, title, note):
        index=len(data['replays']); data['replays'].append(compact(str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else path,title,note)); added.append(index);return index

    # Outcome-selected illustration; aggregate statistics retain all outcomes.
    case='test_2_45'
    old_pair=[]
    for condition,title in [('CAP-RR-003','随机房间：预选单 CAP 失败'),('GDA','随机房间：边界整合成功')]:
        row=next(r for r in old['rows'] if r['case_id']==case and r['condition']==condition)
        old_pair.append(replay(row['result_path'],title,'test_2_45：10个独立CAP均失败，GA/GDA成功；整合动作均属于当时的程序建议并集。结果选例，不代表总体泛化已超过单CAP。'))
    item=selected[0]
    sr=read(str(Path(item['result_path']).relative_to(ROOT).parent/'summary.json'))
    selection=read(BASE+'/repair/selection.json')
    ref=next(x['learner_path'] for x in selection if x['case_id']==sr['case_id'])
    cut=sr['prefix_audit']['ticks']*.05
    repair_pair=[replay(ref,'真实失败状态：扩数据 GDA 继续执行',f"{sr['case_id']}；在第{item['start_boundary']}个边界、{cut:.2f}秒接管。两侧此前物理与公开观测一致。"),replay(item['result_path'],'同一前缀：已有 CAP 接管成功',f"{item['teacher']} 从同一真实前缀继续，成功后缀进入追加监督；这不是从任务初态重新挑选成功教师。")]
    for i in repair_pair:data['replays'][i]['handoff_time']=cut
    if acceptance:
        p=BASE+f"/repair_acceptance/runs/repaired_GDA/{sr['case_id']}__cut{item['start_boundary']:03d}/result.json.gz"
        if (ROOT/p).exists():
            learned=replay(p,'追加监督后：GDA 从同一前缀接管','冻结 repaired GDA 的开发机制验收；不能当作独立留出。')
            data['replays'][learned]['handoff_time']=cut
            data.setdefault('extra_examples',{})['repair_learned']=[repair_pair[1],learned]
    data.setdefault('extra_examples',{}).update(random_integration=old_pair,repair=repair_pair)
    random_scenes=[]
    for count in range(2,6):
        row=next(r for r in old['rows'] if r['split']=='dev' and r['case_id'].startswith(f'dev_{count}_') and r['condition']=='G')
        raw=read(str(Path(row['result_path']).relative_to(ROOT)))
        random_scenes.append(dict(id=row['case_id'],scene=raw['replay']['scene'],frame=raw['replay']['frames'][0],count=count))
    data['random_scenes']=random_scenes
    latest=dict(updated=stamp,first_round=old['summary'],collection=collection,diagnosis=diagnosis,repair=repair,repair_baseline=baseline,fits=fits,progress=progress,training_success=sum(r['success'] for r in training['rows']),final=({k:v for k,v in final.items() if k!='rows'} if final else None),acceptance=acceptance)
    data['random_room']=latest
    sources=[('随机房间：10 CAP 与首次监督整合','docs/agent_exchange/reviews/Random_Room_CAP_Integration_20261117.md'),('扩覆盖与真实接管：预定协议及修订','docs/agent_exchange/tasks/Random_Room_Coverage_Repair_20261117.md'),('首次随机房间：机器可读完整结果',OLD+'/RESULT.json'),('当前九模型：训练动作拟合',HOST+'/FIT_BY_SOURCE.json'),('真实接管修复：330分支统计',HOST+'/REPAIR_ANALYSIS.json')]
    if final:sources.append(('扩覆盖与真实接管：完整闭环结果','docs/agent_exchange/reviews/Random_Room_Coverage_Repair_20261117.md'))
    data['sources'].extend(dict(title=t,path=p) for t,p in sources)
    return latest, added


def render(x):
    old=x['first_round'];f=x['fits'];c=x['collection'];r=x['repair'];b=x['repair_baseline']
    pending = not x['final']
    status = (f"最终固定评估进行中（本快照 {x['progress'].get('completed','—')}/{x['progress'].get('total',4000)}）；新100例最终留出尚无完整结论。" if pending else '最终固定评估已完成。'+('真实接管机制验收已完成。' if x['acceptance'] else '真实接管机制验收仍在进行。'))
    s=f'''<section id="latest"><div class="section-no">新增证据 / 随机房间与真实接管</div><h2>已有程序的能力，能否在更多状态上被正确使用？</h2><p class="caption">更新于 {x['updated']}。这是静态快照；后台进度不随本页自动刷新。</p><div class="callout"><b>{status}</b><p>程序库固定为10个CAP，本轮只扩充成功示范、补真实接管后缀并重新监督拟合，没有新CAP开发或PPO。</p></div>
<div class="two"><div><h3>从手工任务类型，走向随机房间</h3><p>固定6×4.8m房间，随机放置2–5个可动物体（含目标）和0–3个静态障碍。将指定物体运到至少1.8m以外的目标；初态与目标做几何合法性检查，不按动态可解性筛选。</p><p>Sol High先在160个开发初态上逐步补充10个CAP。每个新程序可以针对尚未覆盖的任务修订，旧程序保留，不要求新程序独自兼容全部旧任务。</p><p class="small">两机器人合法局部扫描集中汇总；动作仍按实体、双机器人控制和时长选择。固定90秒任务预算。随机多物体不意味着已经具备清障能力：这批成功示范没有主动操作非目标实体。</p></div><div><div id="random-scene-grid" class="scene-grid"></div><p class="caption">4个开发初态，仅按物体数量选例展示分布；图中真值仅用于人类回放。</p></div></div>
<article class="result-card"><span class="tag good">D · 程序库互补与首次整合</span><h3>开发集能保留互补能力，留出上仍有明显缺口。</h3><div class="two"><div><p>最佳单CAP开发成功65/160；10个程序的成功并集为124/160。每根选择一条成功示范，得到124条轨迹、4477个决策边界。</p><p>GA、GDA分别成功125/160和127/160，均保留教师库全部124个开发成功。但最终留出只有8/20，未超过预选单CAP。</p><p>两者在留出都只保留教师可解的7/14例，另有1例全库失败而整合成功。<a href="#replay" data-example="random_integration">看这一例回放 →</a></p></div><div>'''
    rows=[]
    for key,label in [('CAP-RR-003','预选单CAP-003'),('teacher_union','教师成功并集'),('G','G'),('GA','GA'),('GDA','GDA')]:
        rows.append([label]+[f"{old[split][key]['success']}/{old[split][key]['n']}" for split in ['dev','validation','test']])
    s+=table(['策略','开发','验证','最终留出'],rows)
    s+='''<p class="small">首轮各模型14000更新、单seed。教师并集是事后覆盖参考，不是可部署选择器；旧验证与测试此后仅作已见诊断。</p></div></div></article>
<article class="result-card"><span class="tag neutral">E · 扩数据与修复接管</span><h3>先区分“缺能力”与“在已有能力之间选错”。</h3>'''
    s+=f'''<p>旧留出14个“库里有解、GDA失败”的任务中，在预定截点尝试已有CAP接管，有12个任务可以救回。它们不是都缺少新技能；至少部分失败可以从当前真实状态继续修复。</p><div class="pipeline"><div><b>200新开发初态</b><span>10CAP逐根执行，{c['union']}/200有成功教师</span></div><i>→</i><div><b>270条完整示范</b><span>124旧＋146新，9292个边界</span></div><i>→</i><div><b>100独立修复开发初态</b><span>扩数据GDA {b['success']}/100，教师并集{b['teacher_union']}/100</span></div><i>→</i><div><b>25个成功后缀</b><span>来自19根、822个追加边界</span></div></div>
<p>扩数据GDA在用于收集示范的200根上成功{x['training_success']}/200，并保留所有146个教师成功。为了不在已经拟合好的初态上寻找修复，新加入100个独立的修复开发初态；该调整在新最终留出执行前登记。</p><p>从其中24个失败任务选33个实际执行边界，10CAP分别接管，共330分支，{r['success_branches']}次成功；救回{r['rescuable_cases']}/24个任务、{r['rescuable_cut_states']}/33个状态。每状态只选一条成功后缀，前缀供历史计算，失败前缀动作不作正标签。330次前缀的物理与公开观测一致性全部通过。<a href="#replay" data-example="repair">看失败继续 / CAP接管对照 →</a></p><p class="small">25条后缀不是25个独立新任务。截点按预定动作分歧及其后边界选取，并不声称已经定位因果失败点。未救回也不等于无解。</p></article>
<article class="result-card"><span class="tag neutral">F · 等更新预算的输入对照</span><h3>程序知识改善拟合，但拟合不等于闭环成功。</h3><p>G仅使用合法几何历史；GA增加程序建议与控制状态；GDA再增加描述语义。九模型同为28000更新、同seed与32×32浅层评分网络。数据变多时覆盖面增加，但相同更新预算意味着每条样本的重复学习次数减少。</p>'''
    fitrows=[]
    for tier,label in [('control','124完整轨迹：旧数据等预算'),('expanded','270完整轨迹：扩覆盖'),('repaired','270完整＋25成功后缀')]:
        n=f[tier+'_G']['boundaries']
        fitrows.append([label,n]+[f"{f[tier+'_'+arm]['exact_actions']}/{n} ({f[tier+'_'+arm]['exact_actions']/n:.1%})" for arm in ['G','GA','GDA']])
    s+=table(['监督数据','决策边界','G匹配','GA匹配','GDA匹配'],fitrows,'完整动作匹配：实体、两机器人动作、请求时长全部正确才算一次')
    s+='''<p>当前可说：程序知识让本配置更容易拟合成功行为。G仍有训练误差，GA/GDA又在线使用程序的规划与记忆；不能把差异都归为抽象语义，也不能仅据此证明留出样本效率更高。GDA−GA才对应本批描述语义的增量。</p></article>'''
    if x['final']:
        z=x['final']['summary']['fresh_test'];rows=[]
        for tier,label in [('original','原124示范 / 14000更新'),('control','原124示范 / 28000更新'),('expanded','270完整 / 28000更新'),('repaired','270完整＋25后缀 / 28000更新')]:
            rows.append([label]+[f"{z[tier+'_'+arm]['success']}/{z[tier+'_'+arm]['n']}" for arm in ['G','GA','GDA']])
        s+='<article class="result-card"><span class="tag good">G · 冻结后的新100例</span><h3>同一批未参与开发的初态，比较实际任务成功。</h3>'+table(['版本','G','GA','GDA'],rows)
        s+=f"<p>预选CAP-003：{z['CAP-RR-003']['success']}/100；教师成功并集：{z['teacher_union']}/100。所有模型冻结后才执行这100例，不据其结果选择模型或继续拟合。</p><p class=\"small\">一次随机面板、单训练seed；没有证明任意任务泛化、相同LLM预算下的开发成本优势或完整程序蒸馏。GA/GDA仍在线运行程序库。</p></article>"
    else:
        s+='''<article class="result-card"><span class="tag warning">G · 正在评估</span><h3>新100例最终留出：等待完整结果。</h3><p>九个新模型加三个原版模型，在同一新100例上比较；10CAP提供逐根可解见证。原200例另作已见诊断，合计4000次固定评估。没有用部分成功率提前选优。</p><p>之后另查：追加监督的GDA能否复现25个接管成功、能否改善100根修复开发任务。这是机制验收，与独立泛化分开报告。</p></article>'''
    if x['acceptance']:
        q=x['acceptance'];v=q['roots'];u=q['suffixes']
        s+=f'''<div class="callout"><b>修复监督机制验收：</b>同一真实前缀接管成功{u['success']}/{u['n']}；100根修复开发任务从{v['before']}变化至{v['after']}，新增{len(v['gained'])}、丢失{len(v['lost'])}。该组有19根用于后缀监督，不作为独立留出。全轮正式执行{q['combined_formal_episodes']}次（含前缀重放），另有1次工程预检；未追加CAP或PPO。</div>'''
    s+='''<p class="small">接下来的判断：扩覆盖是否改善新留出；真实接管后缀是否纠正分布偏移；描述语义是否在同数据下比仅有程序建议进一步有效。旧结果与新结果分面板呈现，不把训练、开发验收和最终留出合并成一条提升曲线。</p></section>'''
    return s
