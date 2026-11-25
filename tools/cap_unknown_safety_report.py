"""Publish all frozen small-scene outcomes, including failures and abstentions."""
import csv
import json
from collections import Counter
from pathlib import Path
import shutil
from tools.cap_symbolic_general import ROOT,read,save
from tools.cap_unknown_safety_audit import OUT
from tools.cap_unknown_safety_confirm import verify

REPORT=ROOT/'docs/research/key_findings/2026-11-25_unknown_safety'


def export_csv(path,rows):
    with path.open('w')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def paired_audit(frozen):
    rows=[]
    for pair in frozen['panels']:
        a=read(OUT/'acceptance/empty_observe'/(pair+'.json.gz'))
        b=read(OUT/'acceptance/blocked_observe'/(pair+'.json.gz'))
        same=[];first=None
        for ra,rb in zip(a['records'],b['records']):
            if ra['public_sha256']!=rb['public_sha256']:
                first=ra['tick'];break
            same.append(ra['command']==rb['command']and ra['values']==rb['values'])
        rows.append(dict(pair=pair,identical_prefix_boundaries=len(same),identical_prefix_decisions=all(same),
            first_public_difference_tick=first,empty_termination=a['summary']['termination'],
            blocked_termination=b['summary']['termination']))
    return rows


def replays():
    from tools.cap_symbolic_general_report import HTML
    selected=[]
    for arm,label in [('empty_observe','空场景：安全观察后短推'),('blocked_observe','遮挡障碍：观察后拒绝'),('empty_no_observe','不观察：unknown暂缓')]:
        r=read(OUT/'acceptance'/arm/'pair00.json.gz')
        commands=[dict(tick=x['tick'],option=x['action'],certified=x['certified'],evidence=x['selected_evidence']['evidence'],
            unknown_goal=x['unknown_goal'],blocked_goal=x['blocked_goal'])for x in r['records']]
        commands += [dict(option='brake',reason='完整停止核验')for _ in range(10)]
        selected.append(dict(label=label,room='pair00',version='unknown-v1',reason='预先按编号取首组，不按成功筛选',
            summary=r['summary'],replay=r['replay'],commands=commands))
    html=HTML.replace('__DATA__',json.dumps(selected,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    html=html.replace('CAP 符号策略：逐 tick 回放','Unknown、安全观察与停止：逐 tick 回放').replace('CAP 符号策略：真实执行回放','Unknown、安全观察与完整停止')
    html=html.replace('样例按文件名排序选各版本首个成功、首个失败、首个静态接触；完整分母见报告与 rooms.csv。','固定显示冻结确认集第0组的三个对照；图中隐藏障碍为离线真值展示，并非机器人事先可见。完整分母见报告与 paired.csv。')
    (REPORT/'replay.html').write_text(html)


def plot_evidence():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Polygon as Patch
    from shapely.geometry import shape
    from experiments.planar_tasks.authoring_v2.symbolic_unknown.runtime import Controller
    from experiments.planar_tasks.authoring_v2.random_room_tasks import polygon
    a=read(OUT/'acceptance/empty_observe/pair00.json.gz');b=read(OUT/'acceptance/blocked_observe/pair00.json.gz')
    c=Controller();c.decide(a['public'][0]);initial=c.map.snapshot()
    panels=[('Same initial public evidence',initial,b['replay']['frames'][0],b),
            ('Empty: real observation, push, stop',a['map'],a['replay']['frames'][-1],a),
            ('Blocked: real observation, reject',b['map'],b['replay']['frames'][-1],b)]
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,(title,m,frame,r)in zip(axes,panels):
        ax.set_facecolor('#e5e7eb');g=shape(m['free']);polys=list(g.geoms)if g.geom_type=='MultiPolygon'else[g]
        for p in polys:
            if p.is_empty:continue
            ax.fill(*p.exterior.xy,color='#bde4d2',linewidth=0)
            for ring in p.interiors:ax.fill(*ring.xy,color='#e5e7eb',linewidth=0)
        scene=r['replay']['scene'];p=frame['entities'][scene['primary_id']]
        ax.add_patch(Patch(polygon(p[0],p[1],.25,1.,p[2]),facecolor='#6e9de3',edgecolor='#244b7d'))
        for i,p in enumerate(frame['robots']):ax.add_patch(Circle(p[:2],.1,color=['#ca5858','#3c765a'][i]))
        for o in scene['obstacles']:
            ax.add_patch(Patch(polygon(o['cx'],o['cy'],2*o['half_w'],2*o['half_h'],o['angle']),fill=False,edgecolor='#be3434',linestyle='--',linewidth=2))
        if m['hits']:
            ax.scatter([p[0]for p in m['hits']],[p[1]for p in m['hits']],s=4,color='#bb2929')
        centre=r['case']['parameters'];ax.set_xlim(centre['x']-1.3,centre['x']+1.1);ax.set_ylim(centre['y']-1.2,centre['y']+1.2)
        ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.suptitle('Green: certified static-free under declared priors. Grey: unknown. Dashed red: offline obstacle truth.',fontsize=10)
    fig.savefig(REPORT/'evidence.png',dpi=180);fig.savefig(REPORT/'evidence.svg');plt.close(fig)


def main():
    frozen=verify();m=read(OUT/'acceptance_metrics.json');old=read(OUT/'historical_audit.json');pairs=paired_audit(frozen)
    REPORT.mkdir(parents=True,exist_ok=True)
    export_csv(REPORT/'paired.csv',m['rows']);export_csv(REPORT/'historical.csv',old['rows']);save(REPORT/'paired_prefix_audit.json',pairs)
    save(REPORT/'metrics.json',m);save(REPORT/'historical_audit.json',old)
    for name in ('acceptance_frozen.json','literal_program.json','semantic_schema.json'):shutil.copyfile(OUT/name,REPORT/name)
    dev=[]
    for p in sorted((OUT/'development').glob('*.json.gz')):
        r=read(p);dev.append(dict(case=p.name,**r['summary']))
    save(REPORT/'development.json',dev)
    safety=dict(commands=0,admitted_unknown=0,admitted_nonfree=0,unregistered_selections=0)
    for path in sorted((OUT/'acceptance').glob('*/*.json.gz')):
        r=read(path)
        for row in r['records']:
            safety['commands']+=1
            safety['admitted_unknown']+=bool(row['certified']and row['selected_evidence']['evidence']['status']=='unknown')
            safety['admitted_nonfree']+=bool(row['certified']and row['selected_evidence']['evidence']['status']!='free')
            safety['unregistered_selections']+=bool(row['certified']and row['choice']['option']!=row['action'])
    save(REPORT/'runtime_audit.json',safety)
    a=m['arms'];empty=a['empty_observe'];blocked=a['blocked_observe'];control=a['empty_no_observe']
    text=['# 未知空间先观察：静态证据、短推与完整停止','',
        f"旧14个首次箱体静态接触前缀全部复现，公开边界观测哈希逐一一致：8个碰撞障碍此前未出现在策略收到的边界扫描中，6个已见且仍保留在记忆中。因此问题同时包含未知空间被乐观处理，以及已有证据没有落实到完整外廓和扫掠义务；不能全归因于没看见。",'',
        '## 冻结小场景结果','',
        '| 对照 | 有效推进并完整停止 | 实际接触局数 | 运动范围越界局数 | 完整停止通过 | 未认证紧急停止 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,label in [('empty_observe','空场景，真实观察'),('blocked_observe','遮挡障碍，真实观察'),('empty_no_observe','空场景，不观察')]:
        s=a[arm];text.append(f"| {label} | {s['safe_progress']}/{s['episodes']} | {s['contact_episodes']}/{s['episodes']} | {s['envelope_violation_episodes']}/{s['episodes']} | {s['stopping_passed']}/{s['episodes']} | {s['uncertified_emergency_stops']} |")
    text += ['',f"遮挡障碍组 {blocked['obstacle_rejected']}/12 观察后拒绝。12对中 {m['initial_public_identical_pairs']} 对初始公开输入完全相同；{sum(p['identical_prefix_decisions']for p in pairs)}/12 对在出现公开观测差异前的动作和语义值一致。有效推进要求至少2 cm，并实际执行40 tick制动、检查全程和末端残速。",'',
        f"共检查 {safety['commands']} 个策略决策，认证动作中 unknown 放行 {safety['admitted_unknown']}、非free放行 {safety['admitted_nonfree']}。关闭观察组保守暂缓，所以这里的正例依赖真实获取证据，而不是把全部停止计为任务完成。",'',
        '终止原因逐组计数：`'+str({arm:dict(Counter(r['termination']for r in m['rows']if r['arm']==arm))for arm in a})+'`。',
        '观察命令数/整局物理ticks（包括推进和停止）：`'+str({arm:(s['views'],s['ticks'])for arm,s in a.items()})+'`。完整逐局数据见[配对表](paired.csv)。','',
        '## 具体改变','',
        '- 没有覆盖证据的区域保持unknown；不再用“没找到墙”产生5 m净距。已见障碍命中优先否决。历史自由区域持久保存，证据失效或删除则撤回为unknown。',
        '- 同一符号规则检查公开菜单、整段扫掠的已知与自由、机器人与箱体范围冲突及运动契约。观察和停止也需要证据。预测或有限在线成功样例不能写回真实空间证据。',
        '- 真实观察机器人绕到箱体侧后方；看见后仍需让开箱体预测范围。每段4 tick；停止检查覆盖额外2秒。目标可见长面和CAD先验给出完整外廓集合。',
        '- 所有开发失败保留。其中短BRAKE窗口掩盖的机器人参考速度余动和箱体余动均造成范围越界；这些记录为失败，修订后才冻结。','',
        '## 证据边界','',
        '**这是受限宿主契约下的局部安全闭环验收，不是任意未知环境的无条件安全证明。** 自由区域认证依赖精确静态传感器、最小障碍连通面积0.04 m²、初始分离0.10 m及目标CAD先验。运动范围仍为待独立审计的宿主假设。本次接触计数只能描述这些实测，不能证明所有兼容状态都安全。缺少这些先验时应退回unknown。','',
        '旧三版预测器的C/F/K结果保持历史口径。本轮未重新宣称旧64房间已经安全，也未报告整局成功率或在线仿真减少。这里是版本化宿主修复与局部测台；重新接入作者策略并做整局评估前，仍须checked提交、verify和新冻结。','',
        '[逐tick回放](replay.html) · [空间证据图](evidence.png) · [验收口径](protocol.md) · [冻结源与布局](acceptance_frozen.json) · [符号程序](literal_program.json) · [语义schema](semantic_schema.json) · [旧接触逐例审计](historical.csv) · [全部开发回合](development.json) · [相同前缀输入检查](paired_prefix_audit.json)','',
        '复现：从仓库根目录执行`.venv/bin/python docs/research/key_findings/2026-11-25_unknown_safety/reproduce.py --pair pair00 --arm empty_observe --output /tmp/unknown_pair00_replay.json.gz`。脚本先验证冻结源码，并拒绝覆盖指定输出。报告生成不运行新物理实验。','']
    (REPORT/'README.md').write_text('\n'.join(text));replays();plot_evidence()
    registry=ROOT/'docs/research/experiment_journal/entries.json';entries=read(registry);entry=next(e for e in entries if e['id']=='unknown-safety')
    entry['result']=f"12组冻结对照：空场景有效推进并完整停止 {empty['safe_progress']}/12；遮挡障碍观察后拒绝 {blocked['obstacle_rejected']}/12；不观察有效推进 {control['safe_progress']}/12。"
    entry['analysis']=f"36局实测接触 {sum(s['contact_episodes']for s in a.values())}，范围越界 {sum(s['envelope_violation_episodes']for s in a.values())}。条件依赖静态空间、最小障碍面积、初始分离和运动范围；不宣称旧64房间或整局安全已解决。"
    entry['assets']=[[n,t]for n,t in [('protocol.md','冻结验收口径'),('metrics.json','分组指标'),('paired.csv','逐局配对'),('historical.csv','14个旧接触审计'),('replay.html','真实轨迹回放'),('evidence.png','已知自由与unknown'),('development.json','开发失败与修订'),('acceptance_frozen.json','冻结源与布局'),('literal_program.json','可抽取规则'),('runtime_audit.json','运行期检查')]]
    entry['progress']=[p for p in entry['progress']if p['label']!='完整报告与范围核验']+[dict(label='完整报告与范围核验',file='FINISHED.json',total=1)]
    save(registry,entries);save(OUT/'FINISHED.json',dict(scope='frozen local acceptance',metrics=m['arms'],runtime_audit=safety))
    print(json.dumps(m['arms'],ensure_ascii=False));verify()


if __name__=='__main__':main()
