"""Publish reproducible paired measurements and deterministic replay examples."""
import csv
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

from tools.cap_symbolic_general import ROOT, OUT, AUTHOR, read, save, sha

REPORT=ROOT/'docs/research/key_findings/2026-11-25_symbolic_general'

REPRODUCE='''"""Run from the repository root; writes a new file, preserves frozen evidence."""
import argparse
import gzip
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from tools.cap_symbolic_general import OUT, read
from tools.cap_symbolic_general_evaluate import verify_freeze
from tools.cap_symbolic_general_compile import compile_policy
from tools.cap_symbolic_general_runtime import Controller
from experiments.planar_tasks.authoring_v2.symbolic_general import simulation, scaffold
from experiments.planar_tasks.authoring_v2.symbolic_general.prediction import EffectModel

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--version',type=int,choices=[0,1,2],required=True)
p.add_argument('--room',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
frozen=verify_freeze()
source=Path(frozen['source']).read_text()
fixture=read(OUT/'panels.json')['groups']['confirmation'][a.room]
compile_policy(scaffold.verify_source(source))
model=EffectModel(read(frozen['model_files'][a.version]))
result=simulation.episode(fixture,source,controller=Controller(model),
    keep_observations=False,replay=True,wall_seconds=240)
result['reproduction']=dict(room=a.room,version=a.version,policy_sha256=frozen['source_sha256'],
    model_sha256=frozen['model_sha256'][a.version],
    caveat='Wall-clock budget can make branch counts differ with execution load.')
payload=json.dumps(result,ensure_ascii=False,separators=(',',':')).encode()
with a.output.open('xb')as f:f.write(gzip.compress(payload)if a.output.suffix=='.gz'else payload)
print(json.dumps(result['summary'],ensure_ascii=False))
'''

HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CAP 符号策略：逐 tick 回放</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f7fa;color:#172338}main{max-width:1100px;margin:auto}canvas{width:100%;background:white;border:1px solid #ccd4df}select,button,input{font:inherit;margin:6px}pre{white-space:pre-wrap;background:white;padding:14px}.row{display:flex;gap:16px;align-items:center}#slider{flex:1}small{color:#536174}</style>
<main><h1>CAP 符号策略：真实执行回放</h1><p>目标箱蓝色，同伴箱橙色，机器人红/绿，静态障碍灰色。画面含开发测台真实状态，仅用于离线核对；策略输入为公开观测。</p>
<select id="case"></select><div class="row"><button id="play">播放</button><input id="slider" type="range" min="0" value="0"><span id="clock"></span></div>
<canvas id="view" width="1000" height="800"></canvas><pre id="detail"></pre><small>样例按文件名排序选各版本首个成功、首个失败、首个静态接触；完整分母见报告与 rooms.csv。</small></main>
<script>const cases=__DATA__;const sel=document.getElementById('case'),slider=document.getElementById('slider'),canvas=document.getElementById('view'),ctx=canvas.getContext('2d');let playing=false;
cases.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.label;sel.appendChild(o)});
function polygon(points,color){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#25354a';ctx.lineWidth=.012;ctx.stroke()}
function box(x,y,a,w,h,color){let c=Math.cos(a),s=Math.sin(a);polygon([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]].map(([u,v])=>[x+c*u-s*v,y+s*u+c*v]),color)}
function draw(){let r=cases[+sel.value],replay=r.replay,scene=replay.scene,frame=replay.frames[+slider.value];if(!frame)return;let scale=145;ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,1000,800);ctx.setTransform(scale,0,0,-scale,60,750);ctx.strokeStyle='#dce3ec';ctx.lineWidth=.005;for(let x=0;x<=scene.arena[0];x+=.5){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,scene.arena[1]);ctx.stroke()}for(let y=0;y<=scene.arena[1];y+=.5){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(scene.arena[0],y);ctx.stroke()}ctx.strokeStyle='#344862';ctx.lineWidth=.025;ctx.strokeRect(0,0,...scene.arena);
let goal=scene.goal;ctx.beginPath();ctx.arc(goal.x,goal.y,goal.radius,0,Math.PI*2);ctx.fillStyle='#b8e2b8';ctx.fill();scene.obstacles.forEach(o=>box(o.cx,o.cy,o.angle,2*o.half_w,2*o.half_h,'#8a929e'));
for(let [name,p]of Object.entries(frame.entities)){if(name===scene.primary_id){let c=Math.cos(p[2]),s=Math.sin(p[2]);polygon(scene.object_polygon.map(([x,y])=>[p[0]+c*x-s*y,p[1]+s*x+c*y]),'#699fe7')}else{let o=scene.movable_objects.find(q=>q.label===name);if(o)box(p[0],p[1],p[2],o.width,o.height,'#e9ad64')}}
frame.robots.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],scene.robot_radius,0,Math.PI*2);ctx.fillStyle=i?'#2f9870':'#c75252';ctx.fill();ctx.strokeStyle='#172338';ctx.lineWidth=.015;ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(p[0]+.17*Math.cos(p[2]),p[1]+.17*Math.sin(p[2]));ctx.stroke()});
document.getElementById('clock').textContent='tick '+frame.tick+' / '+replay.frames.at(-1).tick;
let cmd=r.commands[Math.max(0,frame.command_index)];document.getElementById('detail').textContent=JSON.stringify({room:r.room,version:r.version,selection_reason:r.reason,final:r.summary,current:cmd},null,2)}
function choose(){let r=cases[+sel.value];slider.max=r.replay.frames.length-1;slider.value=0;draw()}sel.onchange=choose;slider.oninput=draw;document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'暂停':'播放'};setInterval(()=>{if(playing){if(+slider.value>=+slider.max){playing=false;return}slider.value=Math.min(+slider.max,+slider.value+2);draw()}},50);choose();</script></html>'''


def replays():
    selected=[]
    for version in ('0','1','2','ungated'):
        paths=sorted((OUT/'confirmation/tasks'/version).glob('*.json.gz'));used=set()
        for reason,test in [('首个成功',lambda s:s['success']),('首个失败',lambda s:not s['success']),
                            ('首个箱体静态接触',lambda s:s['object_static_commands']>0)]:
            for path in paths:
                data=read(path)
                if test(data['summary']):
                    if path not in used:
                        used.add(path);room=path.name[:-8]
                        selected.append(dict(label=f'v{version} · {room} · {reason}',room=room,version=version,
                            reason=reason,summary=data['summary'],replay=data['replay'],
                            commands=[dict(tick=r['tick'],option=r['choice']['option'],command=r['command'],
                                           object_static=r['audit']['object_static'],
                                           model_sha256=(r.get('evidence')or{}).get('model_sha256'))for r in data['records']]))
                    break
    if any(r['replay']is None for r in selected):raise ValueError('missing actual replay')
    (REPORT/'replay.html').write_text(HTML.replace('__DATA__',json.dumps(selected,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')))
    save(REPORT/'replay_selection.json',[{k:r[k]for k in ('room','version','reason')}for r in selected])


def pct(x):return '未定义'if x is None else f'{100*x:.2f}%'
def number(x):return '未定义'if x is None else f'{x:.3f}'


def plot_results(metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(13,3.8),layout='constrained')
    for ax,key,title in zip(axes[:3],('C','F','K'),
                            ('Effective coverage C (%)','Successor failure F (%)','Online branches / 100 boundaries')):
        factor=1 if key=='K'else 100
        values=[None if r[key]is None else factor*r[key]for r in metrics['versions']]
        ax.plot(range(3),values,'o-',color='#2563a6',linewidth=2)
        ax.set_xticks(range(3),['v0','v1','v2']);ax.set_title(title,fontsize=10)
        ax.set_ylim(0,max(1,max((x for x in values if x is not None),default=0)*1.25));ax.grid(axis='y',alpha=.2)
        for i,value in enumerate(values):
            if value is not None:ax.annotate(f'{value:.2f}',(i,value),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
        ax.margins(y=.2)
    ax=axes[3];labels=['v0','v1','v2','ungated'];rs=[metrics['tasks'][k]for k in ('0','1','2','ungated')]
    values=[100*r['success']/r['rooms']for r in rs]
    safe_values=[100*sum(row['success']and not row['object_static_commands']for row in r['rows'])/r['rooms']for r in rs]
    ax.bar([i-.18 for i in range(4)],values,width=.36,color='#2563a6',label='All successes')
    ax.bar([i+.18 for i in range(4)],safe_values,width=.36,color='#31916c',label='No target-static contact')
    ax.set_xticks(range(4),labels);ax.set_ylim(0,100);ax.legend(fontsize=6,loc='upper left')
    ax.set_title('Whole-task success (%)',fontsize=10);ax.grid(axis='y',alpha=.2)
    for i,(r,value)in enumerate(zip(rs,values)):
        ax.text(i,value+2,f"{r['success']}/{r['rooms']}",ha='center',fontsize=9)
    fig.suptitle('Frozen CAP, three effect models; 64 held-out rooms in one layout family',fontsize=12)
    fig.savefig(REPORT/'metrics.png',dpi=180);fig.savefig(REPORT/'metrics.svg');plt.close(fig)


def worked_cases():
    rows=[row for path in sorted((OUT/'confirmation/paired').glob('*.json.gz'))for row in read(path)['rows']]
    index={(r['room'],r['boundary'],r['action'],r['version']):r for r in rows}
    tests=[('放行、实际推进并完成后继',lambda r:r['admitted']and r['safe_effective']and(r['root_progress']or 0)>=.02),
           ('预测集合包含反例而暂缓',lambda r:not r['admitted']and r['symbolic_status']=='counterexample'),
           ('经验放行后的实际承诺违约',lambda r:r['admitted']and not r['actual_contract_valid']),
           ('从在线试算改为模型直接放行',lambda r:r['admitted']and r['source']=='model'and r['safe_effective']and
            index[(r['room'],r['boundary'],r['action'],0)]['source']=='online_empirical')]
    selected=[];text=['# 从规则预测到实际后继：四类原始案例','',
        '各类从按房间、边界、候选排序的 v2 记录中取首个符合条件的案例；用于解释机制，不代表各类发生率。完整分母见主报告。','']
    for label,test in tests:
        row=next((r for r in rows if r['version']==2 and test(r)),None)
        if row is None:text+=['## '+label,'','没有符合此条件的案例。',''];continue
        name=row['room']+f"_{row['boundary']:04d}.json.gz"
        predictions=read(OUT/'confirmation/predictions'/name)
        position=predictions['root']['actions'].index(row['action'])
        pred=predictions['versions'][2][position]
        actual=read(OUT/'confirmation/branches/v2'/name)['rows'][position]
        selected.append(dict(label=label,measurement=row,prediction=pred,actual=actual,
            original_predictions=str(OUT/'confirmation/predictions'/name),
            original_physical_branch=str(OUT/'confirmation/branches/v2'/name)))
        symbolic=((pred.get('online')or{}).get('joint_symbolic')if row['source']=='online_empirical'else pred.get('symbolic'))or{}
        distance_ranges=[o['updates'].get('distance')for o in pred['interval_outcomes']]
        if row['source']=='online_empirical':
            distance_ranges += [[s['values']['distance'],s['values']['distance']]for s in pred['online']['samples']]
        text+=['## '+label,'',f"房间 `{row['room']}`，边界 {row['boundary']}，tick {row['tick']}，动作 `{row['action']}`。",'',
            f"放行依据 `{row['source']}`；符号结果 `{symbolic.get('status')}`；预测风险 {pred['predicted_risk']}。",'',
            '用于检查的下一边界目标距离范围（米；在线分支以点值加入联合集合）：`'+str(distance_ranges)+'`。',
            f"实测公开目标距离摘要：{actual['next_semantic']['distance']:.6f} m。实际后继 `{actual['successor'].get('option')}`，可启动 {actual['successor'].get('executable')}，预算内完成 {actual['successor_completed']}。",'',
            f"根动作真实向目标推进 {actual['audit']['goal_gain']:.6f} m；根动作与后继窗口箱体静态接触 {actual['full_object_static']}；实际承诺检查 {row['actual_contract_valid']}。",'',
            '符号问题：`'+str([{'branch':i.get('branch'),'property':i.get('property'),'status':i.get('status')}for i in symbolic.get('issues',[])])+'`。','']
    text+=['完整联合预测、反例见证、实际公开摘要及后继记录见 [案例数据](worked_cases.json)。这些数据包含离线物理核对量，未作为策略输入。','']
    save(REPORT/'worked_cases.json',selected)
    (REPORT/'worked_cases.md').write_text('\n'.join(text))


def action_strata():
    rows=[row for path in sorted((OUT/'confirmation/paired').glob('*.json.gz'))for row in read(path)['rows']]
    groups=[]
    for version in range(3):
        for family in ('push','orbit','move','brake'):
            rs=[r for r in rows if r['version']==version and r['action'].split('_')[0]==family]
            if not rs:continue
            admitted=[r for r in rs if r['admitted']]
            groups.append(dict(version=version,root_action_family=family,candidates=len(rs),admitted=len(admitted),
                effective=sum(r['safe_effective']for r in admitted),successor_failed=sum(not r['actual_completed']for r in admitted),
                successor_options=sorted({(r.get('successor')or{}).get('option')or'none'for r in admitted})))
    newly_legal=[]
    for version in range(3):
        count=0;completed=0;total=0
        for path in sorted((OUT/'confirmation/branches'/f'v{version}').glob('*.json.gz')):
            for r in read(path)['rows']:
                if r.get('error')or not r['action'].startswith('orbit'):continue
                total+=1;successor=(r.get('successor')or{}).get('option')or''
                if successor.startswith('push')and not r['semantic']['legal:'+successor]and r['next_semantic']['legal:'+successor]:
                    count+=1;completed+=bool(r['successor_completed'])
        newly_legal.append(dict(version=version,executed_orbit_candidates=total,newly_legal_actual_selected_push=count,completed=completed))
    packet=dict(root_action_groups=groups,orbit_to_newly_legal_push=newly_legal,
        scope='all forced diagnostic candidates, including rejected roots; newly legal means false before orbit and true after for the actually selected successor')
    save(REPORT/'action_strata.json',packet);return packet


def publish():
    from tools.cap_symbolic_general_evaluate import verify_freeze
    from experiments.planar_tasks.authoring_v2.random_room_author import RoomCampaign
    frozen=verify_freeze();campaign=RoomCampaign(AUTHOR/'campaign').verify_frozen()
    m=read(OUT/'metrics.json');protocol=read(OUT/'protocol.json');models=read(OUT/'models/manifest.json')
    expected=protocol['counts']['confirmation']
    if m['paired_rooms']!=expected or any(v['rooms']!=expected for v in m['tasks'].values()):raise ValueError('incomplete confirmation inventory')
    if len(models)!=3:raise ValueError('need all three prediction versions')
    if any(r['model_covered_contract_violations']for r in m['versions']):
        raise ValueError('covered admitted outcome contradicts the symbolic contract')
    if any(r['pairing_mismatches']for r in m['versions']):raise ValueError('unresolved physical-prefix pairing mismatch')
    if m['tasks']['2']['success']<1:raise ValueError('final deployment has not completed any full task')
    if read(OUT/'runtime_selection_proof.json')['result']['status']!='proved':raise ValueError('runtime selector obligation unresolved')
    if not read(OUT/'host_checked_boundaries.json'):raise ValueError('missing compiled CAP boundary checks')
    rechecks=read(OUT/'model_recheck_summary.json')
    raw=read(OUT/'raw_replay_audit.json');costs=read(OUT/'cost_accounting.json')
    if raw['goal_gain_mismatches']or raw['frame_tick_mismatches']or raw['success_outside_goal']:
        raise ValueError('unresolved independent replay audit discrepancy')
    REPORT.mkdir(parents=True,exist_ok=True)
    compile(REPRODUCE,'reproduce_one.py','exec')
    (REPORT/'reproduce_one.py').write_text(REPRODUCE)
    checks={}
    for name in ('tests_framework.xml','tests_related.xml'):
        path=OUT/name
        suites=list(ET.parse(path).getroot().iter('testsuite'))
        checks[name]={key:sum(int(s.attrib.get(key,0))for s in suites)for key in ('tests','failures','errors','skipped')}
        if checks[name]['failures']or checks[name]['errors']:raise ValueError('recorded tests failed: '+name)
        shutil.copy2(path,REPORT/name)
    save(REPORT/'validation.json',dict(tests=checks,tests_overlap=True,full_repository_suite_run=False,
        evaluation_freeze_verified=True,frozen_host_source_count=len(frozen['host_sources']),
        campaign_sha256=campaign['sha256'],raw_pose_audit={k:v for k,v in raw.items()if k!='rows'},
        paired_errors=[r['errors']for r in m['versions']],paired_prefix_mismatches=[r['pairing_mismatches']for r in m['versions']]))
    for name in ('metrics.json','paired.csv','protocol.json','runtime_selection_proof.json',
                 'host_checked_boundaries.json','model_recheck_summary.json','policy_frozen.json',
                 'raw_replay_audit.json','cost_accounting.json','runtime_protocol.json','host_amendments.json','program_rechecks.json','semantic_registry.json','checker_amendment.json','analysis_environment.json','online_error_audit.json'):
        shutil.copy2(OUT/name,REPORT/name)
    shutil.copy2(OUT/'models/manifest.json',REPORT/'models.json')
    shutil.copy2(AUTHOR/'FINAL.json',REPORT/'author_summary.json')
    shutil.copy2(frozen['source'],REPORT/'policy.py')
    rows=[]
    for version,data in m['tasks'].items():
        for r in data['rows']:
            online=r.get('online',{});rollout=online.get('rollout',{})
            rows.append(dict(version=version,room=r['room'],success=r['success'],ticks=r['ticks'],decisions=r['decisions'],
                task_cost_seconds=r['task_cost_seconds'],object_static_commands=r['object_static_commands'],
                robot_static_commands=r['robot_static_commands'],peer_contact_commands=r['peer_contact_commands'],
                brake_commands=r['brake_commands'],goal_gain=r['goal_gain'],error=r['error'],
                online_branches=rollout.get('branches',0),online_ticks=rollout.get('ticks',0),
                online_seconds=rollout.get('seconds',0),incompatible=rollout.get('incompatible',0),
                budget_exhausted=rollout.get('budget_exhausted',0)))
    with (REPORT/'rooms.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    replays()
    plot_results(m)
    worked_cases()
    strata=action_strata()
    lines=['# 可抽取 CAP 规则与预测模型迭代','',
        f"框架及三版冻结闭环已实现。局部主假设的配对区间支持：{m['main_hypothesis']['paired_intervals_support_all']}。但 v2 整局只成功 {m['tasks']['2']['success']}/{expected}，同一 CAP 不加预测门为 {m['tasks']['ungated']['success']}/{expected}；v2 仍有 {m['tasks']['2']['object_static_rooms']}/{expected} 间出现箱体静态接触。当前结果没有实现安全保证与较高整局完成率的兼顾。",'',
        '同一份字面量规则驱动运行和符号检查。作者程序冻结后，仅更新效果模型；确认集没有用于拟合或选策略。以下结果仅适用于 compact20 的固定对角起点布局族。','',
        '![三版配对指标和整局成功](metrics.png)','',
        '## 配对候选结果','',
        '|模型|候选/边界|放行|有效覆盖 C|接续失败 F|每百边界在线分支 K|放行后静态接触|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for k,r in enumerate(m['versions']):
        lines.append(f"|v{k}|{r['candidates']}/{r['boundaries']}|{r['admitted']}|{pct(r['C'])}|{pct(r['F'])}|{number(r['K'])}|{r['admitted_static']}|")
    lines+=['','C 的分母是全部固定受测候选，分子是获准且无箱体静态接触、实际后继完成的候选。F 只衡量获准后的接续未完成；接触另列。无放行时 F 未定义。停止不算有效覆盖。K 计入根边界与实际后继决策的全部在线分支，分母也包括这两类真实受测边界；一个边界试算五个分支计五次。',
        '候选池包含宿主提供的替代档位；metrics.json 同时给出仅含作者规则动作的 authored_C/F，以及只含原始首选的 nominal_first_C。候选覆盖不能替代下表的真实整局执行结果。','',
        '|比较|ΔC，百分点 [95%区间]|ΔF，百分点 [95%区间]|ΔK [95%区间]|','|---|---|---|---|']
    for d in m['differences']:
        vals=[]
        for key in ('C','F','K'):
            r=d['metrics'][key];ci=r['ci95'];fmt=(lambda x:'未定义'if x is None else f'{100*x:.2f}')if key!='K'else number
            vals.append(fmt(r['difference'])+(' ['+fmt(ci[0])+', '+fmt(ci[1])+']'if ci else ' [未定义]'))
        lines.append(f"|v{d['after']}−v{d['before']}|"+'|'.join(vals)+'|')
    h=m['main_hypothesis']
    lines+=['',f"预定容差为 C 下降不超过 5 个百分点、F 上升不超过 5 个百分点，且 K 严格减少。点估计同时满足：{h['point_criteria_met']}；房间配对区间同时支持：{h['paired_intervals_support_all']}。区间为 2000 次按房间重采样的 95% 百分位区间。",'',
        '## 预测更新与放行依据','',
        '|模型|拟合房间|效果记录|有数值校准的动作|模型直接放行|在线经验放行|模型放行子集联合覆盖|实测承诺违约|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model,r in zip(models,m['versions']):
        lines.append(f"|v{model['version']}|{model['rooms']}|{model['rows']}|{model['calibrated_actions']}|{r['model_admissions']}|{r['online_admissions']}|{pct(r['model_admitted_joint_coverage'])} (n={r['model_admitted_coverage_denominator']})|{r['admitted_contract_violations']}|")
    lines+=['','联合覆盖同时核对所有已预测语义量与过程安全标签。最后一列在全部放行候选中检查实际效果承诺和实际后继启动条件，与 F 的“后继是否按时完成”不同。数值校准动作数只表示有校准数据，不表示满足联合风险上界。','',
        '|版本变化|保留放行|撤回放行|新增放行|继续暂缓|','|---|---:|---:|---:|---:|']
    for change,r in rechecks.items():
        lines.append(f"|{change}|{r['maintained']}|{r['withdrawn']}|{r['relaxed']}|{r['still_withheld']}|")
    lines+=['','|版本|根动作类型|候选|放行|安全且后继完成|放行后继未完成|','|---|---|---:|---:|---:|---:|']
    for r in strata['root_action_groups']:
        lines.append(f"|v{r['version']}|{r['root_action_family']}|{r['candidates']}|{r['admitted']}|{r['effective']}|{r['successor_failed']}|")
    orbit=strata['orbit_to_newly_legal_push'][-1]
    lines+=['',f"v2 的 {orbit['executed_orbit_candidates']} 个实测绕行候选中，实际选中的 PUSH 从绕行前不可启动变为可启动的有 {orbit['newly_legal_actual_selected_push']} 个，其中后继完成 {orbit['completed']} 个。这是全部强制候选的诊断，含未放行候选。作者开发阶段的新法向案例另见 author_summary.json，不能当成新确认房的机制泛化证据。",'']
    lines+=['',
        '## 从头执行的整局结果','',
        '|版本|整局成功|无箱体静态接触的成功|有箱体静态接触的房间|平均任务成本（秒）|在线分支|决策边界|K|执行异常|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for version,r in m['tasks'].items():
        kval=100*r['online_branches']/r['boundaries']if r['boundaries']else None
        safe_success=sum(row['success']and not row['object_static_commands']for row in r['rows'])
        lines.append(f"|{version}|{r['success']}/{r['rooms']}|{safe_success}/{r['rooms']}|{r['object_static_rooms']}/{r['rooms']}|{number(r['mean_cost'])}|{r['online_branches']}|{r['boundaries']}|{number(kval)}|{r['errors']}|")
    lines+=['','ungated 为同一冻结 CAP、不使用预测放行门的参考。失败成本固定为环境时间预算的两倍；运行异常留在分母。逐房间的停止、接触、物理 ticks、在线预算耗尽与仿真起点不相容见 rooms.csv。','',
        '|版本|停止命令/全部边界|出现在线预算耗尽的房间|在线假设构造数|未通过相容检查|在线耗时（秒）|',
        '|---|---:|---:|---:|---:|---:|']
    for version,r in m['tasks'].items():
        ledgers=[row.get('online',{}).get('rollout',{})for row in r['rows']]
        lines.append(f"|{version}|{sum(row['brake_commands']for row in r['rows'])}/{r['boundaries']}|{sum(x.get('budget_exhausted',0)>0 for x in ledgers)}/{r['rooms']}|{sum(x.get('models',0)for x in ledgers)}|{sum(x.get('incompatible',0)for x in ledgers)}|{number(sum(x.get('seconds',0)for x in ledgers))}|")
    lines+=['','K 应与有效覆盖、整局完成和停止量一起解释。在线几何构造及相容检查也消耗计算时间；未执行物理分支的构造不计为 rollout 分支，单独列在这里。墙钟上限在分支边界检查，可能有一个分支的超限；运行负载也可能影响何时触及墙钟预算，因此冻结程序和随机种子不等于逐位重现墙钟受限决策。','']
    online_errors=read(OUT/'online_error_audit.json')
    lines += [f"在线假设仿真另有 {online_errors['errors']} 次被捕获的 `push_no_legal_face`，集中在 {online_errors['queries']} 次接续查询；这些查询的放行数为 {online_errors['admitted_queries']}，均未取得有效后果样本。错误分支保留预算记账，不计为成功证据。这与真实物理评估的执行异常分开报告，详见 online_error_audit.json。",'']
    example=next((row for row in m['tasks']['2']['rows']if row['success']and not row['object_static_commands']),None)
    if example:
        lines+=['## 一个完整任务的配对案例','',
            f"按房间名排序，v2 首个无箱体静态接触的成功是 `{example['room']}`。这是预定确认集中的单个例子，不能替代全部房间的分母；对应真实轨迹见回放。",'',
            '|版本|完成|真实向目标净推进（米）|停止命令|模型放行|在线物理分支|','|---|---:|---:|---:|---:|---:|']
        for version,group in m['tasks'].items():
            row=next(r for r in group['rows']if r['room']==example['room']);ledger=row.get('online',{})
            lines.append(f"|{version}|{row['success']}|{row['goal_gain']:.3f}|{row['brake_commands']}|{ledger.get('model_admissions',0)}|{ledger.get('rollout',{}).get('branches',0)}|")
        lines+=['']
    lines+=['## 实现与证据边界','',
        '- 元表达式带单位；选择规则、评分和并列顺序可抽取。检查器预测联合后果后重新计算实际选择。未预测的受影响值成为 unknown。','- 对任意运行期非停止权限，检查每个启用规则蕴含其执行条件，并检查规则覆盖。该证明只保证启动条件，不保证动作完成或物理安全。编译 CAP 与解释器的首命令在宿主选定的开发中途边界逐项核对。',
        '- 作者的效果承诺位于 policy.py 的 SYMBOLIC_POLICY.expect；模型只预测公开语义摘要与过程接触标签。有限联合模式保留离散关联，模式内连续区间是保守包络。',
        '- v0/v1/v2 逐批增加开发房间；用前一冻结版实际运行，在不确定或停止边界补采。缺少局部支持、预测后果改变放行判断的案例优先进入下一版，并加入固定高置信抽查。校准房与拟合房分离，但校准房也用于开发诊断，不是最终确认集。开发采样与在线仿真分别计费。',
        '- 在线仿真从公开回波拟合箱体，并使用历史静态回波，构造多个未知深度假设。假设须通过声明的 8 cm 测距/自由空间容差检查；这属于近似观测约束，并未证明与全部无噪声历史严格相容。不使用真实执行状态克隆。已有预测反例未被排除时，名义成功不能覆盖它；在线后果与保留的预测模式一起检查。未观测空间、形状完备性与控制器冷启动仍无覆盖保证，所以此处是经验放行。',
        '- 每个候选先保存预测，再实际从同一物理前缀执行。接续由冻结策略选择，最多重复该命令 120 ticks：PUSH 向目标净推进至少 2 cm；绕行得到目标方向可执行 PUSH；MOVE 两台机器人沿指令方向各移动至少 5 cm。整局评估则在每一真实边界重新选择。',
        '- 符号证明相对于给定效果集合成立。0.10 是冻结筛选阈值；逐分量数值残差校准不构成联合 90% 覆盖保证，也不是获准子集的违约率上界。经验集合（含过程安全标签）的真实覆盖率和获准子集违约见 metrics.json。零次观测碰撞也不等于零风险。',
        '- 几何版本为直径 20 cm、绕行目标净距 5 cm、当前可见面 PUSH 法向阈值 45°、72 条机体坐标射线。5 cm 是控制目标，不是已证明的硬安全下界。',
        '', '## 复核入口','',
        '- [逐房间表](rooms.csv) · [同前缀候选](paired.csv) · [真实执行回放](replay.html)',
        '- [四类逐步检查案例](worked_cases.md) · [案例完整数据](worked_cases.json)',
        '- [动作类型与绕行后新可执行 PUSH](action_strata.json) · [作者开发记录与局部探针](author_summary.json)',
        '- [模型迭代指标](metrics.json) · [修改影响复查](model_recheck_summary.json) · [运行期选择条件证明](runtime_selection_proof.json)',
        '- [作者程序修订依赖](program_rechecks.json)（含竞争规则可能被改选的保守影响范围）',
        '- [157 个基础量的单位、来源、刷新与失效约定](semantic_registry.json)',
        '- [逐 tick 原始位姿复算](raw_replay_audit.json) · [开发、离线、在线仿真账目](cost_accounting.json) · [正式冻结前的宿主修订](host_amendments.json)',
        '- [工程与证据复核](validation.json)：25 项框架检查及包含它们的 59 项相关检查通过；没有宣称运行全仓测试。',
        '- [分析依赖](analysis_environment.json)：物理评估结束后补装 SciPy 1.15.3，用于二项风险区间分位数；保留原始缺依赖报错，NumPy 与策略执行源码未改。',
        '- [在线假设异常复查](online_error_audit.json)：逐查询保留发生原因、样本数和最终放行决定。',
        '- [冻结策略](policy.py) · [作者提交与源指纹](policy_frozen.json) · [模型指纹](models.json) · [冻结协议](protocol.json)',
        '- [单房间重放脚本](reproduce_one.py)：在仓库根目录运行，先校验全部冻结输入，结果写到新文件，不覆盖历史记录。',
        '', '```bash',
        '.venv/bin/python docs/research/key_findings/2026-11-25_symbolic_general/reproduce_one.py --version 2 --room diag_symbolic_confirmation_003 --output /tmp/cap-symbolic-room003.json.gz',
        '```',
        '',f"原始记录目录：`{OUT}`。作者 campaign verify_frozen：`{campaign['sha256']}`。",'']
    (REPORT/'README.md').write_text('\n'.join(lines))
    registry=ROOT/'docs/research/experiment_journal/entries.json';entries=read(registry)
    entry=next(e for e in entries if e['id']=='symbolic-general');a,b=m['versions'][0],m['versions'][2]
    entry.update(result=f"{m['paired_rooms']} 间确认房，v0→v2 的有效覆盖 {pct(a['C'])}→{pct(b['C'])}、接续失败 {pct(a['F'])}→{pct(b['F'])}、每百边界在线分支 {number(a['K'])}→{number(b['K'])}。v2 整局成功 {m['tasks']['2']['success']}/{expected}。",
        analysis=f"局部主假设的配对区间支持：{h['paired_intervals_support_all']}。整局 v2 {m['tasks']['2']['success']}/{expected}，同 CAP 无预测门 {m['tasks']['ungated']['success']}/{expected}；v2 箱体静态接触 {m['tasks']['2']['object_static_rooms']}/{expected} 间。局部改善未转化为较高整局完成率，安全保证未成立。",
        assets=[[x,y]for x,y in [('metrics.json','配对指标'),('rooms.csv','逐房间结果'),('paired.csv','逐候选对照'),('replay.html','真实执行回放'),('policy.py','冻结 CAP 规则'),('models.json','三版预测器指纹'),('model_recheck_summary.json','修改影响复查'),('runtime_selection_proof.json','运行期选择证明'),('host_checked_boundaries.json','编译 CAP 发令核对'),('protocol.json','冻结协议'),('policy_frozen.json','作者提交与冻结'),('raw_replay_audit.json','原始位姿复算'),('cost_accounting.json','仿真账目'),('host_amendments.json','宿主修订'),('runtime_protocol.json','在线放行规则')]])
    entry['assets'] += [['program_rechecks.json','程序修订依赖'],['semantic_registry.json','基础量约定'],['checker_amendment.json','检查器等价优化']]
    entry['assets'] += [['metrics.png','指标图'],['metrics.svg','可编辑指标图']]
    entry['assets'] += [['worked_cases.md','从预测到实际后继'],['worked_cases.json','案例原始数据']]
    entry['assets'] += [['validation.json','工程与证据复核'],['tests_framework.xml','框架测试记录'],['tests_related.xml','相关测试记录']]
    entry['assets'] += [['action_strata.json','动作类型与新法向诊断'],['author_summary.json','作者开发与探针']]
    entry['assets'] += [['reproduce_one.py','单房间重放脚本']]
    entry['assets'] += [['analysis_environment.json','统计分析环境']]
    entry['assets'] += [['online_error_audit.json','在线假设异常复查']]
    save(registry,entries)
    receipt=dict(status='complete',at=time.time(),metrics_sha256=sha(OUT/'metrics.json'),
        report=str(REPORT/'README.md'),policy_sha256=frozen['source_sha256'],model_sha256=frozen['model_sha256'],
        main_hypothesis=h,actual_final_task_successes=m['tasks']['2']['success'])
    save(OUT/'FINISHED.json',receipt)
    return receipt


if __name__=='__main__':print(json.dumps(publish(),ensure_ascii=False))
