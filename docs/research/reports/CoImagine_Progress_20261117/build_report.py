"""Build a self-contained supervisor briefing from existing frozen evidence only."""
from pathlib import Path
import base64,gzip,hashlib,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from latest_evidence import enrich, render

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
EVIDENCE=[]

def read(path):
    p=ROOT/path;b=p.read_bytes()
    EVIDENCE.append(dict(path=str(path),sha256=hashlib.sha256(b).hexdigest()))
    return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)

def save(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def compact_result(path,title,note):
    r=read(path);replay=r['replay'];frames=replay['frames'];indices=set(range(0,len(frames),2))|{0,len(frames)-1}
    # Keep every option boundary as well as the 10 Hz visualization samples.
    bounds={x['event']['start_tick'] for x in r['decisions']}
    indices|={i for i,f in enumerate(frames) if f['tick'] in bounds}
    out=[]
    for i in sorted(indices):
        f=frames[i];out.append(dict(t=round(f['t'],4),tick=f['tick'],entities={k:[round(v,6) for v in p[:3]] for k,p in f['entities'].items()},robots=[[round(v,6) for v in p[:3]] for p in f['robots']],active_target=f['active_target']))
    actions=[dict(t=x['event']['start_tick']*.05,end=x['event']['end_tick']*.05,action=x['action'],reason=x['event']['reason']) for x in r['decisions']]
    return dict(title=title,note=note,scene=replay['scene'],frames=out,actions=actions,summary={k:r['summary'][k] for k in ['success','ticks','decisions','stop','error']},source=str(path),fixture_sha256=r['fixture_sha256'])

def choose(campaign,condition,case):
    s=read(campaign+'/campaign.json');r=next(r for r in s['runs'] if r['condition']==condition and r['case_id']==case)
    return campaign+'/runs/'+r['id']+'/result.json.gz'

def diagram_scene(scene,frame):
    width,height=scene['arena'];sx=80;pad=14;W=width*sx+2*pad;H=height*sx+2*pad
    def xy(p):return (pad+p[0]*sx,pad+(height-p[1])*sx)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="双机器人平面运输场景"><rect width="100%" height="100%" fill="#f3f6fa"/>']
    def poly(points,color,stroke):
        parts.append('<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in map(xy,points))+f'" fill="{color}" stroke="{stroke}" stroke-width="1.5"/>')
    def rotated(points,pose):
        x,y,a=pose[:3];c,s=math.cos(a),math.sin(a);return [[x+c*u-s*v,y+s*u+c*v] for u,v in points]
    for b in scene['obstacles']:
        w,h=b['half_w'],b['half_h'];poly(rotated([[-w,-h],[w,-h],[w,h],[-w,h]],[b['cx'],b['cy'],b.get('angle',0)]),'#475569','#334155')
    for b in scene['movable_objects']:
        w,h=b['width']/2,b['height']/2;poly(rotated([[-w,-h],[w,-h],[w,h],[-w,h]],frame['entities'][b['label']]),'#efb75a','#a56b13')
    poly(rotated(scene['object_polygon'],frame['entities'][scene['primary_id']]),'#66a7d7','#18578e')
    g=scene['goal'];x,y=xy([g['x'],g['y']]);parts.append(f'<circle cx="{x}" cy="{y}" r="{g["radius"]*sx}" fill="none" stroke="#9364ce" stroke-width="2" stroke-dasharray="5 3"/><text x="{x}" y="{y-24}" text-anchor="middle" font-size="13" fill="#6b46a0">GOAL</text>')
    for i,p in enumerate(frame['robots']):
        x,y=xy(p);parts.append(f'<circle cx="{x}" cy="{y}" r="{scene["robot_radius"]*sx}" fill="{["#259b8d","#e77960"][i]}" stroke="white" stroke-width="2"/><text x="{x}" y="{y+4}" text-anchor="middle" font-size="11" fill="white">R{i}</text>')
    parts.append('</svg>');return ''.join(parts)

def build():
    h='outputs/semantic_rl_authors/rel_transfer_20261117_01'
    latest=read(h+'/RESULT.json');assert read(h+'/FINISHED.json')['status']=='complete'
    assert read(h+'/eval_resume.exit.json')['code']==0 and len(latest['rows'])==560
    attr=read('outputs/semantic_rl_authors/input_attribution_v2_20261117_01/RESULT.json')
    large='outputs/semantic_rl/large_coverage_evaluation_20261116_01'
    transfer='outputs/semantic_rl/rel_transfer_evaluation_20261117_02'
    coverage_rows=read(large+'/campaign.json')['runs']
    assert [sum(r['success'] for r in coverage_rows if r['condition']==a and r['split']=='test') for a in ['base','repair','control','expanded']]==[92,119,105,136]
    extraction=read('outputs/semantic_rl/rel_transfer_data_20261117_01/extraction.json')
    assert extraction['demonstrations']==45 and extraction['boundaries']==1950
    rows=latest['rows'];caps={r['case_id']:r['success'] for r in rows if r['condition']=='CAP_s006'}
    agreement={a:sum(r['success']==caps[r['case_id']] for r in rows if r['condition']==a) for a in ['GA','DA','GDA']}
    assert agreement==dict(GA=80,DA=80,GDA=80)
    names=['G','GA','DA','GDA'];old={str(n):[attr['groups']['test'][f'{a}_{n}']['success'] for a in names] for n in [50,150]}
    assert old=={'50':[82,113,111,116],'150':[124,124,128,121]}
    final={split:[latest['groups'][split][a]['success'] for a in names+['CAP_s006']] for split in ['iid','transfer']}
    replays=[
      compact_result(choose(transfer,'CAP_s006','root_east_blocked'),'清障 → 换位 → 运输','Sol最终CAP；开发锚点成功。展示可动物体B确实被先移开。'),
      compact_result(choose(large,'expanded','root_wall_release'),'贴墙释放 → 输运','150示范监督模型；原锚点成功。展示已有技能能形成完整操作。'),
      compact_result(choose(large,'repair','test_0025'),'修复CAP：同一初态失败','按结果选取的说明性案例；该初态两份固定CAP均失败。'),
      compact_result(choose(large,'expanded','test_0025'),'监督整合：同一初态成功','与左例相同初态；属于教师并集之外的5个成功之一，不等于已经证明新组合技能。'),
      compact_result(choose(transfer,'CAP_s006','transfer_0036'),'跨布局：统一CAP失败','斜向留出；与右例相同初态。展示程序先验在新布局上的适用边界。'),
      compact_result(choose(transfer,'G','transfer_0036'),'跨布局：几何G成功','G在48个跨布局任务中成功7例之一；结果选例，不代表总体成功率高。'),
    ]
    assert not replays[2]['summary']['success'] and replays[3]['summary']['success']
    assert replays[2]['fixture_sha256']==replays[3]['fixture_sha256']
    assert not replays[4]['summary']['success'] and replays[5]['summary']['success']
    assert replays[4]['fixture_sha256']==replays[5]['fixture_sha256']
    panels=read('outputs/semantic_rl/rel_transfer_parent_20261117_01/panels/roots.json')['cases'];scenes=[]
    for root in panels:
        p=choose(transfer,'CAP_s006',root['id']);r=read(p);scenes.append(dict(id=root['base_id'],layout=root['layout'],relation=root['relation'],scene=r['replay']['scene'],frame=r['replay']['frames'][0]))
    for s in scenes:(HERE/(s['id']+'.svg')).write_text(diagram_scene(s['scene'],s['frame']))
    for i,r in enumerate(replays):(HERE/f'replay_{i}_start.svg').write_text(diagram_scene(r['scene'],r['frames'][0]))
    source_docs=[
      ('预测工具使用审计','docs/agent_exchange/reviews/Independent_Predictor_20261110.md'),
      ('结构化搜索诊断','docs/agent_exchange/reviews/Option_Search_Failure_Analysis_20261110.md'),
      ('CAP三锚点与泛化','docs/agent_exchange/reviews/Authoring_V2_Sol_Generalization_20261111.md'),
      ('完整动作PPO','docs/agent_exchange/reviews/Duration_Stream_PPO_Final_20261115.md'),
      ('有限参数长训PPO','docs/agent_exchange/reviews/Bounded_Direction_Long_Final_20261116.md'),
      ('扩大示范覆盖','docs/agent_exchange/reviews/CAP_Large_Coverage_20261116.md'),
      ('输入归因与编码修复','docs/agent_exchange/reviews/CAP_Input_Attribution_Repair_20261117.md'),
      ('关系迁移任务','docs/research/Relational_Transfer_Task_Suite_v1_20261117.md'),
      ('本轮预定协议与恢复账本','docs/agent_exchange/tasks/Relational_Transfer_CAP_20261117.md'),
      ('研究主张v4','docs/research/CAP_Semantic_Handoff_Research_Plan_v4_20261116.md'),
      ('最新关系迁移完整结果','docs/agent_exchange/reviews/Relational_Transfer_Attribution_20261117.md'),
    ]
    for title,path in source_docs:
        b=(ROOT/path).read_bytes();EVIDENCE.append(dict(path=path,title=title,sha256=hashlib.sha256(b).hexdigest()))
    data=dict(date='2026-11-17',arms=names,attribution=old,transfer=final,coverage=dict(labels=['基础CAP','修复CAP','旧数据等预算','扩大覆盖监督'],counts=[92,119,105,136],n=216),replays=replays,scenes=scenes,labels=read('outputs/testbed/runs/20261117_030148_aa88db.json')['action_labels'],sources=[dict(title=t,path=p) for t,p in source_docs],current_summary=dict(episodes=560,ticks=latest['ticks'],fit_actions=1950,demos=45,success_label_agreement=agreement,error_count=sum(bool(r['error']) for r in rows),seals=latest['seals']))
    latest, added_replays = enrich(data, read, compact_result)
    for source in data['sources']:
        p = ROOT/source['path']
        EVIDENCE.append(dict(path=source['path'], title=source['title'], sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    for i in added_replays:
        r = data['replays'][i]
        (HERE/f'replay_{i}_start.svg').write_text(diagram_scene(r['scene'],r['frames'][0]))
    # Exportable, standalone scientific figure (all results from the cited tables).
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,3.7));colors=['#64748b','#4c82bb','#8f70b6','#218c7e']
    axes[0].bar(range(4),np.array([92,119,105,136])/216*100,color=colors)
    axes[0].set_xticks(range(4),['CAP-1','CAP-2','BC-13','BC-150']);axes[0].set_title('Coverage expansion / n=216')
    for i,v in enumerate([92,119,105,136]):axes[0].text(i,v/216*100+2,f'{v}/216',ha='center',fontsize=9)
    x=np.arange(4)
    for offset,n,color in [(-.18,50,'#9cabc0'),(.18,150,'#218c7e')]:
        axes[1].bar(x+offset,np.array(old[str(n)])/216*100,width=.34,label=f'{n} demonstrations',color=color)
    axes[1].set_xticks(x,names);axes[1].set_title('Input attribution / n=216');axes[1].legend(frameon=False,fontsize=8)
    x=np.arange(5)
    for offset,split,n,color,label in [(-.18,'iid',24,'#9cabc0','East/North held-out'),(.18,'transfer',48,'#218c7e','West/Diagonal held-out')]:axes[2].bar(x+offset,np.array(final[split])/n*100,width=.34,color=color,label=label)
    axes[2].set_xticks(x,names+['CAP']);axes[2].set_title('New layout transfer');axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:ax.set_ylim(0,104);ax.set_ylabel('Success (%)');ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(HERE/'results_overview.png',dpi=180);fig.savefig(HERE/'results_overview.svg');plt.close(fig)
    image=base64.b64encode((HERE/'results_overview.png').read_bytes()).decode()
    template=(HERE/'template.html').read_text();content=template.replace('__LATEST__',render(latest)).replace('__CSS__',(HERE/'report.css').read_text()).replace('__SCRIPT__',(HERE/'report.js').read_text()).replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')).replace('__OVERVIEW__',image)
    (HERE/'index.html').write_text(content)
    unique={e['path']:e for e in EVIDENCE};save('evidence_manifest.json',dict(date=data['date'],files=list(unique.values()),report_sha256=hashlib.sha256(content.encode()).hexdigest(),visualization='existing privileged replay, decimated to 10Hz plus option boundaries; rounded to 1e-6 for display only',selection='illustrative outcome-selected cases; aggregate results include all specified cases; no new physics/fit/model calls'))
    save('summary.json',{k:v for k,v in data.items() if k not in ['replays','scenes','random_scenes','labels']})
    print('built',HERE/'index.html','bytes',len(content.encode()),'replays',len(replays))

if __name__=='__main__':build()
