"""Report coverage, perform the sealed family tests, and publish the live author ledger."""
import csv
import html
import time
from pathlib import Path

from tools import cap_focus_recovery_five as run


def usage(s):
    path=s.get('telemetry')
    if not path:return 0
    return int((run.tail_usage(Path(path)) or {}).get('total_tokens') or 0)


def census_table(s, root):
    members=[(r['number'],r['sid']) for r in s['rounds']]
    active=s.get('active')
    if active and s['stage'] in ('commit','census'):
        sid=active.get('choice')
        if sid is None:
            previous={sid for _,sid in members}
            sid=next((p.name for p in (run.OUT/'census').glob('s*') if p.name not in previous),None)
        members.append((active['number'],sid))
    if not members:
        return '<h2>全开发池扫描</h2><p>首轮成员选定后扫描 160 个原始初态。</p>'
    records=[]
    lines=[]
    union=set()
    for number,sid in members:
        rows=[run.read(p) for p in sorted((run.OUT/'census'/sid).glob('*.json'))] if sid else []
        wins={r['id'] for r in rows if r['complete'] and r['success']}
        errors=sum(not r['complete'] for r in rows)
        complete=len(rows)-errors
        added=len(wins-union)
        union.update(wins)
        state='完成' if complete==160 else '有异常' if errors else '扫描中'
        cells=[number,sid or '待扫描',state,f'{complete}/160',len(wins),complete-len(wins),errors,added,len(union)]
        lines.append('<tr>'+''.join(f'<td>{html.escape(str(c))}</td>' for c in cells)+'</tr>')
        records.extend([number,sid,r['id'],r['complete'],r['success'] if r['complete'] else '',r['ticks'],r['termination'],r['error']] for r in rows)
    tmp=root/'census.csv.tmp'
    with tmp.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['round','sid','case','complete','success','ticks','termination','error'])
        writer.writerows(records)
    tmp.replace(root/'census.csv')
    return ('<h2>全开发池扫描</h2><table><tr><th>轮</th><th>成员</th><th>状态</th><th>已完成</th>'
            '<th>成功</th><th>失败</th><th>异常</th><th>新增成功</th><th>累计并集</th></tr>'
            +''.join(lines)+'</table><p><a href="census.csv" target="_blank">逐场景扫描结果（CSV）</a></p>')


def render(s):
    root=run.OUT/'live'
    root.mkdir(exist_ok=True)
    a=s.get('active')
    done=s['rounds']
    def cells(xs):
        return '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in xs)+'</tr>'
    lines=[]
    for r in done:
        lines.append(cells([r['number'],r['center'],r['sid'],'成功' if r['center_success'] else '失败',
            str(r['visible'])+'/5',len(r['wins']),r['added'],r['total']]))
    extra=''
    if a:
        for r in a['trials']:
            extra+=cells([r['sid'],', '.join(r['wins']) or '无',str(len(r['wins']))+'/5'])
    finish=run.OUT/'FINISHED.json'
    result=''
    if finish.exists():
        m=run.read(finish)
        result='<h2>冻结后未见扰动</h2><p>'+html.escape('各轮指定成员成功数（每轮12例） '+str([r['own_hidden'] for r in m['rounds']]))+'</p>'
    stages={'prepared':'准备','author':'作者开发中','commit':'成员已选定','census':'冻结本轮候选，统计全池覆盖','frozen':'作者库已冻结','incomplete':'未完整完成','complete':'已完成'}
    now=time.time()
    title=('按要求在第 '+str(len(done))+' 轮全池扫描后停止 · 第 4、5 轮已取消'
           if s['stage']=='stopped' else stages[s['stage']])
    if a:title+=' · 第 '+str(a['number'])+' 轮'
    clock=''
    if a:
        clock=f'<p>本轮剩余 {max(0,int(a["deadline"]-now))} 秒 · 中心 {html.escape(a["center"] or "待选择")}</p>'
    if finish.exists():title='已完成五轮及隐藏扰动评测'
    closed=run.OUT/'author_closed.json'
    budget=run.read(closed) if closed.exists() else {}
    seconds=budget.get('wall_seconds',max(0,int(now-(s.get('start') or now))))
    token=budget.get('tokens',usage(s))
    census=census_table(s,root)
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="5"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五轮区域 CAP</title><style>body{{font:16px system-ui,sans-serif;background:#f4f5f0;color:#172d31;max-width:1000px;margin:4vw auto;padding:0 18px}}h1{{font-size:2.1rem}}h2{{margin-top:2rem}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #d3d9d2}}p{{line-height:1.55}}small{{color:#47626a}}</style><h1>区域中心与故障恢复</h1>{census}<h2>{title}</h2>{clock}<p>作者 tokens {token:,} / 40,000,000 · 已用墙钟 {int(seconds)} 秒 / {run.MAX_SECONDS:,} 秒</p><table><tr><th>轮</th><th>中心</th><th>成员</th><th>原初态</th><th>中心族</th><th>单成员 /160</th><th>新增</th><th>并集 /160</th></tr>{''.join(lines)}</table>'''
    if extra:body+='<h2>本轮已测版本</h2><table><tr><th>版本</th><th>成功初态</th><th>中心族</th></tr>'+extra+'</table>'
    body+=result+'<p><small>中心族含原初态与四个可见微扰。tokens 含缓存输入；全池覆盖在成员选定后统计。</small></p></html>'
    tmp=root/'index.tmp'
    tmp.write_text(body)
    tmp.replace(root/'index.html')


def evaluate(s):
    assert run.HOST(run.CAM).verify_frozen()==s['seal']
    members=s['rounds']
    assert len(members)==5 and len({r['sid'] for r in members})==5
    fixtures={f"r{r['number']}_{k}":v for r in members for k,v in run.family(r['center'],True).items()}
    scores={}
    for i,r in enumerate(members):
        rows=run.execute(r['sid'],fixtures,run.OUT/'hidden'/r['sid'],16)
        if len(rows)!=60 or any(not v['complete'] for v in rows):raise RuntimeError('incomplete hidden evaluation')
        scores[r['sid']]={v['id']:int(v['success']) for v in rows}
        run.save(run.OUT/'hidden_progress.json',dict(finished=i+1,members=5))
    lines=[]
    for idx,r in enumerate(members):
        keys=[k for k in fixtures if k.startswith(f"r{idx+1}_")]
        prior={k for p in members[:idx] for k in keys if scores[p['sid']][k]}
        own={k for k in keys if scores[r['sid']][k]}
        lines.append(dict(number=idx+1,center=r['center'],sid=r['sid'],visible=r['visible'],center_success=r['center_success'],
            own_hidden=len(own),prior_hidden=len(prior),added_hidden=len(own-prior),dev_wins=len(r['wins']),added=r['added'],union=r['total']))
    run.REPORT.mkdir(exist_ok=True)
    with (run.REPORT/'hidden.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['case',*[r['sid'] for r in members]])
        for k in sorted(fixtures):w.writerow([k,*[scores[r['sid']][k] for r in members]])
    metrics=dict(rounds=lines,closed=run.read(run.OUT/'author_closed.json'),seal=s['seal'],hidden_runs=300)
    run.save(run.OUT/'FINISHED.json',metrics)
    md=['# 单中心与厘米级扰动的五轮 CAP 开发','',
        '一位 GPT-6.1 Sol High 作者，从空库开始，只访问公共指南和本次已写源代码。160 个原始开发初态沿用上一轮的池。作者每轮选取库中尚未解决的一个中心，只在原初态和四个固定微扰上调试。微扰同时平移两台机器人，各自移动 1、2、3 或 5 厘米；目标物、房间及朝向固定。宿主在每轮确定成员后做一次全池覆盖统计。','',
        '| 轮 | 中心 | 成员 | 中心成功 | 可见族 /5 | 未见扰动 /12 | 未见扰动相对前轮并集新增 | 开发成功 /160 | 开发新增 | 开发并集 /160 |','|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in lines:
        fields=[r[k] for k in ('number','center','sid','center_success','visible','own_hidden','added_hidden','dev_wins','added','union')]
        md.append('| '+' | '.join(map(str,fields))+' |')
    md+=['','每个中心的 12 个隐藏微扰在开发前生成，每个半径另取 3 个方向组合，五轮结束后才执行。每个成员都测五个中心的全部隐藏扰动，共 300 次。60 个隐藏初态共享五个父场景。中心族的成功或新增并集不能替代自然新房间或可部署混合 actor 的闭环结果。开发与隐藏的同族成功也可能来自更稳健的前期动作，独立证明恢复需要同故障前缀续跑。','',
        f"该次作者终结记账见 [预算记录](../../../../outputs/semantic_rl/{run.OUT.name}/author_closed.json)。[逐微扰成败](hidden.csv) · [开发及冻结记录](../../../../outputs/semantic_rl/{run.OUT.name}/state.json) · [各轮源码](../../../../outputs/semantic_rl/{run.OUT.name}/author/semantics)"]
    (run.REPORT/'README.md').write_text('\n'.join(md)+'\n')
    render(s)


if __name__=='__main__':
    while True:
        s=run.read(run.STATE)
        render(s)
        if s['stage'] in ('stopped','incomplete'):break
        if s['stage']=='frozen':
            if not (run.OUT/'author_closed.json').exists():
                end=run.read(run.OUT/'author_end.json') if (run.OUT/'author_end.json').exists() else None
                r=s['rounds'][-1]
                run.save(run.OUT/'author_closed.json',dict(tokens=usage(s),wall_seconds=round((end['at'] if end else r['accepted'])-s['start'],1),timing='final author finish command' if end else 'final accepted trial',final_sweep_separate=True))
            evaluate(s)
            break
        time.sleep(3)
