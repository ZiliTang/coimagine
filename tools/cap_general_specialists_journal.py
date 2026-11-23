"""Public engineering notes and observed development trials for the 6.2 CAP author."""
import difflib
import json
from datetime import datetime
from functools import lru_cache
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/semantic_rl/cap_general_specialists_62_20261120_01'
NOTES=ROOT.parent/'coimagine_author_workspaces/general_specialists_62_01/author_notes.json'
ZONE=ZoneInfo('Asia/Shanghai')


def read(p):
    return json.loads(p.read_text())


def diff(sid,parent):
    known={r['id'] for r in read(OUT/'author/campaign.json')['submissions']}
    if sid not in known or parent not in known:raise KeyError('unknown CAP')
    src=OUT/'author/semantics'
    before,after=[(src/(key+'.py')).read_text().splitlines(keepends=True) for key in (parent,sid)]
    return ''.join(difflib.unified_diff(before,after,fromfile=parent+'.py',tofile=sid+'.py'))


def page():
    book=read(OUT/'state.json')
    subs=read(OUT/'author/campaign.json')['submissions']
    dev=set(read(OUT/'development.json'))
    try:
        notes=read(NOTES)
    except (FileNotFoundError,json.JSONDecodeError):
        notes={}
    rounds=book['rounds']+([book['active']] if book['active'] else [])
    activity=read(OUT/'activity.json') if (OUT/'activity.json').exists() else {}
    by_sid={}
    for r in rounds:
        for t in r['trials']:
            ids=t.get('ids')
            if ids is None and t['kind']=='sweep':ids=dev
            if ids is None and r['n']>1 and t['eligible']:ids=r['ids']
            by_sid.setdefault(t['sid'],[]).append((t,set(ids) if ids is not None else None,r))
    committed={r['sid']:r for r in book['rounds']}
    cards=[]
    for i,sub in enumerate(subs):
        sid=sub['id'];name=sub['summary']
        note=notes.get(name,{}) if isinstance(notes,dict) else {}
        if not isinstance(note,dict):note={}
        prev=subs[i-1]['id'] if i else None
        parent_name=note.get('parent')
        parent=next((r['id'] for r in reversed(subs[:i]) if r['id']==parent_name or r['summary']==parent_name),None)
        ref=parent or prev
        round_no=by_sid[sid][-1][2]['n'] if sid in by_sid else activity.get('round','')
        lines=[f'<article id="{sid}"><header><h2>{sid} <small>{escape(name)}</small></h2><span>第 {round_no} 轮</span></header>']
        if note:
            for key,label in [('observed','观察'),('change','修改'),('check','检验')]:
                if isinstance(note.get(key),str) and note[key]:
                    lines.append(f'<p><b>{label}</b>　{escape(note[key])}</p>')
        else:lines.append('<p class="muted">作者说明待补。</p>')
        lines.append(f'<p><a href="/general62-source?sid={sid}" target="_blank">完整源码</a>')
        if ref:
            desc='对照作者注明的母本' if parent else '对照上一次提交'
            lines.append(f'　<a href="/general62-diff?sid={sid}&amp;parent={ref}" target="_blank">{desc} {ref} 的代码差异</a>')
        lines.append('</p>')
        for j,(trial,ids,r) in enumerate(by_sid.get(sid,[])):
            when=datetime.fromtimestamp(trial['finished'],ZONE).strftime('%H:%M:%S')
            label='全开发池' if trial['kind']=='sweep' else '开发试跑'
            errors=trial.get('errors')
            if errors is None and ids is not None:
                errors=sum(not cached_complete(str(OUT/'runs'/sid/(k+'.json'))) for k in ids)
            lines.append(f'<p><small>{when}</small>　{label}　<strong>{trial["score"]}/{trial["total"]}</strong>'+
                         (f'　异常 {errors}' if errors else '')+'</p>')
            if ids is None:continue
            wins=set(trial['wins'])
            same=next(((t,ks) for t,ks,_ in reversed(by_sid.get(ref,[])) if ks==ids),None)
            detail=[]
            if same and not errors and not same[0].get('errors') and all(cached_complete(str(OUT/'runs'/ref/(k+'.json'))) for k in ids):
                old=set(same[0]['wins']);plus=sorted(wins-old);minus=sorted(old-wins)
                lines.append(f'<p>与 {ref} 同场景比较　救回 <b>{len(plus)}</b>　丢失 <b>{len(minus)}</b></p>')
                detail.extend([('救回',plus),('丢失',minus)])
            detail.extend([('成功',sorted(wins)),('全部测试场景',sorted(ids))])
            lines.append(f'<details id="cases-{sid}-{j}"><summary>逐场景名单</summary>')
            for key,values in detail:
                lines.append(f'<p><b>{key}</b>　'+escape('、'.join(values) or '无')+'</p>')
            lines.append('</details>')
        if activity.get('stage')=='running' and activity['sid']==sid:
            ids=set(activity['ids']); done=sum(p.stem in ids for p in (OUT/'runs'/sid).glob('*.json'))
            lines.append(f'<p>本次试跑正在返回结果　{done}/{len(ids)}</p>')
        if sid in committed:
            c=committed[sid]
            lines.append(f'<p><strong>本轮入库</strong>　全池 {len(c["wins"])}/160，新增 {c["added"]}。{escape(c.get("note",""))}</p>')
        lines.append('</article>')
        cards.append(''.join(lines))
    return SHELL.replace('<!--cards-->',''.join(reversed(cards)) or '<p>等待首个提交。</p>')




@lru_cache(maxsize=16384)
def cached_complete(path):
    return bool(read(Path(path))['complete'])


SHELL="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sol 作者开发记录</title><style>
*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;background:#f4f5f1;color:#172d31;max-width:920px;margin:auto;padding:24px 18px}article{background:white;border:1px solid #d7ded6;border-radius:8px;padding:12px 20px;margin:12px 0}header{display:flex;justify-content:space-between;align-items:baseline;gap:12px}h1,h2,p{margin:7px 0}h2{font-size:21px}small,.muted,header span{color:#55665e;font-weight:400}a,summary{color:#006965}details{overflow-wrap:anywhere}summary{cursor:pointer}
</style><p><a href="/#general62">返回实验面板</a>　<a href="/general62-coverage">候选成功覆盖对照</a></p><h1>Sol 作者开发记录</h1><p class="muted">最新版本在前 · 作者提交说明与开发实测 · 北京时间</p><main><!--cards--></main><script>
async function update(){
 if(document.hidden||window.getSelection()?.toString())return;
 try{
 const resp=await fetch(location.pathname,{cache:'no-store'});if(!resp.ok)return;
 const doc=new DOMParser().parseFromString(await resp.text(),'text/html');
 const main=document.querySelector('main');const open=[...main.querySelectorAll('details[open]')].map(x=>x.id);
 const next=doc.querySelector('main');
 for(const id of open){const d=next.querySelector('#'+id);if(d)d.open=true}
 if(next.innerHTML!==main.innerHTML){const anchor=[...main.children].find(x=>x.getBoundingClientRect().bottom>0);
 const offset=anchor?.getBoundingClientRect().top;main.replaceWith(next);
 if(scrollY>0&&anchor?.id&&document.getElementById(anchor.id))scrollBy(0,document.getElementById(anchor.id).getBoundingClientRect().top-offset)}
 }catch(e){}
}
setInterval(update,5000);</script></html>"""
