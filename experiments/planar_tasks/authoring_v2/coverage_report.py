"""Auditable16-policy coverage matrices, interactive HTML and exportable plots."""
import csv
import json
from pathlib import Path

import numpy as np

from . import random_room_coverage_v2 as V

OUT=V.ROOT/'docs/research/reports/CAP_Coverage_Comparison_20261117'
GD_BASE=V.ROOT/'outputs/semantic_rl/gd_coverage_comparison_20261117_01'
RETRIEVAL=V.ROOT/'outputs/semantic_rl/program_retrieval_comparison_20261117_01'
NAMES=[f'CAP-RR-{i:03d}' for i in range(1,11)]+['G','GA','GD','GDA','R0','Rt']


def overlap_matrices(matrix):
    """Rows may be entirely NaN for pending methods; never turn missing into failure."""
    m=np.asarray(matrix,float);n=len(m)
    result={k:np.full((n,n),np.nan) for k in ('intersection','jaccard','rescue','containment')}
    for i,a in enumerate(m):
        if not np.isfinite(a).all():continue
        a=a.astype(bool)
        for j,b in enumerate(m):
            if not np.isfinite(b).all():continue
            b=b.astype(bool);intersection=int((a&b).sum());union=int((a|b).sum())
            result['intersection'][i,j]=intersection
            result['jaccard'][i,j]=intersection/union if union else np.nan
            result['rescue'][i,j]=int((~a&b).sum())
            result['containment'][i,j]=intersection/a.sum() if a.sum() else np.nan
    return result


def serial(a):
    if isinstance(a,np.ndarray):return serial(a.tolist())
    if isinstance(a,list):return [serial(x) for x in a]
    if isinstance(a,dict):return {k:serial(v) for k,v in a.items()}
    if isinstance(a,float) and not np.isfinite(a):return None
    return a


def gather(include_gd=False):
    source=V.HOST/'RESULT.json'
    # V.HOST may be rebound by another orchestration module in this process.
    source=V.ROOT/'outputs/semantic_rl_authors/random_room_coverage_20261117_02/RESULT.json'
    original=V.read(source)['rows']
    rows=[dict(r,method={'expanded_G':'G','expanded_GA':'GA','expanded_GDA':'GDA'}.get(r['condition'],r['condition']))
          for r in original if r['split']=='fresh_test' and (r['condition'].startswith('CAP-') or r['condition'] in ('expanded_G','expanded_GA','expanded_GDA'))]
    paths=[source,RETRIEVAL/'evaluation/result.json']
    rows += [dict(r,method=r['condition']) for r in V.read(paths[1])['rows']]
    if include_gd:
        V.Campaign(GD_BASE/'evaluation').verify_frozen()
        paths.append(GD_BASE/'evaluation/result.json')
        rows += [dict(r,method='GD') for r in V.read(paths[-1])['rows']]
    cases=V.read(V.PARENT/'fresh_test.json')['cases'];ids=sorted(c['id'] for c in cases)
    assert len(ids)==100
    by={}
    for r in rows:
        key=(r['method'],r['case_id'])
        if key in by:raise ValueError('duplicate method/case')
        by[key]=r
    mat=np.full((len(NAMES),len(ids)),np.nan)
    for i,n in enumerate(NAMES):
        present={case for method,case in by if method==n}
        if not present:
            if n!='GD' or include_gd:raise ValueError(('missing method',n))
            continue
        if present!=set(ids):raise ValueError(('unaligned cases',n))
        for j,case in enumerate(ids):mat[i,j]=int(by[n,case]['success'])
    return mat,ids,by,{str(p):V.sha(p) for p in paths}


def plots(matrix,ids,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap
    figs=[]
    counts=np.nansum(matrix,axis=0)
    order=sorted(range(len(ids)),key=lambda j:(ids[j].split('_')[2],-counts[j],tuple(np.nan_to_num(matrix[:,j],nan=-1)),ids[j]))
    fig,ax=plt.subplots(figsize=(17,6))
    cmap=ListedColormap(['#edf0f4','#087f8c']);cmap.set_bad('#b0b6c0')
    ax.imshow(np.ma.masked_invalid(matrix[:,order]),aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=1)
    ax.set_yticks(range(len(NAMES)),NAMES)
    ax.set_xticks(range(0,100,5),[ids[order[j]].replace('fresh_test_','') for j in range(0,100,5)],rotation=60,ha='right',fontsize=7)
    for x in [24.5,49.5,74.5]:ax.axvline(x,color='white',linewidth=2)
    ax.set_title('Success coverage on the same100 diagnostic scenes (gray = pending)')
    ax.set_xlabel('Grouped by movable count, then coverage pattern; exact IDs in CSV/HTML')
    fig.tight_layout();figs.append(('success',fig))
    for key,title,maximum in [('jaccard','Success-set Jaccard: intersection / union',1),
                              ('rescue','Column method succeeds where row method fails (cases)',100)]:
        fig,ax=plt.subplots(figsize=(12,10))
        cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#e1e4e9')
        image=ax.imshow(np.ma.masked_invalid(metrics[key]),cmap=cmap,vmin=0,vmax=maximum)
        ax.set_xticks(range(len(NAMES)),NAMES,rotation=65,ha='right');ax.set_yticks(range(len(NAMES)),NAMES)
        for i in range(len(NAMES)):
            for j in range(len(NAMES)):
                value=metrics[key][i,j]
                if np.isfinite(value):
                    label=f'{value:.2f}' if key=='jaccard' else str(int(value))
                    ax.text(j,i,label,ha='center',va='center',fontsize=7,color='white' if value>maximum*.6 else '#17202b')
        ax.set_title(title);fig.colorbar(image,ax=ax,shrink=.8);fig.tight_layout();figs.append((key,fig))
    with PdfPages(OUT/'coverage.pdf') as pdf:
        for name,fig in figs:
            fig.savefig(OUT/(name+'.png'),dpi=170)
            fig.savefig(OUT/(name+'.svg'))
            pdf.savefig(fig);plt.close(fig)
    return [ids[j] for j in order]


HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoImagine · CAP与整合策略成功覆盖</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#142133;background:#f3f5f8;margin:24px auto;padding:0 20px;max-width:1600px}h1{font-size:28px}section{background:white;padding:20px;margin:18px 0;border:1px solid #d7dfe8;border-radius:10px}.scroll{overflow:auto;max-height:680px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #dce2e8;padding:6px;text-align:center;white-space:nowrap}th{background:#f0f4f7;position:sticky;top:0;z-index:2}th:first-child{left:0;z-index:3}td:first-child{position:sticky;left:0;background:#f0f4f7;z-index:1;font-weight:600}button,select{padding:8px;margin:5px;cursor:pointer}td.cell{cursor:pointer;min-width:30px}td.cell:focus{outline:3px solid #ed9e33}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:380px;overflow:auto;background:#f5f7fa;padding:15px}a{color:#09647e}.muted{color:#596b80}.pending{color:#945d13}
</style><h1>同一面板上的成功覆盖与互补关系</h1>
<p id="status"></p><p>270条完整成功示范；学习器均为28000更新、seed17。R0/Rt使用同270轨迹，k=1；十个CAP为冻结程序。全部在同100个已查看场景上对齐，属于固定诊断，不是新盲测。</p>
<p class="muted">G/GA/GD/GDA保留相同合法动作支持。检索为紧凑状态快照，学习器含历史；CAP不受监督训练预算约束。覆盖相似不等于动作行为相同，方法差异不能单独归因架构。</p>
<section><b>成功率与教师能力保留</b><div id="summary" class="scroll"></div></section>
<section><b>覆盖矩阵</b><p><select id="mode"><option value="success">方法 × 场景成败</option><option value="jaccard">成功集合Jaccard重合度</option><option value="intersection">两方法共同成功数</option><option value="rescue">列方法补上行方法多少失败</option><option value="containment">列方法覆盖行方法成功集合的比例</option></select>点击单元格查看具体场景。灰色表示尚无结果或分母为零。</p>
<div id="matrix" class="scroll"></div><pre id="details">请选择一个单元格。</pre></section>
<section><b>导出与来源</b><p><a href="coverage.pdf">三页PDF图</a> · <a href="success.png">逐场景PNG</a> · <a href="jaccard.svg">重合度SVG</a> · <a href="rescue.svg">补救覆盖SVG</a> · <a href="success.csv">成败CSV</a> · <a href="coverage.json">完整数据与来源哈希</a></p>
<p>Jaccard = 共同成功数 / 至少一方成功数，不将共同失败视为相似证据。补救覆盖矩阵有方向：行是原方法，列是补充方法；事后并集不代表已实现的路由策略。独有成功只相对于此图已完成的方法集合。</p></section>
<script>const D=__DATA__;
const el=(t,s)=>{let x=document.createElement(t);if(s!==undefined)x.textContent=s;return x};
const names=D.methods,ids=D.case_ids,m=D.success;
document.getElementById('status').textContent=D.complete?'16个方法已全部对齐。':'GD正在训练/评估；灰色GD行是待完成，不是失败。其他15种方法已对齐。';
function tableSummary(){let t=el('table'),h=el('tr');['方法','成功/100','保留CAP并集','CAP并集外成功','所有方法中独有'].forEach(x=>h.append(el('th',x)));t.append(h);names.forEach(n=>{let s=D.summary[n],r=el('tr');[n,s.success===null?'待完成':s.success+'/100',s.retained_teacher??'—',s.outside_teacher??'—',s.exclusive??'—'].forEach(x=>r.append(el('td',x)));t.append(r)});document.getElementById('summary').replaceChildren(t)}
function detail(i,j,key){let out={row:names[i]};if(key==='success'){out.case=ids[j];out.success=m[i][j];out.successful_methods=names.filter((n,k)=>m[k][j]===1)}else{out.column=names[j];out.value=D.metrics[key][i][j];out.both=ids.filter((s,k)=>m[i][k]===1&&m[j][k]===1);out.row_only=ids.filter((s,k)=>m[i][k]===1&&m[j][k]===0);out.column_only=ids.filter((s,k)=>m[i][k]===0&&m[j][k]===1)}document.getElementById('details').textContent=JSON.stringify(out,null,2)}
function draw(){let key=document.getElementById('mode').value,success=key==='success',cols=success?D.case_order.map(x=>ids.indexOf(x)):names.map((_,i)=>i),t=el('table'),h=el('tr');h.append(el('th',success?'方法 / 场景':'行 / 列'));cols.forEach(j=>h.append(el('th',success?ids[j].replace('fresh_test_',''):names[j])));t.append(h);names.forEach((n,i)=>{let r=el('tr');r.append(el('td',n));cols.forEach(j=>{let v=success?m[i][j]:D.metrics[key][i][j],ratio=key==='jaccard'||key==='containment',c=el('td',v===null?'—':success?(v?'✓':'·'):ratio?v.toFixed(2):v);c.className='cell';c.tabIndex=0;let strength=success?v:ratio?v:v/100;c.style.background=v===null?'#d4d9e0':`hsl(190 55% ${98-55*strength}%)`;c.title=success?n+' / '+ids[j]:n+' → '+names[j];c.onclick=()=>detail(i,j,key);c.onkeydown=e=>{if(e.key==='Enter')detail(i,j,key)};r.append(c)});t.append(r)});document.getElementById('matrix').replaceChildren(t)}
tableSummary();draw();document.getElementById('mode').onchange=draw;
</script></html>'''


def render(include_gd=False):
    OUT.mkdir(parents=True,exist_ok=True)
    matrix,ids,by,sources=gather(include_gd)
    metrics=overlap_matrices(matrix)
    union=(matrix[:10]==1).any(0)
    summary={}
    for i,n in enumerate(NAMES):
        if not np.isfinite(matrix[i]).all():
            summary[n]=dict(success=None,retained_teacher=None,outside_teacher=None,exclusive=None);continue
        wins=matrix[i]==1
        others=np.delete(matrix,i,axis=0)
        summary[n]=dict(success=int(wins.sum()),retained_teacher=int((wins&union).sum()),
            outside_teacher=int((wins&~union).sum()),exclusive=int((wins&~(others==1).any(0)).sum()))
    case_order=plots(matrix,ids,metrics)
    data=serial(dict(complete=include_gd,methods=NAMES,case_ids=ids,case_order=case_order,success=matrix,
        metrics=metrics,summary=summary,teacher_union=int(union.sum()),sources=sources,
        report_dir=str(OUT),information='fixed previously viewed diagnostic100; input/architecture restrictions in protocol'))
    V.save(OUT/'coverage.json',data)
    with (OUT/'success.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method',*ids])
        for n,row in zip(NAMES,data['success']):w.writerow([n,*['pending' if x is None else int(x) for x in row]])
    for key,array in data['metrics'].items():
        with (OUT/(key+'.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['row_method',*NAMES])
            for n,row in zip(NAMES,array):w.writerow([n,*row])
    text=HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    temp=OUT/'index.pending.html';temp.write_text(text);temp.replace(OUT/'index.html')
    lines=['# CAP、输入消融与检索方法的统一成功覆盖','',
        '状态：'+('GD补齐，16方法完整。' if include_gd else 'GD待完成，其他15方法已对齐；pending不计为失败。'),'',
        '| 方法 | 成功/100 | 保留CAP并集 | CAP并集外成功 |','|---|---:|---:|---:|']
    for n in NAMES:
        s=summary[n];lines.append(f"| {n} | {s['success'] if s['success'] is not None else 'pending'} | {s['retained_teacher']} | {s['outside_teacher']} |")
    lines+=['','[交互覆盖报告](../../research/reports/CAP_Coverage_Comparison_20261117/index.html)。',
        'G/GA/GD/GDA同270示范、28000更新、seed17；CAP为冻结程序，R0/Rt索引同270轨迹但快照输入不同。此100例已看过，仅固定诊断。',
        'GD屏蔽冻结分组中的A值及缺失标记，但仍运行CAP以计算D；这不是完全不运行CAP的蒸馏模型。',
        '成功覆盖的相似性不等于动作序列或机制相似。共同失败不纳入Jaccard；空成功并集记未定义；补救覆盖有方向。']
    (V.ROOT/'docs/agent_exchange/reviews/GD_Coverage_Comparison_20261117.md').write_text('\n'.join(lines)+'\n')
    return data


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--include-gd',action='store_true');a=p.parse_args()
    r=render(a.include_gd)
    print(json.dumps(dict(complete=r['complete'],summary=r['summary'],report_dir=r['report_dir']),ensure_ascii=False))
