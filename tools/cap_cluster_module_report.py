"""Report the author's frozen single-cluster module against its parent, by room."""
import csv
from math import sqrt
from pathlib import Path
from statistics import median
import json

from tools.cap_cluster_module import OUT, PUBLIC, read, save
from tools.cap_fault_curriculum_author import sha

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/research/key_findings/2026-11-24_single_cluster_cap'
JOURNAL = ROOT/'docs/research/experiment_journal/entries.json'


def wilson(k, n):
    if not n:
        return None
    p = k/n
    z = 1.95996398454
    d = 1 + z*z/n
    half = z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    centre = (p+z*z/(2*n))/d
    return (max(0.,centre-half), min(1.,centre+half))


def percentage_interval(k,n):
    limits=wilson(k,n)
    if not limits:
        return '无样本'
    return f'{100*k/n:.1f}%（95% Wilson 区间 {100*limits[0]:.1f}% 至 {100*limits[1]:.1f}%）'


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inventory(directory):
    return {p.stem:read(p) for p in sorted(directory.glob('*.json'))}


def branch_progress(row):
    effect=row['local_effect']
    return round(effect[0]['gap']-effect[1]['gap'],5) if len(effect)==2 else None


def build():
    lock=read(OUT/'locked.json')
    name=lock['name']
    src=OUT/'trials'/name/'campaign/composed.py'
    assert sha(src.read_text())==lock['sha256']
    base=inventory(OUT/'baseline')
    fixtures=read(OUT/'fixtures.json')
    anchor={k for k,v in base.items() if v['anchor']}
    modes={m:inventory(OUT/'trials'/name/'confirmation'/m) for m in ('full','parent','forced','authored')}
    assert set(base)==set(fixtures)==set(modes['full']) and len(base)==200
    assert all(set(modes[m])==anchor for m in ('parent','forced','authored'))
    rooms,branches=[],[]
    for key,b in base.items():
        old=b['result']; new=modes['full'][key]
        if new['changed']==0:
            assert new['result']['history']==old['history'],key
        rooms.append(dict(room=key,cluster=int(key in anchor),parent=old['success'],module=new['result']['success'],
                          proposed_changes=new['changed'],first_offer_tick=new['first_tick'],
                          history_changed=int(new['result']['history']!=old['history']),
                          parent_ticks=old['ticks'],module_ticks=new['result']['ticks']))
        if key not in anchor:
            continue
        assert modes['parent'][key]['result']==old
        row=dict(room=key,anchor_tick=b['anchor']['tick'],anchor_history=b['anchor']['history'])
        for mode in ('parent','forced','authored'):
            branch=modes[mode][key]
            assert branch['verified_anchor']
            if branch['changed']==0:
                assert branch['result']['history']==old['history']
            row.update({mode:branch['result']['success'],mode+'_offer':int(branch['first_tick'] is not None),
                        mode+'_changed':branch['changed'],mode+'_advance_m':branch_progress(branch)})
        branches.append(row)
    doc_rows=[]
    for label,subset in [('簇内',[r for r in rooms if r['cluster']]),
                         ('簇外',[r for r in rooms if not r['cluster']]),('全部',rooms)]:
        n=len(subset)
        parent=sum(r['parent'] for r in subset); new=sum(r['module'] for r in subset)
        rescue=sum(r['module']>r['parent'] for r in subset); loss=sum(r['module']<r['parent'] for r in subset)
        doc_rows.append((label,n,parent,new,rescue,loss))
    _,n,p,_,rescued,lost=doc_rows[0]
    _,total,win,new,total_rescued,total_loss=doc_rows[-1]
    criterion=bool(wilson(rescued,n-p) and wilson(rescued,n-p)[0]>.5 and total_loss==0)
    result=(f'新房间簇内原失败救回 {rescued}/{n-p}，簇内原成功丢失 {lost}/{p}；'
            f'簇外原成功丢失 {doc_rows[1][-1]}/{doc_rows[1][2]}。整局 {win} → {new}/{total}。')
    fork_lines=[]
    for m,label in [('parent','原程序继续'),('forced','请求直接接手'),('authored','按作者入口接手')]:
        offered=sum(r[m+'_offer'] for r in branches)
        pos=sum(r[m]>r['parent'] for r in branches)
        neg=sum(r[m]<r['parent'] for r in branches)
        fork_lines.append(f'| {label} | {offered} | {sum(r[m] for r in branches)}/{len(branches)} | {pos} | {neg} |')
    analysis=('达到' if criterion else '未达到')+'预先登记的系统性解决标准。'
    DOC.mkdir(parents=True,exist_ok=True)
    write_csv(DOC/'rooms.csv',rooms);write_csv(DOC/'branches.csv',branches)
    # Archive the exact deployable source and author explanation referenced by this report.
    (DOC/'module.py').write_text((PUBLIC/(name+'.py')).read_text())
    report=(DOC/'README.md').read_text().split('\n## 冻结验收\n')[0].replace('作者开发中。','作者已交卷。').rstrip()
    report+='\n\n## 冻结验收\n\n'+result+analysis+'\n\n'
    report+=f"作者选定 {name}，提交用时 {lock['elapsed_seconds']/60:.1f} 分钟。{lock['description']}\n\n"
    report+='| 母程序上的分组 | 房间 | 母程序成功 | 模块程序成功 | 救回 | 丢失 |\n|---|---:|---:|---:|---:|---:|\n'
    report+='\n'.join('| '+' | '.join(map(str,r))+' |' for r in doc_rows)
    report+=f'\n\n簇内原失败救回率为 {percentage_interval(rescued,n-p)}。簇外原成功的丢失率为 {percentage_interval(doc_rows[1][-1],doc_rows[1][2])}。这些区间描述该固定布局生成规则下的房间差异。仅运行一次独立作者开发。\n\n'
    outside_wins=[r for r in rooms if not r['cluster'] and r['parent']]
    different=[r for r in outside_wins if r['history_changed']]
    kept=[r['module_ticks']-r['parent_ticks'] for r in different if r['module']]
    report+=(f"簇外原成功的 {len(outside_wins)} 间中，实际执行历史发生改变的有 {len(different)} 间。"
             + (f"其中仍成功的房间，完成耗时变化中位数为 {median(kept):+g} tick。" if kept else "")
             + "\n\n")
    report+='## 同一时刻的三条后续路线\n\n'
    report+='| 从停滞窗口末尾起步 | 模块实际给出动作的房间 | 整局成功 | 相对原程序救回 | 相对原程序丢失 |\n|---|---:|---:|---:|---:|\n'+'\n'.join(fork_lines)
    report+='\n\n每间房只取第一次合格窗口。三条路线到达相同的公开历史哈希，原程序重放的整局终点历史也逐项一致。请求直接接手时仍须通过作者的可执行条件。此表衡量这一处机会；从初态运行的程序还可以更早介入或反复尝试。逐例表中的局部推进是在起点与至少 120 tick 后首个命令边界之间，目标物到终点距离的减少量（米）；不足一个窗口留空。\n\n'
    target_failures=[r for r in rooms if r['cluster'] and not r['parent']]
    changed=sum(r['history_changed'] for r in target_failures)
    report+=f"原程序发生目标故障并最终失败的 {len(target_failures)} 间里，整局启用模块后有 {changed} 间的终点公开历史发生变化。"
    for r in target_failures:
        if r['module']:
            tick=base[r['room']]['anchor']['tick']
            report+=f"在救回的 {r['room']} 中，模块首次给出动作的时刻为 {r['first_offer_tick']} tick，母程序停滞窗口末尾为 {tick} tick。"
    report+='\n\n'
    report+='## 作者开发记录\n\n| 提交 | 已测开发房间 | 簇内原失败救回 | 原成功丢失 |\n|---|---:|---:|---:|\n'
    dev=inventory(OUT/'snapshots')
    for directory in sorted((OUT/'trials').glob('trial*')):
        rows=inventory(directory/'packet/full')
        if rows:
            plus=sum(bool(dev[k]['anchor']) and r['result']['success']>dev[k]['result']['success'] for k,r in rows.items())
            minus=sum(r['result']['success']<dev[k]['result']['success'] for k,r in rows.items())
            report+=f'| {directory.name} | {len(rows)} | {plus} | {minus} |\n'
    meter=read(DOC/'worklog.json')
    selected=next(c for c in meter['candidates'] if c['name']==name)
    if selected['usage']:
        report+=(f"\n截至写入所选版本，作者累计使用 {selected['usage']['total_tokens']:,} token，"
                 f"其中缓存输入 {selected['usage']['cached_input_tokens']:,}。选版之后的试作与交卷开销另计入工作记录的作者总量。\n")
    usage=meter.get('usage')
    if usage:
        report+=(f"\n全会话作者用量为 {usage['total_tokens']:,} token，其中输入 {usage['input_tokens']:,}"
                 f"（含缓存 {usage['cached_input_tokens']:,}），输出 {usage['output_tokens']:,}。"
                 "实验以墙钟时间限制作者，未设 token 上限。\n")
    report+='\n[作者工作记录与逐版用量](worklog.html) · [模块源码](module.py) · [全部新房间](rooms.csv) · [同起点三分支](branches.csv)\n'
    (DOC/'README.md').write_text(report)
    raw=JOURNAL.read_text()
    start=raw.rindex('\n  {',0,raw.index('"id": "single-cluster-cap"'))+3
    entry,length=json.JSONDecoder().raw_decode(raw[start:])
    assert entry['id']=='single-cluster-cap'
    entry.update(result=result,analysis=analysis)
    if ['module.py','冻结模块源码'] not in entry['assets']:entry['assets'].append(['module.py','冻结模块源码'])
    new_text=raw[:start]+json.dumps(entry,ensure_ascii=False,indent=2).replace('\n','\n  ')+raw[start+length:]
    json.loads(new_text)
    JOURNAL.write_text(new_text)
    save(OUT/'finished.json',dict(summary=result,criterion=criterion,sha256=lock['sha256'],per_group=doc_rows))
    print(result,analysis)


if __name__=='__main__':
    build()
