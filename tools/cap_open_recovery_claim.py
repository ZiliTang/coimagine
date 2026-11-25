"""Score the author's two-observation local goal on paid, frozen, paired executions."""
import csv
from math import dist
from tools.cap_open_recovery_audit import load_record,timing
from tools.cap_open_recovery_round import OUT,DOC,episode,old,read,save,seal
from tools.cap_open_recovery_finish import journal


def sustained(record,t,baseline,horizon,amount,min_span,goal,frames=False):
    end=min(t+horizon,len(record['frames'])-1)
    samples=[]
    for row in record['boundaries']:
        at=row['tick']
        if t<at<=end:
            gap=dist(record['frames'][at][:2],goal) if frames else row['gap']
            if gap is not None and gap>=0:
                samples.append((at,gap))
    return any(u-t0>=min_span and 0<=g<=baseline-amount+1e-9 and 0<=g0<=baseline-amount+1e-9
               for (t0,g0),(u,g) in zip(samples,samples[1:]))


def write_table(name,rows):
    if not rows:return
    with (DOC/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def assess(horizon=320,amount=.30,min_span=8):
    lock=read(OUT/'locked.json')
    name=lock['name']
    rooms=read(OUT/'fixtures.json')
    splits=timing(name,rooms)
    write_table('timing.csv',splits)
    rows=[]
    for s in splits:
        t=s['first_change']
        if t is None:continue
        k=s['room']
        base=load_record('P',k); cand=load_record(name,k)
        ba=next((r for r in base['boundaries'] if r['tick']==t),{})
        ca=next((r for r in cand['boundaries'] if r['tick']==t),{})
        matched=bool(ba and ca and ba['public_history']==ca['public_history'] and ca['gap']>=0)
        ev=next((e for e in cand['events'] if e['t']==t and e['kind'] in ('pending','boundary')),{})
        state=ev.get('state',{}).get('recovery',{})
        marked=bool(state.get('pending') and state.get('command')==ev.get('action'))
        matched=matched and marked
        released=any(e['t']>t and not e['state'].get('recovery',{}).get('command') and
                     e['state'].get('recovery',{}).get('release',-1)>=t and
                     e['kind']=='boundary' for e in cand['events'])
        goal=[rooms[k]['goal'][c] for c in ('x','y')]
        assert base['frames'][t]==cand['frames'][t]
        values=dict(room=k,tick=t,common_visible_boundary=int(matched),candidate_entry=int(marked),released_to_parent=int(released),parent=s['parent'],candidate=s['candidate'])
        for policy,record in [('P',base),('candidate',cand)]:
            values[policy+'_observed']=int(sustained(record,t,ca['gap'],horizon,amount,min_span,goal)) if matched else None
            values[policy+'_physical']=int(sustained(record,t,dist(record['frames'][t][:2],goal),
                                                 horizon,amount,min_span,goal,True))
            values[policy+'_available_ticks']=min(horizon,len(record['frames'])-1-t)
        rows.append(values)
    write_table('local_claim.csv',rows)
    return rows,splits


def paired_forks():
    rows,splits=assess()
    q=[r for r in rows if r['common_visible_boundary']]
    name=read(OUT/'locked.json')['name']
    fixtures=read(OUT/'fixtures.json')
    program=seal('P')
    verified=[]
    for tag in ['P',name]:
        jobs=[(r['room'],fixtures[r['room']],False,r['tick'],'boundary',False,False,program) for r in q]
        if jobs: old.collect(seal(tag),jobs,episode,12,OUT/'claim_forks'/tag)
    for r in q:
        k=r['room']
        pa=read(OUT/'claim_forks/P'/(k+'.json'))
        ca=read(OUT/'claim_forks'/name/(k+'.json'))
        assert pa['reached']==ca['reached']
        assert pa['result']==read(OUT/'runs/confirm/P/full'/(k+'.json'))['result']
        ok=ca['result']==read(OUT/'runs/confirm'/name/'full'/(k+'.json'))['result']
        verified.append(dict(room=k,tick=r['tick'],history=pa['reached']['history'],
                             observation=pa['reached']['observation_sha256'],reproduced_candidate=int(ok)))
    write_table('forks.csv',verified)
    return rows,splits,verified


def finish():
    rows,splits,verified=paired_forks()
    lock=read(OUT/'locked.json')
    assert lock['name']=='trial06' and lock['sha256']=='0571e1eb86cedc147c6faeccebbf405cce082614a842440cc1898dfc8711fa48'
    q=[r for r in rows if r['common_visible_boundary']]
    def totals(left,right):
        return dict(parent=sum(r[left] for r in q),candidate=sum(r[right] for r in q),
                    gain=sum(r[right]>r[left] for r in q),loss=sum(r[right]<r[left] for r in q))
    observed=totals('P_observed','candidate_observed')
    phys=totals('P_physical','candidate_physical')
    release=sum(r['released_to_parent'] for r in q)
    parity=sum(r['reproduced_candidate'] for r in verified)
    early=sum(s['relative_tick'] is not None and s['relative_tick']<0 for s in splits)
    late=sum(s['relative_tick'] is not None and s['relative_tick']>=0 for s in splits)
    usage=read(DOC/'worklog.json')['usage']
    qtext=(f"新房间实际改变命令 {len(rows)}/200，其中 {len(q)} 间在相同可见边界满足作者定义的入口。"
           f"在这 {len(q)} 间，母程序持续缩短 30 cm 的有 {observed['parent']} 间，候选有 {observed['candidate']} 间"
           f"（新增 {observed['gain']}、丢失 {observed['loss']}）。")
    text='\n## 作者交卷及局部承诺\n\n'
    text+=('作者选择保留母策略原有绕行动作的优先级，新增局部换推位规则。条件包括近期推进缓慢、观测到直推路线受固定障碍限制，以及有可用的替代推位。'
           '动作得到真实执行反馈后才确认接手，取得位移后可有限次数再次尝试。作者也开发了 PUSH 事件中途接手版，定稿沿用普通决策与原有复核时机。\n\n')
    text+=f"共提交 8 份源文件，其中 1 份超过部署源码上限。作者用时 {lock['elapsed_seconds']/60:.1f} 分钟，累计 {usage['total_tokens']:,} token，含 {usage['cached_input_tokens']:,} 缓存输入。选定 trial06，另保留 trial05。22 间开发房间从 14 间成功增至 16 间，救回 2 间、丢失 0 间。\n\n"
    text+='定稿时给出的局部目标是从首次实际改选起，320 tick 内有两个连续有效观测比起点更接近目标至少 0.30 m，两个边界相隔至少 8 tick。该标准在开发时形成，独立房间的测量发生在冻结后。开发的 6 个自然分叉中，母程序 1 间达标，候选 4 间达标。\n\n'
    text+=qtext+f" 同一起点分叉的公开观测和历史哈希逐项核对，候选重放复现整局结果及历史 {parity}/{len(verified)}。以同一时刻的真实物体位姿复核，母程序 {phys['parent']}/{len(q)}，候选 {phys['candidate']}/{len(q)}。在这些起点后，观察到模块释放且普通边界由母策略恢复决策 {release}/{len(q)}。\n\n"
    text+=f"母策略有 16 间进入预定义的停滞簇，其中 {early} 间首次改选早于停滞窗口确认，{late} 间在确认时或更晚才首次改选。窗口确认时刻是离线诊断量，不能单凭先后关系判定提前干预避免了碰撞。\n\n"
    text+='## 采用决定\n\n作为局部候选归档，继续使用前轮 trial09 作为默认母程序。目标簇内救回 2 间，全部 200 间中丢失 4 间；本轮没有证明能系统性解决该簇并保住既有成功。\n'
    p=DOC/'README.md'
    p.write_text(p.read_text().split('\n## 作者交卷及局部承诺')[0].rstrip()+'\n'+text)
    result='开发 14 → 16/22；新房间 129 → 127/200，救回 2 间、丢失 4 间。'
    journal(result=result,analysis=qtext,assets=[['worklog.html','作者开发记录与 token'],['rooms.csv','整局配对结果'],
             ['local_claim.csv','作者承诺的局部效果'],['forks.csv','同前缀核对'],['local.csv','固定停滞时刻续跑'],
             ['timing.csv','首次改选时机'],['policy.py','冻结完整策略']])
    save(OUT/'finished.json',dict(whole=result,local=qtext,physical=phys,verified=parity,
                                  released=release,before=early,at_or_after=late,default='P'))
    print(qtext,phys,parity,release,early,late)


if __name__=='__main__':
    finish()
