"""After model selection, measure branch effects at the first accepted local intervention."""
import time
import traceback
from unittest.mock import patch
from tools import cap_free_repair_round as r
from tools.cap_free_repair_finish import update_journal

def profile(job):
    k,f,source,probe=job
    proposed={}
    accepted=[]
    calls=[]
    original=r.prev.ISO.IsolatedCAP.update
    def update(member,h):
        previous=member.memory or {}
        ev=r.prev.PREFIX.s.phase.cap.current_event(h)
        tick=ev['observation']['tick']
        actual=ev.get('requested_action') or ev.get('action')
        if h.previous_tick in proposed and actual==proposed[h.previous_tick] and tick>h.previous_tick:
            accepted.append(h.previous_tick)
        original(member,h)
        s=member.memory.get(probe['slot'],{})
        a=s.get(probe['action'])
        if a and a==member.proposal:proposed[tick]=a
        calls.append(dict(tick=tick,command=bool(a),plan=bool(s.get(probe['plan']))))
    with patch.object(r.prev.ISO.IsolatedCAP,'update',update):
        _,x=r.host.episode((k,f,True,None,'boundary',False,r.host.enabled(source),source))
    baseline=r.read(r.OUT/'evaluation/selected'/(k+'.json'))
    assert x['result']==baseline['result']
    return k,dict(accepted=accepted,offered_ticks=list(proposed),roles=calls)

def metric(x,t):
    start=next(b for b in x['boundaries'] if b['tick']==t)
    after=[b for b in x['boundaries'] if t<b['tick']<=t+120]
    return dict(touched=any(b['touch'] for b in after),success=int(x['result']['success']),
        gain_at_last_boundary_m=round(start['gap']-after[-1]['gap'],3) if after and min(start['gap'],after[-1]['gap'])>=0 else None,
        last_tick=after[-1]['tick'] if after else None)

def main():
    while not (r.OUT/'whole_room.json').is_file():
        if (r.OUT/'ERROR.json').exists():return
        time.sleep(8)
    r.configure()
    lock=r.read(r.OUT/'locked.json')
    src=r.host.seal(lock['name'])
    base=r.host.seal('P')
    fixtures=r.read(r.OUT/'fixtures.json')
    probe=lock.get('role_probe')
    local=[]
    if probe:
        r.host.old.collect(src,[(k,f,src,probe) for k,f in fixtures.items()],profile,16,r.OUT/'roles')
        profiles={k:r.read(r.OUT/'roles'/(k+'.json')) for k in fixtures}
        anchors={k:v['accepted'][0] for k,v in profiles.items() if v['accepted']}
        jobs=[(k,fixtures[k],True,t,'boundary',False,r.host.enabled(base),src) for k,t in anchors.items()]
        if jobs:r.host.old.collect(base,jobs,r.host.episode,16,r.OUT/'local_base')
        for k,t in anchors.items():
            new=r.read(r.OUT/'evaluation/selected'/(k+'.json'))
            old=r.read(r.OUT/'local_base'/(k+'.json'))
            b=next(v for v in new['boundaries'] if v['tick']==t)
            assert b['public_history']==old['reached']['history']
            release=next((v['tick'] for v in profiles[k]['roles'] if v['tick']>t and not v['plan'] and not v['command']),None)
            local.append(dict(room=k,t=t,release=release,old=metric(old,t),new=metric(new,t),after_release=metric(new,release) if release is not None and any(v['tick']==release for v in new['boundaries']) else None))
        r.save(r.OUT/'local_results.json',local)
    results=r.read(r.OUT/'whole_room.json')
    import csv
    with (r.DOC/'rooms.csv').open() as f:rows=list(csv.DictReader(f))
    lines=[]
    if 'without_old_recovery' in rows[0]:
        old_only=sum(int(z['parent']) and not int(z['without_old_recovery']) for z in rows)
        alternative_only=sum(int(z['without_old_recovery']) and not int(z['parent']) for z in rows)
        lines.append(f"原组合独胜 {old_only} 间，定稿关闭原顶层恢复后独胜 {alternative_only} 间。两路从初态独立运行的成功并集为 {old_only+results['scores']['without_old_recovery']}/200。")
    if probe:
        lines.append(f"在新分支首次得到实际执行的 {len(local)} 间，从同一公开前缀改回原组合续跑。120 tick 内公开边界观测到接触的房间，原组合 {sum(v['old']['touched'] for v in local)} 间、定稿 {sum(v['new']['touched'] for v in local)} 间。记录到首次释放计划和命令的有 {sum(v['release'] is not None for v in local)} 间。")
    with (r.DOC/'README.md').open('a') as f:f.write('\n'+'\n\n'.join(lines)+'\n')
    r.save(r.OUT/'finished.json',dict(**results,local_pairs=len(local),role_probe=probe))
    update_journal(analysis=' '.join(lines) or '本轮结果见逐房间配对。')
if __name__=='__main__':
    try:main()
    except Exception:
        r.save(r.OUT/'ERROR.json',dict(error=traceback.format_exc()))
        raise
