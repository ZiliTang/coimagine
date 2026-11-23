"""Record audited physical replays for selected wins and losses in the paired diagnostic."""
import json
import math
from pathlib import Path

import torch

from tools import cap_regional_advantage as ref

study=ref.study
DOC=study.DOCS.parent/'2026-11-20_regional_advantage'
OUT=ref.OUT


@torch.inference_mode()
def record(fixture, step, first, suffix, expected, prefix):
    dq=study.dq
    env=dq.fast_runtime.ScopedEnvironment(fixture,dq.study.CFG,fast_feedback=True,record_details=True,
                                         history_factory=dq.fast_runtime.AuditedCachedHistory)
    lib=ref.cpp.Proposals()
    events=[]
    try:
        for idx in range(300):
            panel=lib.panel(env.history)
            i,j=ref.choices(panel)[:2]
            obs=dq.current_event(env.history)['observation']
            if idx==step and prefix:
                assert prefix==env.history.sha256 and i!=j
            select=0 if idx<step else first if idx==step else suffix
            act=panel[4][(i,j)[select]]
            contacts=[r['contacting_object'] for r in obs['robots']]
            caps=[ref.cpp.SID_ORDER[k] for k in panel[2][(i,j)[select]].nonzero()[0]]
            ev=env.step(act,remaining_ticks=1800-env.env.total_periods)
            events.append(dict(t=ev['start_tick'],end=ev['end_tick'],a=act,contacts=contacts,
                               caps=caps,segments=ev['segments']))
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        assert (env.history.sha256,env.env.total_periods,int(env.env.success))==(expected['history'],expected['ticks'],expected['success'])
        replay=env.replay()
        scene=replay['scene']
        frames=replay['frames']
        assert [v['tick'] for v in frames]==list(range(expected['ticks']+1))
        ids=[scene['primary_id'],*[m['label'] for m in scene['movable_objects']]]
        tracks=[[round(float(x),5) for entity in [*(k['entities'][i] for i in ids),*k['robots']]
                 for x in entity[:3]] for k in frames]
        return dict(events=events,frames=tracks,success=expected['success'],ticks=expected['ticks'],
                    history=expected['history'],reason=env.env.termination or 'decision_budget',scene=scene)
    finally:
        env.close_reads()
        dq.fast_runtime.ACTIVE=None


def first_split(a,b):
    for x,y in zip(a['events'],b['events']):
        if x['t']!=y['t'] or x['a']!=y['a']:
            return min(x['t'],y['t'])
        if x['end']!=y['end']:
            return min(x['end'],y['end'])
    return min(a['ticks'],b['ticks'])


def main():
    ref.init()
    parents={r['parent']:r for r in study.read(OUT/'panel.json')}
    comparisons=[]
    for cid,title,desc,suffix in [
        ('dev_2_23','换一步，救回失败','只更换首次分歧的动作，从下一次决策起两边都由编号 actor 接手。',0),
        ('dev_2_07','换一步，丢掉成功','两边均由编号 actor 接手，观察较晚出现的一个不同动作。',0),
        ('dev_5_11','同一分歧，编号 actor 接手','与下一例共用同一对首步。先看编号 actor 如何续跑。',0),
        ('dev_5_11','同一分歧，动作 actor 接手','首步与上一例完全相同，只更换后续策略，再比较哪一边成功。',1),
        ('dev_2_14','初态移动 1 厘米','两边全程使用同一编号 actor，右边仅把机器人 0 的初始 x 坐标增加 1 厘米。',None)]:
        original=study.read(OUT/'cases'/f'{cid}_0.json')
        p=parents[cid]
        if suffix is None:
            neighbor=study.read(OUT/'cases'/f'{cid}_1.json')
            assert p['axis']=='x' and p['sign']==1
            pair=[record(p['fixtures'][k],0,0,0,r['baseline'],None) for k,r in enumerate((original,neighbor))]
            marker=0
            labels=['原始初态','机器人 0 向东 1 厘米']
        else:
            c=study.read(OUT/'cross'/f'{cid}.json')
            labels=['编号 actor 提出的首步','动作 actor 提出的首步']
            expected=[original['baseline'],c['crosses'][0]] if suffix==0 else [c['crosses'][1],original['labels']['initial']]
            pair=[record(p['fixtures'][0],c['step'],first,suffix,expected[first],c['prefix']) for first in (0,1)]
            marker=original['stages']['divergence']['tick']
            assert pair[0]['frames'][:marker+1]==pair[1]['frames'][:marker+1]
        assert pair[0]['success']!=pair[1]['success']
        scene=pair[0].pop('scene')
        assert scene==pair[1].pop('scene')
        comparisons.append(dict(cid=cid,title=title,description=desc,marker=marker,first_split=first_split(*pair),
                                labels=labels,scene=scene,runs=pair))
    study.save(OUT/'replay_data.json',comparisons)
    export(comparisons)


def export(comparisons):
    template=Path(__file__).with_name('cap_regional_advantage_replays.html').read_text()
    DOC.mkdir(parents=True,exist_ok=True)
    (DOC/'replays.html').write_text(template.replace('/* EMBED_REPLAYS */',json.dumps(comparisons,ensure_ascii=False,separators=(',',':'))))
    summary=[]
    for c in comparisons:
        g=c['scene']['goal']
        def audit(r):
            return dict(success=r['success'],ticks=r['ticks'],history=r['history'],reason=r['reason'],
                        distances=[round(math.dist(r['frames'][min(t,r['ticks'])][:2],[g['x'],g['y']]),3)
                                   for t in [c['marker'],c['marker']+40,r['ticks']]])
        summary.append(dict(case=c['cid'],title=c['title'],intervention=c['marker'],first_split=c['first_split'],
                            left=audit(c['runs'][0]),right=audit(c['runs'][1])))
    study.save(DOC/'replay_audit.json',summary)


if __name__=='__main__':
    main()
