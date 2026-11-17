"""Bounded ENGINEERING acceptance; authored by hand, no LLM or efficacy claim."""
import argparse
from pathlib import Path
from .pipeline import Campaign, read, save
from .pipeline_execution import commands
from .testbed import default_sequence


INITIAL='''import math
NAMES=['observed_goal_distance']
def initial_memory(): return {}
def update_memory(memory,event): return memory
def semantic_features(memory,observation,action):
    points=next((e['points'] for e in observation['entities'] if e['id']==observation['goal']['target_id']),[])
    if not points:return {'observed_goal_distance':None}
    x=sum(p[0] for p in points)/len(points);y=sum(p[1] for p in points)/len(points)
    gx,gy=observation['goal']['xy']
    return {'observed_goal_distance':math.hypot(gx-x,gy-y)}
'''

REVISED='''import math
NAMES=['observed_goal_distance','requested_push_alignment']
def initial_memory(): return {}
def update_memory(memory,event): return memory
def semantic_features(memory,observation,action):
    points=next((e['points'] for e in observation['entities'] if e['id']==observation['goal']['target_id']),[])
    if not points:return dict.fromkeys(NAMES,None)
    x=sum(p[0] for p in points)/len(points);y=sum(p[1] for p in points)/len(points)
    gx,gy=observation['goal']['xy'];dx=gx-x;dy=gy-y;d=math.hypot(dx,dy)
    alignment=0.
    if action['target_id']==observation['goal']['target_id']:
        for a in (action['a0'],action['a1']):
            if 0<=a<32:
                direction,speed=divmod(a,4);angle=direction*math.pi/4
                alignment+=(dx*math.cos(angle)+dy*math.sin(angle))/max(d,1e-6)*.1*(speed+1)
    return dict(observed_goal_distance=d,requested_push_alignment=alignment*action['ticks']*.05)
'''


def brief(c,rid,summary,output,evidence=()):
    return c.author_turn(rid,dict(actor='engineering',summary=summary,
        rationale='人工编写的管线验收步骤；不是 LLM 发现或方法效果实验。',
        evidence=list(evidence),output=output))['turn_id']


def run(path):
    c=Campaign.create(path,group='clear-transport-engineering-01')
    expected=dict(boundary=80,kind='action_legal',action=dict(target_id='entity-a',a0=2,a1=2,ticks=4))
    c.new_round(expectations=[expected]);seed=c.replay('r000');print('seed',seed,flush=True)
    if seed['status']!='success':raise RuntimeError('seed witness regression')
    c.new_round({'entity-b.x':.3,'robot1.x':.3},parent='r000',expectations=[expected])
    original=c.replay('r001');print('perturbed replay',original,flush=True)
    success=None
    for decisions in (40,36):
        tid=brief(c,'r001',f'从换位开始处修复：将向西移动时长改为 {decisions*.2:.1f} 秒。',
                  dict(repair_boundary=45,west_decisions=decisions),['r001/a000:boundary80'])
        seq=default_sequence();seq[2]['decisions']=decisions
        repaired=c.repair('r001',attempt_id='a000',boundary=45,suffix=commands(seq)[45:],turn_id=tid)
        print('repair',repaired,flush=True)
        if repaired['status']=='success':success=repaired;break
    # All failed repairs remain visible; acceptance can still diagnose missing
    # repair success instead of inventing it. Physical comparisons use seed.
    for rid,aid,boundary in [('r000','a000',83)]+([('r001',success['id'],89)] if success else []):
        seq=read(c._apath(rid,aid)/'request.json')['commands']
        candidates=[dict(target_id='entity-a',a0=a,a1=a,ticks=40) for a in (2,0,44)]
        suffix=seq[boundary+10:]
        brief(c,rid,'在相同前缀之后比较两种东推速度与停稳动作，后续序列保持一致。',
              dict(boundary=boundary,candidates=candidates),[rid+'/'+aid])
        result=c.compare(rid,attempt_id=aid,boundary=boundary,candidates=candidates,continuation=suffix)
        print('comparison',rid,result['preference_pairs'],flush=True)
    rid='r001';t=brief(c,rid,'提交仅含观测目标距离的初始特征，检查其能否区分同一状态下的候选。',dict(names=['observed_goal_distance']))
    s0=c.submit_semantics(rid,source=INITIAL,turn_id=t)
    f0=c.fit(rid,s0['id']);print('fit0',f0,flush=True)
    t=brief(c,rid,'在同一批比较数据上加入请求推动方向与目标方向的对齐量，再次拟合。',
            dict(names=['observed_goal_distance','requested_push_alignment']),[f0['fit_id']])
    s1=c.submit_semantics(rid,source=REVISED,turn_id=t)
    refs=read(c.path/'fits'/(f0['fit_id']+'.json'))['comparisons']
    f1=c.fit(rid,s1['id'],comparison_refs=refs);print('fit1',f1,flush=True)
    seq=read(c._apath('r000','a000')/'request.json')['commands']
    selected=c.apply_ranker('r000',fit_id=f1['fit_id'],attempt_id='a000',boundary=83,
        candidates=[dict(target_id='entity-a',a0=a,a1=a,ticks=40) for a in (44,0,2)],
        continuation=seq[93:],condition='S')
    print('ranked execution',selected,flush=True)
    save(c.path/'acceptance.json',dict(kind='hand-authored engineering acceptance',llm_calls=0,
        seed_success=seed['success'],perturbed_status=original['status'],repair_success=bool(success),
        ranked_execution_success=selected['success'],fits=[f0,f1],
        limitations=['one development mother group; no held-out generalization',
            'features/repairs manually authored, not discovered by an LLM',
            'fixed continuation preferences, not optimal Q',
            'fixed E summary and linear rankers; no neural history encoder',
            'ranked continuation is not full deployment search']))
    print(c.freeze(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);run(p.parse_args().output)
