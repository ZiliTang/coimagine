"""Bounded engineering acceptance with the published seed sequence, not an author experiment."""
from pathlib import Path
import argparse
import json
from experiments.planar_tasks.relational_search import pipeline_execution as X, testbed as T
from .environment import EventConfig
from .runner import rollout, save, train_run


def engineering_source():
    stages=[]; tick=0
    for segment in T.default_sequence():
        tick += segment['decisions']*4
        stages.append((tick, segment['target_id'], segment['a0'], segment['a1']))
    return 'STAGES = '+repr(stages)+'''\nNAMES = ['command_match','duration_error','stage_time']
def initial_memory():return {}
def update_memory(m,e):return m
def propose_actions(m,o):
    tick=o['tick']
    for end,target,a0,a1 in STAGES:
        if tick<end:return [dict(target_id=target,a0=a0,a1=a1,ticks=min(4,end-tick))]
    return [dict(target_id=None,a0=44,a1=44,ticks=4)]
def semantic_features(m,o,a,p):
    expected=p[0]
    return dict(command_match=float(all(a[k]==expected[k] for k in ('target_id','a0','a1'))),
                duration_error=abs(a['ticks']-expected['ticks']),stage_time=o['tick']*.05)
def prefix_semantic_features(m,o,a,p):
    expected=p[0]
    return dict(command_match=float(all(a[k]==expected[k] for k in a if k!='ticks')),
                duration_error=None if 'ticks' not in a else abs(a['ticks']-expected['ticks']),stage_time=o['tick']*.05)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    source=engineering_source(); fixture=X.seed_fixture()
    (args.out/'engineering_source.py').write_text(source);save(args.out/'fixture.json',fixture)
    result,_=rollout(fixture,source,config=EventConfig(mode='progress'),max_ticks=900,max_decisions=300,wall_seconds=240)
    result.update(split='train',provenance='published engineering seed sequence adapted by clock; not a new LLM solution')
    save(args.out/'demonstration.json',result)
    print(json.dumps(result['summary']),flush=True)
    if not result['summary']['success']:
        raise RuntimeError('progress witness did not succeed; do not invent imitation labels')
    reports=train_run(args.out/'train',fixture,source,result,mode='progress',episodes=1,max_ticks=80,
                      max_decisions=6,supervised_epochs=30,max_demo_samples=4)
    save(args.out/'acceptance.json',dict(purpose='engineering only; four demonstration histories and six RL decisions',
         witness=result['summary'],training=reports))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
