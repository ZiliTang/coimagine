"""Explicit bounded replay of a supplied physical witness; no author or learning call.

Run with --out NEW_DIRECTORY --witness prior_wall_physical_witness.json.
Checks fixed-time execution and a clock-preserving policy under progress boundaries.
The latter is a seed replay, not a newly learned/generalizing CAP solution.
"""
import argparse
from pathlib import Path

from .semantic_rl import SemanticRLCampaign, SemanticRLSession, development_fixtures
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.pipeline import save, read, pack
from experiments.planar_tasks.relational_search.contract import digest


def replay_source(witness):
    # The policy has only public time, goal identity and legal menus at runtime.
    rows = [(r['end_tick'], *r['actions']) for r in witness['commands']]
    return '''NAMES=['recommendation_match','duration_error','phase_time_margin']
SCHEDULE = ''' + repr(rows) + '''
def initial_memory(): return {}
def update_memory(memory,event): return memory
def propose_actions(memory,observation):
    t=observation['tick']
    end,a,b=next((r for r in SCHEDULE if t<r[0]),(t+40,44,44))
    target=observation['goal']['target_id']
    return [dict(target_id=target,a0=a,a1=b,ticks=min(40,end-t))]
def prefix_semantic_features(memory,observation,prefix,proposals):
    q=proposals[0]
    return dict(recommendation_match=float(all(q[k]==v for k,v in prefix.items())),
                duration_error=-abs(prefix['ticks']-q['ticks']) if 'ticks' in prefix else None,
                phase_time_margin=q['ticks']/40.)
def semantic_features(memory,observation,action,proposals):
    return prefix_semantic_features(memory,observation,action,proposals)
'''


def run(out, witness_path):
    out = Path(out); witness = read(witness_path)
    fixture = development_fixtures('stage1-wall-anchor-v2')['wall_release']
    # Fail before physics if a caller supplies a different geometric witness.
    scene, boxes = X.fixture(fixture)
    from experiments.planar_tasks.tasks import scene_metadata
    if scene_metadata(scene) != witness['scene']:
        raise ValueError('physical witness geometry does not match the declared anchor')
    c = SemanticRLCampaign.create(out/'campaign', fixture_profile='stage1-wall-anchor-v2',
            feature_runtime='cached_fork',
            model=dict(provider='none',name='host physical witness replay; no new author calls'))
    save(out/'witness.json', witness)
    save(out/'protocol.json', dict(purpose='geometry/controller compatibility; not independent CAP or learning',
         fixture_profile='stage1-wall-anchor-v2', max_attempts=2, max_ticks_per_attempt=400,
         max_total_ticks=800, witness_sha256=digest(witness), llm_calls=0, fits=0, rl_updates=0))
    source = replay_source(witness); (out/'replay_policy.py').write_text(source)
    session = SemanticRLSession(c.path)
    sid = session.tool('submit', dict(source=source,summary='Supplied physical schedule, public-time adaptation only'))['id']
    commands = [dict(target_id=X.T.PRIMARY_ID,a0=r['actions'][0],a1=r['actions'][1],ticks=r['requested_ticks'])
                for r in witness['commands']]
    save(out/'time_request.json',dict(fixture=fixture,commands=commands,max_ticks=400))
    timed = X.execute(fixture, commands, max_ticks=400)
    pack(out/'time_result.json.gz',timed)
    progress = session.tool('test',dict(sid=sid,task='wall_release',max_ticks=400,max_decisions=150))
    result = dict(time=timed['summary'],progress=progress,
                  executed_ticks=timed['summary']['ticks']+progress['ticks'],
                  accounting='time replay separately charged here; progress metered in campaign, no double count',
                  source_id=sid,fixture_sha256=digest(fixture))
    save(out/'summary.json',result)
    save(out/'verification.json',c.freeze())
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--witness',type=Path,required=True)
    args=p.parse_args()
    import json
    print(json.dumps(run(args.out,args.witness)))
