"""Bounded movement/visibility/branch acceptance, not a task-success benchmark."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import math
import time

from coimagine.continuous2d.scenes import Pose, Box
from experiments.planar_tasks.tasks import task_catalog
from .environment import MultiObjectOptionEnv
from .motion import move_id, VERSION, CODEBOOK_HASH
from .contract import History, canonical
from .predictor import NominalPredictor
from .search import choose, Budget
from .evidence import Ledger


def run(destination):
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();models=[]
    ledger=Ledger(dict(execution_ticks=4000,prediction_ticks=1000))
    report=dict(status='running',llm_calls=0,task_benchmark=False,
                action_version=VERSION,action_codebook_hash=CODEBOOK_HASH,cases=[])
    anchor=task_catalog()['planar_open_anchor'].instance
    unseen=replace(anchor,arena=(10.,8.),object_pose=Pose(6.6,2.,0),
        robot_poses=(Pose(1.,2.,0),Pose(1.,6.,0)),goal_centre=(8.,2.))
    def move(d='E',v=.30,t=40,other=44):
        return dict(target_id=None,a0=move_id(d,v),a1=other,ticks=t)
    brake=dict(target_id=None,a0=44,a1=44,ticks=40)
    def step(env,a):
        if time.monotonic()-started>900:raise TimeoutError('acceptance wall budget')
        ledger.charge(execution_ticks=a['ticks'])
        return env.step_command(a,record=True)
    cases=[('discover',unseen,[move()]*3),
           ('wall_and_retreat',replace(unseen,obstacles=(Box(2.3,2.,.1,.7),)),[move(v=.40)]*5+[move('W')]),
           ('independent_robots',unseen,[move(other=move_id('S',.20))]),
           ('push_move_push',anchor,[dict(target_id='entity-a',a0=2,a1=2,ticks=20),
                                     move('W',t=1),move('W',t=20),brake,
                                     dict(target_id='entity-a',a0=2,a1=2,ticks=20)])]
    try:
        for name,fixture,commands in cases:
            env=MultiObjectOptionEnv(fixture);initial=env.offline_audit();initial_public=env.public();events=[]
            for command in commands:
                result=step(env,command)
                events.append(dict(result=result,observation=env.public(),offline_audit=env.offline_audit()))
            # Replay through precisely the same entry and account every request.
            replay=MultiObjectOptionEnv(fixture)
            for command in commands:step(replay,command)
            assert replay.offline_audit()==env.offline_audit()
            if name=='discover':
                assert initial_public['entities']==[] and env.public()['entities']
                assert initial['entities']==env.offline_audit()['entities']
            if name=='wall_and_retreat':
                assert events[-2]['observation']['robots'][0]['execution']['direction_unavailable']
                assert events[-1]['observation']['robots'][0]['own']['x']<events[-2]['observation']['robots'][0]['own']['x']-.3
            if name=='push_move_push':assert env.slots[0].skill.kind=='push'
            data=dict(fixture=asdict(fixture),initial_public=initial_public,events=events,
                      initial_offline_audit=initial,rows=env.rows,boundaries=env.boundaries)
            (out/(name+'.json')).write_text(canonical(data)+'\n')
            report['cases'].append(dict(name=name,execution_ticks=env.total_periods,replay_ticks=replay.total_periods,
                exact_replay=True,initial_seen=[e['id'] for e in initial_public['entities']],
                final_seen=[e['id'] for e in env.public()['entities']],
                robot_displacements=[math.dist(a[:2],b[:2]) for a,b in zip(initial['robots'],env.offline_audit()['robots'])],
                termination=env.termination))
        env=MultiObjectOptionEnv(anchor);h=History();h.append(env.public());before=env.offline_audit()
        model=NominalPredictor(h,ticks=120,calls=3,seconds=20.);models.append(model)
        a=move('W',t=20)
        first=model.predict(model.root,a);second=model.predict(first['handle'],a);repeat=model.predict(first['handle'],a)
        assert second['observation']==repeat['observation'] and env.offline_audit()==before
        (out/'prediction.json').write_text(canonical(dict(history=h.events,model=model.record(),result=second))+'\n')
        model=NominalPredictor(h,ticks=160,calls=4,seconds=20.);models.append(model)
        selected=choose(h,model.root,model,lambda h,a:float(a['target_id'] is None),lambda h:10.,
                        Budget(prediction_ticks=160,calls=4,seconds=10.))
        assert selected['action'] is not None and selected['action']['target_id'] is None
        assert env.offline_audit()==before
        result=step(env,selected['action'])
        (out/'search.json').write_text(canonical(dict(search=selected,model=model.record(),execution=result,
                                                      observation=env.public()))+'\n')
        report['search']=dict(action=selected['action'],execution_ticks=result['end_tick'],prediction_ticks=selected['prediction_ticks'])
        report['status']='complete'
    except Exception as exc:
        report.update(status='incomplete',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        ledger.charge(prediction_ticks=sum(m.ledger.used['ticks'] for m in models))
        for model in models:model.close()
        report.update(usage=ledger.record(),wall_seconds=time.monotonic()-started)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        source=out/'source';source.mkdir()
        for p in Path(__file__).parent.glob('*.py'):(source/p.name).write_bytes(p.read_bytes())
        for p in Path('tests').glob('test_relational*.py'):(source/p.name).write_bytes(p.read_bytes())
        files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
        (out/'manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('destination')
    print(json.dumps(run(parser.parse_args().destination),indent=2))
