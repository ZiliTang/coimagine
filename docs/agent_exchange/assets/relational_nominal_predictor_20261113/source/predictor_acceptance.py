"""Bounded nominal/execution comparison and physical search wiring check."""
from pathlib import Path
import hashlib
import json
import math
import time
from experiments.planar_tasks.tasks import task_catalog
from .environment import MultiObjectOptionEnv,MovableBox
from .contract import History
from .predictor import NominalPredictor
from .search import choose,Budget
from .evidence import Ledger


def run(destination):
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    ledger=Ledger(dict(execution_ticks=2000,prediction_ticks=4000))
    report=dict(status='running',llm_calls=0,task_benchmark=False,cases=[])
    models=[]
    instance=task_catalog()['planar_open_anchor'].instance
    def fresh():return MultiObjectOptionEnv(instance,[MovableBox('b',1.55,2.95,width=.6,height=.3)])
    push=dict(target_id='entity-a',a0=2,a1=2,ticks=10)
    switch=dict(target_id='b',a0=44,a1=44,ticks=10)
    cases=[('cold_A',[],push),('moving_A',[dict(push,ticks=20)],push),
           ('committed_B',[switch,switch],dict(target_id='b',a0=44,a1=10,ticks=10))]
    try:
        for name,prefix,action in cases:
            env=fresh();h=History();h.append(env.public())
            for a in prefix:
                ledger.charge(execution_ticks=a['ticks']);env.step_command(a);h.append(env.public(),a)
            root_audit=env.offline_audit()
            m=NominalPredictor(h,ticks=400,calls=8,seconds=15.)
            models.append(m)
            first=m.predict(m.root,action)
            second=m.predict(first['handle'],action)
            repeat=m.predict(first['handle'],action)
            assert repeat['observation']==second['observation']
            assert env.offline_audit()==root_audit
            for _ in range(2):
                ledger.charge(execution_ticks=action['ticks']);env.step_command(action)
            actual=env.offline_audit()
            # These are explicitly offline diagnostics, never predictor input.
            nominal=m.live[second['handle']].offline_audit()
            errors={label:math.dist(actual['entities'][label][:2],pose[:2])
                    for label,pose in nominal['entities'].items()}
            data=dict(name=name,prefix=prefix,action=action,root_public_history=h.events,
                      predictor=m.record(),predicted=second['observation'],executed_public=env.public(),
                      offline_audit=dict(actual=actual,nominal=nominal,position_errors_m=errors))
            (out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
            usage=m.ledger.used['ticks']
            report['cases'].append(dict(name=name,root_tick=h.events[-1]['observation']['tick'],
                predicted_tick=second['observation']['tick'],prediction_ticks=usage,
                repeated_prefix_equal=True,offline_position_errors_m=errors))
            m.close()
        env=fresh();h=History();h.append(env.public())
        m=NominalPredictor(h,ticks=160,calls=8,seconds=15.)
        models.append(m)
        selected=choose(h,m.root,m,lambda h,a:float(a['a0']==2 and a['a1']==2),lambda h:10.,
                        Budget(prediction_ticks=160,calls=4,seconds=5.,batch=8))
        assert selected['action'] is not None
        assert selected['prediction_ticks']==m.ledger.used['ticks']
        ledger.charge(execution_ticks=selected['action']['ticks'])
        result=env.step_command(selected['action'])
        (out/'search.json').write_text(json.dumps(dict(search=selected,predictor=m.record(),execution=result,
                                                      executed_public=env.public()),indent=2)+'\n')
        report['search']=dict(action=selected['action'],calls=selected['calls'],
                              prediction_ticks=selected['prediction_ticks'],actual_execution_ticks=result['end_tick'])
        report['status']='complete'
    except Exception as exc:
        report.update(status='incomplete',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        # Include failed/partial model calls, not just completed case summaries.
        ledger.charge(prediction_ticks=sum(m.ledger.used['ticks'] for m in models))
        for m in models:m.close()
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
    p=argparse.ArgumentParser();p.add_argument('destination')
    print(json.dumps(run(p.parse_args().destination),indent=2))
