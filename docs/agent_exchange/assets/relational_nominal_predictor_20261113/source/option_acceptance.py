"""Bounded real Option wiring acceptance. No learned policy or task benchmark."""
from pathlib import Path
import hashlib
import json
import time
from .environment import MultiObjectOptionEnv,MovableBox
from .evidence import Ledger
from .contract import canonical
from experiments.planar_tasks.tasks import task_catalog


def run(destination):
    out=Path(destination); out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    ledger=Ledger(dict(execution_ticks=2000))
    instance=task_catalog()['planar_open_anchor'].instance
    # One simple reachable B, and one physically constrained B. These are
    # development wiring fixtures, not a selected success-rate test set.
    cases=[('reachable',MovableBox('b',1.55,2.95,width=.6,height=.3)),
           ('constrained',MovableBox('b',2.,2.85,width=.3,height=.3))]
    report=dict(status='running',observation='partial centralized sensor sharing',llm_calls=0,
                task_evaluations=0,cases=[])
    try:
        for name,box in cases:
            env=MultiObjectOptionEnv(instance,[box])
            initial=env.offline_audit()
            events=[]
            def execute(target,a0,a1,ticks):
                if time.monotonic()-started>900:raise RuntimeError('wall limit')
                ledger.charge(execution_ticks=ticks)
                result=env.step_command(dict(target_id=target,a0=a0,a1=a1,ticks=ticks))
                events.append(dict(result=result,public_observation=env.public(),offline_audit=env.offline_audit()))
            execute('b',44,44,10);execute('b',44,44,10)
            for _ in range(2):
                if [44,10] not in env.public()['entity_legal_pairs'].get('b',[]):break
                execute('b',44,10,30)
            # Account for stopping; do not switch while force references persist.
            execute(None,44,44,20)
            for _ in range(3):
                if not any(e['id']=='entity-a' for e in env.public()['entities']):break
                execute('entity-a',44,44,10)
                if env.active_target=='entity-a':break
            final=env.offline_audit()
            ledger.charge(execution_ticks=env.total_periods)
            replay=env.replay_commands()
            assert replay.offline_audit()==final
            assert env.active_target=='entity-a','handoff did not return to transport target'
            delta=final['entities']['b'][1]-initial['entities']['b'][1]
            if name=='reachable':assert delta>.01,'reachable fixture did not move B'
            report['cases'].append(dict(name=name,fixture=box.__dict__,execution_ticks=env.total_periods,
                replay_ticks=replay.total_periods,exact_replay=True,blocker_dy=delta,
                final_target=env.active_target,last_b_feedback=[
                    r['execution'] for e in events if e['result']['handoff']['target']=='b'
                    for r in e['public_observation']['robots']][-2:],
                b_push_feedback=[e['public_observation']['robots'][1]['execution'] for e in events
                                 if e['result']['requested']['target_id']=='b' and e['result']['requested']['a1']<32]))
            with (out/(name+'.jsonl')).open('w') as f:
                for event in events:f.write(canonical(event)+'\n')
        report['status']='complete'
    except Exception as exc:
        report.update(status='incomplete',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(ledger=ledger.record(),wall_seconds=time.monotonic()-started)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        source=out/'source';source.mkdir()
        for p in Path(__file__).parent.glob('*.py'):(source/p.name).write_bytes(p.read_bytes())
        tests=Path('tests/test_relational_environment.py');(source/tests.name).write_bytes(tests.read_bytes())
        files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in out.rglob('*') if p.is_file()}
        (out/'manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('destination')
    print(json.dumps(run(parser.parse_args().destination),indent=2))
