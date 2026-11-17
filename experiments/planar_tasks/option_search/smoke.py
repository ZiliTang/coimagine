"""Bounded framework acceptance, NOT new policy optimization or test performance."""
from pathlib import Path
import json
import shutil
import time
from experiments.planar_tasks.baseline import seal
from experiments.planar_tasks.baseline.session import package_command, verify_frozen
from experiments.planar_tasks.baseline.panels import load_panel
from experiments.planar_tasks.runner import run_episode, RunBudget
from .build import build_package, verify_package, verify
from .search import Budget

ROOT=seal.ROOT
SOURCE=ROOT/'outputs/agentic/prediction_sol_runtime_20261110_01'
OUT=ROOT/'outputs/agentic/option_search_framework_20261110_01'
LIMITS=dict(execution_ticks=750,prediction_ticks=6000,wall_seconds=180,episodes=9)


def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def load(path):return json.loads(path.read_text())


def main():
    verify();verify_frozen(SOURCE/'session')
    OUT.mkdir(parents=True,exist_ok=False)
    _,tasks=load_panel(SOURCE/'evaluation/panel.json')
    save(OUT/'preflight.json',dict(kind='framework_acceptance_not_research_evaluation',limits=LIMITS,
        panel_sha256=seal.digest(SOURCE/'evaluation/panel.json'),framework_release_sha256=seal.digest(Path(__file__).with_name('release_v1.json')),
        source_submission_sha256=seal.digest(SOURCE/'session/frozen/manifest.json'),selected_indices=[0,10,21],
        comparison='same 3-decision prefixes, original free policy vs framework with action-selecting numeric heuristic; reference heuristic separately'))
    reference=OUT/'reference_package';build_package(reference,budget=Budget(expansions=3,prediction_ticks=240,prediction_calls=20))
    started=time.monotonic();deadline=started+LIMITS['wall_seconds'];results=[];used_ticks=used_prediction=0
    try:
        for index in (0,10,21):
            source_row=load(SOURCE/'evaluation/summary.json')['results'][index]
            decisions=load(SOURCE/'evaluation'/source_row['episode_path']/'decisions.json')[:3]
            commands=[dict(a0=d['actions'][0],a1=d['actions'][1],ticks=d['requested_ticks']) for d in decisions]
            heuristic=OUT/f'faithfulness_{index}.py'
            heuristic.write_text('commands='+repr(commands)+'\n'
                'def score(context,nodes,memory):\n'
                ' desired=commands[context["observation"]["decision"]]\n'
                ' return [-10000. if n["plan"][0]==desired else 10000. for n in nodes]\n')
            package=OUT/f'package_{index}';build_package(package,heuristic,Budget(expansions=3,prediction_ticks=240,prediction_calls=20))
            entries={}
            for condition,source in [('free',SOURCE/'session/frozen/package'),('framework',package),('reference',reference)]:
                tick_cap=3 if condition=='reference' else 120
                if used_ticks+tick_cap>LIMITS['execution_ticks']:raise RuntimeError('smoke physical budget')
                if condition!='free':verify_package(source)
                destination=OUT/f'{index}_{condition}'
                summary=run_episode(tasks[index],package_command(source),destination,
                    RunBudget(3,20,5),max_control_ticks=tick_cap,wall_deadline=deadline,
                    prediction_profile=dict(version='nominal-v1',runtime=True),
                    provenance=dict(kind='framework_acceptance',condition=condition),
                    security_sandbox=dict(kind='bubblewrap',read_only_package=True))
                used_ticks+=summary['executed_control_ticks'];used_prediction+=summary.get('prediction_usage',{}).get('ticks',0)
                if used_prediction>LIMITS['prediction_ticks']:raise RuntimeError('smoke prediction budget')
                assert summary['error'] is None,summary['error']
                entries[condition]=dict(summary=summary,path=str(destination.relative_to(OUT)))
            equality={name:(OUT/entries['free']['path']/name).read_bytes()==(OUT/entries['framework']['path']/name).read_bytes()
                      for name in ('decisions.json','sensors.jsonl','truth.json')}
            assert all(equality.values()),equality
            traces=[]
            for line in (OUT/entries['framework']['path']/'policy.stderr').read_text().splitlines():
                try:record=json.loads(line)
                except ValueError:continue
                if record.get('kind')=='option_search':traces.append(record)
            results.append(dict(index=index,family=source_row['family'],commands=commands,equality=equality,entries=entries,
                search_depths=[max([len(e['plan']) for e in t['events']] or [0]) for t in traces],
                root_sizes=[t['root_action_count'] for t in traces]))
        summary=dict(status='complete',limits=LIMITS,execution_ticks=used_ticks,prediction_ticks=used_prediction,
                     wall_seconds=time.monotonic()-started,results=results)
    except Exception as exc:
        summary=dict(status='incomplete',error=dict(type=type(exc).__name__,message=str(exc)),limits=LIMITS,
                     execution_ticks=used_ticks,prediction_ticks=used_prediction,wall_seconds=time.monotonic()-started,results=results)
    save(OUT/'summary.json',summary)
    save(OUT/'artifacts_sha256.json',{str(p.relative_to(OUT)):seal.digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifacts_sha256.json'})
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},indent=2))


if __name__=='__main__':main()
