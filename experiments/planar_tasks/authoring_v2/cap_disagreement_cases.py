"""Audit selected discovery examples, with privileged frames isolated from the gate."""
import ast
import json
import math
from . import cap_disagreement_features as f
from .cap_disagreement_report import KEY,read


def program(name):
    path=f.SOURCE.with_name('CAP-RR-'+name+'.py')
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name!='evaluate_candidates']
    env={'math':math}
    exec(compile(tree,str(path),'exec'),env)
    return env


def main():
    rows={r['id']:r for r in read(f.OUT/'discovery_prefixes.json')}
    examples={}
    for suffix in ('dev_3_20','validation_3_43','dev_2_38','dev_4_13'):
        cid='inverse_test_0_'+suffix
        x=rows[cid]; index=x['step']-1
        models=[f.load(path/cid/'result.json.gz') for path in f.OLD]
        prefix=models[0]['public_rows'][:index+1]; o=prefix[-1]['observation']
        cases={}
        for arm in ('G','T'):
            a=x[arm]; name=a['caps'][0]; env=program(name)
            mem=env['initial_memory']();prev=0
            for row in prefix:
                tick=row['observation']['tick']
                mem=env['update_memory'](mem,dict(row,executed_ticks=tick-prev))
                prev=tick
            action,signal,phase,k,_,_=env['choose'](mem,o)
            assert [action['a0'],action['a1'],action['ticks'],action['target_id']]==[*a['codes'],a['ticks'],a['target']]
            pairs=o['entity_legal_pairs'].get(o['goal']['target_id'],[])
            cases[arm]=dict(action=a,representative_cap=name,phase=phase,desired_dir=k,
                            route_direction=signal[2:4],legal_fast_drivers=[j for j in range(2)
                            if any(p[j]==4*k+3 and p[1-j]==46 for p in pairs)],
                            stall=mem.get('stall'),recover_until=mem.get('until'),still=mem['still'])
        trajectories={}
        for name,model in zip(('GG','TT','GT','TG'),[*models,*[f.load(f.OUT/'cross'/cid/n/'result.json.gz') for n in ('GT','TG')]]):
            tr=model['replay']['frames'];target=o['goal']['target_id'];goal=o['goal']['xy']
            def snapshot(frame):
                return dict(tick=frame['tick'],distance=math.dist(frame['entities'][target][:2],goal))
            tracks=[snapshot(min(tr,key=lambda frame:abs(frame['tick']-t))) for t in (o['tick'],o['tick']+40,o['tick']+120)]
            d=model['decisions'][index:index+3]
            trajectories[name]=dict(success=model['summary']['success'],termination=model['summary']['termination'],
                                    first_three=[dict(tick=e['event']['start_tick'],periods=e['event']['periods'],action=e['action']) for e in d],
                                    frames=tracks+[snapshot(tr[-1])])
        examples[cid]=dict(tick=x['tick'],step=x['step'],goal=x['goal'],geometry=x['geometry'],robots=x['robots'],
                           contacts=x['contacts'],cases=cases,trajectories=trajectories)
    (KEY/'case_audits.json').write_text(json.dumps(examples,indent=2)+'\n')
    for k,v in examples.items():
        print(k,v['cases'])
        for a,s in v['trajectories'].items():
            print(a,s['frames'],s['first_three'][1]['action'])


if __name__=='__main__':
    main()
