"""Summarize the frozen semantic gate, controlled interventions and local stability."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from .cap_disagreement_features import OUT,ROOT
from .cap_disagreement_gate import choose,action_only

KEY=Path('docs/research/key_findings/2026-11-19_semantic_disagreement')


def read(path):
    return json.loads(path.read_text())


def write_csv(name,rows):
    with (KEY/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    return dict(n=len(rows),ga=sum(r['ga'] for r in rows),attention=sum(r['att'] for r in rows),
                union=sum(max(r['att'],r['ga']) for r in rows),gate=sum(r['value'] for r in rows),
                always_push=sum(r['push_value'] for r in rows),decisive=sum(r['ga']!=r['att'] for r in rows),
                correct=sum(r['ga']!=r['att'] and r['value'] for r in rows),
                rescue=sum(r['value']>r['ga'] for r in rows),loss=sum(r['value']<r['ga'] for r in rows),
                picked_attention=sum(r['pick']=='T' for r in rows))


def main():
    stats={};output=[]; near=[]
    oldpanel=read(ROOT/'cap_inverse_scale_20261119_01/evaluation_panel.json')
    cases={r['case_id']:r for r in oldpanel+read(OUT/'fresh_panel.json')+read(OUT/'neighbors_panel.json')}
    for split in ('discovery','oldcheck','fresh','neighbors'):
        rows=[]
        for x in read(OUT/(split+'_prefixes.json')):
            gate,reason=choose(x['G'],x['T']) if 'step' in x else ('G','identical')
            push=action_only(x['G'],x['T']) if 'step' in x else 'G'
            def result(arm):
                return x['ga'] if arm=='G' else x['att']
            row=dict(split=split,id=x['id'],movable=cases[x['id']]['movable_count'],ga=x['ga'],att=x['att'],
                     step=x.get('step'),tick=x.get('tick'),pick=gate,reason=reason,value=result(gate),
                     push_value=result(push))
            rows.append(row)
        output.extend(rows)
        if split=='neighbors':
            near=rows
        eligible=[r for r in rows if r['reason'] in ('ready','reposition')]
        stats[split]=dict(all=summarize(rows),eligible=summarize(eligible),
                          branches={k:summarize([r for r in rows if r['reason']==k]) for k in sorted({r['reason'] for r in rows})})
        if split=='fresh':
            rng=np.random.default_rng(33)
            deltas=[np.array([r['value']-r['ga'] for r in rows if r['movable']==k]) for k in sorted({r['movable'] for r in rows})]
            draws=sum(rng.choice(d,size=(20000,len(d))).sum(axis=1) for d in deltas)/len(rows)
            stats[split]['delta_vs_ga_ci_pp']=(100*np.quantile(draws,[.025,.975])).tolist()
    orig={r['id']:r for r in output if r['split']=='discovery'}
    crossed=defaultdict(dict)
    for r in read(OUT/'cross.json'):
        crossed[r['case']][r['first']+r['suffix']]=r['success']
    neighborhood=defaultdict(Counter)
    for r in near:
        parent=cases[r['id']]['parent']
        key=('both' if r['ga'] else 'neither') if r['ga']==r['att'] else ('same' if r['ga']==orig[parent]['ga'] else 'reversed')
        neighborhood[parent][key]+=1
    roots=[]
    for parent,alt in sorted(crossed.items()):
        gg=orig[parent]['ga']; tt=orig[parent]['att']; gt=alt['GT']; tg=alt['TG']
        assert tt!=gg and len(alt)==2 and sum(neighborhood[parent].values())==4
        effect=('first' if gt==gg and tg==tt else 'suffix' if gt==tt and tg==gg else
                'either_winner_component' if gt==tg==1 else 'both_winner_components')
        roots.append(dict(id=parent,GG=gg,GT=gt,TG=tg,TT=tt,effect=effect,
                          **{k:neighborhood[parent][k] for k in ('same','reversed','both','neither')}))
    stats['cross']=Counter(r['effect'] for r in roots)
    stats['near_counts']={k:sum(r[k] for r in roots) for k in ('same','reversed','both','neither')}
    stats['near_roots']={k:sum(r[k]>0 for r in roots) for k in ('same','reversed','both','neither')}
    stats['near_roots_stable_four']=sum(r['same']==4 for r in roots)
    write_csv('per_scene.csv',output)
    write_csv('interventions.csv',roots)
    (KEY/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    for key,value in stats.items():
        print(key,value.get('all',value) if isinstance(value,dict) else value)


if __name__=='__main__':
    main()
