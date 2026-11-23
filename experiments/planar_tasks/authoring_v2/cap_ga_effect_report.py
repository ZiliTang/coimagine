"""Paired scene and informative held-fork comparisons for the frozen A/B/C study."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .cap_ga_effect_data import CAPS, OUT, old, publish, read, save

KEY=Path('docs/research/key_findings/2026-11-19_ga_effect')
ARMS=['original','A','B','C']


def export_csv(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def compare(rows,a,b,groups):
    """a minus b. Pair by scene, stratify according to original movable-count strata."""
    if not rows:return dict(n=0)
    delta=np.array([r[a]-r[b] for r in rows])
    rng=np.random.default_rng(71)
    draws=np.zeros(20000)
    for g in sorted(set(groups)):
        local=delta[np.array(groups)==g]
        draws+=rng.choice(local,(len(draws),len(local)),replace=True).sum(1)
    return dict(n=len(rows),rescue=int((delta>0).sum()),loss=int((delta<0).sum()),
                delta_pp=float(100*delta.mean()),ci_pp=np.round(np.quantile(100*draws/len(rows),[.025,.975]),2).tolist())


def training_counts():
    by=defaultdict(Counter)
    for p in sorted((OUT/'samples').glob('*.npz')):
        with np.load(p) as d:
            group='augment' if p.stem.startswith('inv_') else 'original'
            for k in (group,'all'):
                counts=by[k];counts['scenes']+=1;counts['planned']+=int(d['planned'])
                counts['mismatch']+=int(d['mismatch']>=0);counts['verified']+=int(d['verified'])
                counts['forks']+=len(np.unique(d['index']));counts['transitions']+=len(d['index'])
                counts['contributing_scenes']+=bool(len(d['index']))
                counts['terminal_or_no_legal']+=int((d['top']==-1).sum())
                counts['v009_top']+=int((d['top']==0).sum())
    return dict(by)


def effect_differences():
    # Diagnostic only: does the predictor distinguish two actions at one public prefix?
    import torch
    from .cap_ga_effect_fit import Heads,samples
    u,e=samples('held',('u','e'))
    base,_=old.load_checkpoint(old.Q)
    frozen=read(OUT/'models_frozen.json')
    truth=(e[:,1]-e[:,0]).numpy()
    result={}
    for arm in 'BC':
        model=Heads(base,arm)
        model.load_state_dict(torch.load(OUT/'models'/(arm+'.pt'),map_location='cpu',weights_only=True))
        with torch.no_grad():
            pred=model(u)[1]
            diff=((pred[:,1]-pred[:,0])*torch.tensor(frozen['effect_scale'])).numpy()
        result[arm]=[]
        for j in range(truth.shape[1]):
            target=truth[:,j];valid=np.isfinite(target)
            result[arm].append(dict(observed_pairs=int(valid.sum()),
                zero_delta_mae=float(np.abs(target[valid]).mean()),
                predicted_delta_mae=float(np.abs(target[valid]-diff[valid,j]).mean()),
                mean_abs_pred_delta=float(np.abs(diff[valid,j]).mean())))
    nominal=[];actual=[]
    for p in sorted((OUT/'held').glob('*.npz')):
        branches=read(old.OUT/'cases'/(p.stem+'.json'))['branches']['ga']
        fork=branches[0]['fork']
        nominal.append([min(a['ticks'],1800-fork['tick']) for a in fork['candidates']])
        actual.append([r['successor']['periods'] for r in branches])
    err=np.array(nominal)-np.array(actual)
    result['nominal_ticks_baseline']=dict(mae=float(np.abs(err).mean()),
                                         paired_delta_mae=float(np.abs(err[:,1]-err[:,0]).mean()))
    return result


def report():
    KEY.mkdir(parents=True,exist_ok=True)
    summary=dict(collection=training_counts(),effect_differences=effect_differences())
    evals=read(OUT/'evaluation.json');by=defaultdict(dict)
    for r in evals:by[r['case_id']][r['arm']]=r
    assert len(by)==200 and all(len(v)==11 for v in by.values())
    scenes=[]
    for cid,values in sorted(by.items()):
        row={s:values[s]['success'] for s in [*ARMS,*CAPS]}
        scenes.append(dict(case_id=cid,movable=values['A']['movable_count'],ncap=sum(row[c] for c in CAPS),**row))
    records={}
    for k,rr in [('all',scenes),('union',[r for r in scenes if r['ncap']]),
                 ('unique',[r for r in scenes if r['ncap']==1]),('outside',[r for r in scenes if not r['ncap']])]:
        records[k]=dict(n=len(rr),success={a:sum(r[a] for r in rr) for a in [*ARMS,*CAPS]},
                        comparisons={a+'-'+b:compare(rr,a,b,[r['movable'] for r in rr])
                                     for a,b in [('A','original'),('B','A'),('C','B'),('C','A'),('C','original')]})
    summary['fresh']=records
    hr=read(OUT/'held_results.json')
    held=[]
    panel={c['case_id']:c for c in read(old.OUT/'panel.json')}
    for i,p in enumerate(sorted((OUT/'held').glob('*.npz'))):
        with np.load(p) as d:
            z=d['z']
            row=dict(case_id=p.stem,movable=panel[p.stem]['movable_count'],z0=int(z[0]),z1=int(z[1]),original=int(z[0]))
            for arm in 'ABC':
                pick=hr[arm]['choices'][i];row[arm]=int(z[pick]);row['pick_'+arm]=pick
            held.append(row)
    info=[r for r in held if r['z0']!=r['z1']]
    summary['held']={a:{k:v for k,v in h.items() if k not in ('scores','choices')} for a,h in hr.items()}
    summary['held_paired']={a+'-'+b:compare(info,a,b,[r['movable'] for r in info]) for a,b in [('A','original'),('B','A'),('C','B')]}
    mask=(OUT/'models_frozen.json')
    save(KEY/'models_frozen.json',read(mask));save(KEY/'protocol.json',read(OUT/'protocol.json'))
    save(KEY/'summary.json',summary);export_csv(KEY/'per_scene.csv',scenes);export_csv(KEY/'per_fork.csv',held)
    export_csv(KEY/'training_curve.csv',read(OUT/'training_curve.json'))
    save(OUT/'FINISHED.json',dict(scenes=200,closed_loops=len(evals),held_pairs=len(held),summary=old.sha(KEY/'summary.json')))
    lines=['全新 200 个场景','　'.join(f"{a} {records['all']['success'][a]}/200" for a in ARMS),
           f"有胜负差的留出动作对 {len(info)}/{len(held)}",'　'.join(f"{a} {hr[a]['correct']}/{len(info)}" for a in ARMS)]
    publish('实验完成','所有冻结模型均完成',3,len(evals),result='\n'.join(lines))
    return summary


if __name__=='__main__':report()
