"""Read-only follow-up: where the retention tradeoffs occur and example trajectories."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import numpy as np
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.runtime import CAPS
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read,save
from tools.cap_trace_value_long_report import write_csv


def illustrate(out,routing):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle,Polygon
    examples=[]
    for from_label,to_label in [(1,0),(0,1)]:
        selected=sorted([r for r in routing if r['arm']=='paired' and r['seed']=='17' and r['mode']=='free'
            and int(r['once'])==from_label and int(r['success'])==to_label],key=lambda r:r['case_id'])[:2]
        examples+=selected
    if not examples:return []
    fig,axes=plt.subplots(1,len(examples),figsize=(4.4*len(examples),4.1),squeeze=False,layout='constrained')
    details=[]
    for ax,r in zip(axes.flat,examples):
        cid=r['case_id']; paths={}
        for mode in ('once','free'):
            f=out/'evaluation'/f'paired_s17_{mode}'/cid/'result.json.gz'
            with gzip.open(f,'rt') as stream:paths[mode]=json.load(stream)
        j=int(r['first_switch'])
        assert all(a['history_hash']==b['history_hash'] for a,b in zip(paths['once']['public_rows'][:j+1],paths['free']['public_rows'][:j+1]))
        a=paths['once']['decisions'][j];b=paths['free']['decisions'][j]
        assert a['requested']!=b['requested']
        entry=dict(case_id=cid,first_divergence=j,public_history_hash=paths['once']['public_rows'][j]['history_hash'],
            initial_cap=r['first_cap'],challenger=CAPS[b['cap']],once_action=a['requested'],free_action=b['requested'],
            score_margin=b['scores'][b['cap']]-b['scores'][a['cap']],once_success=int(r['once']),free_success=int(r['success']))
        details.append(entry)
        scene=paths['once']['replay']['scene']
        for ob in scene['obstacles']:
            rect=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*[ob['half_w'],ob['half_h']]
            c,s=math.cos(ob['angle']),math.sin(ob['angle'])
            ax.add_patch(Polygon(rect@np.array([[c,s],[-s,c]])+[ob['cx'],ob['cy']],color='0.85'))
        ax.add_patch(Circle((scene['goal']['x'],scene['goal']['y']),scene['goal']['radius'],color='green',alpha=.35))
        for mode,color in [('once','#2075ab'),('free','#dc7c2e')]:
            data=paths[mode]
            xy=np.array([fr['entities'][scene['primary_id']][:2] for fr in data['replay']['frames']])
            ax.plot(*xy.T,color=color,label=f'{mode}: '+('success' if data['summary']['success'] else 'failure'))
            ax.scatter(*xy[-1],s=18,color=color)
        ax.set(title=f'{cid}\n{r["first_cap"]} / first switch at boundary {j}',aspect='equal',
               xlim=(0,scene['arena'][0]),ylim=(0,scene['arena'][1]),xlabel='x (m)',ylabel='y (m)')
        ax.legend(fontsize=8)
    fig.savefig(out/'examples.png',dpi=145);plt.close(fig)
    save(out/'examples.json',details)
    return details


def switch_margins(out,routing):
    # Only public scores before the first action deviation; endpoint outcomes
    # compare entire deployed policies, not the isolated effect of this action.
    rows=[]
    for r in routing:
        if r['mode']!='free' or not r['first_switch'] or int(r['forced']):continue
        name=f"{r['arm']}_s{r['seed']}_free"
        with gzip.open(out/'evaluation'/name/r['case_id']/'result.json.gz','rt') as fp:
            trace=json.load(fp)
        j=int(r['first_switch'])
        decision=trace['decisions'][j]
        if j==0 or decision['reason']!='switch':continue
        previous=trace['decisions'][j-1]['cap']
        margin=decision['scores'][decision['cap']]-decision['scores'][previous]
        rows.append(dict(arm=r['arm'],seed=r['seed'],case_id=r['case_id'],boundary=j,
            margin=margin,band='below_015' if margin<.15 else 'at_least_015',
            first_cap=r['first_cap'],new_cap=CAPS[decision['cap']],once=int(r['once']),free=int(r['success'])))
    write_csv(out/'first_switches.csv',rows)
    stat=[]
    for arm in ('boundary','root','paired'):
        for seed in ('17','29'):
            for band in ('below_015','at_least_015'):
                ss=[r for r in rows if (r['arm'],r['seed'],r['band'])==(arm,seed,band)]
                stat.append(dict(arm=arm,seed=seed,band=band,n=len(ss),once_success=sum(r['once'] for r in ss),
                    once_failure=sum(1-r['once'] for r in ss),rescued=sum(r['free']>r['once'] for r in ss),
                    lost=sum(r['once']>r['free'] for r in ss)))
    write_csv(out/'margin_groups.csv',stat)


def main(out):
    assets=Path('docs/agent_exchange/assets')/out.name
    with (out/'routing.csv').open() as f:routing=list(csv.DictReader(f))
    with (out/'per_scene.csv').open() as f:cases=list(csv.DictReader(f))
    table={r['case_id']:r for r in cases}
    signals=[]
    for arm in ('boundary','root','paired'):
        for seed in (17,29):
            for mode in ('once','free','guard'):
                rows=[r for r in routing if r['arm']==arm and r['mode']==mode and r['seed']==str(seed)]
                for label,test in [('only_one_solver',lambda c:int(c['union7_count'])==1),
                    ('multiple_solvers',lambda c:int(c['union7_count'])>=2),('outside_union',lambda c:int(c['union7_count'])==0),
                    *[(cap,lambda c,cap=cap:int(c[cap])==1) for cap in CAPS]]:
                    subset=[r for r in rows if test(table[r['case_id']])]
                    signals.append(dict(arm=arm,seed=seed,mode=mode,subset=label,n=len(subset),
                        success=sum(int(r['success']) for r in subset),initial_cap_success=sum(int(r['once']) for r in subset),
                        rescues=sum(r['success']=='1' and r['once']=='0' for r in subset),
                        losses=sum(r['success']=='0' and r['once']=='1' for r in subset)))
    write_csv(out/'coverage_strata.csv',signals)
    switch_margins(out,routing)
    illustrations=illustrate(out,routing)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta=read(out/'summary.json')
    inv=meta['coverage']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for i,arm in enumerate(('boundary','root','paired')):
        for j,mode in enumerate(('once','free','guard')):
            rows=[r for r in meta['metrics'] if (r['arm'],r['mode'])==(arm,mode)]
            pos=i+(j-1)*.25
            color=('#436f9e','#db863b','#489077')[j]
            axes[0].bar(pos,np.mean([r['retained7'] for r in rows])/inv['union7'],width=.23,color=color,label=mode if i==0 else None)
            axes[0].plot([pos]*2,[r['retained7']/inv['union7'] for r in rows],'k.',ms=4)
            if j:
                axes[1].bar(pos,np.mean([r['rescued_vs_once'] for r in rows]),width=.22,color=color,label=mode if i==0 else None)
                axes[1].bar(pos,-np.mean([r['lost_vs_once'] for r in rows]),width=.22,color=color,alpha=.45)
    for ax,title in zip(axes,('Retained fraction of 7-CAP union','Per 200 scenes: rescued (+) / lost (-) vs once')):
        ax.set_title(title);ax.set_xticks(range(3),['Boundary','Root','Paired prefixes']);ax.legend();ax.axhline(0,color='0.35',lw=.8)
    axes[0].set_ylim(0,1)
    fig.savefig(out/'retention.png',dpi=150);plt.close(fig)
    for name in ('retention.png','coverage_strata.csv','first_switches.csv','margin_groups.csv','examples.png','examples.json','provenance.json'):
        if (out/name).exists():shutil.copyfile(out/name,assets/name)
    print('Paired replay examples',illustrations,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
