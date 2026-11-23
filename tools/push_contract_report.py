"""Score matched PUSH contracts, local sensitivity, and public geometry predictability."""
import argparse
import csv
from pathlib import Path

import numpy as np

from tools import push_contract_bench as bench

OUT=bench.OUT
DOC=Path(__file__).resolve().parents[1]/'docs/research/key_findings/2026-11-23_push_contract'
KINDS=('standoff','push_low','push_progress','dual')


def load(split,arm,partial=False):
    cases=[c for c in bench.read(OUT/('protocol_stress.json' if split=='stress' else 'protocol.json'))['cases'] if c['split']==split]
    rows=[]
    for c in cases:
        for off in bench.OFFSETS:
            p=OUT/'runs'/arm/off/(c['id']+'.json')
            if not p.exists() and partial:continue
            rows.append(dict(bench.read(p),case=c))
    return rows


def execution(rows):
    return [r for r in rows if r['status']=='executed']


def bootstrap(values):
    a=np.asarray(values)
    return [float(x) for x in np.quantile(np.random.default_rng(23).choice(a,(5000,len(a))).mean(1),[.025,.975])]


def summarize(rows,bases):
    base={(r['id'],r['offset']):r for r in bases}
    assert all(r['status']==base[r['id'],r['offset']]['status'] and r['prefix']==base[r['id'],r['offset']]['prefix'] for r in rows)
    stats=[]
    for kind in KINDS:
        cohort=[r for r in rows if r['case']['kind']==kind]
        active=execution(cohort)
        local=[]
        for key in dict.fromkeys(r['id'] for r in active):
            g=[r for r in active if r['id']==key]
            if len(g)>1:
                v=[r['along'] for r in g]
                local.append((sum(r['effect'] for r in g),len(g),max(v)-min(v)))
        before=[base[r['id'],r['offset']] for r in active]
        no_contact=lambda r:r['stable'] is None
        new_contact=lambda k:sum(r[k]>0 and not b[k] for r,b in zip(active,before))
        lost_contact=lambda k:sum(r[k]==0 and b[k]>0 for r,b in zip(active,before))
        stats.append(dict(kind=kind,attempts=len(cohort),n=len(active),effect=sum(r['effect'] for r in active),
                          stable=sum(not no_contact(r) for r in active),
                          stable_no_effect=sum(not no_contact(r) and not r['effect'] for r in active),
                          reasons={k:sum(r['reason']==k for r in active) for k in sorted({r['reason'] for r in active})},
                          gain=sum(r['effect'] and not b['effect'] for r,b in zip(active,before)),
                          loss=sum(b['effect'] and not r['effect'] for r,b in zip(active,before)),
                          contact_gain=sum(no_contact(b) and not no_contact(r) for r,b in zip(active,before)),
                          contact_loss=sum(no_contact(r) and not no_contact(b) for r,b in zip(active,before)),
                          center_gain=sum(r['offset']=='zero' and r['effect'] and not b['effect'] for r,b in zip(active,before)),
                          center_loss=sum(r['offset']=='zero' and b['effect'] and not r['effect'] for r,b in zip(active,before)),
                          peer=sum(r['peer']>0 for r in active),other=sum(r['other']>0 for r in active),
                          peer_new=new_contact('peer'),other_new=new_contact('other'),peer_lost=lost_contact('peer'),
                          other_lost=lost_contact('other'),recovery=sum(bool(r['recovery']) for r in active),
                          varying=sum(0<n<m for n,m,_ in local),neighborhoods=len(local),
                          all_fulfill=sum(n==m for n,m,_ in local),all_fail=sum(n==0 for n,_,_ in local),
                          range_cm=100*float(np.median([v for _,_,v in local])) if local else None))
    return stats


def public_predictability(dev,validation):
    dev=[r for r in execution(dev) if r['case']['kind']!='dual']
    held=[r for r in execution(validation) if r['case']['kind']!='dual']
    def design(r):
        g=r['geometry']
        return np.array([r['case']['command']['ticks']*.05,ACTIONS_speed(r),g['contact'],g['visible'],
                         g['points'],g['goal_gap'],g['tangent'],g['station_clear'],g['peer']])
    a=np.stack(list(map(design,dev)));b=np.stack(list(map(design,held)))
    y=np.array([r['effect'] for r in dev],float);z=np.array([r['effect'] for r in held],float)
    delta=b-a.mean(0);std=np.maximum(a.std(0),.02);a=(a-a.mean(0))/std
    out={}
    for arm,n in [('command',2),('geometry',b.shape[1])]:
        x=a[:,:n];v=delta[:,:n]/std[:n]
        beta=np.linalg.solve(x.T@x+10*np.eye(n),x.T@(y-y.mean()))
        score=np.clip(y.mean()+v@beta,0,1)
        by={}
        for j,row in enumerate(held):
            by.setdefault(row['case']['room'],[]).append(float((score[j]-z[j])**2))
        mask=z==1
        diff=score[mask,None]-score[None,~mask]
        out[arm]=dict(brier=float(np.mean((score-z)**2)),brier_room=float(np.mean([np.mean(v) for v in by.values()])),
                      auc=float(np.mean((diff>0)+.5*(diff==0))))
    out['n']=len(held)
    return out


def ACTIONS_speed(r):
    return bench.ACTIONS[r['case']['command'][f'a{r["robot"]}']].speed


def render(example):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle, Circle
    from coimagine.random_start.spec import ROBOT_RADIUS
    from coimagine.random_start.geometry import box_polygon
    fig,axes=plt.subplots(len(example),2,figsize=(9,3.15*len(example)),squeeze=False)
    for (base,new),axs in zip(example,axes):
        case=base['case']
        obstacle_poses=[]
        with bench.world(bench.DEV[case['room']]) as (world,_):
            for step in bench.archive(case['sid'],case['room'])[:case['index']]:
                world.step(step['a'],remaining_ticks=1800-world.env.total_periods)
            assert world.history.sha256==base['prefix']
            for box in bench.DEV[case['room']]['movable_objects']:
                pose=bench.position(world.env,box['label'])
                obstacle_poses.append((pose,box['width']/2,box['height']/2))
        for r,ax in zip((base,new),axs):
            case=r['case'];fixture=bench.DEV[case['room']]
            for b in fixture['room']['obstacles']:
                ax.add_patch(Polygon(box_polygon(b['cx'],b['cy'],b['angle'],b['half_w'],b['half_h']),fc='#d0d5d9',ec='#5a6370'))
            for pose,w,h in obstacle_poses:
                ax.add_patch(Polygon(box_polygon(*pose,w,h),fc='#ded3b5',ec='#8b7954',alpha=.72))
            # Sand polygons show movable obstacles at the matched option start.
            for xy,style in [(r['start'],'--'),(r['frames'][-1]['xy'],'-')]:
                ax.add_patch(Polygon(box_polygon(*xy,.125,.5),fill=False,ls=style,ec='#4a507e',lw=1.8))
            for i,color in enumerate(('#147e8c','#be6550')):
                xy=np.array([f['bots'][i] for f in r['frames']])
                ax.plot(*xy.T,color=color,lw=2.3)
                ax.scatter(*xy[0],c=color,marker='x',s=55)
                ax.add_patch(Circle(xy[-1],ROBOT_RADIUS,fill=False,ec=color,lw=1.5))
                q=np.array([f['q'][i] for f in r['frames'] if f['q'][i] is not None])
                if len(q):ax.plot(*q.T,color=color,lw=1,ls=':',marker='.',ms=2)
            start=np.array(r['start'][:2]);d=np.array(r['dir'])
            ax.quiver(*start,*d,angles='xy',scale_units='xy',scale=3,color='#26313c')
            ax.add_patch(Rectangle((0,0),6,4.8,fill=False,color='#7a8289'))
            ax.set(xlim=(start[0]-1.25,start[0]+1.25),ylim=(start[1]-1.,start[1]+1.),aspect='equal',
                   title=f'{case["room"]}  {r["arm"]}  {r["offset"]}  |  {100*r["along"]:+.1f} cm  |  contact {r["stable"] or "–"}')
            ax.grid(alpha=.18)
    fig.suptitle('Audited option trajectories. Dotted: anchors. Dashed box: initial target.',fontsize=11)
    fig.tight_layout()
    fig.savefig(DOC/'trajectories.png',dpi=155)
    plt.close(fig)


def draw_neighborhoods(arms, before, after):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',sharey=True)
    cases=sorted({r['id'] for r in execution(before) if r['case']['kind']=='standoff'})
    for arm,rows,ax in zip(arms, (before,after),axs):
        for i,k in enumerate(cases):
            entries=[r for r in execution(rows) if r['id']==k]
            for mark,r in enumerate(entries):
                ax.scatter(i+(mark-2)*.10,100*r['along'],s=26,marker='o' if r['offset']=='zero' else 's',
                           color='#16837b' if r['effect'] else '#8a5365',alpha=.82)
        ax.axhline(2.,ls='--',lw=1,c='#454d57')
        ax.set(xticks=range(len(cases)),xticklabels=[v.split('standoff_')[1] for v in cases],title=arm)
        ax.tick_params(axis='x',labelrotation=45)
        ax.grid(axis='y',alpha=.18)
    axs[0].set_ylabel('Target displacement along the commanded direction (cm)')
    axs[0].legend(handles=[Line2D([],[],ls='',marker='o',color='#16837b',label='Contract fulfilled'),
                           Line2D([],[],ls='',marker='o',color='#8a5365',label='Contract missed')],loc='best')
    fig.suptitle('Held-out standoff prefixes: nominal pose (circle) and legal perturbations (squares)')
    fig.savefig(DOC/'neighborhoods.png',dpi=155)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--select',action='store_true')
    p.add_argument('--preview',action='store_true')
    a=p.parse_args()
    base=load('development','base',a.preview)
    if a.preview:
        for arm in bench.CONFIGS:
            subset=load('development',arm,True)
            keys={(r['id'],r['offset']) for r in base}
            matched=[r for r in subset if (r['id'],r['offset']) in keys]
            print(arm,[(r['kind'],r['n'],r['gain'],r['loss'],r['recovery'],r['peer_new']+r['other_new']) for r in summarize(matched,base)])
        return
    dev={a:summarize(load('development',a),base) for a in bench.CONFIGS}
    decision=OUT/'selected.json'
    if a.select:
        def value(rows):
            normal=next(r for r in rows if r['kind']=='push_progress')
            fail=[r for r in rows if r['kind'] in ('standoff','push_low')]
            net=sum(r['gain']-r['loss'] for r in fail)
            regressions=sum(r['other_new']+r['peer_new']-r['other_lost']-r['peer_lost'] for r in rows)
            friction=sum(r['varying'] for r in fail)
            contacts=sum(r['contact_gain']-r['contact_loss'] for r in fail)
            eligible=(normal['loss']<=2 and normal['center_loss']==0 and regressions<=2)
            return (eligible,net,-friction,contacts,-regressions)
        winner=max(dev,key=lambda arm:value(dev[arm]))
        if value(dev[winner])[1]<=0:winner='base'
        bench.save(decision,dict(candidate=winner,development=dev))
        print('selected',winner,flush=True)
        return
    winner=bench.read(decision)['candidate']
    before=load('validation','base')
    after=load('validation',winner)
    validation={k:summarize(v,before) for k,v in [('base',before),(winner,after)]}
    raw={(r['id'],r['offset']):r for r in execution(before)}
    groups={}
    for r in execution(after):
        b=raw[r['id'],r['offset']]
        family=groups.setdefault(r['case']['room'],[])
        family.append(float(r['effect'])-float(b['effect']))
    data=dict(dev=dev,validation=validation,selected=winner,
              geometry={'base':public_predictability(base,before),winner:public_predictability(load('development',winner),after)},
              held_effect_room_ci=bootstrap([np.mean(v) for v in groups.values()]))
    DOC.mkdir(parents=True,exist_ok=True)
    bench.save(DOC/'result.json',data)
    keys=['split','room','kind','arm','offset','status','reason','ticks','stable','effect','along','lateral',
          'contacts','peer','other','recovery_status','recovery']
    with (DOC/'branches.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        for r in before+after:
            c=r['case']
            w.writerow({k:r.get(k,c.get(k)) for k in keys})
    examples=[];seen=set()
    changed=[(raw[r['id'],r['offset']],r) for r in execution(after) if bool(r['recovery'])]
    for sign in (1,-1,0):
        pairs=[(b,r) for b,r in changed if int(r['effect'])-int(b['effect'])==sign and r['id'] not in seen]
        if pairs:
            pair=max(pairs,key=lambda x:abs(x[1]['along']-x[0]['along']))
            examples.append(pair);seen.add(pair[1]['id'])
    if not examples:examples=[(raw[k],raw[k]) for k in list(raw)[:1]]
    render(examples)
    draw_neighborhoods(('Frozen PUSH',winner),before,after)
    print(winner,validation,flush=True)


if __name__=='__main__':
    main()
