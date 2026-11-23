"""Describe frozen expert success regions and test geometry-only readouts on separate rooms."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/transfers/cap_trace_value_dev_v1'
SCALE = ROOT/'outputs/semantic_rl/cap_inverse_scale_20261119_01'
OUT = ROOT/'docs/research/key_findings/2026-11-19_cap_geometry'
CAPS = ('009','003','001','010','006','007','002')
COLORS = ('#0072b2','#e69f00','#009e73','#d55e00','#7655aa','#b64b84','#25717c')


def read(path):
    return json.loads(path.read_text())


def geometry(f):
    def xy(b):
        return np.array([b['x'], b['y']])
    t, goal = xy(f['object']), xy(f['goal'])
    route = goal-t
    length = np.linalg.norm(route)
    along = route/length
    side = np.array([-along[1], along[0]])
    arena = np.array(f['room']['arena'])
    g = dict(goal_distance=length, goal_dir_x=along[0], goal_dir_y=along[1])
    for name, pos in [('target',t),('goal',goal)]:
        g[name+'_x'],g[name+'_y'] = pos/arena
        g[name+'_wall'] = min(*pos, *(arena-pos))
    theta = f['object']['angle'] - np.arctan2(along[1],along[0])
    g.update(axis_cos2=np.cos(2*theta),axis_sin2=np.sin(2*theta))
    for i, robot in enumerate(f['robots']):
        d = xy(robot)-t
        heading = np.array([np.cos(robot['angle']),np.sin(robot['angle'])])
        for name, feature in [('along',d@along),('side',d@side),('heading_along',heading@along),('heading_side',heading@side)]:
            g[f'r{i}_{name}'] = feature
    g['robot_separation'] = np.linalg.norm(xy(f['robots'][0])-xy(f['robots'][1]))
    points = t + np.linspace(0,1,41)[:,None]*route
    for name, bodies in [('wall_obstacle',f['room']['obstacles']),('loose_object',f['movable_objects'])]:
        g[name+'_count'] = len(bodies)
        clearance = np.full(len(points),2.)
        for b in bodies:
            center = [b.get('cx',b.get('x')), b.get('cy',b.get('y'))]
            half = [b['half_w'], b['half_h']] if 'half_w' in b else [b['width']/2,b['height']/2]
            c,s = np.cos(b['angle']),np.sin(b['angle'])
            local = np.abs((points-center)@np.array([[c,-s],[s,c]]))-half
            signed = np.linalg.norm(np.maximum(local,0),axis=1)+np.minimum(local.max(1),0)
            clearance = np.minimum(clearance,signed)
        g.update({name+'_start':clearance[0],name+'_end':clearance[-1],name+'_route':min(clearance)})
    return g


def content(f):
    return hashlib.sha256(json.dumps({k:v for k,v in f.items() if k!='scene_id'},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def auc(labels,pred):
    pos,neg=pred[labels==1],pred[labels==0]
    return float(((pos[:,None]>neg).sum()+0.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))


def table(name, rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def near_examples(checks,scenes):
    from matplotlib.patches import Polygon, Circle
    children={r['case_id']:r['fixture'] for file in sorted((SCALE/'batches').glob('*_plan.json')) for r in read(file)}
    parents={r['case_id']:r['fixture'] for r in scenes}
    palette={'target':'#0072b2','movable':'#bad5de','fixed':'#777f88','robot':'#cd6746'}
    def body(ax,xy,angle,w,h,color,heading=False):
        c,s=np.cos(angle),np.sin(angle)
        corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*[w/2,h/2]
        ax.add_patch(Polygon(corners@np.array([[c,s],[-s,c]])+xy,closed=True,
                             facecolor=color,edgecolor='#37404a',linewidth=.7))
        if heading:
            ax.plot([xy[0],xy[0]+c*w*.5],[xy[1],xy[1]+s*w*.5],c='white',lw=1.3)
    def plot(ax,f):
        obj=f['object']; goal=f['goal']
        ax.plot([goal['x'],obj['x']],[goal['y'],obj['y']],ls=':',c='#6d8285',lw=1)
        ax.add_patch(Circle([goal['x'],goal['y']],goal['radius'],edgecolor='#25876b',facecolor='none',lw=1.5))
        for b in f['room']['obstacles']:
            body(ax,[b['cx'],b['cy']],b['angle'],2*b['half_w'],2*b['half_h'],palette['fixed'])
        for b in f['movable_objects']:
            body(ax,[b['x'],b['y']],b['angle'],b['width'],b['height'],palette['movable'])
        body(ax,[obj['x'],obj['y']],obj['angle'],.25,1.,palette['target'])
        for b in f['robots']:
            body(ax,[b['x'],b['y']],b['angle'],.36,.36,palette['robot'],True)
        ax.set(xlim=(0,f['room']['arena'][0]),ylim=(0,f['room']['arena'][1]),aspect='equal',xticks=[],yticks=[])
    examples=[]
    for field,criterion in [('same sole winner',lambda r:r['still_sole']),('other CAPs only',lambda r:r['any_success'] and not r['retained']),('all seven fail',lambda r:not r['any_success'])]:
        examples.append((field,next(r for r in checks if criterion(r))))
    fig,axes=plt.subplots(2,3,figsize=(11,6.4))
    for j,(title,r) in enumerate(examples):
        plot(axes[0,j],parents[r['parent']]);plot(axes[1,j],children[r['child']])
        axes[0,j].set_title(r['parent']+'  /  sole '+r['cap'],fontsize=10)
        axes[1,j].set_title(r['child']+'  /  '+title,fontsize=10)
    axes[0,0].set_ylabel('Original room');axes[1,0].set_ylabel('Perturbed neighbor')
    fig.suptitle('Small perturbations of original sole-winner rooms')
    fig.text(.5,.015,'Dark blue: target   ·   Green circle: goal   ·   Orange: robots   ·   Light blue: movable clutter   ·   Gray: fixed obstacles',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94))
    fig.savefig(OUT/'nearby_rooms.png',dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scenes=read(BUNDLE/'scenes.json')
    panel=read(SCALE/'evaluation_panel.json')
    assert len({content(r['fixture']) for r in scenes+panel})==1360
    with (SCALE/'per_scene.csv').open() as f:
        reference={r['case_id']:r for r in csv.DictReader(f)}
    outcomes={(r['case_id'],r['cap_id']):int(r['summary']['success']) for r in read(BUNDLE/'index.json')}
    assert len(outcomes)==5320
    y=np.array([[outcomes[s['case_id'],'CAP-RR-'+c] for c in CAPS] for s in scenes])
    test=np.array([[int(reference[s['case_id']]['CAP-RR-'+c]) for c in CAPS] for s in panel])
    features=[geometry(r['fixture']) for r in scenes+panel]
    names=list(features[0])
    x=np.array([list(f.values()) for f in features])
    n=len(y)
    z=(x-x[:n].mean(0))/x[:n].std(0)
    _,sv,vectors=np.linalg.svd(z[:n],full_matrices=False)
    for v in vectors:
        if v[np.argmax(abs(v))]<0:v*=-1
    pcs=z@vectors[:2].T
    variance=sv**2/(sv**2).sum()
    view=[dict(id=r['case_id'],f=r['fixture'],y=y[i].tolist(),pc=pcs[i].tolist(),
               dist=features[i]['loose_object_route'],goal=features[i]['goal_distance']) for i,r in enumerate(scenes)]
    template=Path(__file__).with_name('cap_initial_geometry_view.html').read_text()
    (OUT/'explore.html').write_text(template.replace('__DATA__',json.dumps(view,separators=(',',':')).replace('<',r'\u003c')))
    totals=y.sum(1)
    fig,axes=plt.subplots(2,4,figsize=(14.6,7.7),sharex=True,sharey=True)
    for i,ax in enumerate(axes.flat):
        if i<7:
            hit=y[:,i].astype(bool)
            ax.scatter(*pcs[:n][~hit].T,s=10,c='#c7cdd2',alpha=.60,linewidth=0)
            ax.scatter(*pcs[:n][hit].T,s=13,c=COLORS[i],alpha=.62,linewidth=0)
            single=hit&(totals==1)
            ax.scatter(*pcs[:n][single].T,s=49,facecolors='none',edgecolors='#232c37',lw=1)
            ax.set_title(f"CAP {CAPS[i]}   ·   {hit.sum()} successes, {single.sum()} sole",fontsize=10.5)
        else:
            ax.scatter(*pcs[:n].T,s=10,c='#dce0e2',lw=0)
            for j in range(7):
                ax.scatter(*pcs[:n][(totals==1)&(y[:,j]==1)].T,s=24,c=COLORS[j],lw=.35,edgecolors='#222222',label=CAPS[j])
            ax.set_title('Sole winners across all seven CAPs',fontsize=10.5)
            ax.legend(ncol=4,loc='upper right',fontsize=7,handletextpad=.1,columnspacing=.5)
        ax.grid(alpha=.15)
    for ax in axes[-1]:ax.set_xlabel(f'PC1 ({variance[0]:.1%})')
    for ax in axes[:,0]:ax.set_ylabel(f'PC2 ({variance[1]:.1%})')
    fig.suptitle('Seven expert policies on the same 760 initial rooms',fontweight='bold',fontsize=17)
    fig.legend(handles=[Line2D([],[],ls='',marker='o',color=c,label=t,markerfacecolor=m) for c,t,m in
                        [('#b8c0c7','Failure','#b8c0c7'),('#587181','Success (color identifies expert)','#587181'),('#232c37','Sole success','none')]],
               loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,.055,1,.95])
    fig.savefig(OUT/'initial_geometry_pca.png',dpi=170)
    plt.close(fig)
    distance=np.linalg.norm(z[n:,None]-z[None,:n],axis=2)
    def scores(mask,labels):
        ix=np.argsort(distance[:,mask],axis=1)[:,:15]
        return labels[mask][ix].mean(axis=1)
    pred=scores(np.ones(n,bool),y)
    rows=[]
    for j,c in enumerate(CAPS):
        row=dict(cap=c,dev_success=int(y[:,j].sum()),dev_sole=int(((totals==1)&(y[:,j]==1)).sum()),test_success=int(test[:,j].sum()),auc=auc(test[:,j],pred[:,j]))
        if j:
            train_diff=y[:,j]!=y[:,0]
            test_diff=test[:,j]!=test[:,0]
            relative=scores(train_diff,y[:,j])[test_diff]
            row.update(vs009_test=int(test_diff.sum()),vs009_cap_wins=int(test[test_diff,j].sum()),vs009_auc=auc(test[test_diff,j],relative))
        else:row.update(vs009_test='',vs009_cap_wins='',vs009_auc='')
        rows.append(row)
    # Select each 1D comparison within development only, then transfer feature and direction.
    one_dim=[]
    bins=[]
    route=x[n:,names.index('loose_object_route')]
    for j in range(1,7):
        a,b=y[:,j]!=y[:,0],test[:,j]!=test[:,0]
        measure=np.array([auc(y[a,j],col[a]) for col in x[:n].T])
        k=int(abs(measure-.5).argmax())
        sign=1 if measure[k]>.5 else -1
        one_dim.append(dict(cap=CAPS[j],feature=names[k],sign=sign,
                            development_auc=max(measure[k],1-measure[k]),
                            held_auc=auc(test[b,j],sign*x[n:,k][b])))
        for label,mask in [('intersects',route<0),('within_0.5m',(route>=0)&(route<.5)),('farther',route>=.5)]:
            take=mask&b
            bins.append(dict(cap=CAPS[j],clearance=label,cap_wins=int(test[take,j].sum()),discordant=int(take.sum())))
    table('one_feature_transfer.csv',one_dim)
    table('corridor.csv',bins)
    fig,ax=plt.subplots(figsize=(9.4,4.7))
    for k,title in enumerate(['Centerline intersects','0 to 0.5 m','At least 0.5 m']):
        vals=bins[k::3]
        p=np.array([r['cap_wins']/r['discordant'] for r in vals])
        h=ax.bar(np.arange(6)+(k-1)*.26,p,width=.245,color=['#0072b2','#67b6bc','#dbab67'][k],label=title)
        ax.bar_label(h,[f"{r['cap_wins']}/{r['discordant']}" for r in vals],fontsize=8,padding=2)
    ax.set_xticks(range(6),['CAP '+c for c in CAPS[1:]])
    ax.axhline(.5,c='#555555',lw=1,ls='--')
    ax.set_ylim(0,.89)
    ax.set_ylabel('Fraction won by the named CAP')
    ax.set_title('When exactly one of the named CAP and CAP 009 succeeds')
    ax.legend(ncol=3,frameon=False,loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT/'corridor_vs_009.png',dpi=170)
    plt.close(fig)
    table('held_geometry_scores.csv',rows)
    table('scenes.csv',[dict(case_id=r['case_id'],panel='development' if i<n else 'held',pc1=pcs[i,0],pc2=pcs[i,1],**dict(zip(['cap_'+c for c in CAPS],np.vstack((y,test))[i])),**features[i]) for i,r in enumerate(scenes+panel)])
    table('loadings.csv',[dict(feature=name,PC1=vectors[0,j],PC2=vectors[1,j]) for j,name in enumerate(names)])
    train_ix={r['case_id']:i for i,r in enumerate(scenes)}
    neighbors={}
    for file in sorted((SCALE/'batches').glob('*_results.json')):
        for r in read(file):
            key=r['case_id']
            if key not in neighbors:neighbors[key]=(r['parent'],{})
            neighbors[key][1][r['cap'][-3:]]=int(r['success'])
    prefix=read(SCALE/'doses/1001.json')['neighbors']
    checks=[]
    for cid,(parent,result) in list(neighbors.items())[:prefix]:
        a=y[train_ix[parent]]
        if a.sum()!=1:continue
        b=np.array([result[c] for c in CAPS]);i=a.argmax()
        checks.append(dict(parent=parent,child=cid,cap=CAPS[i],retained=int(b[i]),still_sole=int(b[i] and b.sum()==1),any_success=int(b.any())))
    table('nearby_singletons.csv',checks)
    near_examples(checks,scenes)
    counts=test.sum(1)
    commit=int(test[np.arange(len(test)),pred.argmax(1)].sum())
    summary=dict(dim=len(names),variance=variance[:2].tolist(),dev_union=int((totals>0).sum()),dev_unique=int((totals==1).sum()),test_union=int((counts>0).sum()),test_unique=int((counts==1).sum()),commit=commit,fixed_development_best=CAPS[int(y.sum(0).argmax())],fixed_success=int(test[:,y.sum(0).argmax()].sum()),knn_singleton=int(test[counts==1,pred[counts==1].argmax(1)].sum()),rows=rows,neighbors=len(checks),parents=len({r['parent'] for r in checks}),retained=sum(r['retained'] for r in checks),sole=sum(r['still_sole'] for r in checks),near_union=sum(r['any_success'] for r in checks))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':
    main()
