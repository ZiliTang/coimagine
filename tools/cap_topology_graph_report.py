"""Fit graph representations on development rooms and evaluate paired transfers."""
import csv
import itertools
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from tools import cap_topology_graph as g

K=6
METHODS=('physical','supervised','public')
NAMES=g.LABELS
read,save=g.study.read,g.study.save


def load(fold):
    paths=sorted((g.OUT/'runs'/fold).glob('*.json'))
    assert len(paths)==200
    return [read(p) for p in paths]


def matrix(points,name):
    return np.asarray([s['x'][:29] if name=='x' else s[name] for s in points],float)


def nearest(x,centres):
    return ((x[:,None]-centres)**2).sum(2).argmin(1)


def kmeans(x):
    best=np.inf
    rng=np.random.default_rng(447)
    for _ in range(20):
        c=[x[rng.integers(len(x))]]
        for _ in range(1,K):
            dist=((x[:,None]-c)**2).sum(2).min(1)
            c.append(x[rng.choice(len(x),p=dist/dist.sum())])
        c=np.asarray(c)
        for _ in range(100):
            groups=nearest(x,c)
            nextc=np.stack([x[groups==i].mean(0) if (groups==i).any() else c[i] for i in range(K)])
            if np.allclose(nextc,c):break
            c=nextc
        loss=((x-c[nearest(x,c)])**2).sum()
        if loss<best:
            best=loss
            saved=c.copy()
    return saved


def auc(y,p):
    score=p[y==1,None]-p[y==0]
    return float(((score>0)+.5*(score==0)).mean()) if score.size else None


def train(dev):
    x,y=matrix(dev,'x'),matrix(dev,'y')
    mu,scale=x.mean(0),x.std(0).clip(.05)
    x=np.clip((x-mu)/scale,-5,5)
    pc,tc=kmeans(x),kmeans(y)
    # Deterministic display order from little contact to frequent contact.
    tc=tc[np.lexsort((tc[:,6],tc[:,1],tc[:,3]))]
    torch.manual_seed(83)
    torch.set_num_threads(1)
    model=nn.Sequential(nn.Linear(x.shape[1],48),nn.ReLU(),nn.Linear(48,32),nn.ReLU(),nn.Linear(32,len(NAMES)))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    inputs,labels=(torch.as_tensor(a,dtype=torch.float32) for a in (x,y))
    rng=np.random.default_rng(83)
    for _ in range(1500):
        idx=rng.integers(len(x),size=128)
        loss=nn.functional.binary_cross_entropy_with_logits(model(inputs[idx]),labels[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    def predict(points):
        z=np.clip((matrix(points,'x')-mu)/scale,-5,5)
        with torch.no_grad():
            prob=model(torch.tensor(z,dtype=torch.float32)).sigmoid().numpy()
        return prob,nearest(prob,tc),nearest(z,pc)
    torch.save(dict(state=model.state_dict(),mu=mu.tolist(),scale=scale.tolist(),public_centres=pc.tolist(),
                    teacher_centres=tc.tolist()),g.OUT/'frozen_readout.pt')
    return tc,predict


def extract(rooms):
    points={}
    for r in rooms:
        for rec in r['records']:
            for h,s in rec['points'].items():
                # History hashes can repeat across deterministic branches.
                key=(r['room'],h)
                if key in points:
                    assert points[key]['x'][:29]==s['x'][:29] and points[key]['y']==s['y']
                points[key]=s
    return points


def ci(pairs):
    if not pairs:return None
    by_room=defaultdict(list)
    for room,d in pairs:by_room[room].append(d)
    v=np.array([(sum(ds),len(ds)) for ds in by_room.values()])
    idx=np.random.default_rng(921).integers(len(v),size=(3000,len(v)))
    boot=v[idx].sum(1)
    return dict(n=len(pairs),rooms=len(v),mean=float(v[:,0].sum()/v[:,1].sum()),
                ci=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist())


def edge_records(rooms):
    edges=[]
    for room in rooms:
        key=room['room']
        seen={}
        for rec in room['records']:
            rr=rec['branches']
            for n,origin,a,b in [(0,'root','AA','BA'),(1,'A','AA','AB'),(1,'B','BA','BB')]:
                if len(rr[a]['states'])<=n:continue
                h=rr[a]['states'][n]
                assert rr[b]['states'][n]==h
                def dest(v):
                    return (key,v['states'][n+1]) if len(v['states'])>n+1 else K+int(not v['success'])
                pair=[dest(rr[a]),dest(rr[b])]
                if h in seen:
                    assert seen[h]==pair
                    continue
                seen[h]=pair
                edges.append(dict(room=key,source=(key,h),dest=pair,origin=origin,depth=rec['depth']))
    return edges


def node(assign,key):
    return key if isinstance(key,int) else assign[key]


def counts(edges,assign,dest_assign):
    arr=np.zeros((K,2,K+2),int)
    for r in edges:
        for action,dest in enumerate(r['dest']):
            arr[assign[r['source']],action,node(dest_assign,dest)]+=1
    return arr


def pick_edges(dev,da):
    selected=[]
    for i in range(K):
        rows=[r for r in dev if da[r['source']]==i]
        if len({r['room'] for r in rows})<20:continue
        vals=np.array([[node(da,d) for d in r['dest']] for r in rows])
        diff=(vals[:,1,None]==np.arange(K+2)).mean(0)-(vals[:,0,None]==np.arange(K+2)).mean(0)
        j=int(np.argmax(abs(diff)))
        if abs(diff[j])>=.10:
            selected.append((i,j))
    return selected


def transfer_scores(dev,test,labels):
    # All representations predict the SAME physical destination incl. two terminal nodes.
    ref=counts(dev,labels['dev']['physical'],labels['dev']['physical']).sum(axis=0)
    marginal=(ref+1)/(ref.sum(1,keepdims=True)+K+2)
    metrics={}
    scores={}
    for m in METHODS:
        arrays=counts(dev,labels['dev'][m],labels['dev']['physical'])
        pred=(arrays+10*marginal)/(arrays.sum(2,keepdims=True)+10)
        pooled=arrays.sum(1)
        pooled=(pooled+10*marginal.mean(0))/(pooled.sum(1,keepdims=True)+10)
        loss=[]
        for e in test:
            i=labels['fresh'][m][e['source']]
            y=np.eye(K+2)[[node(labels['fresh']['physical'],d) for d in e['dest']]]
            brier=lambda p: ((p-y)**2).sum(1).mean()
            loss.append([e['room'],brier(pred[i]),brier(pooled[i]),brier(marginal)])
        scores[m]=[(v[0],v[1]) for v in loss]
        metrics[m]=dict(brier=ci([(r,p) for r,p,_,_ in loss]),gain_over_node_only=ci([(r,base-p) for r,p,base,_ in loss]),
                        gain_over_cap_only=ci([(r,base-p) for r,p,_,base in loss]))
    for m in METHODS:
        metrics[m]['gain_over_public_partition']=ci([(r,base-p) for (r,p),(_,base) in zip(scores[m],scores['public'])])
    return metrics


def coarsening(rooms,assign):
    groups={'same':[],'different':[]}
    for r in rooms:
        for rec in r['records']:
            rr=rec['branches']
            for a,b in [('AA','AB'),('BA','BB')]:
                va,vb=rr[a],rr[b]
                if len(va['states'])<3 or len(vb['states'])<3:continue
                key=r['room']
                same=assign[(key,va['states'][2])]==assign[(key,vb['states'][2])]
                groups['same' if same else 'different'].append((key,int(va['success']!=vb['success'])))
    return {k:dict(n=len(v),discord=sum(d for _,d in v),effect=ci(v)) for k,v in groups.items()}


def switch_rows(rooms,assign):
    rows=[]
    for r in rooms:
        key=r['room']
        for rec in r['records']:
            a,b=[rec['branches'][route] for route in ('AA','AB')]
            state=a['states']
            i=assign[(key,state[1])] if len(state)>1 else -1
            rows.append(dict(room=key,node=i,a=a['success'],b=b['success'],depth=rec['depth']))
    return rows


def rule_stats(dev,fresh):
    selected=[]
    details=[]
    for n in range(K):
        u=[r for r in dev if r['node']==n]
        v=[r for r in fresh if r['node']==n]
        diff=lambda rows: ci([(r['room'],r['b']-r['a']) for r in rows])
        d,t=diff(u),diff(v)
        if d and d['rooms']>=20 and d['mean']>0:
            selected.append(n)
        details.append(dict(node=n,dev=d,fresh=t,fresh_gain=sum(r['b']>r['a'] for r in v),
                            fresh_loss=sum(r['b']<r['a'] for r in v)))
    y=[r['b'] if r['node'] in selected else r['a'] for r in fresh]
    return dict(nodes=selected,per_node=details,n=len(fresh),eligible=sum(r['node']>=0 for r in fresh),
                switched=sum(r['node'] in selected for r in fresh),AA=sum(r['a'] for r in fresh),
                AB=sum(r['b'] for r in fresh),selected=sum(y),
                hindsight_node=sum(r['a'] for r in fresh)+sum(max(0,d['fresh_gain']-d['fresh_loss']) for d in details),
                hindsight_pair=sum(max(r['a'],r['b']) for r in fresh),gain=sum(z>r['a'] for z,r in zip(y,fresh)),
                loss=sum(z<r['a'] for z,r in zip(y,fresh)),effect=ci([(r['room'],z-r['a']) for z,r in zip(y,fresh)]),
                by_depth={str(d):[sum(r['a'] for r in fresh if r['depth']==d),sum(z for z,r in zip(y,fresh) if r['depth']==d)] for d in g.PREFIX})


def composition(rooms,assign):
    rows=[]
    for r in rooms:
        key=r['room']
        for rec in r['records']:
            rr=rec['branches']
            root=rr['AA']['states'][:1]
            if not root:continue
            def arrival(v):
                return assign[(key,v['states'][1])] if len(v['states'])>1 else K+int(not v['success'])
            def local(v):
                if len(v['states'])<=1:return 0
                states=rec['points']
                gap=states[v['states'][2]]['gap'] if len(v['states'])>2 else v['gap']
                return int(states[v['states'][1]]['gap']-gap>=.02)
            rows.append(dict(room=key,start=assign[(key,root[0])],midA=arrival(rr['AA']),midB=arrival(rr['BA']),
                             **{k:v['success'] for k,v in rr.items()},local={k:local(v) for k,v in rr.items()}))
    return rows


def path_checks(dev,fresh):
    def path(data,i,j):
        root=[r for r in data if r['start']==i]
        made=[r for r in root if r['midA']==j]
        return dict(arrival=ci([(r['room'],int(r['midA']==j)-int(r['midB']==j)) for r in root]),
                    end=ci([(r['room'],r['AB']-r['AA']) for r in made]),
                    local=ci([(r['room'],r['local']['AB']-r['local']['AA']) for r in made]),
                    gain=sum(r['AB']>r['AA'] for r in made),loss=sum(r['AB']<r['AA'] for r in made))
    qualified=[]
    for i,j in itertools.permutations(range(K),2):
        d=path(dev,i,j)
        if d['end'] and d['end']['rooms']>=12 and d['arrival']['mean']>=.05 and d['end']['mean']>=.05:
            qualified.append((d['arrival']['mean']*d['end']['mean'],i,j,d))
    qualified.sort(key=lambda item:(-item[0],item[1],item[2]))
    tests=[dict(source=i,mid=j,dev=d,fresh=path(fresh,i,j)) for _,i,j,d in qualified]
    origins=[]
    for i in range(K):
        v={}
        for route in 'AB':
            keep=[r for r in fresh if r['mid'+route]==i]
            v[route]=dict(end=ci([(r['room'],r[route+'B']-r[route+'A']) for r in keep]),
                          local=ci([(r['room'],r['local'][route+'B']-r['local'][route+'A']) for r in keep]))
        origins.append(dict(node=i,origins=v))
    return dict(paths=tests,by_preparer=origins)


def write_csv(name,rows):
    with (g.DOC/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def freeze_dev():
    import hashlib
    rooms=load('dev')
    mapping=extract(rooms)
    pts=list(mapping.values())
    centres,predict=train(pts)
    _,student,public=predict(pts)
    truth=nearest(matrix(pts,'y'),centres)
    edges=edge_records(rooms)
    lock={}
    for m,c in zip(METHODS,(truth,student,public)):
        assign=dict(zip(mapping,map(int,c)))
        switch=switch_rows(rooms,assign)
        comp=composition(rooms,assign)
        lock[m]=dict(edges=pick_edges(edges,assign),switch_nodes=rule_stats(switch,[])['nodes'],
                     paths=[[v['source'],v['mid']] for v in path_checks(comp,[])['paths']])
    sha=hashlib.sha256((g.OUT/'frozen_readout.pt').read_bytes()).hexdigest()
    save(g.OUT/'dev_freeze.json',dict(model_sha256=sha,selected=lock,training_rooms=len(rooms),unique_points=len(pts)))


def frozen_model():
    import hashlib
    path=g.OUT/'frozen_readout.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==read(g.OUT/'dev_freeze.json')['model_sha256']
    model_data=torch.load(path,map_location='cpu',weights_only=True)
    centres,pc,mu,scale=(np.asarray(model_data[k]) for k in ('teacher_centres','public_centres','mu','scale'))
    model=nn.Sequential(nn.Linear(len(mu),48),nn.ReLU(),nn.Linear(48,32),nn.ReLU(),nn.Linear(32,len(NAMES)))
    model.load_state_dict(model_data['state'])
    model.eval()
    def predict(points):
        z=np.clip((matrix(points,'x')-mu)/scale,-5,5)
        with torch.no_grad():
            prob=model(torch.tensor(z,dtype=torch.float32)).sigmoid().numpy()
        return prob,nearest(prob,centres),nearest(z,pc)
    return centres,predict


def main():
    g.DOC.mkdir(parents=True,exist_ok=True)
    panels={k:load(k) for k in ('dev','fresh')}
    points={k:extract(v) for k,v in panels.items()}
    centres,predict=frozen_model()
    labels={}
    readouts={}
    cluster_profiles={}
    for fold,mapping in points.items():
        pts=list(mapping.values())
        y=matrix(pts,'y')
        prob,student,public=predict(pts)
        truth=nearest(y,centres)
        labels[fold]={m:dict(zip(mapping,map(int,c))) for m,c in zip(METHODS,(truth,student,public))}
        confusion=np.zeros((K,K),int)
        np.add.at(confusion,(truth,student),1)
        readouts[fold]=dict(n=len(pts),acc=float((truth==student).mean()),majority=float(np.bincount(truth,minlength=K).max()/len(truth)),
                            balanced=float((np.diag(confusion)/confusion.sum(1).clip(1))[confusion.sum(1)>0].mean()),
                            confusion=confusion.tolist(),relations={k:dict(positive=int(y[:,i].sum()),auc=auc(y[:,i],prob[:,i]),
                            brier=float(np.mean((y[:,i]-prob[:,i])**2))) for i,k in enumerate(NAMES)})
        cluster_profiles[fold]=[dict(node=n,n=int(sum(truth==n)),bits=np.round(y[truth==n].mean(0),3).tolist()) for n in range(K)]
    edges={k:edge_records(rows) for k,rows in panels.items()}
    summary=dict(labels=NAMES,profiles=cluster_profiles,readouts=readouts,transfers={},handoffs={},
                 prediction=transfer_scores(edges['dev'],edges['fresh'],labels),centres=centres.tolist())
    frequencies=[]
    pairs=[]
    for m in METHODS:
        selected=pick_edges(edges['dev'],labels['dev'][m])
        selected_stats=[]
        for i,j in selected:
            item=dict(source=i,destination=j)
            for fold,rows in edges.items():
                arr=labels[fold][m]
                keep=[r for r in rows if arr[r['source']]==i]
                item[fold]=ci([(r['room'],int(node(arr,r['dest'][1])==j)-int(node(arr,r['dest'][0])==j)) for r in keep])
            selected_stats.append(item)
        summary['transfers'][m]=selected_stats
        for fold,rows in edges.items():
            arr=labels[fold][m]
            cube=counts(rows,arr,arr)
            for i in range(K):
                keep=[r for r in rows if arr[r['source']]==i]
                for j in range(K+2):
                    frequencies.append(dict(method=m,panel=fold,source=i,dest=j,n=len(keep),rooms=len({r['room'] for r in keep}),
                                    A=int(cube[i,0,j]),B=int(cube[i,1,j])))
            if m=='physical':
                for r in rows:
                    pairs.append(dict(panel=fold,room=r['room'],history=r['source'][1],origin=r['origin'],depth=r['depth'],
                                start=arr[r['source']],A=node(arr,r['dest'][0]),B=node(arr,r['dest'][1])))
        stats={fold:switch_rows(rows,labels[fold][m]) for fold,rows in panels.items()}
        summary['handoffs'][m]=rule_stats(stats['dev'],stats['fresh'])
        c={fold:composition(rows,labels[fold][m]) for fold,rows in panels.items()}
        summary.setdefault('composition',{})[m]=path_checks(c['dev'],c['fresh'])
    physical_check={}
    for fold,rows in edges.items():
        subj=labels[fold]['supervised']
        keep=[e for e in rows if subj[e['source']]==1]
        physical_check[fold]={}
        for key,assignment in [('pred',subj),('truth',labels[fold]['physical'])]:
            counts_by_branch=[[int(node(assignment,d)==5) for d in e['dest']] for e in keep]
            physical_check[fold][key]=dict(a=sum(v[0] for v in counts_by_branch),b=sum(v[1] for v in counts_by_branch),
                                            diff=ci([(e['room'],v[1]-v[0]) for e,v in zip(keep,counts_by_branch)]))
    save(g.DOC/'edge_physical_check.json',physical_check)
    summary['coarsening']={fold:coarsening(rows,labels[fold]['physical']) for fold,rows in panels.items()}
    summary['panels']={k:dict(rooms=len(rows),paired_transitions=len(edges[k]),points=len(points[k]),
                              valid_prefixes=sum(bool(rec['branches']['AA']['states']) for r in rows for rec in r['records']),
                              episodes_by_order={route:sum(rec['branches'][route]['success'] for r in rows for rec in r['records'])
                                                 for route in g.ROUTES})
                        for k,rows in panels.items()}
    lock=read(g.OUT/'dev_freeze.json')['selected']
    for m in METHODS:
        assert lock[m]['switch_nodes']==summary['handoffs'][m]['nodes']
        assert lock[m]['edges']==[[x['source'],x['destination']] for x in summary['transfers'][m]]
        assert lock[m]['paths']==[[x['source'],x['mid']] for x in summary['composition'][m]['paths']]
    write_csv('edges.csv',frequencies)
    write_csv('prefix_pairs.csv',pairs)
    hand=[]
    for fold,rows in panels.items():
        for r in rows:
            for rec in r['records']:
                rr=rec['branches']
                h=rr['AA']['states'][1] if len(rr['AA']['states'])>1 else None
                hand.append(dict(panel=fold,room=r['room'],parent='AB'[r['origin']],depth=rec['depth'],A_made_history=h,
                                 **{m:labels[fold][m].get((r['room'],h)) for m in METHODS},
                                 **{k:rr[k]['success'] for k in g.ROUTES}))
    write_csv('handoffs.csv',hand)
    save(g.DOC/'results.json',summary)
    from tools import cap_topology_graph_figures as figs
    figs.make(panels,labels,points,centres,frequencies)
    save(g.OUT/'FINISHED.json',summary['panels'])


if __name__=='__main__':
    import argparse
    args=argparse.ArgumentParser()
    args.add_argument('--freeze',action='store_true')
    if args.parse_args().freeze:freeze_dev()
    else:main()
