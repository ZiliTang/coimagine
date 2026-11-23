"""Analyze one-commit success, future contrast and handoff outcomes at matched prefixes."""
import csv
import json
from pathlib import Path

import numpy as np
from tools import cap_first_choice_transfer as exp

r=exp.r


def read(panel,routes):
    pairs=[]
    for key in r.e.read(exp.OUT/"protocol.json")[panel]:
        rec={k:r.e.read(exp.OUT/panel/k/(key+".json")) for k in routes}
        assert all(row["start"]==rec["A"]["start"] for row in rec.values())
        if rec["A"]["start"] is None:
            assert len({row["history"] for row in rec.values()})==1
        for name in routes:
            if len(name)>1:
                base=rec[name[0]]
                for k in ("0","1","5"):
                    assert rec[name]["frames"].get(k)==base["frames"].get(k)
                if "5" not in base["frames"]:
                    assert base["history"]==rec[name]["history"]
        pairs.append(dict(rec,key=key))
    return pairs


def node(row):
    f=row["frames"].get("5")
    return f["z"] if f is not None else 48+row["success"]


def fit(dev):
    rows=[q for q in dev if q["A"]["start"] is not None]
    x=np.array([q["A"]["proposals"]["x"] for q in rows])
    y=np.array([q["B"]["success"]-q["A"]["success"] for q in rows])
    mean=x.mean(axis=0)
    scale=x.std(axis=0)
    scale[scale<1e-7]=1.
    x=(x-mean)/scale
    w=np.linalg.solve(x.T@x+len(x)*np.eye(x.shape[1]),x.T@(y-y.mean()))
    edges=np.zeros((48,2,50),int)
    for q in rows:
        z=q["A"]["start"]["z"]
        for i,k in enumerate(("A","B")):
            edges[z,i,node(q[k])]+=1
    global_counts=edges.sum(axis=(0,1))+1
    baseline=global_counts/global_counts.sum()
    shared=(edges.sum(axis=1)+8*baseline)/(edges.sum(axis=(1,2))[:,None]+8)
    specific=(edges+8*shared[:,None,:])/(edges.sum(axis=2)[:,:,None]+8)
    candidates=[]
    for z,ns in enumerate(edges):
        n=int(ns[0].sum())
        if n<12:continue
        for t in range(48):
            if t==z:continue
            delta=int(ns[1,t]-ns[0,t])
            if delta:candidates.append((abs(delta)/n,n,-z,-t,delta))
    _,_,nz,nt,delta=max(candidates)
    edge=dict(z=-nz,t=-nt,arm="B" if delta>0 else "A",dev=edges[-nz,:,-nt].tolist(),n=int(edges[-nz,0].sum()))
    return mean,scale,w,y.mean(),shared,specific,edge


def paired(vals):
    vals=np.asarray(vals,dtype=float)
    if not len(vals):return dict(n=0)
    rng=np.random.default_rng(233421)
    reps=vals[rng.integers(len(vals),size=(5000,len(vals)))].mean(axis=1)
    return dict(n=len(vals),mean=float(vals.mean()),ci=np.quantile(reps,[.025,.975]).tolist())


def auc(y,score):
    y=np.array(y);s=np.array(score)
    pos=s[y==1];neg=s[y==-1]
    if not len(pos) or not len(neg):return dict(auc=None)
    dif=pos[:,None]-neg
    matrix=(dif>0)+.5*(dif==0)
    rng=np.random.default_rng(2251)
    # Resample rooms separately within the two unique-winner classes.
    u=rng.multinomial(len(pos),np.full(len(pos),1/len(pos)),5000)/len(pos)
    v=rng.multinomial(len(neg),np.full(len(neg),1/len(neg)),5000)/len(neg)
    return dict(auc=float(matrix.mean()),ci=np.quantile(((u@matrix)*v).sum(axis=1),[.025,.975]).tolist())


def analyze():
    dev=read("dev",("A","B"))
    mean,scale,w,bias,shared,specific,edge=fit(dev)
    held=read("held",("A","B","AB","BA"))
    scores={m:[] for m in ("17","29","43","ridge")}
    results={k:[] for k in ("A","B","AB","BA")}
    local=[]
    brier=[]
    later={h:[] for h in exp.WINDOWS[1:]}
    source=edge["z"];dest=edge["t"];preferred=edge["arm"]
    edgeout=[]
    indices=[]
    n_different=n_both_live=0
    live_diff=0
    for i,q in enumerate(held):
        for k in results:results[k].append(q[k]["success"])
        a,b=q["A"],q["B"]
        if not a["start"]:continue
        indices.append(i)
        p=a["proposals"]
        for m in scores:
            scores[m].append(float(((np.array(p["x"])-mean)/scale)@w+bias) if m=="ridge" else p[m])
        z=a["start"]["z"]
        ns=[node(a),node(b)]
        n_different+=ns[0]!=ns[1]
        if max(ns)<48:
            n_both_live+=1
            live_diff+=ns[0]!=ns[1]
        err=[]
        for arm,t in enumerate(ns):
            truth=np.eye(50)[t]
            err.append(float(np.square(shared[z]-truth).sum()-np.square(specific[z,arm]-truth).sum()))
        brier.append(np.mean(err))
        if z==source:
            edgeout.append([int(n==dest) for n in ns])
            if node(q[preferred])==dest:
                other="B" if preferred=="A" else "A"
                local.append([q[preferred]["success"],q[preferred+other]["success"]])
        for h in later:
            fa=a["frames"].get(str(h));fb=b["frames"].get(str(h))
            if fa and fb and min(fa["gap"],fb["gap"])>=0 and a["success"]!=b["success"]:
                later[h].append((b["success"]-a["success"],fa["gap"]-fb["gap"],scores["ridge"][-1]))
    if (exp.OUT/"edge_lock.json").is_file():
        assert edge==r.e.read(exp.OUT/"edge_lock.json")
    av,bv,ab,ba=[np.array(v) for v in results.values()]
    y=(bv-av)[indices]
    predictors={}
    for m,s in scores.items():
        decisions=np.array(s)>0
        picked=bv.copy()
        picked[indices]=np.where(decisions,bv[indices],av[indices])
        predictors[m]=dict(auc=auc(y,s),correct=int(((y!=0)&((y>0)==decisions)).sum()),
                           success=int(picked.sum()),vsB=paired(picked-bv))
        if m!="ridge":
            full=np.array([r.e.read(exp.current.OUT/"fresh"/f"s{m}_no_replay"/(q["key"]+".json"))["success"] for q in held])
            predictors[m].update(full_success=int(full.sum()),full_gained=int(((full==1)&(picked==0)).sum()),
                                 full_lost=int(((full==0)&(picked==1)).sum()),full_vs_commit=paired(full-picked))
    late=[]
    for q in held:
        a,b=q["A"],q["B"]
        if a["success"]==b["success"]:continue
        fa,fb=a["frames"],b["frames"]
        if all(h in f and f[h]["gap"]>=0 for h in ("10","30") for f in (fa,fb)):
            late.append((b["success"]-a["success"],fa["10"]["gap"]-fb["10"]["gap"],fa["30"]["gap"]-fb["30"]["gap"]))
    recorded={q["key"]:{} for q in held}
    for j,i in enumerate(indices):
        recorded[held[i]["key"]]={m:ss[j] for m,ss in scores.items()}
    exp.DOC.mkdir(exist_ok=True,parents=True)
    with (exp.DOC/"rooms.csv").open("w") as f:
        wr=csv.writer(f)
        wr.writerow(["room","A","B","AB","BA","start_node","A_node5","B_node5","first_tick",*scores])
        for q in held:
            start=q["A"]["start"]
            wr.writerow([q["key"],*[q[k]["success"] for k in results],start["z"] if start else None,
                         node(q["A"]) if start else None,node(q["B"]) if start else None,
                         start["tick"] if start else None,*[recorded[q["key"]].get(m) for m in scores]])
    stats=dict(dev=len(dev),held=len(held),diverged=len(indices),one_winner=int((y!=0).sum()),
               fixed={k:int(np.sum(v)) for k,v in results.items()},predictors=predictors,
               future={h:dict(n=len(q),future=auc([t[0] for t in q],[t[1] for t in q]),
                              initial=auc([t[0] for t in q],[t[2] for t in q])) for h,q in later.items()},
               matched_late=dict(n=len(late),early=auc([t[0] for t in late],[t[1] for t in late]),late=auc([t[0] for t in late],[t[2] for t in late])),
               different_states=n_different,both_live=n_both_live,live_different=live_diff,
               successor_brier_improvement=paired(brier),edge=edge,edge_held=len(edgeout),
               edge_outcomes=np.sum(np.array(edgeout).reshape(-1,2),axis=0).tolist(),
               edge_paired=paired([(v[1]-v[0])*(1 if preferred=="B" else -1) for v in edgeout]),
               handoff_on_edge=local,
               union=int((av|bv).sum()), reachable=int((av|bv|ab|ba).sum()),
               bothfail=int(((av|bv)==0).sum()),
               rescued_ab=int(((av+bv==0)&(ab==1)).sum()),rescued_ba=int(((av+bv==0)&(ba==1)).sum()),
               ab_gain=int(((av==0)&(ab==1)).sum()),ab_loss=int(((av==1)&(ab==0)).sum()),
               ba_gain=int(((bv==0)&(ba==1)).sum()),ba_loss=int(((bv==1)&(ba==0)).sum()))
    r.e.save(exp.DOC/"results.json",stats)
    return stats


if __name__=="__main__":
    print(json.dumps(analyze(),ensure_ascii=False,indent=2))
