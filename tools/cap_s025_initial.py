"""Binary prediction of frozen s025 success from its initial public observation."""
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch
from torch import nn

from tools import cap_actor_inputs as previous
from tools import cap_regional_five as geo
from experiments.planar_tasks.semantic_rl.relational_transfer_features import RelativeGeometry

run = previous.run
read, save = run.ref.read, run.ref.save
ROOT = run.campaign.OUT
OUT = ROOT/'s025_initial_binary_20261122'
DOC = previous.DOC.parent/'2026-11-22_s025_initial_binary'
CHECKS = (1, 5, 10, 20, 40, 80, 160, 320)
ARMS = ('compact_linear','compact_mlp','scan_mlp','scan_ln','scan_natural')
SEED = 17


def inventory():
    scale = run.ref.scale
    old, held, members = scale.inputs()
    assert members[0]=='s025'
    natural, variants = (read(scale.OUT/(k+'.json')) for k in ('natural','variants'))
    parents = read(ROOT/'extra_parents.json') | read(scale.OUT/'parents.json')
    assert len(parents)==412
    archive = run.ref.sets.unzip(scale.OUT/'new_outcomes.json.gz')
    current = read(previous.prior.OUT/'panel.json')
    assert run.campaign.base.digest(current)==read(previous.prior.OUT/'protocol.json')['geometry']
    audit = {r['room']:r for r in read(previous.OUT/'priority_audit.json') if r['arm']=='priority'}
    groups = {'development':old|natural|variants,'original':held,'current':current}
    records, fixtures, seen = [], [], set()
    for group, rooms in groups.items():
        for room, fixture in sorted(rooms.items()):
            digest = run.campaign.input_hash(fixture)
            assert digest not in seen
            seen.add(digest)
            if group=='current':
                label = read(previous.OUT/'evaluation/priority'/(room+'.json'))
                a = audit[room]
                assert a['rank_counts']==[label['decisions'],0,0,0,0] and a['fallback']==0
                assert label['room']==room and (label['success'] or label['ticks']==1800 or label['decisions']==300)
            else:
                label = archive[room]['s025'] if room in archive else read(ROOT/'runs/s025'/(room+'.json'))
                assert label['id']==room and label['complete'] and not label['error']
                assert (label['success'] and label['termination']=='success') or (not label['success'] and
                        (label['termination']=='task_budget_exhausted' or label['ticks']==1800 or label['decisions']==300))
            records.append(dict(room=room,panel=group,parent=parents.get(room,room),variant=room in parents,
                                objects=1+len(fixture['movable_objects']),content_sha256=digest,y=int(label['success'])))
            fixtures.append(fixture)
    assert len(records)==1200 and set(parents.values())<=set(groups['development'])-set(parents)
    return records,fixtures


def initial(fixture):
    with run.runtime.world(fixture) as (env,_):
        obs = run.ref.dq.current_event(env.history)['observation']
        assert obs['tick']==obs['decision']==env.decision_count==0
        g = RelativeGeometry(1)
        g.advance(env.history)
        keep = previous.gkeep[previous.gkeep<previous.FRAME]
        return geo.geom(obs).astype('float32'),g.initial[keep].astype('float32')


def folds(records):
    rng=np.random.default_rng(SEED)
    assignment={}
    for n in range(2,6):
        parents=sorted(r['room'] for r in records if r['panel']=='development' and not r['variant'] and r['objects']==n)
        for j,k in enumerate(rng.permutation(parents)): assignment[k]=j%5
    for r in records:
        r['fold']=assignment[r['parent']] if r['panel']=='development' else -1
    assert len(assignment)==388


def transform(x,train):
    valid=np.isfinite(x)
    safe=np.where(valid,x,0).astype('float64')
    count=np.maximum(1,valid[train].sum(0))
    mean=safe[train].sum(0)/count
    sd=np.sqrt(np.maximum(0.,np.square(safe[train]).sum(0)/count-mean**2))
    z=np.where(valid,(safe-mean)/np.maximum(.1,sd),0).clip(-10,10)
    v=np.concatenate((z,valid),axis=1).astype('float32')
    # Remove constants using fitting rows only, including constant validity flags.
    kept=np.ptp(v[train],axis=0)>1e-7
    return torch.from_numpy(v[:,kept].copy()),dict(mean=torch.from_numpy(mean),scale=torch.from_numpy(np.maximum(.1,sd)),kept=torch.from_numpy(kept))


def auc(y,s):
    unique, inv, counts=np.unique(s,return_inverse=True,return_counts=True)
    mid=np.cumsum(counts)-(counts-1)/2
    pos=y.astype(bool)
    n=pos.sum()
    return float((mid[inv[pos]].sum()-n*(n+1)/2)/(n*(len(y)-n)))


def metrics(y,p,t):
    pred=p>=t
    pos=y==1
    tp=int(np.sum(pred & pos)); tn=int(np.sum(~pred & ~pos))
    return dict(n=len(y),wins=int(pos.sum()),auc=auc(y,p),brier=float(np.mean((y-p)**2)),
                accuracy=float((tp+tn)/len(y)),ba=float((tp/pos.sum()+tn/(~pos).sum())/2),
                tp=tp,tn=tn,fp=int((~pos).sum()-tn),fn=int(pos.sum()-tp))


def train_curve(arm,x,y,fit,ev,export=None):
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    z,normalization=transform(x,fit)
    if arm=='compact_linear':
        model=nn.Linear(z.shape[1],1)
    else:
        model=nn.Sequential(nn.Linear(z.shape[1],64),nn.LayerNorm(64,elementwise_affine=False) if arm=='scan_ln' else nn.Identity(),
                            nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    labels=torch.tensor(y,dtype=torch.float32)
    series={}
    for epoch in range(1,CHECKS[-1]+1):
        for batch in np.array_split(rng.permutation(fit),int(np.ceil(len(fit)/64))):
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(z[batch]).flatten(),labels[batch])
            loss.backward()
            opt.step()
        if epoch in CHECKS:
            with torch.inference_mode():
                series[epoch]=model(z[ev]).flatten().sigmoid().numpy().astype(float)
            if export and epoch==export['epoch']:
                torch.save(dict(**export,arm=arm,normalization=normalization,weights=model.state_dict()),OUT/'selected.pt')
                break
    return series


def fit_arm(job):
    arm,compact,scan,records=job
    torch.set_num_threads(1)
    x=compact if arm.startswith('compact') else scan
    y=np.array([r['y'] for r in records])
    fold=np.array([r['fold'] for r in records])
    natural=np.array([not r['variant'] for r in records])
    train=fold>=0
    fitting=train & (natural if arm=='scan_natural' else True)
    # Validate on independent natural parents, matching the two evaluation panels.
    val=train & natural
    oof={e:np.full(len(y),np.nan) for e in CHECKS}
    for i in range(5):
        ix=np.flatnonzero(fitting & (fold!=i)); ev=np.flatnonzero(val & (fold==i))
        for e,p in train_curve(arm,x,y,ix,ev).items(): oof[e][ev]=p
    scores={e:float(np.mean((oof[e][val]-y[val])**2)) for e in CHECKS}
    best=min(CHECKS,key=lambda e:(scores[e],e))
    thresholds=np.linspace(.05,.95,181)
    t=max(thresholds,key=lambda v:(metrics(y[val],oof[best][val],v)['ba'],-abs(v-.5)))
    full=train_curve(arm,x,y,np.flatnonzero(fitting),np.arange(len(y)))
    p=full[best]
    panel=np.array([r['panel'] for r in records])
    subsets=dict(fit=fitting,natural_dev=val,original=panel=='original',current=panel=='current',held=~train)
    report=dict(epoch=int(best),threshold=float(t),oof_brier=scores,
                family_oof=metrics(y[val],oof[best][val],t),
                selected={k:metrics(y[m],p[m],t) for k,m in subsets.items()},
                late={k:metrics(y[m],full[CHECKS[-1]][m],t) for k,m in subsets.items()},
                curve=[dict(epoch=e,oof_brier=scores[e],fit_brier=float(np.mean((full[e][fitting]-y[fitting])**2)),
                            fit_auc=auc(y[fitting],full[e][fitting])) for e in CHECKS])
    return arm,report,p,full[CHECKS[-1]]


def main():
    OUT.mkdir(exist_ok=True)
    DOC.mkdir(exist_ok=True,parents=True)
    records,fixtures=inventory()
    folds(records)
    protocol=dict(arms=list(ARMS),checks=list(CHECKS),seed=SEED,selection='minimum five-fold natural-parent OOF Brier; earliest tie',
                  threshold='maximum OOF balanced accuracy, prefer closest to 0.5',target='s025 success under frozen full episode budget',
                  source_hash=run.campaign.sha(run.campaign.CAM/'semantics/s025.py'),inputs='initial public observation only',
                  manifests={g:run.campaign.base.digest([r for r in records if r['panel']==g])
                             for g in ('development','original','current')})
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json')==protocol
    save(OUT/'protocol.json',protocol)
    with ProcessPoolExecutor(8,mp_context=mp.get_context('fork')) as pool: raw=list(pool.map(initial,fixtures,chunksize=8))
    compact,scan=(np.stack(v) for v in zip(*raw))
    report={}; predictions={}
    with ProcessPoolExecutor(5,mp_context=mp.get_context('fork')) as pool:
        for f in as_completed([pool.submit(fit_arm,(a,compact,scan,records)) for a in ARMS]):
            arm,r,p,late=f.result()
            report[arm]=r; predictions[arm]=p; predictions[arm+'_late']=late
            save(OUT/(arm+'.json'),r)
    champion=min(ARMS,key=lambda a:report[a]['family_oof']['brier'])
    y=np.array([r['y'] for r in records])
    dev=np.array([r['panel']=='development' for r in records])
    natural=dev & np.array([not r['variant'] for r in records])
    chosen=report[champion]
    torch.set_num_threads(1)
    check=train_curve(champion,compact if champion.startswith('compact') else scan,y,
                      np.flatnonzero(natural if champion=='scan_natural' else dev),np.arange(len(y)),
                      export=dict(epoch=chosen['epoch'],threshold=chosen['threshold']))
    np.testing.assert_allclose(check[chosen['epoch']],predictions[champion],atol=1e-6,rtol=0)
    prior=float(y[natural].mean())
    base={}
    for name in ('development','original','current','held'):
        m=~dev if name=='held' else np.array([r['panel']==name for r in records])
        base[name]=metrics(y[m],np.full(sum(m),prior),.5)
    # Label-based room bootstrap within panel and obstacle-count strata (room is the unit).
    rng=np.random.default_rng(SEED)
    strata=[np.array([i for i,r in enumerate(records) if r['panel']==panel and r['objects']==n])
            for panel in ('original','current') for n in range(2,6)]
    draws=np.concatenate([rng.choice(v,size=(4000,len(v))) for v in strata],axis=1)
    cis={}
    for arm in ARMS:
        p=predictions[arm]
        aa=[auc(y[idx],p[idx]) for idx in draws]
        diff=((p[draws]-y[draws])**2-(prior-y[draws])**2).mean(1)
        cis[arm]=dict(auc=np.quantile(aa,[.025,.975]).tolist(),brier_minus_prior=np.quantile(diff,[.025,.975]).tolist())
    labels={r['room']:r['y'] for r in records if r['panel']=='development'}
    variants=[r for r in records if r['variant']]
    transition=Counter((labels[r['parent']],r['y']) for r in variants)
    parents={r['parent'] for r in variants}
    mixed=sum(len({labels[k]}|{r['y'] for r in variants if r['parent']==k})>1 for k in parents)
    for i,row in enumerate(records):
        row.update({k:float(v[i]) for k,v in predictions.items()})
    with (DOC/'rooms.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    quantiles=[]
    for panel in ('original','current'):
        ranked=sorted([r for r in records if r['panel']==panel],key=lambda r:(r[champion],r['room']))
        quantiles.append([sum(r['y'] for r in ranked[i:i+40]) for i in range(0,200,40)])
    save(OUT/'FINISHED.json',dict(champion=champion,arms=report,ci=cis,prior=prior,baseline=base,
                variant_counts={str(k):v for k,v in sorted(transition.items())},parents=len(parents),mixed_parents=mixed,
                quintile_wins=quantiles,raw_widths=[compact.shape[1],scan.shape[1]]))


if __name__=='__main__': main()
