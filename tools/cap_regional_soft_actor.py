"""Matched time-weighted imitation on the original fastest-teacher prefixes."""
import gzip
import json

import numpy as np
import torch

from tools import cap_regional_fastest_actor as hard

study = hard.study
OUT = hard.OUT.parent/'cap_regional_soft_actor_20261120_01'
DOCS = hard.DOCS.parent/'2026-11-20_regional_soft_actor'
PANELS = {'dev':160,'aug':55,'fresh':200,'near':90}
EXTRA = 4500


def trace(sid,cid):
    with gzip.open(study.ext.refs(sid,cid)/(cid+'.json.gz'),'rt') as f:
        return json.load(f)


def labels(teachers,data,sids):
    a = hard.Store(hard.BASE/'arrays/train').arrays
    mass = np.zeros(data['valid'].shape,dtype=np.float64)
    contributors = np.zeros(len(mass),dtype=int)
    successful_runs = 0
    for cid,t in teachers.items():
        ix = np.flatnonzero(data['root']==t['root'])
        reference = trace(t['sid'],cid)
        assert len(ix)==len(reference)
        for j,sid in enumerate(sids):
            r = study.read(hard.BASE/'shards'/sid/(cid+'.json'))
            if not r['success']:
                continue
            successful_runs += 1
            other = trace(sid,cid)
            source = np.flatnonzero((a['root']==t['root']) & (a['cap']==j))
            source = source[np.argsort(a['boundary'][source])]
            assert len(source)==len(other) and r['ticks']==other[-1]['tick']
            elapsed = 0
            for idx,src,step,ref in zip(ix,source,other,reference):
                chosen = a['chosen'][src]
                # Require agreement of public features and the ordered candidate panel.
                for key in ('x','valid','sources','candidates'):
                    np.testing.assert_array_equal(a[key][src],data[key][idx])
                assert data['valid'][idx,chosen] and data['sources'][idx,chosen,j]
                remaining = r['ticks']-elapsed
                assert remaining>0
                mass[idx,chosen] += 1./remaining
                contributors[idx] += 1
                if step['a']!=ref['a']:
                    assert chosen!=data['chosen'][idx]
                    break
                assert chosen==data['chosen'][idx] and step['tick']==ref['tick']
                elapsed = step['tick']
    assert (contributors>=1).all()
    prob = (mass/mass.sum(1,keepdims=True)).astype(np.float32)
    np.testing.assert_allclose(prob.sum(1),1.,rtol=1e-6)
    assert not prob[~data['valid']].any()
    teacher_prob=prob[np.arange(len(prob)),data['chosen']]
    assert (teacher_prob>0).all()
    mixed=(prob>0).sum(1)>1
    start=data['boundary']==0
    changed=prob.max(1)>teacher_prob+1e-6
    room_fraction=[float(mixed[data['root']==r].mean()) for r in np.unique(data['root'])]
    stats=dict(rooms=len(teachers),successful_runs=successful_runs,boundaries=len(prob),
        informative=int((data['valid'].sum(1)>1).sum()),multiple_experts=int((contributors>1).sum()),
        mixed=int(mixed.sum()),rooms_mixed=int(np.unique(data['root'][mixed]).size),
        starts_mixed=int((mixed & start).sum()),soft_mode_beats_teacher=int(changed.sum()),
        sampler_mixed=float(np.mean(room_fraction)),mean_teacher_mass_mixed=float(teacher_prob[mixed].mean()))
    study.save(OUT/'label_stats.json',stats)
    np.savez_compressed(OUT/'targets.npz',prob=prob,root=data['root'],boundary=data['boundary'],
                        chosen=data['chosen'],contributors=contributors)
    print('LABELS',stats,flush=True)
    return prob


def cross_entropy(logits,valid,prob):
    logp=logits.masked_fill(~valid,-torch.inf).log_softmax(1).masked_fill(~valid,0.)
    return -(prob*logp).sum(1)


@torch.inference_mode()
def fit_metrics(model,data,prob):
    old_hits=[]
    mass=[]
    losses=[]
    for i in range(0,len(prob),128):
        sl=slice(i,i+128)
        x,c,s,v,y=hard.batch(data,sl)
        scores=model.values(x,c,s)
        pred=scores.masked_fill(~v,-torch.inf).argmax(1)
        p=torch.from_numpy(prob[sl])
        losses.extend(cross_entropy(scores,v,p).tolist())
        mass.extend(p.gather(1,pred[:,None]).flatten().tolist())
        old_hits.extend((pred==y).tolist())
    hit=np.asarray(old_hits)
    mixed=(prob>0).sum(1)>1
    top=np.isclose(prob.max(1),mass,atol=1e-6,rtol=0)
    return dict(ce=float(np.mean(losses)),hard_hits=int(hit.sum()),top_soft=int(top.sum()),
                mixed_top=int(top[mixed].sum()),mixed_hard=int(hit[mixed].sum()),
                perfect=int(sum(hit[data['root']==r].all() for r in np.unique(data['root']))))


def train(data,prob):
    torch.manual_seed(hard.SEED)
    model=hard.actor()
    buckets=[np.flatnonzero(data['root']==r) for r in np.unique(data['root'])]
    sizes=np.array([len(b) for b in buckets])
    offsets=np.cumsum(np.r_[0,sizes[:-1]])
    indices=np.concatenate(buckets)
    curve=[]
    for name,steps,lr in [('fixed',hard.STEPS,3e-4),('continued',EXTRA,3e-5)]:
        rng=np.random.default_rng(hard.SEED)
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        for step in range(steps+1):
            if step in hard.CHECKS or step==steps:
                line=dict(checkpoint=name,step=step,**fit_metrics(model,data,prob))
                curve.append(line)
                study.save(OUT/'fit.json',curve)
                print('FIT',line,flush=True)
            if step==steps:
                break
            roots=rng.integers(len(buckets),size=64)
            ix=indices[offsets[roots]+(rng.random(64)*sizes[roots]).astype(int)]
            x,c,s,v,_=hard.batch(data,ix)
            loss=cross_entropy(model.values(x,c,s),v,torch.from_numpy(prob[ix])).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        torch.save(model.state_dict(),OUT/(name+'.pt'))


def evaluate(fixtures,sids):
    study.init_models(sids)
    for key in ('fixed','continued'):
        study.dq.OUT=OUT/key
        study.dq.OUT.mkdir(exist_ok=True)
        model=hard.actor()
        model.load_state_dict(torch.load(OUT/(key+'.pt'),map_location='cpu',weights_only=True))
        study.dq.MODEL=model.eval()
        for panel,n in PANELS.items():
            jobs=[(cid,f,key,None,None,panel) for cid,f in sorted(fixtures.items()) if cid.startswith(panel+'_')]
            assert len(jobs)==n
            rows=study.dq.batch(jobs,16,'evaluate_'+panel)
            assert {r['case_id'] for r in rows}=={job[0] for job in jobs}
            print('EVAL',key,panel,n,sum(r['success'] for r in rows),flush=True)


def run():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    dev,teachers,data=hard.load_data()
    frozen=study.read(study.OUT/'frozen.json')
    protocol=dict(seed=17,weight='inverse remaining actual ticks per successful CAP at exact common prefix; sum by action',
        rows='frozen fastest-teacher rows; original root-balanced sampler and same RNG streams',panels=PANELS,
        primary='fixed 24000 vs hard fixed 24000, natural held closed-loop',secondary='additional 4500 fixed in advance vs hard selected at same update',
        frozen=frozen['development'],held=frozen['held'],norms=study.sha(hard.BASE/'normalizers.npz'),
        hard_fixed=study.sha(hard.OUT/'actor.pt'),hard_continued=study.sha(hard.OUT/'polish.pt'))
    if (OUT/'protocol.json').exists():
        assert study.read(OUT/'protocol.json')==protocol
    else:
        study.save(OUT/'protocol.json',protocol)
    if not (OUT/'continued.pt').is_file():
        prob=labels(teachers,data,frozen['sids'])
        train(data,prob)
    held=study.read(study.OUT/'held.json')
    assert study.ext.old.digest(held)==frozen['held']
    evaluate(dev|held,frozen['sids'])


if __name__=='__main__':
    run()
