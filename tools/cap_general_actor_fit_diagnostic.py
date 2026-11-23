"""Controlled, train-only selection of interventions for the 800-room hard-label actor."""
import argparse
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from tools import cap_general_actor_sets as sets
from tools.cap_general_actor_sets import campaign,dq,imitation,original,scale,read,save

OUT = campaign.OUT/'action_actor_800_fit_diagnostic_s17'
ARMS = ('replicate','init29','order29','lr1e4','layernorm','state_scale','continue')


def prepare(work):
    old,held,sids = scale.inputs()
    dq.OUT = work
    dq.MODULES = [dq.load_native((campaign.CAM/'semantics'/(sid+'.py')).read_text())[0] for sid in sids]
    def progress(stage,**info):
        save(OUT/'progress.json',dict(stage=stage,**info))
    dq.progress = progress
    records,_ = sets.training_records(work,old,sids)
    held_records = sets.replay(held,sets.sources_for(held,sids),sids,campaign.OUT/'runs','重建留出教师')
    for panel,shards in [('train',records),('held',held_records)]:
        data = scale.load_teachers(shards)
        dest=work/panel
        dest.mkdir()
        for k,a in data.items():
            np.save(dest/(k+'.npy'),a)
        del data
    save(OUT/'progress.json',dict(stage='拟合对照'))


def load(work,panel):
    return {p.stem:np.load(p,mmap_mode='c') for p in (work/panel).glob('*.npy')}


def encoded_key(x,mean,scale):
    valid=np.isfinite(x)
    z=np.where(valid,(x-mean)/scale,0.).clip(-10,10)
    z[z==0]=0.
    return z.tobytes()+np.packbits(valid).tobytes()


def conflict_audit(data):
    norms=np.load(original.BASE/'normalizers.npz')
    groups=defaultdict(Counter)
    tied=0
    for x,c,s,v,y in zip(*(data[k] for k in ('x','candidates','sources','valid','chosen'))):
        state=sha256(encoded_key(x,norms['mean'],norms['scale'])).digest()
        encoded=[sha256(encoded_key(a,norms['candidate_mean'],norms['candidate_scale'])+b.tobytes()).digest()
                 for a,b in zip(c[v],s[v])]
        target=encoded[np.flatnonzero(np.flatnonzero(v)==y).item()]
        tied+=encoded.count(target)>1
        groups[(state,tuple(sorted(encoded)))][target]+=1
    return dict(rows=len(data['root']),unique_inputs=len(groups),contradictory_inputs=sum(len(c)>1 for c in groups.values()),
                unavoidable_disagreements=sum(sum(c.values())-max(c.values()) for c in groups.values()),
                target_indistinguishable_from_another_candidate=int(tied))


def normalize_fused(module,args):
    return (F.layer_norm(args[0],(64,)),)


def configure(arm):
    torch.manual_seed(29 if arm=='init29' else 17)
    m=imitation.actor(original.BASE)
    if arm=='continue':
        m.load_state_dict(torch.load(scale.OUT/'actor.pt',weights_only=True))
    if arm=='layernorm':
        m.net[1].register_forward_pre_hook(normalize_fused)
    if arm=='state_scale':
        original_encode=m.encode
        m.encode=lambda x: original_encode(x)/16
    return m


def load_best(directory=OUT):
    result=read(directory/'FINISHED.json')
    path=directory/'best_train.pt'
    if campaign.sha(path)!=result['best_sha']:
        raise ValueError(f'Checkpoint digest mismatch at {path}')
    model=configure(result['best_train_arm'])
    model.load_state_dict(torch.load(path,weights_only=True))
    return model.eval()


@torch.inference_mode()
def inspect_model(model,data):
    roots=data['root']
    rng=np.random.default_rng(17)
    other=rng.permutation(len(roots))
    same=roots[other]==roots
    while same.any():
        other[same]=rng.integers(len(roots),size=int(same.sum()))
        same=roots[other]==roots
    width=model.layout['geometry_width']
    eligible=correct=changed=identical=0
    ever=[np.zeros(64,dtype=bool),np.zeros(64,dtype=bool)]
    for start in range(0,len(roots),128):
        ix=np.arange(start,min(start+128,len(roots)))
        x,c,s,v,labels=imitation.batch(data,ix)
        hidden=[]
        hooks=[model.net[j].register_forward_hook(lambda mod,ins,out:hidden.append(out)) for j in (1,3)]
        scores=model.values(x,c,s).masked_fill(~v,-torch.inf)
        for hook in hooks:hook.remove()
        multi=v.sum(1)>1
        pattern_change=torch.zeros(len(x),dtype=torch.bool)
        for j,z in enumerate(hidden):
            active=z>0
            ever[j]|=active[v&multi[:,None]].any(0).numpy()
            pattern_change|=((active!=active[:,:1,:]) & v[:,:,None]).any((1,2))
        pred=scores.argmax(1)
        eligible+=int(multi.sum())
        correct+=int(((pred==labels)&multi).sum())
        identical+=int((~pattern_change&multi).sum())
        swapped=x.clone()
        swapped[:,:width]=torch.from_numpy(data['x'][other[ix],:width])
        changed+=int(((model.values(swapped,c,s).masked_fill(~v,-torch.inf).argmax(1)!=pred)&multi).sum())
    return dict(multi=eligible,correct=correct,geometry_swap_changed=changed,
                identical_relu_gates=identical,ever_active=[int(a.sum()) for a in ever])


def probe_rows(data):
    eligible=np.flatnonzero(data['valid'].sum(1)>1)
    ids=np.random.default_rng(39).choice(eligible,size=min(1024,len(eligible)),replace=False)
    return {k:np.array(v[ids]) for k,v in data.items()}


def margin_gradient(model,data):
    """Derivative of the top-two logit margin for a common 64-D perturbation."""
    slopes=[]
    for ix in np.array_split(np.arange(len(data['root'])),8):
        x,c,s,v,_=imitation.batch(data,ix)
        captured=[]
        def intercept(mod,args):
            shared=torch.zeros_like(args[0][:,0],requires_grad=True)
            captured.append(shared)
            return (args[0].detach()+shared[:,None,:],)
        # Insert before optional LayerNorm so all arms probe the same fused linear term.
        hook=model.net[1].register_forward_pre_hook(intercept,prepend=True)
        try:
            logits=model.values(x,c,s).masked_fill(~v,-torch.inf)
            best=logits.topk(2,dim=1).values
            grad,=torch.autograd.grad((best[:,0]-best[:,1]).sum(),captured)
            slopes.extend(grad.norm(dim=1).tolist())
        finally:
            hook.remove()
    return dict(n=len(slopes),zero=sum(v==0 for v in slopes),median_norm=float(np.median(slopes)))


def train_arm(job, *, trace=False):
    arm,work=job
    torch.set_num_threads(1)
    data=load(work,'train')
    model=configure(arm)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4 if arm=='lr1e4' else 3e-4)
    roots=data['root']
    buckets=[np.flatnonzero(roots==r) for r in np.unique(roots)]
    lengths=np.array([len(b) for b in buckets])
    offsets=np.cumsum(np.r_[0,lengths[:-1]])
    all_ids=np.concatenate(buckets)
    rng=np.random.default_rng(29 if arm=='order29' else 17)
    if arm=='continue':
        for _ in range(24000):
            rng.integers(len(buckets),size=64)
            rng.random(64)
    steps=3000 if arm=='replicate' else 24000
    curve=[]
    if trace:
        early=[]
        sample=probe_rows(data)
    for step in range(steps+1):
        if trace and step in (0,10,50,200,1000,3000):
            early.append(dict(step=step,**inspect_model(model,sample),margin=margin_gradient(model,sample)))
            save(OUT/'early_baseline.json',early)
        if step in imitation.CHECKS:
            fit,_=imitation.assess(model,data)
            curve.append(dict(step=step,**fit))
            save(OUT/(arm+'_curve.json'),curve)
            if step==steps:break
        ri=rng.integers(len(buckets),size=64)
        ix=all_ids[offsets[ri]+(rng.random(64)*lengths[ri]).astype(int)]
        x,c,s,v,y=imitation.batch(data,ix)
        loss=F.cross_entropy(model.values(x,c,s).masked_fill(~v,-torch.inf),y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
    if arm=='replicate':
        historical=read(scale.OUT/'fit.json')[:len(curve)]
        assert curve==historical, (curve,historical)
        return arm,dict(reproduced_steps=steps)
    train=inspect_model(model,data)
    held=inspect_model(model,load(work,'held'))
    # Only a train-selected representative is kept after the comparison.
    torch.save(model.state_dict(),work/(arm+'.pt'))
    result=dict(train=train,held=held,steps=steps,ce=curve[-1]['ce'])
    save(OUT/(arm+'_metrics.json'),result)
    return arm,result


def priority_reference(work):
    """Fit one constant CAP priority order on training boundaries only."""
    panels={}
    for panel in ('train','held'):
        data=load(work,panel)
        multi=data['valid'].sum(1)>1
        src=data['sources'][multi].astype(bool) & data['valid'][multi,:,None]
        assert src.any((1,2)).all() and (src.sum(axis=1)<=1).all()
        panels[panel]=(src,data['chosen'][multi])
    names=read(scale.OUT/'protocol.json')['sids']
    def choices(sources,order):
        available=sources[:,:,order].any(1)
        first=np.asarray(order)[available.argmax(1)]
        return sources[np.arange(len(sources)),:,first].argmax(1)
    src,target=panels['train']
    order=max(permutations(range(len(names))),key=lambda o:(choices(src,o)==target).sum())
    result=dict(priority=[names[i] for i in order])
    baseline=sets.checkpoint(scale.OUT)
    for panel,(src,target) in panels.items():
        data=load(work,panel)
        _,prediction=imitation.assess(baseline,data)
        chosen=choices(src,order)
        result[panel]=dict(multi=len(target),correct=int((chosen==target).sum()),
                           matching_baseline=int((chosen==prediction[data['valid'].sum(1)>1]).sum()))
    save(OUT/'priority_reference.json',result)


def final_probes(work):
    priority_reference(work)
    train=load(work,'train')
    held=load(work,'held')
    sample=probe_rows(train)
    comparison={}
    references={'hard_reference':scale.OUT,'set_reference':sets.OUT,'hard_330':original.OUT}
    for arm in (*references,*ARMS[1:]):
        if arm in references:
            model=sets.checkpoint(references[arm])
        else:
            model=configure(arm)
            model.load_state_dict(torch.load(work/(arm+'.pt'),weights_only=True))
            model.eval()
        comparison[arm]=dict(margin=margin_gradient(model,sample))
        for name,data in [('train',train),('held',held)]:
            fit,_=imitation.assess(model,data)
            comparison[arm][name]=dict(fully_matched_rooms=fit['perfect'],rooms=len(np.unique(data['root'])),
                                       correct=fit['informative_matched'],multi=fit['informative'])
    save(OUT/'margin_comparison.json',comparison)


def run(work):
    torch.set_num_threads(1)
    data=load(work,'train')
    assert len(data['root'])==15601 and (data['valid'].sum(1)>1).sum()==11254
    save(OUT/'protocol.json',dict(arms=list(ARMS),train=15601,teachers=491,batch=64,default_lr=.0003,
        reference_model=campaign.sha(scale.OUT/'actor.pt'),normalizers=campaign.sha(original.BASE/'normalizers.npz'),
        selection='keep highest train multi-action accuracy among new hard-label arms; never use held scores to choose',
        continuation='new Adam because old fit stored only weights; sampler advanced past original 24000 batches'))
    # Start the independent fits before the input audit.
    with ProcessPoolExecutor(max_workers=len(ARMS)) as pool:
        futures=[pool.submit(train_arm,(arm,work),trace=(arm=='replicate')) for arm in ARMS]
        save(OUT/'conflicts.json',conflict_audit(data))
        ref=sets.checkpoint(scale.OUT)
        baseline=dict(train=inspect_model(ref,data),held=inspect_model(ref,load(work,'held')))
        save(OUT/'baseline.json',baseline)
        arms=dict(f.result() for f in as_completed(futures))
    from tools.cap_general_actor_fit_centering import run as center_control
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(center_control,[(mode,work) for mode in ('baseline','mask','values')]))
    final_probes(work)
    winner=max((a for a in ARMS if a!='replicate'),key=lambda a:arms[a]['train']['correct'])
    shutil.copyfile(work/(winner+'.pt'),OUT/'best_train.pt')
    result=dict(arms=arms,baseline=baseline,conflicts=read(OUT/'conflicts.json'),best_train_arm=winner,
                best_sha=campaign.sha(OUT/'best_train.pt'))
    save(OUT/'FINISHED.json',result)
    save(OUT/'progress.json',dict(stage='完成'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',type=Path)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if not (OUT/'FINISHED.json').exists():
        if args.prepared:
            run(args.prepared)
        else:
            with TemporaryDirectory(prefix='coimagine-hardfit-',dir='/dev/shm') as tmp:
                prepare(Path(tmp))
                run(Path(tmp))
