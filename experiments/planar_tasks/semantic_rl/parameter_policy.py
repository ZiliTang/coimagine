"""Duration-only adaptation conditioned on an immutable supervised actor.

Separate checkpoint format prevents loading this actor as an unrestricted Policy.
The supervisor is a function of the current legal history, not a fixed trajectory.
"""
from copy import deepcopy
import hashlib
import math
import torch
from .policy import Policy
from .learning import advantages
from .batched_ppo import prepare, evaluate_prepared

VERSION = 'supervised-structure-duration-v1'


def state_digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def subset(panel, indices):
    p=dict(panel)
    for key in ('choices','representatives','semantic','raw'):
        p[key]=[panel[key][i] for i in indices]
    return p


def encode(model):
    return dict(condition=model.condition,semantic_sha256=model.semantic_sha256,
                semantic_dim=len(model.s.mean),raw_dim=len(model.e.mean),
                state=model.state_dict(),schema=model.schema,action_protocol=model.action_protocol,
                semantic_protocol=model.semantic_protocol)


def decode(blob, sha):
    if blob['semantic_sha256']!=sha:raise ValueError('semantic source mismatch')
    dummy=dict(semantic=[[0.]*blob['semantic_dim']],raw=[[0.]*blob['raw_dim']],
               semantic_protocol=blob['semantic_protocol'],action_protocol=blob['action_protocol'])
    model=Policy([dummy],sha,blob['condition']);model.load_state_dict(blob['state']);model.schema=blob['schema']
    return model


class DurationPolicy(Policy):
    @classmethod
    def from_supervisor(cls,supervisor,radius=4):
        if type(radius) is not int or not 0<=radius<=39:raise ValueError('integer duration radius0..39 required')
        if not supervisor.action_protocol or supervisor.action_protocol['version']!='duration-cap-fault-interrupt-v1':
            raise ValueError('duration execution protocol required')
        model=deepcopy(supervisor);model.__class__=cls
        frozen=deepcopy(supervisor).eval();frozen.requires_grad_(False)
        # Explicitly unregistered: optimizer/model.parameters cannot update it.
        object.__setattr__(model,'supervisor',frozen)
        model.radius=radius;model.supervisor_sha256=state_digest(frozen)
        return model

    def decide(self,history,features,*,teacher=None,deterministic=False,generator=None,temperature=1.):
        if teacher is not None:raise ValueError('duration actor cannot override frozen structure with a teacher')
        with features.decision(history),torch.no_grad():
            tr=self.supervisor._decide(history,features,teacher=None,deterministic=True,generator=None,temperature=1.)
            anchor=tr['action'];panels=[subset(p,[i]) for p,i in zip(tr['panels'][:3],tr['indices'][:3])]
            q=tr['panels'][3];legal=[i for i,c in enumerate(q['choices']) if abs(c['ticks']-anchor['ticks'])<=self.radius]
            last=subset(q,legal);panels.append(last)
            dist,_=self.distribution(last,temperature)
            selected=int(dist.logits.argmax()) if deterministic else int(torch.multinomial(dist.probs,1,generator=generator))
            action=dict(last['choices'][selected])
            assert all(action[k]==anchor[k] for k in ('target_id','a0','a1'))
            tr.update(action=action,panels=panels,indices=[0,0,0,selected],temperature=temperature,
                      logp=float(dist.log_prob(torch.tensor(selected))),value=float(self.value(panels[0])),
                      recommended=action in panels[0]['proposals'],legal_action_count=len(legal),scored_prefixes=3+len(legal),
                      supervisor_action=anchor,parameter_protocol=dict(version=VERSION,radius=self.radius,supervisor_sha256=self.supervisor_sha256))
            return tr

    def value(self,root):
        if len(root['semantic'])>1:
            with torch.no_grad():selected=int(self.supervisor.distribution(root)[0].logits.argmax())
            root=subset(root,[selected])
        return super().value(root)

    def save(self,path,metadata):
        if state_digest(self.supervisor)!=self.supervisor_sha256:raise ValueError('frozen supervisor changed')
        torch.save(dict(version=VERSION,actor=encode(self),supervisor=encode(self.supervisor),radius=self.radius,
                        supervisor_sha256=self.supervisor_sha256,metadata=metadata),path)

    @classmethod
    def load(cls,path,semantic_sha256):
        v=torch.load(path,map_location='cpu',weights_only=True)
        if v.get('version')!=VERSION:raise ValueError('duration checkpoint required')
        sup=decode(v['supervisor'],semantic_sha256)
        if state_digest(sup)!=v['supervisor_sha256']:raise ValueError('supervisor checksum')
        model=cls.from_supervisor(sup,v['radius']);actor=decode(v['actor'],semantic_sha256)
        if (actor.schema,actor.action_protocol,actor.semantic_protocol,actor.condition)!=(sup.schema,sup.action_protocol,sup.semantic_protocol,sup.condition):
            raise ValueError('actor/supervisor protocol mismatch')
        model.load_state_dict(actor.state_dict());return model,v['metadata']


def duration_ppo_update(model,optimizer,episodes,*,epochs=4,minibatch_size=64,seed=17,residual_coef=.05,clip=.2,target_kl=.03):
    if not isinstance(model,DurationPolicy):raise ValueError('duration policy required')
    steps=[];targets=[];advs=[]
    for ep in episodes:
        if not ep['steps']:continue
        y,a=advantages(ep['steps'],bootstrap=ep['bootstrap']);steps+=ep['steps'];targets+=y.tolist();advs+=a.tolist()
    if not steps:raise ValueError('empty episodes')
    rows=prepare(model,steps)
    if any(any(len(p[0])!=1 for p in panels[:3]) for panels,_ in rows):raise ValueError('unfrozen structural support')
    old=torch.tensor([s['logp'] for s in steps]);targets=torch.tensor(targets);adv=torch.tensor(advs)
    if len(adv)>1 and adv.std(unbiased=False)>1e-8:adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    with torch.no_grad():error=max(float((evaluate_prepared(model,rows[i:i+minibatch_size])[0]-old[i:i+minibatch_size]).abs().max()) for i in range(0,len(rows),minibatch_size))
    if error>2e-5:raise ValueError('rollout/PPO log-prob mismatch: '+str(error))
    actor_parameters=list(model.semantic.parameters())+list(model.residual.parameters())
    critic_parameters=list(model.critic.parameters());curve=[];stopped=False;gen=torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        for idx in torch.randperm(len(rows),generator=gen).split(minibatch_size):
            lp,value,penalty,entropy=evaluate_prepared(model,[rows[i] for i in idx.tolist()])
            delta=lp-old[idx];ratio=delta.exp();kl=((ratio-1)-delta).mean()
            if target_kl and float(kl.detach())>1.5*target_kl:stopped=True;break
            actor=-torch.minimum(ratio*adv[idx],ratio.clamp(1-clip,1+clip)*adv[idx]).mean()
            critic=(value-targets[idx]).square().mean();residual=penalty.mean();loss=actor+.5*critic+residual_coef*residual
            optimizer.zero_grad();loss.backward()
            actor_norm=torch.nn.utils.clip_grad_norm_(actor_parameters,1.)
            critic_norm=torch.nn.utils.clip_grad_norm_(critic_parameters,1.)
            if not all(torch.isfinite(x) for x in (loss,actor_norm,critic_norm)):raise ValueError('non-finite update')
            optimizer.step()
            curve.append(dict(epoch=epoch,loss=float(loss.detach()),actor=float(actor.detach()),critic=float(critic.detach()),
                              residual=float(residual.detach()),kl=float(kl.detach()),entropy=float(entropy.mean().detach()),
                              actor_grad_norm=float(actor_norm),critic_grad_norm=float(critic_norm)))
        if stopped:break
    if state_digest(model.supervisor)!=model.supervisor_sha256:raise ValueError('supervisor changed during update')
    return dict(updates=len(curve),samples=len(steps),logp_recompute_error=error,early_stop_kl=stopped,curve=curve)
