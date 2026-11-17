"""Frozen entity/family choice with public-menu-bounded speed/direction/time RL."""
import torch
from .learning import advantages
from .batched_ppo import prepare,evaluate_prepared
from .policy import Policy
from .parameter_policy import DurationPolicy,encode,decode,state_digest,subset
from .actions import ActionTree,FIELDS
from .memory_flow import current_event,MemoryHistory

VERSION='supervised-family-bounded-parameters-v1'


def action_parameters(action):
    if type(action) is not int:raise ValueError('integer action required')
    if 0<=action<32:return 'push',action//4,action%4
    if 47<=action<=78:return 'move',(action-47)//4,(action-47)%4
    if 32<=action<=34:return 'rotate_ccw',0,action-32
    if 35<=action<=37:return 'rotate_cw',0,action-35
    if action in (40,43,44,46):return str(action),0,0
    raise ValueError('unsupported public action')


def compatible(candidate,anchor,direction_radius):
    f,d,_=action_parameters(candidate);g,e,_=action_parameters(anchor)
    return f==g and (f not in ('push','move') or min((d-e)%8,(e-d)%8)<=direction_radius)


class BoundedPolicy(DurationPolicy):
    @classmethod
    def from_supervisor(cls,supervisor,radius=4,direction_radius=0):
        if type(direction_radius) is not int or direction_radius not in (0,1):raise ValueError('direction radius0/1 required')
        model=super().from_supervisor(supervisor,radius);model.direction_radius=direction_radius;return model

    def decide(self,history,features,*,teacher=None,deterministic=False,generator=None,temperature=1.):
        if teacher is not None:raise ValueError('frozen family cannot be overridden')
        with features.decision(history),torch.no_grad():
            original=self.supervisor._decide(history,features,teacher=None,deterministic=True,generator=None,temperature=1.)
            anchor=original['action'];o=current_event(history)['observation']
            tree=ActionTree(o,_projected=isinstance(history,MemoryHistory),_action_index=history.action_index if isinstance(history,MemoryHistory) else None)
            pairs=[(a,b) for a,b in tree.menus[anchor['target_id']] if compatible(a,anchor['a0'],self.direction_radius) and compatible(b,anchor['a1'],self.direction_radius)]
            assert (anchor['a0'],anchor['a1']) in pairs
            first={a for a,b in pairs};prefix={'target_id':anchor['target_id']}
            panels=[subset(original['panels'][0],[original['indices'][0]])];indices=[0];logps=[]
            for depth,key in enumerate(FIELDS[1:],1):
                # Preserve original public-tree representatives, so restricting
                # support cannot change zero-update greedy supervisor behavior.
                panel=original['panels'][depth] if prefix==original['panels'][depth]['prefix'] else features.panel(history,tree,prefix)
                allowed=first if key=='a0' else {b for a,b in pairs if a==prefix['a0']} if key=='a1' else set(range(max(1,anchor['ticks']-self.radius),min(40,anchor['ticks']+self.radius)+1))
                ids=[i for i,q in enumerate(panel['choices']) if q[key] in allowed]
                p=subset(panel,ids);dist,_=self.distribution(p,temperature)
                selected=int(dist.logits.argmax()) if deterministic else int(torch.multinomial(dist.probs,1,generator=generator))
                prefix=p['choices'][selected];panels.append(p);indices.append(selected);logps.append(dist.log_prob(torch.tensor(selected)))
            return dict(original,action=prefix,panels=panels,indices=indices,temperature=temperature,
                        logp=float(torch.stack(logps).sum()),value=float(self.value(panels[0])),
                        recommended=prefix in panels[0]['proposals'],legal_action_count=len(pairs)*len(panels[3]['choices']),
                        scored_prefixes=sum(len(p['choices']) for p in panels),supervisor_action=anchor,
                        parameter_protocol=dict(version=VERSION,radius=self.radius,direction_radius=self.direction_radius,supervisor_sha256=self.supervisor_sha256))

    def save(self,path,metadata):
        if state_digest(self.supervisor)!=self.supervisor_sha256:raise ValueError('frozen supervisor changed')
        torch.save(dict(version=VERSION,actor=encode(self),supervisor=encode(self.supervisor),radius=self.radius,direction_radius=self.direction_radius,supervisor_sha256=self.supervisor_sha256,metadata=metadata),path)

    @classmethod
    def load(cls,path,semantic_sha256):
        v=torch.load(path,map_location='cpu',weights_only=True)
        if v.get('version')!=VERSION:raise ValueError('bounded parameter checkpoint required')
        sup=decode(v['supervisor'],semantic_sha256)
        if state_digest(sup)!=v['supervisor_sha256']:raise ValueError('supervisor checksum')
        model=cls.from_supervisor(sup,v['radius'],v['direction_radius']);actor=decode(v['actor'],semantic_sha256)
        if (actor.schema,actor.action_protocol,actor.semantic_protocol,actor.condition)!=(sup.schema,sup.action_protocol,sup.semantic_protocol,sup.condition):raise ValueError('actor/supervisor protocol mismatch')
        model.load_state_dict(actor.state_dict());return model,v['metadata']


def bounded_ppo_update(model,optimizer,episodes,*,epochs=4,minibatch_size=64,seed=17,residual_coef=.05,clip=.2,target_kl=.03):
    if not isinstance(model,BoundedPolicy):raise ValueError('bounded parameter policy required')
    steps=[];targets=[];advs=[]
    for ep in episodes:
        if not ep['steps']:continue
        y,a=advantages(ep['steps'],bootstrap=ep['bootstrap']);steps+=ep['steps'];targets+=y.tolist();advs+=a.tolist()
    if not steps:raise ValueError('empty episodes')
    rows=prepare(model,steps)
    if any(len(panels[0][0])!=1 for panels,_ in rows):raise ValueError('unfrozen target support')
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
