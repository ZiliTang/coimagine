"""Versioned legal inputs and configurable policy/value and joint-action models."""
from dataclasses import asdict
import copy
from functools import partial
import hashlib,json
import numpy as np
import torch
from torch import nn
from experiments.option_rl_v13 import interface as old
from experiments.option_rl_v14.controller import ACTIONS,CONFIG,ROTATION_CONFIG,VERSION as ACTION_VERSION,make_env,mask
from experiments.option_rl_v14.mode_env import CONTEXT_PER_ROBOT,EXTRA_FIELDS

VERSION='learning-baseline-v14'
TYPE_COUNT,SCALE_COUNT=14,4
ACTION_CODEBOOK=old.ACTION_CODEBOOK
CODEBOOK_HASH=old.CODEBOOK_HASH
CONTROLLER_HASH=old.CONTROLLER_HASH
PAIR_IDS=old.PAIR_IDS


def observe(env):
    obs=env.boundary_observations()
    return np.asarray(env.observation_vector(*obs),dtype=np.float32),*(np.asarray(mask(o),dtype=bool) for o in obs)


def metadata():
    return dict(old.metadata(),interface_version=VERSION,controller_reuse='frozen option_rl_v13',
        extra_feedback_fields=EXTRA_FIELDS,observation_normalization='original760 only; exact scaled controller flags',
        policy_distribution='configured independent legal product OR learned autoregressive; conditional greedy evaluation')


def validate_metadata(data):
    for key in ('interface_version','action_version','action_codebook_hash','controller_config_hash'):
        if data.get(key)!=metadata()[key]:raise ValueError('incompatible checkpoint: '+key)
    if data.get('actions')!=ACTION_CODEBOOK:raise ValueError('action descriptors changed')


class Features(nn.Module):
    def __init__(self,obs_dim,hidden,encoder):
        super().__init__();self.encoder=encoder
        if encoder=='scan':
            self.scan=nn.Sequential(nn.Conv1d(5,16,5,padding=2,padding_mode='circular'),nn.ReLU(),
                nn.Conv1d(16,16,5,padding=2,padding_mode='circular'),nn.ReLU(),nn.Flatten(),nn.Linear(16*72,64),nn.Tanh())
            dim=obs_dim-720+128
        elif encoder=='flat':dim=obs_dim
        else:raise ValueError('unknown encoder')
        self.mlp=nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())

    def forward(self,x):
        if self.encoder=='scan':
            # Keep body-fixed ray order and each robot's orientation features.
            scans=x[:,40:760].reshape(-1,2,72,5).permute(0,1,3,2)
            z=self.scan(scans.reshape(-1,5,72)).reshape(len(x),128)
            x=torch.cat((x[:,:40],z,x[:,760:]),-1)
        return self.mlp(x)


class Policy(nn.Module):
    def __init__(self,obs_dim,hidden=256,*,separate=True,encoder='flat',autoregressive=False):
        super().__init__();self.hidden=hidden;self.obs_dim=obs_dim
        self.model_config=dict(separate=separate,encoder=encoder,autoregressive=autoregressive)
        self.body=Features(obs_dim,hidden,encoder)
        # Start both variants with exactly the same actor AND value functions.
        # Creating a second random body here would also shift actor-head RNG.
        self.critic_body=copy.deepcopy(self.body) if separate else None
        self.types=nn.ModuleList(nn.Linear(hidden,TYPE_COUNT) for _ in (0,1))
        self.scales=nn.ModuleList(nn.Linear(hidden,TYPE_COUNT*SCALE_COUNT) for _ in (0,1))
        if autoregressive:
            self.action_embedding=nn.Embedding(len(ACTIONS),16)
            self.conditional=nn.Sequential(nn.Linear(hidden+16,64),nn.Tanh(),nn.Linear(64,TYPE_COUNT+TYPE_COUNT*SCALE_COUNT))
            nn.init.zeros_(self.conditional[-1].weight);nn.init.zeros_(self.conditional[-1].bias)
        self.value=nn.Linear(hidden,1)
        for head in (*self.types,*self.scales):nn.init.orthogonal_(head.weight,.01);nn.init.zeros_(head.bias)
        self.persistence_bias=nn.Parameter(torch.tensor(2.5))
        families=torch.zeros(TYPE_COUNT,TYPE_COUNT);families[:8,:8]=1;families[8,8]=1;families[9,9]=1;families[10:12,10:12]=1;families[13,13]=1
        self.register_buffer('mode_families',families)
        self.register_buffer('type_ids',torch.tensor([a.type_index for a in ACTIONS]))
        self.register_buffer('scale_ids',torch.tensor([a.scale_index for a in ACTIONS]))
        follow=torch.tensor([a.kind=='follow' for a in ACTIONS]);push=torch.tensor([a.kind=='push' for a in ACTIONS])
        self.register_buffer('legal_pairs',(~follow[:,None]|push[None,:])&(~follow[None,:]|push[:,None]))

    def actor_parameters(self):
        return [p for n,p in self.named_parameters() if not n.startswith(('critic_body.','value.'))]

    def critic_parameters(self):
        params=list(self.value.parameters())
        if self.critic_body is not None:params+=list(self.critic_body.parameters())
        return params

    def _logits(self,x,h,masks,robot,extra=None):
        mask=torch.as_tensor(masks[robot],device=x.device,dtype=torch.bool)
        if mask.ndim==1:mask=mask[None,:].expand(len(x),-1)
        if not bool(mask.any(-1).all()):raise ValueError('empty action mask')
        pairs=torch.zeros(len(x),TYPE_COUNT,SCALE_COUNT,device=x.device,dtype=torch.bool)
        pairs[:,self.type_ids,self.scale_ids]=mask;types=pairs.any(-1)
        safe=pairs.clone();safe[:,:,0]|=~types
        start=760+robot*CONTEXT_PER_ROBOT
        current=x[:,start:start+14]*(1-x[:,start+29:start+30].clamp(0,1))
        logits=self.types[robot](h)+self.persistence_bias*(current@self.mode_families)
        scales=self.scales[robot](h)
        if extra is not None:logits=logits+extra[:,:TYPE_COUNT];scales=scales+extra[:,TYPE_COUNT:]
        a=torch.log_softmax(logits.masked_fill(~types,-torch.inf),-1)
        b=torch.log_softmax(scales.reshape(-1,TYPE_COUNT,SCALE_COUNT).masked_fill(~safe,-torch.inf),-1)
        return (a[:,:,None]+b)[:,self.type_ids,self.scale_ids].masked_fill(~mask,-torch.inf)

    def distributions(self,x,masks,a0=None):
        h=self.body(x);log0=self._logits(x,h,masks,0)
        if self.model_config['autoregressive']:
            n=len(ACTIONS);xx=x[:,None,:].expand(-1,n,-1).reshape(-1,x.shape[-1])
            hh=h[:,None,:].expand(-1,n,-1);ee=self.action_embedding.weight[None,:,:].expand(len(x),-1,-1)
            extra=self.conditional(torch.cat((hh,ee),-1).reshape(-1,self.hidden+16))
            mm=[]
            for m in masks:
                m=torch.as_tensor(m,device=x.device,dtype=torch.bool)
                if m.ndim==1:m=m[None,:].expand(len(x),-1)
                mm.append(m[:,None,:].expand(-1,n,-1).reshape(-1,n))
            log1=self._logits(xx,hh.reshape(-1,self.hidden),mm,1,extra).reshape(len(x),n,n)
            joint=log1.masked_fill(~self.legal_pairs[None,:,:],-torch.inf)
            row_valid=torch.isfinite(joint).any(-1)
            safe=torch.where(row_valid[:,:,None],joint,torch.zeros_like(joint))
            cond_all=torch.distributions.Categorical(logits=safe)
            first=torch.distributions.Categorical(logits=log0.masked_fill(~row_valid,-torch.inf))
        else:
            log1=self._logits(x,h,masks,1)
            joint=(log0[:,:,None]+log1[:,None,:]).masked_fill(~self.legal_pairs[None,:,:],-torch.inf)
            row_valid=torch.isfinite(joint).any(-1)
            safe=torch.where(row_valid[:,:,None],joint,torch.zeros_like(joint))
            first=torch.distributions.Categorical(logits=torch.logsumexp(safe,-1).masked_fill(~row_valid,-torch.inf))
            cond_all=torch.distributions.Categorical(logits=safe)
        if a0 is None:a0=first.probs.argmax(-1)
        conditional=torch.distributions.Categorical(logits=safe[torch.arange(len(x),device=x.device),a0])
        conditional.entropy=lambda:(first.probs*cond_all.entropy()).sum(-1)
        vh=self.critic_body(x) if self.critic_body is not None else h
        return [first,conditional],self.value(vh).squeeze(-1)

    def act(self,x,masks,*,deterministic=False,generator=None):
        ds,value=self.distributions(x,masks)
        a0=ds[0].probs.argmax(-1) if deterministic else torch.multinomial(ds[0].probs,1,generator=generator).squeeze(-1)
        ds,value=self.distributions(x,masks,a0=a0)
        a1=ds[1].probs.argmax(-1) if deterministic else torch.multinomial(ds[1].probs,1,generator=generator).squeeze(-1)
        return [a0,a1],ds[0].log_prob(a0)+ds[1].log_prob(a1),value
