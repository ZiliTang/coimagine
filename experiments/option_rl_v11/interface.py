"""One explicit menu/controller/mask contract for rollout, PPO and greedy replay."""
from dataclasses import asdict
from functools import partial
import hashlib,json,math
import numpy as np
import torch
import torch.nn as nn
from coimagine.random_start import env as E, spec
from coimagine.random_start.effect_push import action_mask

from experiments.option_rl_v11.controller import ACTIONS,VERSION as ACTION_VERSION,CONFIG,ROTATION_CONFIG,make_env
VERSION='option-rl-fast-v2-joint-mask'
TYPE_COUNT,SCALE_COUNT=14,5
ACTION_CODEBOOK=[asdict(a) for a in ACTIONS]
CODEBOOK_HASH=hashlib.sha256(json.dumps(ACTION_CODEBOOK,sort_keys=True).encode()).hexdigest()
def controller_descriptor(config):return asdict(config)
CONTROLLER_HASH=hashlib.sha256(json.dumps({'translation':asdict(CONFIG),'rotation':asdict(ROTATION_CONFIG),'normal_deceleration':.25,'solo_only':True,'total_reference_cap':.5},sort_keys=True).encode()).hexdigest()
PAIR_IDS=np.full((TYPE_COUNT,SCALE_COUNT),-1,dtype=np.int64)
for a in ACTIONS:PAIR_IDS[a.type_index,a.scale_index]=a.index


def observe(env):
    obs=env.boundary_observations()
    return (np.asarray(env.observation_vector(*obs),dtype=np.float32),
            *(np.asarray(action_mask(o,CONFIG,actions=ACTIONS),dtype=bool) for o in obs))


def metadata():
    return dict(interface_version=VERSION,action_version=ACTION_VERSION,
        actions=ACTION_CODEBOOK,action_count=len(ACTIONS),action_codebook_hash=CODEBOOK_HASH,controller_config=controller_descriptor(CONFIG),
        rotation_controller=asdict(ROTATION_CONFIG),normal_deceleration=.25,total_reference_cap=.5,normal_deceleration_scope='solo translation only',env_config=E.EnvConfig(lidar_backend='numpy_batch',peer_clearance_recovery='boundary_v1').as_dict(),controller_config_hash=CONTROLLER_HASH,type_count=TYPE_COUNT,scale_count=SCALE_COUNT,
        policy_distribution='type/scale factors; joint valid-pair normalization; P(a0)P(a1|a0)',
        information_class='centralized full sharing of local observations; no B-bit deployment claim')


def validate_metadata(data):
    for key in ('interface_version','action_version','action_codebook_hash','controller_config_hash'):
        if data.get(key)!=metadata()[key]:raise ValueError(f'incompatible checkpoint: {key}')
    if hashlib.sha256(json.dumps(data.get('actions'),sort_keys=True).encode()).hexdigest()!=CODEBOOK_HASH:
        raise ValueError('checkpoint action descriptors do not match codebook')
    if data.get('controller_config')!=controller_descriptor(CONFIG):
        raise ValueError('checkpoint controller descriptors do not match configuration')


class Policy(nn.Module):
    def __init__(self,obs_dim,hidden=256):
        super().__init__();self.hidden=hidden
        self.body=nn.Sequential(nn.Linear(obs_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.types=nn.ModuleList([nn.Linear(hidden,TYPE_COUNT) for _ in range(2)])
        self.scales=nn.ModuleList([nn.Linear(hidden,TYPE_COUNT*SCALE_COUNT) for _ in range(2)])
        self.value=nn.Linear(hidden,1)
        for head in (*self.types,*self.scales):nn.init.orthogonal_(head.weight,.01);nn.init.zeros_(head.bias)
        self.register_buffer('type_ids',torch.tensor([a.type_index for a in ACTIONS]))
        self.register_buffer('scale_ids',torch.tensor([a.scale_index for a in ACTIONS]))

    def distributions(self,x,masks,a0=None):
        h=self.body(x);result=[]
        for robot in (0,1):
            mask=torch.as_tensor(masks[robot],device=x.device,dtype=torch.bool)
            if mask.ndim==1:mask=mask.unsqueeze(0).expand(x.shape[0],-1)
            if not bool(mask.any(-1).all()):raise ValueError('every sample needs a valid action')
            pairs=torch.zeros((len(x),TYPE_COUNT,SCALE_COUNT),dtype=torch.bool,device=x.device)
            pairs[:,self.type_ids,self.scale_ids]=mask
            types=pairs.any(-1)
            safe=pairs.clone();safe[:,:,0]|=~types  # finite unused conditional rows
            type_logp=torch.log_softmax(self.types[robot](h).masked_fill(~types,-torch.inf),-1)
            scale_logits=self.scales[robot](h).reshape(-1,TYPE_COUNT,SCALE_COUNT)
            scale_logp=torch.log_softmax(scale_logits.masked_fill(~safe,-torch.inf),-1)
            joint=type_logp[:,:,None]+scale_logp
            logits=joint[:,self.type_ids,self.scale_ids].masked_fill(~mask,-torch.inf)
            result.append(torch.distributions.Categorical(logits=logits))
        log0,log1=[d.logits for d in result]
        legal=torch.ones((len(ACTIONS),len(ACTIONS)),dtype=torch.bool,device=x.device)
        follow=torch.tensor([a.kind=='follow' for a in ACTIONS],device=x.device)
        push=torch.tensor([a.kind=='push' for a in ACTIONS],device=x.device)
        legal &= (~follow[:,None] | push[None,:]) & (~follow[None,:] | push[:,None])
        joint=(log0[:,:,None]+log1[:,None,:]).masked_fill(~legal[None,:,:],-torch.inf)
        row_valid=torch.isfinite(joint).any(-1)
        safe_joint=torch.where(row_valid[:,:,None],joint,torch.zeros_like(joint))
        first=torch.distributions.Categorical(logits=torch.logsumexp(safe_joint,-1).masked_fill(~row_valid,-torch.inf))
        if a0 is None:a0=first.probs.argmax(-1)
        conditional=torch.distributions.Categorical(logits=joint[torch.arange(len(x),device=x.device),a0])
        all_conditional=torch.distributions.Categorical(logits=safe_joint)
        conditional.entropy=lambda:(first.probs*all_conditional.entropy()).sum(-1)
        return [first,conditional],self.value(h).squeeze(-1)

    def act(self,x,masks,*,deterministic=False,generator=None):
        ds,value=self.distributions(x,masks)
        a0=ds[0].probs.argmax(-1) if deterministic else torch.multinomial(ds[0].probs,1,generator=generator).squeeze(-1)
        ds,value=self.distributions(x,masks,a0=a0)
        a1=ds[1].probs.argmax(-1) if deterministic else torch.multinomial(ds[1].probs,1,generator=generator).squeeze(-1)
        return [a0,a1],ds[0].log_prob(a0)+ds[1].log_prob(a1),value
