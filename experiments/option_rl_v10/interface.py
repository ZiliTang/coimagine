"""One explicit menu/controller/mask contract for rollout, PPO and greedy replay."""
from dataclasses import asdict
from functools import partial
import hashlib,json,math
import numpy as np
import torch
import torch.nn as nn
from coimagine.random_start import env as E, spec
from coimagine.random_start.effect_push import (
    ACTIONS_REPOSITION_V2,REPOSITION_V2_VERSION,EffectPushConfig,action_mask,make_effect_skill)

VERSION='option-rl-effect-v1'
ACTIONS=ACTIONS_REPOSITION_V2
TYPE_COUNT,SCALE_COUNT=14,5
CONFIG=EffectPushConfig(endpoint_tracking=True,fast_approach=True,same_face_allocation=True,
    reposition_boundary='rounded_scan',reposition_peer_detour='outer_arc',
    translation_completion='normal_travel',post_push='contact_follow',contact_recovery='joint_approach',
    follow_completion='maintenance_duration')
ACTION_CODEBOOK=[asdict(a) for a in ACTIONS]
CODEBOOK_HASH=hashlib.sha256(json.dumps(ACTION_CODEBOOK,sort_keys=True).encode()).hexdigest()


def controller_descriptor(config):
    data=asdict(config)
    if config.contact_tangent_damping==0.:data.pop('contact_tangent_damping')
    if not config.progress_guard:data.pop('progress_guard')
    if config.solo_approach_gap==spec.COOP_READY_GAP:data.pop('solo_approach_gap')
    if config.preparation_policy=='legacy':data.pop('preparation_policy')
    if config.standalone_follow_speed==spec.REPOSITION_SPEED_MAX:data.pop('standalone_follow_speed')
    if config.approach_normal_speed==.10:data.pop('approach_normal_speed')
    if not config.mixed_contact_allocation:data.pop('mixed_contact_allocation')
    # Keep archived run10 descriptors exact when the new optional FOLLOW controls
    # are disabled. Enabling either control creates a distinct explicit contract.
    if config.standalone_follow_gap==.05 and config.standalone_follow_peer_guard=='off':
        data.pop('standalone_follow_gap');data.pop('standalone_follow_peer_guard')
    return data


CONTROLLER_HASH=hashlib.sha256(json.dumps(controller_descriptor(CONFIG),sort_keys=True).encode()).hexdigest()
PAIR_IDS=np.full((TYPE_COUNT,SCALE_COUNT),-1,dtype=np.int64)
for a in ACTIONS:PAIR_IDS[a.type_index,a.scale_index]=a.index


def make_env(instance):
    from experiments.option_rl_v10.scenes import geometry_audit
    check=geometry_audit(instance)
    if not check['legal']:
        raise ValueError(f"illegal initial geometry: {instance.name}: {check['reasons']}")
    return E.RandomStartEnv(instance,skill_factory=partial(make_effect_skill,config=CONFIG),
                           action_table=ACTIONS,action_version=REPOSITION_V2_VERSION)


def observe(env):
    obs=env.boundary_observations()
    return (np.asarray(env.observation_vector(*obs),dtype=np.float32),
            *(np.asarray(action_mask(o,CONFIG,actions=ACTIONS),dtype=bool) for o in obs))


def metadata():
    return dict(interface_version=VERSION,action_version=REPOSITION_V2_VERSION,
        actions=ACTION_CODEBOOK,action_count=len(ACTIONS),action_codebook_hash=CODEBOOK_HASH,controller_config=controller_descriptor(CONFIG),
        controller_config_hash=CONTROLLER_HASH,type_count=TYPE_COUNT,scale_count=SCALE_COUNT,
        policy_distribution='P(type|joint_observation) P(scale|type,joint_observation)',
        information_class='centralized full sharing of local observations; no B-bit deployment claim')


def validate_metadata(data):
    for key in ('interface_version','action_version','action_codebook_hash','controller_config_hash'):
        if data.get(key)!=metadata()[key]:raise ValueError(f'incompatible checkpoint: {key}')
    if hashlib.sha256(json.dumps(data.get('actions'),sort_keys=True).encode()).hexdigest()!=CODEBOOK_HASH:
        raise ValueError('checkpoint action descriptors do not match codebook')
    if hashlib.sha256(json.dumps(data.get('controller_config'),sort_keys=True).encode()).hexdigest()!=CONTROLLER_HASH:
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

    def distributions(self,x,masks):
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
        return result,self.value(h).squeeze(-1)

    def act(self,x,masks,*,deterministic=False,generator=None):
        distributions,value=self.distributions(x,masks)
        actions=[d.probs.argmax(-1) if deterministic else torch.multinomial(d.probs,1,generator=generator).squeeze(-1)
                 for d in distributions]
        logp=sum(d.log_prob(a) for d,a in zip(distributions,actions))
        return actions,logp,value
