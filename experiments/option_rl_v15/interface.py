"""A learned joint type residual over the frozen v14 legal pair distribution.

Zero residual is exactly the independent baseline, including FOLLOW legality.
Scales retain their per-type conditional factors. Neither masks nor controller
rules enforce agreement: joint preferences are learned only from task returns.
"""
import torch
from torch import nn
from experiments.option_rl_v14 import interface as V14
from experiments.option_rl_v14.interface import ACTIONS,CONFIG,ROTATION_CONFIG,ACTION_VERSION,TYPE_COUNT,SCALE_COUNT,ACTION_CODEBOOK,CODEBOOK_HASH,CONTROLLER_HASH,PAIR_IDS,observe,make_env

VERSION='learning-baseline-v15-joint-type-residual'


def metadata():
    return dict(V14.metadata(),interface_version=VERSION,
        policy_distribution='legal joint type residual over type/scale factors; exact joint entropy; conditional greedy',
        coupling='state-conditioned14x14 additive scores; zero initial logits; no goal heuristic or privileged inputs')


def validate_metadata(data):
    for key in ('interface_version','action_version','action_codebook_hash','controller_config_hash'):
        if data.get(key)!=metadata()[key]:raise ValueError('incompatible checkpoint: '+key)
    if data.get('actions')!=ACTION_CODEBOOK:raise ValueError('action descriptors changed')


class Policy(V14.Policy):
    def __init__(self,obs_dim,hidden=256,*,separate=True,encoder='flat',autoregressive=False,coupled=False):
        if coupled and autoregressive:raise ValueError('one joint representation at a time')
        super().__init__(obs_dim,hidden,separate=separate,encoder=encoder,autoregressive=autoregressive)
        self.model_config['coupled']=coupled
        if coupled:
            # Construction cannot shift any subsequent stochastic sampling RNG.
            with torch.random.fork_rng(devices=[]):self.joint_types=nn.Linear(hidden,TYPE_COUNT*TYPE_COUNT)
            nn.init.zeros_(self.joint_types.weight);nn.init.zeros_(self.joint_types.bias)

    def distributions(self,x,masks,a0=None):
        if not self.model_config['coupled']:return super().distributions(x,masks,a0=a0)
        h=self.body(x)
        log0=self._logits(x,h,masks,0);log1=self._logits(x,h,masks,1)
        scores=self.joint_types(h).reshape(-1,TYPE_COUNT,TYPE_COUNT)
        residual=scores[:,self.type_ids[:,None],self.type_ids[None,:]]
        joint=(log0[:,:,None]+log1[:,None,:]+residual).masked_fill(~self.legal_pairs[None,:,:],-torch.inf)
        row_valid=torch.isfinite(joint).any(-1)
        safe=torch.where(row_valid[:,:,None],joint,torch.zeros_like(joint))
        first=torch.distributions.Categorical(logits=torch.logsumexp(safe,-1).masked_fill(~row_valid,-torch.inf))
        if a0 is None:a0=first.probs.argmax(-1)
        conditional=torch.distributions.Categorical(logits=safe[torch.arange(len(x),device=x.device),a0])
        cond_all=torch.distributions.Categorical(logits=safe)
        conditional.entropy=lambda:(first.probs*cond_all.entropy()).sum(-1)
        vh=self.critic_body(x) if self.critic_body is not None else h
        return [first,conditional],self.value(vh).squeeze(-1)
