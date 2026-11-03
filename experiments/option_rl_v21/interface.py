"""V2 15-type policy. Explicitly incompatible with the frozen V1 checkpoint."""
from dataclasses import asdict
import torch
from torch import nn

from experiments.option_rl_v14.interface import Policy as BasePolicy
from experiments.option_rl_v13.mode_env import CONTEXT_PER_ROBOT
from experiments.option_rl_v21 import spec as S
from experiments.option_rl_v21.system import YIELD_FIELDS
from experiments.option_rl_v21.event_env import VERSION, EventConfig, OBS_EXTRA, legal_pair

TYPE_COUNT, SCALE_COUNT = 15, 4


def metadata():
    return dict(interface_version=VERSION, action_version=S.VERSION,
        action_codebook_hash=S.CODEBOOK_HASH, actions=[asdict(a) for a in S.ACTIONS],
        legal_pairs='V1 pairs plus P+Y, R+Y, W+Y, Y+Y, ordered', centralized=True,
        observation_extra=list(OBS_EXTRA), yield_fields=list(YIELD_FIELDS), event_config=asdict(EventConfig()),
        progress='signed robot travel for PUSH/REPOSITION only; YIELD excluded',
        push_mask='current visible normals, signed 35 degree cone; no rectangle dimensions')


def validate_metadata(saved):
    if saved != metadata():
        raise ValueError('incompatible V2 event/action/observation contract')


class Policy(BasePolicy):
    def __init__(self, obs_dim, hidden=256):
        super().__init__(obs_dim, hidden, separate=True, encoder='flat', autoregressive=False)
        self.types = nn.ModuleList(nn.Linear(hidden, TYPE_COUNT) for _ in (0, 1))
        self.scales = nn.ModuleList(nn.Linear(hidden, TYPE_COUNT*SCALE_COUNT) for _ in (0, 1))
        for head in (*self.types, *self.scales):
            nn.init.orthogonal_(head.weight, .01); nn.init.zeros_(head.bias)
        families = torch.zeros(TYPE_COUNT, TYPE_COUNT)
        families[:14, :14] = self.mode_families
        families[14, 14] = 1.
        self.mode_families = families
        self.type_ids = torch.tensor([a.type_index for a in S.ACTIONS])
        scales = [a.scale_index for a in S.ACTIONS]
        for index, value in ((38, 1), (39, 2), (41, 1), (42, 2)):
            scales[index] = value
        self.scale_ids = torch.tensor(scales)
        self.legal_pairs = torch.tensor([[legal_pair(a, b) and a.index not in S.ACTION_ALIASES
            and b.index not in S.ACTION_ALIASES for b in S.ACTIONS] for a in S.ACTIONS])

    def _logits(self, x, h, masks, robot, extra=None):
        mask = torch.as_tensor(masks[robot], device=x.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask[None, :].expand(len(x), -1)
        if not bool(mask.any(-1).all()):
            raise ValueError('empty action mask')
        pairs = torch.zeros(len(x), TYPE_COUNT, SCALE_COUNT, device=x.device, dtype=torch.bool)
        pairs[:, self.type_ids, self.scale_ids] = mask
        types = pairs.any(-1); safe = pairs.clone(); safe[:, :, 0] |= ~types
        start = 760 + robot*CONTEXT_PER_ROBOT
        current = torch.cat((x[:, start:start+14], x[:, 944+robot*len(YIELD_FIELDS):945+robot*len(YIELD_FIELDS)]), -1)
        current = current*(1-x[:, start+29:start+30].clamp(0, 1))
        logits = self.types[robot](h)+self.persistence_bias*(current@self.mode_families)
        type_logp = torch.log_softmax(logits.masked_fill(~types, -torch.inf), -1)
        scale_logp = torch.log_softmax(self.scales[robot](h).reshape(-1, TYPE_COUNT, SCALE_COUNT).masked_fill(~safe, -torch.inf), -1)
        return (type_logp[:, :, None]+scale_logp)[:, self.type_ids, self.scale_ids].masked_fill(~mask, -torch.inf)
