"""Centralized event policy: versioned v19 actions and explicit legal pairs."""
from dataclasses import asdict
import torch

from experiments.option_rl_v15.interface import Policy as OldPolicy
from experiments.option_rl_v19 import spec as S
from experiments.option_rl_v20.event_env import VERSION, EventConfig, OBS_EXTRA, legal_pair


def metadata():
    return dict(interface_version=VERSION, action_version=S.VERSION,
                action_codebook_hash=S.CODEBOOK_HASH,
                actions=[asdict(a) for a in S.ACTIONS],
                legal_pairs='P+P, P+W, R+R, R+W, W+W, ordered',
                observation_extra=list(OBS_EXTRA), event_config=asdict(EventConfig()),
                centralized=True, progress='signed robot travel; not object displacement')


def validate_metadata(saved):
    if saved != metadata():
        raise ValueError('incompatible event/action/observation contract')


class Policy(OldPolicy):
    def __init__(self, obs_dim, hidden=256):
        super().__init__(obs_dim, hidden, separate=True, encoder='flat', coupled=False)
        self.type_ids = torch.tensor([a.type_index for a in S.ACTIONS])
        scales = [a.scale_index for a in S.ACTIONS]
        # The legacy aliases are masked, but must occupy distinct slots during
        # advanced-index assignment in the inherited type/scale factorization.
        for index, scale in ((38, 1), (39, 2), (41, 1), (42, 2)):
            scales[index] = scale
        self.scale_ids = torch.tensor(scales)
        self.legal_pairs = torch.tensor([[legal_pair(a, b) and a.index not in S.ACTION_ALIASES
                                         and b.index not in S.ACTION_ALIASES
                                         for b in S.ACTIONS] for a in S.ACTIONS])
