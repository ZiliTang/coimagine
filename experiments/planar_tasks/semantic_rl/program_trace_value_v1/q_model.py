"""Candidate Q with common continuation, exact multi-source LOCAL masking."""
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import ValueModel


def all_v(model, x):
    """Factor the shared affine state term; same result as seven V forwards."""
    z = model.encode(x)
    width = model.layout['width']
    layer = model.net[0]
    common = model.channel_mask[0].clone()
    if model.arm == 'LOCAL':
        offset = width-model.layout['trace_width']
        common[offset:width] = 0
        common[width+offset:] = 0
    base = F.linear(z*common, layer.weight[:, :2*width], layer.bias)
    first = base[:, None, :] + layer.weight[:, 2*width:].T[None, :, :]
    if model.arm == 'LOCAL':
        for i in range(model.cap_count):
            extra = model.channel_mask[i]-common
            active = extra.bool()
            first[:, i] += F.linear(z[:, active], layer.weight[:, :2*width][:, active])
    return model.net[1:](first).squeeze(-1)


class QModel(ValueModel):
    def __init__(self, layout, mean, scale, arm, candidate_mean, candidate_scale):
        super().__init__(layout, mean, scale, arm)
        self.register_buffer('candidate_mean', torch.as_tensor(candidate_mean, dtype=torch.float32).clone())
        self.register_buffer('candidate_scale', torch.as_tensor(candidate_scale, dtype=torch.float32).clone())
        width = layout['width']
        self.net = nn.Sequential(nn.Linear(2*width+170+self.cap_count, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def initialize_from_v(self, value):
        width = self.layout['width']
        with torch.no_grad():
            self.net[0].weight[:, :2*width].copy_(value.net[0].weight[:, :2*width])
            self.net[0].weight[:, -self.cap_count:].copy_(value.net[0].weight[:, -self.cap_count:])
            self.net[0].bias.copy_(value.net[0].bias)
            self.net[2].load_state_dict(value.net[2].state_dict())
            self.net[4].load_state_dict(value.net[4].state_dict())

    def values(self, x, candidates, sources):
        z = self.encode(x)
        width = self.layout['width']
        offset = width-self.layout['trace_width']
        layer = self.net[0]
        if self.arm == 'GLOBAL':
            base = F.linear(z, layer.weight[:, :2*width], layer.bias)
        else:
            common = torch.ones(2*width, dtype=z.dtype, device=z.device)
            common[offset:width] = 0
            common[width+offset:] = 0
            base = F.linear(z*common, layer.weight[:, :2*width], layer.bias)
        first = base[:, None, :].expand(-1, candidates.shape[1], -1)
        if self.arm == 'LOCAL':
            pieces = []
            for size in self.layout['trace_widths']:
                indices = torch.cat([torch.arange(offset, offset+size), torch.arange(width+offset, width+offset+size)]).to(z.device)
                pieces.append(F.linear(z[:, indices], layer.weight[:, indices]))
                offset += size
            first = first + torch.einsum('bci,bih->bch', sources, torch.stack(pieces, dim=1))
        valid = torch.isfinite(candidates)
        encoded = torch.cat([torch.where(valid, (candidates-self.candidate_mean)/self.candidate_scale, 0.).clamp(-10, 10),
                             valid.float(), sources], dim=-1)
        first = first + F.linear(encoded, layer.weight[:, 2*width:])
        return self.net[1:](first).squeeze(-1)


def tensors(batch):
    return {k: torch.as_tensor(v, dtype=torch.bool if k == 'valid' else torch.float32) for k, v in batch.items()}


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['model']
    if checkpoint['kind'] == 'Q':
        model = QModel(checkpoint['layout'], state['mean'], state['scale'], checkpoint['arm'],
                       state['candidate_mean'], state['candidate_scale'])
    else:
        model = ValueModel(checkpoint['layout'], state['mean'], state['scale'], checkpoint['arm'])
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def double_dqn_target(online, target, batch, reward, done):
    with torch.no_grad():
        choice = online.values(batch['x'], batch['candidates'], batch['sources']).masked_fill(~batch['valid'], -torch.inf).argmax(1)
        continuation = target.values(batch['x'], batch['candidates'], batch['sources']).gather(1, choice[:, None]).squeeze(1)
        return reward+(~done).float()*continuation  # gamma=1, duration retained separately.
