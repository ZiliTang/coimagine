"""Shared conditional scorer; a separate critic, frozen train-only scaling."""
import numpy as np
import math
import torch
from torch import nn
from .actions import ActionTree, FIELDS
from . import VERSION
from .scale import check_trace


def mlp(dim):
    return nn.Sequential(nn.Linear(dim, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))


class Scale(nn.Module):
    def __init__(self, rows):
        super().__init__()
        x = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)
        count = np.isfinite(x).sum(0)
        mean = np.nansum(x, 0) / np.maximum(count, 1)
        std = np.sqrt(np.nansum((x-mean)**2, 0) / np.maximum(count, 1))
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(np.where(std < 1e-5, 1., std), dtype=torch.float32))

    def forward(self, rows):
        # Panels contain validated numbers/None. NumPy converts None to NaN in
        # one numeric-array pass; from_numpy avoids a second tensor copy.
        x = torch.from_numpy(np.asarray(rows, dtype=np.float32))
        missing = torch.isnan(x)
        if not torch.isfinite(x[~missing]).all():
            raise ValueError('non-finite features')
        return torch.cat((((torch.where(missing, self.mean, x)-self.mean)/self.std).clamp(-10, 10), missing.float()), -1)


class Policy(nn.Module):
    def __init__(self, panels, semantic_sha256, condition='M'):
        super().__init__()
        if condition not in ('S', 'E', 'M'):
            raise ValueError('S/E/M required')
        self.condition, self.semantic_sha256 = condition, semantic_sha256
        self.action_protocol = panels[0].get('action_protocol')
        if any(p.get('action_protocol') != self.action_protocol for p in panels):
            raise ValueError('mixed action-scale protocol')
        self.semantic_protocol=panels[0].get("semantic_protocol")
        if any(p.get("semantic_protocol")!=self.semantic_protocol for p in panels):
            raise ValueError("mixed semantic history protocol")
        self.schema = panels[0].get('schema')
        if any(p.get('schema') != self.schema for p in panels):
            raise ValueError('mixed named semantic slots')
        self.s = Scale([r for p in panels for r in p['semantic']])
        self.e = Scale([r for p in panels for r in p['raw']])
        self.semantic = mlp(2*len(self.s.mean))
        self.residual = mlp(2*len(self.e.mean))
        self.critic = mlp(2*(len(self.s.mean)+len(self.e.mean)))
        if condition == 'M':
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def distribution(self, panel, temperature=1.):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('positive finite sampling temperature required')
        if panel.get('semantic_protocol')!=self.semantic_protocol:raise ValueError('semantic history protocol mismatch')
        if panel['semantic_sha256'] != self.semantic_sha256:
            raise ValueError('semantic version mismatch')
        if self.schema is not None and panel.get('schema') != self.schema:
            raise ValueError('named semantic slots mismatch')
        if panel.get('action_protocol') != self.action_protocol:
            raise ValueError('action-scale protocol mismatch: feature panel')
        sem = self.semantic(self.s(panel['semantic'])).flatten()
        raw = self.residual(self.e(panel['raw'])).flatten()
        logits = sem if self.condition == 'S' else raw if self.condition == 'E' else sem+raw
        penalty = (raw-raw.mean()).square().mean() if self.condition == 'M' else raw.sum()*0
        return torch.distributions.Categorical(logits=logits/temperature), penalty

    def value(self, root):
        if root.get("semantic_protocol")!=self.semantic_protocol:raise ValueError("semantic history protocol mismatch")
        features = torch.cat((self.s(root['semantic']).mean(0), self.e(root['raw']).mean(0)))
        return self.critic(features).squeeze()

    def decide(self, history, features, *, teacher=None, deterministic=False, generator=None, temperature=1.):
        with features.decision(history):
            return self._decide(history, features, teacher=teacher, deterministic=deterministic,
                                generator=generator, temperature=temperature)

    def _decide(self, history, features, *, teacher, deterministic, generator, temperature):
        from .memory_flow import current_event,MemoryHistory
        tree = ActionTree(current_event(history)['observation'], _projected=isinstance(history,MemoryHistory),
                          _action_index=history.action_index if isinstance(history,MemoryHistory) else None)
        prefix, panels, indices, logps = {}, [], [], []
        for key in FIELDS:
            panel = features.panel(history, tree, prefix)
            dist, _ = self.distribution(panel, temperature)
            if teacher is not None:
                selected = panel['choices'].index(dict(prefix, **{key: teacher[key]}))
            elif deterministic:
                selected = int(dist.logits.argmax())
            else:
                selected = int(torch.multinomial(dist.probs, 1, generator=generator))
            prefix = panel['choices'][selected]
            panels.append(panel); indices.append(selected)
            logps.append(dist.log_prob(torch.tensor(selected)))
        trace = dict(action=prefix, panels=panels, indices=indices, temperature=temperature,
                     legal_action_count=tree.count, scored_prefixes=sum(len(p['choices']) for p in panels),
                     recommended=prefix in panels[0]['proposals'], history_hash=panels[0]['history_hash'])
        if self.action_protocol is not None:trace['action_protocol']=self.action_protocol
        # Use the exact distributions that selected these four coordinates.
        # PPO evaluate() still recomputes from saved panels with current weights;
        # collection need not repeat all four forwards and the critic twice.
        check_trace(self, trace)
        logp = torch.stack(logps).sum()
        trace.update(logp=float(logp.detach()), value=float(self.value(panels[0]).detach()))
        return trace

    def evaluate(self, trace):
        check_trace(self, trace)
        logps, penalties = [], []
        for p, i in zip(trace['panels'], trace['indices']):
            dist, penalty = self.distribution(p, trace.get('temperature', 1.))
            logps.append(dist.log_prob(torch.tensor(i)))
            penalties.append(penalty)
        return torch.stack(logps).sum(), self.value(trace['panels'][0]), torch.stack(penalties).mean()

    def save(self, path, metadata):
        torch.save(dict(version=VERSION, condition=self.condition, semantic_sha256=self.semantic_sha256,
                        semantic_dim=len(self.s.mean), raw_dim=len(self.e.mean),
                        state=self.state_dict(), metadata=metadata, schema=self.schema, action_protocol=self.action_protocol,semantic_protocol=self.semantic_protocol), path)

    @classmethod
    def load(cls, path, semantic_sha256):
        value = torch.load(path, map_location='cpu', weights_only=True)
        if value['version'] != VERSION or value['semantic_sha256'] != semantic_sha256:
            raise ValueError('incompatible checkpoint')
        dummy = dict(semantic=[[0.]*value['semantic_dim']], raw=[[0.]*value['raw_dim']], action_protocol=value.get('action_protocol'),semantic_protocol=value.get('semantic_protocol'))
        model = cls([dummy], semantic_sha256, value['condition'])
        model.load_state_dict(value['state'])
        model.schema = value.get('schema')
        return model, value['metadata']
