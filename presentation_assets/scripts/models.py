"""Round-5 models: MLP-current, MLP-history and Transformer-history.

The three architectures read the **same** window tensor and the **same** masks and produce the
**same** two heads, so an architecture comparison is not confounded by different inputs:

* ``MLP-current`` sees only the anchor token O0 plus the most recent token (equivalent to "current
  frame + start anchor + command"), i.e. no intermediate history;
* ``MLP-history`` sees the full flattened window with a validity mask;
* ``Transformer-history`` sees the same window with causal attention.

Parameter budgets are reported, never assumed: the two history models are the ones compared for the
architecture question, and the single-frame model is listed separately because its parameter count
differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from experiments.night_sequence_v1.sequences import EXIT_CODES
from experiments.night_sequence_v1.windows import FEATURES, WINDOW

# head output widths
PROCESS_GEOMETRY = 8
PROCESS_POSE = 6
PROCESS_VELOCITY = 6
PROCESS_VALIDITY = 4
PROCESS_EXIT = 1
TERMINAL_REMAINING = 1
#: the full exit vocabulary, not a truncated one: a class the model cannot emit would be scored as
#: an impossible answer rather than a wrong one
TERMINAL_EXIT_CLASSES = len(EXIT_CODES)

#: equal weight per task; inside a task every subgroup counts equally (task book 4.2)
TASK_WEIGHT = 0.5
GROUP_NAMES = ["process_geometry", "process_pose", "process_velocity", "process_validity",
               "process_exit", "terminal_geometry", "terminal_pose", "terminal_velocity",
               "terminal_validity", "terminal_remaining", "terminal_exit_class"]


class Heads(nn.Module):
    """The eleven output heads, each a small two-layer MLP.

    Head width is deliberately small (``head_hidden``) because eleven heads would otherwise
    dominate the parameter budget and make the MLP/Transformer comparison a comparison of head
    sizes rather than of sequence encoders.
    """

    def __init__(self, hidden: int, dropout: float = 0.1, head_hidden: int = 64):
        super().__init__()
        widths = {"process_geometry": PROCESS_GEOMETRY, "process_pose": PROCESS_POSE,
                  "process_velocity": PROCESS_VELOCITY, "process_validity": PROCESS_VALIDITY,
                  "process_exit": PROCESS_EXIT, "terminal_geometry": PROCESS_GEOMETRY,
                  "terminal_pose": PROCESS_POSE, "terminal_velocity": PROCESS_VELOCITY,
                  "terminal_validity": PROCESS_VALIDITY,
                  "terminal_remaining": TERMINAL_REMAINING,
                  "terminal_exit_class": TERMINAL_EXIT_CLASSES}
        for name, out in widths.items():
            setattr(self, name, nn.Sequential(
                nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(head_hidden, out)))


class ModelBase(nn.Module):
    kind = "base"

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout

    def forward(self, window: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class MLPCurrent(ModelBase):
    """Anchor token + most recent token only: the no-intermediate-history baseline."""

    kind = "mlp_current"

    def __init__(self, hidden: int = 512, dropout: float = 0.1, layers: int = 2):
        super().__init__(dropout)
        blocks: List[nn.Module] = []
        width = 2 * FEATURES
        for _ in range(max(1, layers - 1)):
            blocks += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        blocks += [nn.Linear(width, hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*blocks)
        self.heads = Heads(hidden, dropout)

    def forward(self, window: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        anchor = window[:, 0, :]
        # the last real token of the window
        lengths = valid.sum(dim=1).clamp(min=1).long() - 1
        current = window[torch.arange(window.shape[0]), lengths, :]
        features = self.trunk(torch.cat([anchor, current], dim=-1))
        return {name: getattr(self.heads, name)(features) for name in GROUP_NAMES}


class MLPHistory(ModelBase):
    """The full window flattened in fixed order, with the mask concatenated."""

    kind = "mlp_history"

    def __init__(self, hidden: int = 512, dropout: float = 0.1, layers: int = 2):
        super().__init__(dropout)
        width = WINDOW * FEATURES + WINDOW
        blocks: List[nn.Module] = []
        for _ in range(max(1, layers - 1)):
            blocks += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        blocks += [nn.Linear(width, hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*blocks)
        self.heads = Heads(hidden, dropout)

    def forward(self, window: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat = torch.cat([window.reshape(window.shape[0], -1), valid], dim=-1)
        features = self.trunk(flat)
        return {name: getattr(self.heads, name)(features) for name in GROUP_NAMES}


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, length: int = WINDOW + 8):
        super().__init__()
        table = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        table[:, 0::2] = torch.sin(position * div)
        table[:, 1::2] = torch.cos(position * div)
        self.register_buffer("table", table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.table[:x.shape[1]].unsqueeze(0)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a per-sample additive mask and **no** -inf.

    Using ``-1e9`` instead of ``-inf`` matters: with a boolean key-padding mask PyTorch computes
    ``softmax`` over an entirely masked row, which is NaN, and the residual stream then carries that
    NaN into the next block where ``0 * NaN`` leaks it into real rows. With a finite additive mask
    every row keeps a well-defined distribution, padded rows are then zeroed, and changing padding
    content provably cannot change a real prediction.
    """

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        bias = torch.where(allow.unsqueeze(1), torch.zeros((), device=x.device),
                           torch.full((), -1e9, device=x.device))
        weights = torch.softmax(scores + bias, dim=-1)
        out = weights @ value
        out = out.transpose(1, 2).reshape(batch, length, dim)
        return self.drop(self.out(out))


class CausalBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), allow)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerHistory(ModelBase):
    """Pre-LN causal transformer encoder with learned positions and a last-real-token readout."""

    kind = "transformer_history"

    def __init__(self, d_model: int = 128, heads: int = 4, layers: int = 4, ffn: int = 512,
                 dropout: float = 0.1):
        super().__init__(dropout)
        self.input = nn.Linear(FEATURES + 1, d_model)
        self.position = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([CausalBlock(d_model, heads, ffn, dropout)
                                     for _ in range(layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.heads = Heads(d_model, dropout)

    def forward(self, window: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.position(self.input(torch.cat([window, valid.unsqueeze(-1)], dim=-1)))
        length = window.shape[1]
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=window.device))
        # a padded query row may attend to itself, so every row stays finite; padded rows are then
        # zeroed so they can neither be read nor leak into a real row's next block
        padding = valid < 0.5
        # query index is dimension 1, key index is dimension 2: the two need different unsqueezes
        query_padded = padding.unsqueeze(2)
        key_padded = padding.unsqueeze(1)
        eye = torch.eye(length, dtype=torch.bool, device=window.device).unsqueeze(0)
        allow = (causal.unsqueeze(0) & ~key_padded & ~query_padded) | (query_padded & eye)
        for block in self.blocks:
            x = block(x, allow)
            x = x * valid.unsqueeze(-1)
        x = self.final_norm(x) * valid.unsqueeze(-1)
        indices = valid.sum(dim=1).clamp(min=1).long() - 1
        features = x[torch.arange(x.shape[0]), indices, :]
        return {name: getattr(self.heads, name)(features) for name in GROUP_NAMES}


def build_model(kind: str, **kwargs) -> ModelBase:
    if kind == "mlp_current":
        return MLPCurrent(**kwargs)
    if kind == "mlp_history":
        return MLPHistory(**kwargs)
    if kind == "transformer_history":
        return TransformerHistory(**kwargs)
    raise ValueError(f"unknown model kind {kind!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
