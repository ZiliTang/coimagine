"""The three model families this round compares.

``DirectPredictor``      the round-6 architecture, unchanged, used for the capacity curve.
``DirectMultiStep``      one real-prefix encoding, then a **shared** decoder queried by a horizon
                         encoding; future steps are independent of each other.
``LatentAutoregressive`` one real-prefix encoding, then a **shared** residual transition that rolls
                         the latent forward; step k+1 sees only the latent produced at step k, the
                         public command and the clock.

Both multi-step families consume the *same* future labels, masks, loss weights and windows, and their
total parameter counts are matched in the harness, so the comparison is between two ways of spending
one supervision budget rather than between two budgets.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from experiments.night_sequence_v1.models import (  # noqa: E402
    CausalBlock,
    PositionalEncoding,
    last_valid_index,
)
from experiments.night_sequence_v1.windows import FEATURES, WINDOW  # noqa: E402
from experiments.transformer_scale_latent_v1.data import OBSERVATION_WIDTH  # noqa: E402

OBSERVATION_ORDER = ("geometry", "pose", "velocity", "validity")
OBSERVATION_TOTAL = sum(OBSERVATION_WIDTH.values())


class Encoder(nn.Module):
    """Causal encoder over the real prefix, returning every position and the last real readout."""

    def __init__(self, d_model: int = 128, heads: int = 4, layers: int = 4, ffn: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(FEATURES + 1, d_model)
        self.position = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([CausalBlock(d_model, heads, ffn, dropout)
                                     for _ in range(layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, window: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = self.position(self.input(torch.cat([window, valid.unsqueeze(-1)], dim=-1)))
        length = window.shape[1]
        causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=window.device))
        padding = valid < 0.5
        query_padded = padding.unsqueeze(2)
        key_padded = padding.unsqueeze(1)
        eye = torch.eye(length, dtype=torch.bool, device=window.device).unsqueeze(0)
        allow = (causal.unsqueeze(0) & ~key_padded & ~query_padded) | (query_padded & eye)
        for block in self.blocks:
            x = block(x, allow)
        x = self.final_norm(x)
        # a padded row holds only self-attention noise; zero it before it can be read
        return x * valid.unsqueeze(-1)


def last_real(hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """The hidden state of the most recent **real** token.

    Windows are left-padded, so the newest observation sits at the last slot and the *first* slot is
    padding whenever fewer than ``WINDOW`` tokens exist. ``valid.sum() - 1`` is only correct for
    right padding: it returns the position counting from the left, which for a short history lands
    inside the zero padding (at t=0 it returns slot 0, whose encoder row is zeroed), so the summary
    became a constant independent of the input. The index is taken from the mask itself through the
    shared helper, and there is deliberately only one implementation of this rule.
    """

    index = last_valid_index(valid)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, index, :]


def horizon_encoding(horizon: int, dim: int, device) -> torch.Tensor:
    """A fixed sinusoidal table over the prediction step, indexed by k = 0 .. horizon-1."""

    position = torch.arange(horizon, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device)
                    * (-math.log(10000.0) / dim))
    table = torch.zeros(horizon, dim, device=device)
    table[:, 0::2] = torch.sin(position * div)
    table[:, 1::2] = torch.cos(position * div)
    return table


class ObservationDecoder(nn.Module):
    """One shared decoder: (latent, horizon code, command) -> one observation vector."""

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 6 + 16, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, OBSERVATION_TOTAL))

    def forward(self, latent: torch.Tensor, horizon_code: torch.Tensor,
                command: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, horizon_code, command], dim=-1))


def split_observation(raw: torch.Tensor) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    start = 0
    for name in OBSERVATION_ORDER:
        width = OBSERVATION_WIDTH[name]
        out[name] = raw[..., start:start + width]
        start += width
    return out


class DirectMultiStep(nn.Module):
    """Independent per-horizon decoding from one real-prefix encoding."""

    kind = "direct_multistep"

    def __init__(self, d_model: int = 128, heads: int = 4, layers: int = 4, ffn: int = 512,
                 decoder_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(d_model, heads, layers, ffn, dropout)
        self.decoder = ObservationDecoder(d_model, decoder_hidden, dropout)
        self.horizon_code = 16
        # the auxiliary direct-terminal head reads the same real-prefix representation
        self.terminal = nn.Sequential(nn.Linear(d_model, decoder_hidden), nn.GELU(),
                                      nn.Linear(decoder_hidden, 8))

    def forward(self, window: torch.Tensor, valid: torch.Tensor, command: torch.Tensor,
                horizon: int) -> Dict[str, object]:
        hidden = self.encoder(window, valid)
        summary = last_real(hidden, valid)
        codes = horizon_encoding(horizon, self.horizon_code, window.device)
        batch = window.shape[0]
        raw = self.decoder(summary.unsqueeze(1).expand(batch, horizon, -1),
                           codes.unsqueeze(0).expand(batch, horizon, -1),
                           command.unsqueeze(1).expand(batch, horizon, -1))
        return {"steps": split_observation(raw), "terminal_geometry": self.terminal(summary)}


class LatentAutoregressive(nn.Module):
    """One real-prefix encoding, then a shared residual transition rolled forward freely."""

    kind = "latent_autoregressive"

    def __init__(self, d_model: int = 128, heads: int = 4, layers: int = 4, ffn: int = 512,
                 transition_hidden: int = 256, decoder_hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(d_model, heads, layers, ffn, dropout)
        self.d_model = d_model
        self.transition = nn.Sequential(
            nn.Linear(d_model + 6 + 2, transition_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(transition_hidden, d_model))
        self.decoder = ObservationDecoder(d_model, decoder_hidden, dropout)
        self.terminal = nn.Sequential(nn.Linear(d_model, decoder_hidden), nn.GELU(),
                                      nn.Linear(decoder_hidden, 8))
        self.register_buffer("code_table", horizon_encoding(512, 16, torch.device("cpu")),
                             persistent=False)

    def rollout(self, window: torch.Tensor, valid: torch.Tensor, command: torch.Tensor,
                horizon: int, elapsed: torch.Tensor,
                freeze_latent: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Free rollout: step k+1 sees only the latent from step k, the command and the clock."""

        hidden = self.encoder(window, valid)
        summary = last_real(hidden, valid)
        latent = summary
        latents: List[torch.Tensor] = []
        for step in range(horizon):
            if freeze_latent:
                # The ablation decodes the *initial* encoding at every future instant, so the
                # readout for t+(k+1)dt uses z0 rather than z_(k+1).
                latents.append(summary)
                continue
            # ``z_(k+1) = z_k + Transition(z_k, command, clock, dt)`` and the readout for
            # t+(k+1)dt is ``Decoder(z_(k+1))``: the transition is applied *before* the latent is
            # handed to the decoder. The previous version appended first and therefore decoded the
            # un-advanced z0 at k=1, which made the ablation identical to the normal path at the
            # first step by construction.
            clock = torch.stack([elapsed + step * 0.05, torch.full_like(elapsed, 0.05)], dim=-1)
            delta = self.transition(torch.cat([latent, command, clock], dim=-1))
            latent = latent + delta
            latents.append(latent)
        codes = self.code_table[:horizon].to(window.device)
        stacked = torch.stack(latents, dim=1)
        batch = window.shape[0]
        raw = self.decoder(stacked, codes.unsqueeze(0).expand(batch, horizon, -1),
                           command.unsqueeze(1).expand(batch, horizon, -1))
        # the summary is returned so the terminal head does not encode the prefix a second time
        return split_observation(raw), summary

    def forward(self, window: torch.Tensor, valid: torch.Tensor, command: torch.Tensor,
                horizon: int, elapsed: torch.Tensor,
                freeze_latent: bool = False) -> Dict[str, object]:
        steps, summary = self.rollout(window, valid, command, horizon, elapsed, freeze_latent)
        return {"steps": steps, "terminal_geometry": self.terminal(summary)}


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """Encoder / transition / decoder / head parameters, separately, as the task book requires."""

    groups: Dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder"):
            key = "encoder"
        elif name.startswith("transition"):
            key = "transition"
        elif name.startswith("decoder"):
            key = "decoder"
        else:
            key = "heads"
        groups[key] = groups.get(key, 0) + parameter.numel()
    groups["total"] = sum(value for key, value in groups.items() if key != "total")
    return groups
