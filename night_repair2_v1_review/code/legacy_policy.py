"""A small recurrent policy and value network over the legal observation history.

Deliberately minimal (task book section 4.3): a light observation encoder, a GRU, a discrete policy
head over the public joint skill menu, and a value head that reads the **same** history. The point of
this round is to generate data and produce checkable learning evidence, not to study policy
architectures, so no Transformer is used and no privileged feature is accepted.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

#: input keys the policy is allowed to read. Anything absent from this list cannot reach the actor.
LEGAL_INPUT_KEYS = ("observation", "hidden", "action_mask")


class RecurrentActorCritic(nn.Module):
    """Encoder + GRU + {policy, value} heads.

    ``forward`` takes one 0.05 s observation plus the previous hidden state and returns the new hidden
    state. Actions are sampled only at skill boundaries, but the hidden state advances on every
    control period because the caller replays the intra-skill observation sequence.
    """

    def __init__(self, observation_width: int, action_count: int, hidden: int = 128,
                 encoder: int = 128):
        super().__init__()
        self.observation_width = observation_width
        self.action_count = action_count
        self.hidden_size = hidden
        self.encoder = nn.Sequential(nn.Linear(observation_width, encoder), nn.Tanh(),
                                     nn.Linear(encoder, encoder), nn.Tanh())
        self.gru = nn.GRUCell(encoder, hidden)
        self.policy = nn.Linear(hidden, action_count)
        self.value = nn.Linear(hidden, 1)

    def initial_hidden(self, batch: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device or "cpu")

    def step(self, observation: torch.Tensor, hidden: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(observation)
        hidden = self.gru(features, hidden)
        return self.policy(hidden), self.value(hidden).squeeze(-1), hidden

    def forward(self, observations: torch.Tensor, hidden: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """``observations`` is (batch, time, width); returns logits and values for every period."""

        batch, length, _width = observations.shape
        hidden = hidden if hidden is not None else self.initial_hidden(batch, observations.device)
        logits, values, hiddens = [], [], []
        for index in range(length):
            logit, value, hidden = self.step(observations[:, index, :], hidden)
            logits.append(logit)
            values.append(value)
            hiddens.append(hidden)
        return {"logits": torch.stack(logits, dim=1), "values": torch.stack(values, dim=1),
                "hiddens": torch.stack(hiddens, dim=1)}


@dataclass
class EpisodeMemory:
    """One episode's observation history, per-episode hidden state, and boundary bookkeeping."""

    observations: List[np.ndarray]
    boundaries: List[int]                      # index into ``observations`` of each skill boundary
    actions: List[int]
    rewards: List[float]
    discounts: List[float]
    log_probs: List[float]
    values: List[float]
    terminated: bool
    truncated: bool
    success: bool
    episode_id: str
    root_id: str
    scene: str
    goal: Tuple[float, float]

    def new(self, episode_id: str, root_id: str, scene: str, goal) -> "EpisodeMemory":
        return EpisodeMemory(observations=[], boundaries=[], actions=[], rewards=[], discounts=[],
                             log_probs=[], values=[], terminated=False, truncated=False,
                             success=False, episode_id=episode_id, root_id=root_id, scene=scene,
                             goal=(float(goal[0]), float(goal[1])))
