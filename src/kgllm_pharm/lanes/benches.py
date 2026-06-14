from __future__ import annotations

import torch
from torch import nn


class StratumHead(nn.Module):
    def __init__(self, hidden_dim: int, expert_hidden: int, n_pts: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expert_hidden),
            nn.GELU(),
            nn.Linear(expert_hidden, n_pts),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.net(hidden)
        return logits


class GatingNetwork(nn.Module):
    def __init__(self, hidden_dim: int, n_experts: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, n_experts)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weights: torch.Tensor = torch.softmax(self.linear(hidden), dim=-1)
        return weights
