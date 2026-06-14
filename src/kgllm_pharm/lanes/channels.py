from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from kgllm_pharm.lanes.benches import GatingNetwork, StratumHead
from kgllm_pharm.lanes.screen import FrequencyFeatureMask


@dataclass
class RouterOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    weights: torch.Tensor


class FsMoeRouter(nn.Module):
    def __init__(
        self, hidden_dim: int, expert_hidden: int, n_pts: int, mask_fraction: float
    ) -> None:
        super().__init__()
        self.gating = GatingNetwork(hidden_dim, 3)
        self.rare_head = StratumHead(hidden_dim, expert_hidden, n_pts)
        self.mid_head = StratumHead(hidden_dim, expert_hidden, n_pts)
        self.common_head = StratumHead(hidden_dim, expert_hidden, n_pts)
        self.mask = FrequencyFeatureMask(hidden_dim, mask_fraction)

    def calibrate_mask(self, hidden: torch.Tensor, scores: torch.Tensor) -> None:
        self.mask.set_from_scores(scores)

    def forward(self, hidden: torch.Tensor) -> RouterOutput:
        weights = self.gating(hidden)
        rare_logits = self.rare_head(self.mask(hidden))
        mid_logits = self.mid_head(hidden)
        common_logits = self.common_head(hidden)
        blended = (
            weights[:, 0:1] * rare_logits
            + weights[:, 1:2] * mid_logits
            + weights[:, 2:3] * common_logits
        )
        probabilities = torch.sigmoid(blended)
        return RouterOutput(logits=blended, probabilities=probabilities, weights=weights)
