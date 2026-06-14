from __future__ import annotations

import torch
from torch import nn


def frequency_mutual_information(
    hidden: torch.Tensor, frequency_load: torch.Tensor
) -> torch.Tensor:
    centred = hidden - hidden.mean(dim=0, keepdim=True)
    load = frequency_load - frequency_load.mean()
    numerator = (centred * load.unsqueeze(-1)).mean(dim=0)
    denom = centred.std(dim=0) * load.std() + 1e-9
    score: torch.Tensor = (numerator / denom).abs()
    return score


class FrequencyFeatureMask(nn.Module):
    keep: torch.Tensor

    def __init__(self, hidden_dim: int, fraction: float) -> None:
        super().__init__()
        self.fraction = fraction
        self.register_buffer("keep", torch.ones(hidden_dim, dtype=torch.float32))

    def set_from_scores(self, scores: torch.Tensor) -> None:
        dim = scores.shape[0]
        cut = int(round(self.fraction * dim))
        keep = torch.ones(dim, dtype=torch.float32)
        if cut > 0:
            top = torch.topk(scores, cut).indices
            keep[top] = 0.0
        self.keep = keep

    def masked_dims(self) -> int:
        return int((self.keep == 0.0).sum().item())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * self.keep
