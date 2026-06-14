from __future__ import annotations

import math

import torch
from torch import nn


class InductiveGraphToken(nn.Module):
    def __init__(self, total_dim: int, token_dim: int, blend_init: float = 0.5) -> None:
        super().__init__()
        self.project = nn.Linear(total_dim, token_dim, bias=False)
        init = math.log(blend_init / (1.0 - blend_init))
        self.blend_logit = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def blend_coefficient(self) -> torch.Tensor:
        coeff: torch.Tensor = torch.sigmoid(self.blend_logit)
        return coeff

    def forward(
        self, features: torch.Tensor, kg_embedding: torch.Tensor, resident: torch.Tensor
    ) -> torch.Tensor:
        inductive: torch.Tensor = self.project(features)
        beta = self.blend_coefficient()
        blended = beta * inductive + (1.0 - beta) * kg_embedding
        mask = resident.to(inductive.dtype).unsqueeze(-1)
        token: torch.Tensor = mask * blended + (1.0 - mask) * inductive
        return token

    def construct(
        self, features: torch.Tensor, kg_embedding: torch.Tensor, resident: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(features, kg_embedding, resident)
