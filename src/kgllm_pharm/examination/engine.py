from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from kgllm_pharm.declaration.port import RealBackboneUnavailable


@dataclass(frozen=True)
class LoraConfigSpec:
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05


class BackboneModel(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, lora: LoraConfigSpec) -> None:
        super().__init__()
        self.base = nn.Linear(token_dim, hidden_dim)
        self.base.requires_grad_(False)
        self.lora_a = nn.Linear(token_dim, lora.rank, bias=False)
        self.lora_b = nn.Linear(lora.rank, hidden_dim, bias=False)
        nn.init.zeros_(self.lora_b.weight)
        self.dropout = nn.Dropout(lora.dropout)
        self.scaling = lora.alpha / lora.rank

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        frozen: torch.Tensor = torch.tanh(self.base(token))
        delta: torch.Tensor = self.lora_b(self.dropout(self.lora_a(token)))
        return frozen + self.scaling * delta

    def lora_parameters(self) -> list[nn.Parameter]:
        return list(self.lora_a.parameters()) + list(self.lora_b.parameters())


class RealBackbone(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, lora: LoraConfigSpec) -> None:
        super().__init__()
        raise RealBackboneUnavailable(
            "BioMistral/BioMistral-7B weights are not provisioned in this environment"
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        raise RealBackboneUnavailable("real backbone is unavailable")
