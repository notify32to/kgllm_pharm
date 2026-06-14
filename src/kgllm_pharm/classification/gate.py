from __future__ import annotations

import torch


class KgDensityGate:
    def __init__(self, open_default: bool = True, threshold: int = 1) -> None:
        self.open_default = open_default
        self.threshold = threshold

    def transductive_mask(
        self, resident: torch.Tensor, edge_count: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not self.open_default:
            return torch.zeros_like(resident, dtype=torch.bool)
        if edge_count is None:
            return resident.to(torch.bool)
        dense = edge_count >= self.threshold
        return resident.to(torch.bool) & dense
