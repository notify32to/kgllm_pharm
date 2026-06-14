from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FeatureProvider(Protocol):
    total_dim: int

    def scale_blocks(self, factors: torch.Tensor) -> dict[str, torch.Tensor]: ...

    def encode(
        self, factors: torch.Tensor, disabled: frozenset[str] = frozenset()
    ) -> torch.Tensor: ...


class RealBackboneUnavailable(RuntimeError):
    pass
