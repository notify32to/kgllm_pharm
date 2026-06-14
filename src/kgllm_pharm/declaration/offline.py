from __future__ import annotations

import numpy as np
import torch

from kgllm_pharm.declaration.attributes import ScaleSpec


class SurrogateFeatureProvider:
    def __init__(self, specs: list[ScaleSpec], rank: int, mechanistic_dims: int, seed: int) -> None:
        self.specs = specs
        self.total_dim = sum(spec.dim for spec in specs)
        self._order = [spec.name for spec in specs]
        self._signal = {spec.name: spec.signal for spec in specs}
        self._noise_level = {spec.name: spec.noise for spec in specs}
        self._proj: dict[str, torch.Tensor] = {}
        self._noise: dict[str, torch.Tensor] = {}
        self._att: dict[str, torch.Tensor] = {}
        rng = np.random.default_rng(seed + 101)
        for spec in specs:
            scale = 1.0 / np.sqrt(rank)
            proj = rng.standard_normal((rank, spec.dim)) * scale
            noise = rng.standard_normal((rank, spec.dim)) * scale
            att = np.full(rank, spec.gen_attenuation, dtype=np.float64)
            att[:mechanistic_dims] = spec.mech_attenuation
            self._proj[spec.name] = torch.tensor(proj, dtype=torch.float32)
            self._noise[spec.name] = torch.tensor(noise, dtype=torch.float32)
            self._att[spec.name] = torch.tensor(att, dtype=torch.float32)

    def scale_blocks(self, factors: torch.Tensor) -> dict[str, torch.Tensor]:
        base = factors.to(torch.float32)
        blocks: dict[str, torch.Tensor] = {}
        for name in self._order:
            signal = (base * self._att[name]) @ self._proj[name] * self._signal[name]
            nuisance = (base @ self._noise[name]) * self._noise_level[name]
            blocks[name] = signal + nuisance
        return blocks

    def encode(self, factors: torch.Tensor, disabled: frozenset[str] = frozenset()) -> torch.Tensor:
        blocks = self.scale_blocks(factors)
        parts: list[torch.Tensor] = []
        for name in self._order:
            block = blocks[name]
            if name in disabled:
                block = torch.zeros_like(block)
            parts.append(block)
        return torch.cat(parts, dim=-1)
