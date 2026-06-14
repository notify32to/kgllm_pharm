from __future__ import annotations

import importlib.util

import torch

from kgllm_pharm.declaration.attributes import ScaleSpec
from kgllm_pharm.declaration.port import RealBackboneUnavailable

_REQUIRED = ("transformers", "peft", "sentencepiece")
_CHECKPOINTS = {
    "molecular": "seyonec/ChemBERTa-77M-MTR",
    "target": "facebook/esm2_t33_650M_UR50D",
    "clinical_text": "BioMistral/BioMistral-7B",
}


class RealFeatureProvider:
    def __init__(self, specs: list[ScaleSpec]) -> None:
        missing = [name for name in _REQUIRED if importlib.util.find_spec(name) is None]
        if missing:
            raise RealBackboneUnavailable("real four-scale encoders require " + ", ".join(missing))
        self.specs = specs
        self.total_dim = sum(spec.dim for spec in specs)
        self._checkpoints = _CHECKPOINTS

    def scale_blocks(self, factors: torch.Tensor) -> dict[str, torch.Tensor]:
        raise RealBackboneUnavailable(
            "pretrained checkpoints are not provisioned in this environment: "
            + ", ".join(sorted(set(self._checkpoints.values())))
        )

    def encode(self, factors: torch.Tensor, disabled: frozenset[str] = frozenset()) -> torch.Tensor:
        return self.scale_blocks(factors)["molecular"]
