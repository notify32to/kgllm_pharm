from __future__ import annotations

from typing import Any

import torch
from torch import nn

from kgllm_pharm.classification.coder import InductiveGraphToken
from kgllm_pharm.declaration.desk import build_feature_provider
from kgllm_pharm.declaration.port import FeatureProvider
from kgllm_pharm.examination.engine import BackboneModel, LoraConfigSpec
from kgllm_pharm.lanes.channels import FsMoeRouter, RouterOutput
from kgllm_pharm.lanes.screen import frequency_mutual_information
from kgllm_pharm.tariff.schedule import ExperimentConfig

KgLlmPharmConfig = ExperimentConfig


class KgLlmPharm(nn.Module):
    def __init__(self, config: ExperimentConfig, feature_provider: FeatureProvider) -> None:
        super().__init__()
        self.config = config
        self.provider = feature_provider
        self.disabled = frozenset(config.ablation.disabled_scales)
        lora = LoraConfigSpec(
            config.model.lora_rank, config.model.lora_alpha, config.model.lora_dropout
        )
        self.adapter = InductiveGraphToken(
            feature_provider.total_dim, config.model.token_dim, config.model.blend_init
        )
        self.backbone = BackboneModel(config.model.token_dim, config.model.hidden_dim, lora)
        self.router = FsMoeRouter(
            config.model.hidden_dim,
            config.model.expert_hidden,
            config.data.n_pts,
            config.model.mask_fraction,
        )
        if config.ablation.shared_head:
            self.router.rare_head = self.router.common_head
            self.router.mid_head = self.router.common_head

    @classmethod
    def build(cls, config: ExperimentConfig, use_real: bool = False) -> KgLlmPharm:
        provider = build_feature_provider(config.data, use_real=use_real)
        return cls(config, provider)

    def features_from_factors(self, factors: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.provider.encode(factors, self.disabled)

    def token(
        self, features: torch.Tensor, kg_embedding: torch.Tensor, resident: torch.Tensor
    ) -> torch.Tensor:
        if not self.config.ablation.use_adapter:
            return kg_embedding
        token: torch.Tensor = self.adapter(features, kg_embedding, resident)
        return token

    def hidden(
        self, features: torch.Tensor, kg_embedding: torch.Tensor, resident: torch.Tensor
    ) -> torch.Tensor:
        hidden_state: torch.Tensor = self.backbone(self.token(features, kg_embedding, resident))
        return hidden_state

    def forward(
        self, features: torch.Tensor, kg_embedding: torch.Tensor, resident: torch.Tensor
    ) -> RouterOutput:
        out: RouterOutput = self.router(self.hidden(features, kg_embedding, resident))
        return out

    def calibrate_mask(
        self,
        features: torch.Tensor,
        kg_embedding: torch.Tensor,
        resident: torch.Tensor,
        frequency_load: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            hidden = self.hidden(features, kg_embedding, resident)
        scores = frequency_mutual_information(hidden, frequency_load)
        self.router.calibrate_mask(hidden, scores)

    def parameter_groups(
        self, adapter_lr: float, lora_lr: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        adapter_params = list(self.adapter.parameters())
        lora_params = self.backbone.lora_parameters()
        router_params = list(self.router.parameters())
        return [
            {"params": adapter_params, "lr": adapter_lr, "weight_decay": weight_decay},
            {"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay},
            {"params": router_params, "lr": lora_lr, "weight_decay": weight_decay},
        ]
