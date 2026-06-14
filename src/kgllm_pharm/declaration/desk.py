from __future__ import annotations

from kgllm_pharm.declaration.attributes import four_scale_specs
from kgllm_pharm.declaration.bonded import RealFeatureProvider
from kgllm_pharm.declaration.offline import SurrogateFeatureProvider
from kgllm_pharm.declaration.port import FeatureProvider
from kgllm_pharm.tariff.schedule import DataConfig


def build_feature_provider(cfg: DataConfig, use_real: bool = False) -> FeatureProvider:
    specs = four_scale_specs(
        cfg.mol_dim,
        cfg.tgt_dim,
        cfg.path_dim,
        cfg.text_dim,
        cfg.molecular_signal,
        cfg.target_signal,
        cfg.pathway_signal,
        cfg.text_signal,
    )
    if use_real:
        return RealFeatureProvider(specs)
    mechanistic = max(1, cfg.rank // 2)
    return SurrogateFeatureProvider(specs, cfg.rank, mechanistic, cfg.seed)
