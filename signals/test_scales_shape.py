import torch

from kgllm_pharm.declaration.desk import build_feature_provider
from kgllm_pharm.tariff.schedule import DataConfig


def _cfg() -> DataConfig:
    return DataConfig(rank=8, mol_dim=16, tgt_dim=16, path_dim=24, text_dim=16)


def test_total_dim_and_blocks() -> None:
    provider = build_feature_provider(_cfg())
    assert provider.total_dim == 16 + 16 + 24 + 16
    factors = torch.randn(5, 8)
    blocks = provider.scale_blocks(factors)
    assert set(blocks) == {"molecular", "target", "pathway", "clinical_text"}
    assert blocks["pathway"].shape == (5, 24)
    assert provider.encode(factors).shape == (5, provider.total_dim)


def test_disabled_scale_is_zeroed() -> None:
    provider = build_feature_provider(_cfg())
    factors = torch.randn(4, 8)
    full = provider.encode(factors, frozenset({"molecular"}))
    assert int(torch.count_nonzero(full[:, :16])) == 0
    assert int(torch.count_nonzero(full[:, 16:])) > 0
