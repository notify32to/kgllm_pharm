import numpy as np
import torch

from kgllm_pharm.gateway import line
from kgllm_pharm.tariff.schedule import DataConfig, ExperimentConfig, ModelConfig


def test_same_seed_same_outputs() -> None:
    cfg = ExperimentConfig(
        data=DataConfig(
            n_drugs=50, n_pts=16, rank=8, mol_dim=16, tgt_dim=16, path_dim=16, text_dim=16, seed=11
        ),
        model=ModelConfig(token_dim=24, hidden_dim=32, expert_hidden=24, lora_rank=8),
    )
    first = line.assemble(cfg)
    second = line.assemble(cfg)
    assert np.array_equal(first.cohort.labels, second.cohort.labels)
    assert torch.allclose(first.features, second.features)
