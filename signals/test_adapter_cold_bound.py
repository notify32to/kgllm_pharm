import numpy as np
import torch

from kgllm_pharm.classification.tolerance import cold_row_error_bound, empirical_cold_error
from kgllm_pharm.consignments.generation import synthesise
from kgllm_pharm.declaration.desk import build_feature_provider
from kgllm_pharm.tariff.schedule import DataConfig


def test_empirical_cold_error_within_bound() -> None:
    cfg = DataConfig(
        n_drugs=500, n_pts=80, rank=12, mol_dim=24, tgt_dim=24, path_dim=32, text_dim=24, seed=5
    )
    cohort = synthesise(cfg, token_dim=24)
    provider = build_feature_provider(cfg)
    features = provider.encode(torch.from_numpy(cohort.factors.astype(np.float32)))
    features_np = features.numpy().astype(np.float64)
    association = cohort.factors @ cohort.pt_factors.T
    warm = ~cohort.cold
    cold = cohort.cold
    empirical = empirical_cold_error(
        features_np[warm], association[warm], features_np[cold], association[cold]
    )
    m_train = int(warm.sum()) * cfg.n_pts
    bound = cold_row_error_bound(cfg.rank, cfg.n_drugs, m_train)
    assert empirical <= bound
