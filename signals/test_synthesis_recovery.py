import numpy as np
import torch

from kgllm_pharm.consignments.generation import synthesise
from kgllm_pharm.consignments.sorting import FrequencyStratum
from kgllm_pharm.declaration.desk import build_feature_provider
from kgllm_pharm.tariff.schedule import DataConfig


def _fit_residual(features: np.ndarray, labels: np.ndarray) -> float:
    solution, _, _, _ = np.linalg.lstsq(features, labels, rcond=None)
    prediction = features @ solution
    return float(np.linalg.norm(prediction - labels))


def test_pathway_carries_rare_signal_better_than_molecular() -> None:
    cfg = DataConfig(
        n_drugs=400, n_pts=60, rank=16, mol_dim=24, tgt_dim=24, path_dim=32, text_dim=24, seed=3
    )
    cohort = synthesise(cfg, token_dim=24)
    provider = build_feature_provider(cfg)
    blocks = provider.scale_blocks(torch.from_numpy(cohort.factors.astype(np.float32)))
    rare = np.where(cohort.strata == int(FrequencyStratum.RARE))[0]
    labels = cohort.labels[:, rare]
    pathway = blocks["pathway"].numpy().astype(np.float64)
    molecular = blocks["molecular"].numpy().astype(np.float64)
    assert _fit_residual(pathway, labels) < _fit_residual(molecular, labels)


def test_strata_partition_is_complete() -> None:
    cfg = DataConfig(n_drugs=300, n_pts=50, rank=12, seed=4)
    cohort = synthesise(cfg, token_dim=16)
    present = set(int(value) for value in np.unique(cohort.strata))
    assert present == {0, 1, 2}
