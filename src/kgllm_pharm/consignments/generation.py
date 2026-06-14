from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from kgllm_pharm.tariff.schedule import DataConfig

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Cohort:
    factors: FloatArray
    pt_factors: FloatArray
    labels: FloatArray
    pt_frequency: FloatArray
    strata: IntArray
    kg_embedding: FloatArray
    kg_resident: BoolArray
    cold: BoolArray
    mechanistic_dims: int


def _zipf_prevalence(n_pts: int, exponent: float, floor: float) -> FloatArray:
    ranks = np.arange(1, n_pts + 1, dtype=np.float64)
    weights = ranks ** (-exponent)
    prevalence = weights / weights[0] * 0.2
    clipped: FloatArray = np.clip(prevalence, floor, 0.3)
    return clipped


def _stratify(frequency: FloatArray, rare_q: float, mid_q: float) -> IntArray:
    order = np.argsort(frequency)
    strata = np.full(frequency.shape[0], 2, dtype=np.int64)
    n = frequency.shape[0]
    rare_cut = max(1, int(round(rare_q * n)))
    mid_cut = max(rare_cut + 1, int(round(mid_q * n)))
    strata[order[:rare_cut]] = 0
    strata[order[rare_cut:mid_cut]] = 1
    return strata


def synthesise(cfg: DataConfig, token_dim: int) -> Cohort:
    rng = np.random.default_rng(cfg.seed)
    n, p, r = cfg.n_drugs, cfg.n_pts, cfg.rank
    mech = max(1, r // 2)

    drug_factors = rng.standard_normal((n, r))
    drug_factors /= np.linalg.norm(drug_factors, axis=1, keepdims=True) + 1e-9

    target = _zipf_prevalence(p, cfg.zipf_exponent, cfg.min_positives / max(n, 1))
    order = np.argsort(target)[::-1]
    provisional = np.empty(p, dtype=np.int64)
    provisional[order[: max(1, int(round(0.6 * p)))]] = 2
    rare_cut = max(1, int(round(cfg.rare_quantile * p)))
    mid_cut = max(rare_cut + 1, int(round(cfg.mid_quantile * p)))
    asc = np.argsort(target)
    provisional[asc[:rare_cut]] = 0
    provisional[asc[rare_cut:mid_cut]] = 1
    provisional[asc[mid_cut:]] = 2

    mech_weight = np.where(provisional == 0, 2.0, np.where(provisional == 1, 1.3, 0.6))
    gen_weight = np.where(provisional == 0, 0.5, np.where(provisional == 1, 0.9, 1.4))
    pt_factors = rng.standard_normal((p, r))
    pt_factors[:, :mech] *= mech_weight[:, None]
    pt_factors[:, mech:] *= gen_weight[:, None]

    scores = drug_factors @ pt_factors.T
    labels = np.zeros((n, p), dtype=np.float64)
    for j in range(p):
        k = min(n, max(cfg.min_positives, int(round(target[j] * n))))
        winners = np.argpartition(scores[:, j], n - k)[n - k :]
        labels[winners, j] = 1.0

    if cfg.label_noise > 0.0:
        flips = rng.random(labels.shape) < cfg.label_noise
        labels[flips] = 1.0 - labels[flips]

    frequency = labels.mean(axis=0)
    strata = _stratify(frequency, cfg.rare_quantile, cfg.mid_quantile)

    projection = rng.standard_normal((r, token_dim)) / np.sqrt(r)
    kg_resident = rng.random(n) < cfg.kg_resident_fraction
    kg_embedding = drug_factors @ projection
    kg_embedding[~kg_resident] = 0.0
    cold = ~kg_resident

    return Cohort(
        factors=drug_factors,
        pt_factors=pt_factors,
        labels=labels,
        pt_frequency=frequency,
        strata=strata,
        kg_embedding=kg_embedding,
        kg_resident=kg_resident,
        cold=cold,
        mechanistic_dims=mech,
    )
