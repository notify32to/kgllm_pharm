from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kgllm_pharm.audit.concordance import kappa_vs_prr
from kgllm_pharm.audit.measures import macro_aupr, macro_auroc, macro_f1, sensitivity_at_specificity
from kgllm_pharm.audit.reliability import brier_score, expected_calibration_error
from kgllm_pharm.bureau.processing import TrainTensors, fit
from kgllm_pharm.bureau.seal import set_seed
from kgllm_pharm.clearance.certificate import SplitConformal
from kgllm_pharm.consignments.generation import Cohort, synthesise
from kgllm_pharm.consignments.sorting import FrequencyStratum, split_drug_level
from kgllm_pharm.frontier import KgLlmPharm
from kgllm_pharm.tariff.schedule import ExperimentConfig


@dataclass
class Assembly:
    model: KgLlmPharm
    cohort: Cohort
    features: torch.Tensor
    kg_embedding: torch.Tensor
    resident: torch.Tensor
    labels: torch.Tensor
    frequency_load: torch.Tensor


def assemble(config: ExperimentConfig, use_real: bool = False) -> Assembly:
    set_seed(config.data.seed)
    cohort = synthesise(config.data, config.model.token_dim)
    model = KgLlmPharm.build(config, use_real=use_real)
    factors = torch.from_numpy(cohort.factors.astype(np.float32))
    features = model.features_from_factors(factors)
    kg_embedding = torch.from_numpy(cohort.kg_embedding.astype(np.float32))
    resident = torch.from_numpy(cohort.kg_resident)
    labels = torch.from_numpy(cohort.labels.astype(np.float32))
    load = cohort.labels @ cohort.pt_frequency
    frequency_load = torch.from_numpy(load.astype(np.float32))
    return Assembly(model, cohort, features, kg_embedding, resident, labels, frequency_load)


def _subset(assembly: Assembly, idx: np.ndarray) -> TrainTensors:
    sel = torch.from_numpy(idx)
    return TrainTensors(
        features=assembly.features[sel],
        kg_embedding=assembly.kg_embedding[sel],
        resident=assembly.resident[sel],
        labels=assembly.labels[sel],
    )


def train_model(
    config: ExperimentConfig, max_steps: int | None = None
) -> tuple[Assembly, list[float]]:
    assembly = assemble(config)
    split = split_drug_level(
        assembly.cohort.cold,
        config.data.calibration_fraction,
        config.data.test_fraction,
        config.data.seed,
    )
    train_idx = split.train
    sel = torch.from_numpy(train_idx)
    assembly.model.calibrate_mask(
        assembly.features[sel],
        assembly.kg_embedding[sel],
        assembly.resident[sel],
        assembly.frequency_load[sel],
    )
    history = fit(assembly.model, _subset(assembly, train_idx), config.optim, max_steps=max_steps)
    return assembly, history


def _probabilities(assembly: Assembly, idx: np.ndarray) -> np.ndarray:
    sel = torch.from_numpy(idx)
    assembly.model.train(False)
    with torch.no_grad():
        out = assembly.model.forward(
            assembly.features[sel], assembly.kg_embedding[sel], assembly.resident[sel]
        )
    return out.probabilities.detach().numpy().astype(np.float64)


def evaluate(config: ExperimentConfig, max_steps: int | None = None) -> dict[str, float]:
    assembly, _ = train_model(config, max_steps=max_steps)
    split = split_drug_level(
        assembly.cohort.cold,
        config.data.calibration_fraction,
        config.data.test_fraction,
        config.data.seed,
    )
    strata = assembly.cohort.strata
    labels_np = assembly.cohort.labels

    test_prob = _probabilities(assembly, split.test)
    test_tgt = labels_np[split.test]
    rare_cols = np.where(strata == int(FrequencyStratum.RARE))[0]
    rare_prob = test_prob[:, rare_cols].reshape(-1)
    rare_tgt = test_tgt[:, rare_cols].reshape(-1)

    report: dict[str, float] = {
        "auroc": macro_auroc(test_prob, test_tgt),
        "aupr": macro_aupr(test_prob, test_tgt),
        "f1": macro_f1(test_prob, test_tgt),
        "rare_sensitivity": sensitivity_at_specificity(rare_prob, rare_tgt, 0.95),
        "ece": expected_calibration_error(test_prob, test_tgt),
        "brier": brier_score(test_prob, test_tgt),
        "kappa_vs_prr": kappa_vs_prr(test_prob, test_tgt),
    }

    cold_test = assembly.cohort.cold[split.test]
    if cold_test.any():
        report["cold_auroc"] = macro_auroc(test_prob[cold_test], test_tgt[cold_test])

    if config.ablation.use_conformal:
        cal_prob = _probabilities(assembly, split.calibration)
        cal_tgt = labels_np[split.calibration]
        conformal = SplitConformal(config.conformal.alpha)
        conformal.calibrate(cal_prob, cal_tgt, strata)
        report["coverage"] = conformal.marginal_coverage(test_prob, test_tgt, strata)
        per_stratum = conformal.per_stratum_coverage(test_prob, test_tgt, strata)
        report["coverage_rare"] = float(per_stratum[int(FrequencyStratum.RARE)])
    return report


def predict_sets(config: ExperimentConfig, max_steps: int | None = None) -> dict[str, float]:
    assembly, _ = train_model(config, max_steps=max_steps)
    split = split_drug_level(
        assembly.cohort.cold,
        config.data.calibration_fraction,
        config.data.test_fraction,
        config.data.seed,
    )
    strata = assembly.cohort.strata
    cal_prob = _probabilities(assembly, split.calibration)
    cal_tgt = assembly.cohort.labels[split.calibration]
    conformal = SplitConformal(config.conformal.alpha)
    conformal.calibrate(cal_prob, cal_tgt, strata)
    test_prob = _probabilities(assembly, split.test)
    included = conformal.predict_set(test_prob, strata)
    return {
        "mean_set_size": float(included.sum(axis=1).mean()),
        "n_test_drugs": float(split.test.shape[0]),
    }
