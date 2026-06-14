import numpy as np
from sklearn.metrics import cohen_kappa_score

from kgllm_pharm.audit.concordance import cohen_kappa
from kgllm_pharm.audit.measures import macro_auroc
from kgllm_pharm.audit.significance import (
    delong_test,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_ci,
)


def test_cohen_kappa_matches_sklearn() -> None:
    rng = np.random.default_rng(2)
    a = rng.integers(0, 2, 200).astype(bool)
    b = rng.integers(0, 2, 200).astype(bool)
    assert abs(cohen_kappa(a, b) - float(cohen_kappa_score(a, b))) < 1e-9


def test_delong_pvalue_in_range() -> None:
    rng = np.random.default_rng(3)
    labels = (rng.uniform(size=300) < 0.4).astype(np.float64)
    scores_a = labels * 0.6 + rng.uniform(size=300) * 0.4
    scores_b = rng.uniform(size=300)
    pvalue = delong_test(scores_a, scores_b, labels)
    assert 0.0 <= pvalue <= 1.0


def test_bootstrap_brackets_point() -> None:
    rng = np.random.default_rng(4)
    n, p = 120, 8
    labels = (rng.uniform(size=(n, p)) < 0.3).astype(np.float64)
    probs_a = np.clip(labels * 0.5 + rng.uniform(size=(n, p)) * 0.5, 0.0, 1.0)
    probs_b = rng.uniform(size=(n, p))
    point, lo, hi = paired_bootstrap_ci(macro_auroc, probs_a, probs_b, labels, n_boot=200, seed=0)
    assert lo <= hi
    assert lo - 1e-6 <= point <= hi + 1e-6


def test_holm_rejects_smallest() -> None:
    pvalues = np.array([0.001, 0.2, 0.04, 0.5])
    rejected = holm_bonferroni(pvalues, 0.05)
    assert bool(rejected[0])


def test_mcnemar_in_range() -> None:
    rng = np.random.default_rng(5)
    a = rng.integers(0, 2, 120).astype(bool)
    b = rng.integers(0, 2, 120).astype(bool)
    assert 0.0 <= mcnemar_test(a, b) <= 1.0
