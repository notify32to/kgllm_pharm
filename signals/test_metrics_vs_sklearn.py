import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from kgllm_pharm.audit.measures import macro_aupr, macro_auroc


def _data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = 200, 10
    labels = (rng.uniform(size=(n, p)) < 0.3).astype(np.float64)
    probs = rng.uniform(size=(n, p))
    return probs, labels


def test_macro_auroc_matches_sklearn() -> None:
    probs, labels = _data(1)
    mine = macro_auroc(probs, labels)
    reference = [
        roc_auc_score(labels[:, j], probs[:, j])
        for j in range(labels.shape[1])
        if 0 < labels[:, j].sum() < labels.shape[0]
    ]
    assert abs(mine - float(np.mean(reference))) < 1e-9


def test_macro_aupr_matches_sklearn() -> None:
    probs, labels = _data(2)
    mine = macro_aupr(probs, labels)
    reference = [
        average_precision_score(labels[:, j], probs[:, j])
        for j in range(labels.shape[1])
        if labels[:, j].sum() > 0
    ]
    assert abs(mine - float(np.mean(reference))) < 1e-9
