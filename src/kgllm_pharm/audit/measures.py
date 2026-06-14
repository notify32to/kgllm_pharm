from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]


def _column_auroc(scores: FloatArray, labels: FloatArray) -> float | None:
    positives = labels > 0.5
    n_pos = int(positives.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = stats.rankdata(scores)
    rank_sum = float(ranks[positives].sum())
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def macro_auroc(probabilities: FloatArray, targets: FloatArray) -> float:
    values = [_column_auroc(probabilities[:, j], targets[:, j]) for j in range(targets.shape[1])]
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else float("nan")


def _column_average_precision(scores: FloatArray, labels: FloatArray) -> float | None:
    n_pos = int((labels > 0.5).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores)
    ordered = labels[order] > 0.5
    cumulative = np.cumsum(ordered)
    ranks = np.arange(1, ordered.shape[0] + 1)
    precision = cumulative / ranks
    return float((precision * ordered).sum() / n_pos)


def macro_aupr(probabilities: FloatArray, targets: FloatArray) -> float:
    values = [
        _column_average_precision(probabilities[:, j], targets[:, j])
        for j in range(targets.shape[1])
    ]
    kept = [v for v in values if v is not None]
    return float(np.mean(kept)) if kept else float("nan")


def macro_f1(probabilities: FloatArray, targets: FloatArray, threshold: float = 0.5) -> float:
    predictions = probabilities >= threshold
    positives = targets > 0.5
    scores: list[float] = []
    for j in range(targets.shape[1]):
        tp = int((predictions[:, j] & positives[:, j]).sum())
        fp = int((predictions[:, j] & ~positives[:, j]).sum())
        fn = int((~predictions[:, j] & positives[:, j]).sum())
        denom = 2 * tp + fp + fn
        if denom == 0:
            continue
        scores.append(2.0 * tp / denom)
    return float(np.mean(scores)) if scores else float("nan")


def sensitivity_at_specificity(
    scores: FloatArray, labels: FloatArray, specificity: float = 0.95
) -> float:
    negatives = scores[labels < 0.5]
    positives = scores[labels > 0.5]
    if negatives.size == 0 or positives.size == 0:
        return float("nan")
    threshold = float(np.quantile(negatives, specificity, method="higher"))
    return float((positives >= threshold).mean())
