from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

MetricFn = Callable[[FloatArray, FloatArray], float]


def paired_bootstrap_ci(
    metric: MetricFn,
    probabilities_a: FloatArray,
    probabilities_b: FloatArray,
    targets: FloatArray,
    n_boot: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    rows = targets.shape[0]
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, rows, size=rows)
        deltas[b] = metric(probabilities_a[idx], targets[idx]) - metric(
            probabilities_b[idx], targets[idx]
        )
    point = metric(probabilities_a, targets) - metric(probabilities_b, targets)
    tail = (1.0 - level) / 2.0
    lo = float(np.quantile(deltas, tail))
    hi = float(np.quantile(deltas, 1.0 - tail))
    return point, lo, hi


def _midrank(values: FloatArray) -> FloatArray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = values.shape[0]
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks + 1.0
    return out


def delong_test(scores_a: FloatArray, scores_b: FloatArray, labels: FloatArray) -> float:
    order = np.argsort(-labels)
    m = int((labels > 0.5).sum())
    n = labels.shape[0] - m
    if m == 0 or n == 0:
        return float("nan")
    predictions = np.vstack([scores_a[order], scores_b[order]])
    pos = predictions[:, :m]
    neg = predictions[:, m:]
    tx = np.vstack([_midrank(pos[r]) for r in range(2)])
    ty = np.vstack([_midrank(neg[r]) for r in range(2)])
    tz = np.vstack([_midrank(predictions[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.asarray(np.cov(v01), dtype=np.float64)
    sy = np.asarray(np.cov(v10), dtype=np.float64)
    cov = sx / m + sy / n
    variance = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if variance <= 0.0:
        return 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(variance))
    return float(2.0 * stats.norm.sf(abs(z)))


def mcnemar_test(correct_a: BoolArray, correct_b: BoolArray) -> float:
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    statistic = (abs(only_a - only_b) - 1.0) ** 2 / discordant
    return float(stats.chi2.sf(statistic, 1))


def holm_bonferroni(pvalues: FloatArray, alpha: float = 0.05) -> BoolArray:
    n = pvalues.shape[0]
    order = np.argsort(pvalues)
    rejected = np.zeros(n, dtype=np.bool_)
    for rank, idx in enumerate(order):
        threshold = alpha / (n - rank)
        if pvalues[idx] <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


def conformal_ks_statistic(calibration: FloatArray, test: FloatArray) -> float:
    result = stats.ks_2samp(calibration, test)
    return float(result.statistic)
