from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def cohen_kappa(first: BoolArray, second: BoolArray) -> float:
    a = first.reshape(-1)
    b = second.reshape(-1)
    n = a.shape[0]
    if n == 0:
        return float("nan")
    observed = float((a == b).mean())
    pa = float(a.mean())
    pb = float(b.mean())
    expected = pa * pb + (1.0 - pa) * (1.0 - pb)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def proportional_reporting_ratio(a: float, b: float, c: float, d: float) -> float:
    exposed = a / (a + b) if (a + b) > 0 else 0.0
    background = c / (c + d) if (c + d) > 0 else 0.0
    if background == 0.0:
        return float("inf") if exposed > 0 else 0.0
    return exposed / background


def prr_chi_square(a: float, b: float, c: float, d: float) -> float:
    n = a + b + c + d
    if n == 0:
        return 0.0
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    expected = np.array([row1 * col1, row1 * col2, row2 * col1, row2 * col2], dtype=np.float64) / n
    observed = np.array([a, b, c, d], dtype=np.float64)
    safe = expected > 0
    return float((((observed - expected) ** 2)[safe] / expected[safe]).sum())


def prr_calls(counts: FloatArray, prr_min: float = 2.0, chi2_min: float = 4.0) -> BoolArray:
    total = float(counts.sum())
    row_totals = counts.sum(axis=1)
    col_totals = counts.sum(axis=0)
    calls = np.zeros(counts.shape, dtype=np.bool_)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            a = float(counts[i, j])
            b = float(row_totals[i] - a)
            c = float(col_totals[j] - a)
            d = float(total - a - b - c)
            prr = proportional_reporting_ratio(a, b, c, d)
            chi2 = prr_chi_square(a, b, c, d)
            calls[i, j] = prr >= prr_min and chi2 >= chi2_min
    return calls


def kappa_vs_prr(probabilities: FloatArray, counts: FloatArray, threshold: float = 0.5) -> float:
    model_calls = probabilities >= threshold
    reference = prr_calls(counts)
    return cohen_kappa(model_calls, reference)
