from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def expected_calibration_error(
    probabilities: FloatArray, labels: FloatArray, n_bins: int = 15
) -> float:
    probs = probabilities.reshape(-1)
    targets = labels.reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = probs.shape[0]
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper >= 1.0:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(targets[mask].mean())
        error += (count / total) * abs(confidence - accuracy)
    return error


def brier_score(probabilities: FloatArray, labels: FloatArray) -> float:
    diff = probabilities.reshape(-1) - labels.reshape(-1)
    return float(np.mean(diff * diff))
