from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def cold_row_error_bound(
    rank: int, n_drugs: int, n_train_pairs: int, ris_constant: float = 0.2, failure: float = 0.05
) -> float:
    c_delta = np.log(1.0 / failure) / (1.0 - 3.0 * ris_constant)
    return float(c_delta * np.sqrt(rank * np.log(max(n_drugs, 2)) / max(n_train_pairs, 1)))


def empirical_cold_error(
    features_warm: FloatArray,
    association_warm: FloatArray,
    features_cold: FloatArray,
    association_cold: FloatArray,
) -> float:
    solution, _, _, _ = np.linalg.lstsq(features_warm, association_warm, rcond=None)
    prediction = features_cold @ solution
    residual = float(np.linalg.norm(prediction - association_cold))
    scale = float(np.linalg.norm(association_cold)) + 1e-9
    return residual / scale
