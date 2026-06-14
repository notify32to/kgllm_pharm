from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class FrequencyStratum(IntEnum):
    RARE = 0
    MID = 1
    COMMON = 2


@dataclass(frozen=True)
class Split:
    train: IntArray
    calibration: IntArray
    test: IntArray


def split_drug_level(
    cold: BoolArray, calibration_fraction: float, test_fraction: float, seed: int
) -> Split:
    rng = np.random.default_rng(seed + 7)
    n = cold.shape[0]
    all_idx = np.arange(n, dtype=np.int64)
    cold_idx = all_idx[cold]
    warm_idx = all_idx[~cold]
    rng.shuffle(warm_idx)

    n_test = int(round(test_fraction * warm_idx.shape[0]))
    n_cal = int(round(calibration_fraction * warm_idx.shape[0]))
    warm_test = warm_idx[:n_test]
    calibration = warm_idx[n_test : n_test + n_cal]
    train = warm_idx[n_test + n_cal :]
    test = np.concatenate([warm_test, cold_idx])
    return Split(train=np.sort(train), calibration=np.sort(calibration), test=np.sort(test))


def assign_strata(frequency: FloatArray, rare_quantile: float, mid_quantile: float) -> IntArray:
    order = np.argsort(frequency)
    n = frequency.shape[0]
    strata = np.full(n, int(FrequencyStratum.COMMON), dtype=np.int64)
    rare_cut = max(1, int(round(rare_quantile * n)))
    mid_cut = max(rare_cut + 1, int(round(mid_quantile * n)))
    strata[order[:rare_cut]] = int(FrequencyStratum.RARE)
    strata[order[rare_cut:mid_cut]] = int(FrequencyStratum.MID)
    return strata
