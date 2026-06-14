from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

N_STRATA = 3


class SplitConformal:
    def __init__(self, alpha: float = 0.15) -> None:
        self.alpha = alpha
        self.quantiles: FloatArray = np.ones(N_STRATA, dtype=np.float64)
        self.counts: IntArray = np.zeros(N_STRATA, dtype=np.int64)

    def calibrate(self, probabilities: FloatArray, targets: FloatArray, strata: IntArray) -> None:
        for stratum in range(N_STRATA):
            columns = np.where(strata == stratum)[0]
            if columns.size == 0:
                continue
            block_prob = probabilities[:, columns]
            block_tgt = targets[:, columns]
            scores = 1.0 - block_prob[block_tgt > 0.5]
            m = scores.shape[0]
            self.counts[stratum] = m
            if m == 0:
                self.quantiles[stratum] = 1.0
                continue
            level = np.ceil((m + 1) * (1.0 - self.alpha)) / m
            if level >= 1.0:
                self.quantiles[stratum] = 1.0
            else:
                self.quantiles[stratum] = float(np.quantile(scores, level, method="higher"))

    def predict_set(self, probabilities: FloatArray, strata: IntArray) -> BoolArray:
        thresholds = self.quantiles[strata]
        included: BoolArray = (1.0 - probabilities) < thresholds[None, :]
        return included

    def per_stratum_coverage(
        self, probabilities: FloatArray, targets: FloatArray, strata: IntArray
    ) -> FloatArray:
        included = self.predict_set(probabilities, strata)
        coverage = np.full(N_STRATA, np.nan, dtype=np.float64)
        for stratum in range(N_STRATA):
            columns = np.where(strata == stratum)[0]
            positives = targets[:, columns] > 0.5
            total = int(positives.sum())
            if total == 0:
                continue
            covered = int((included[:, columns] & positives).sum())
            coverage[stratum] = covered / total
        return coverage

    def marginal_coverage(
        self, probabilities: FloatArray, targets: FloatArray, strata: IntArray
    ) -> float:
        included = self.predict_set(probabilities, strata)
        positives = targets > 0.5
        total = int(positives.sum())
        if total == 0:
            return float("nan")
        return float((included & positives).sum() / total)
