import numpy as np

from kgllm_pharm.clearance.assurance import coverage_lower_bound, coverage_upper_bound
from kgllm_pharm.clearance.certificate import N_STRATA, SplitConformal


def _draw(rng: np.random.Generator, true: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    p = true.shape[0]
    labels = (rng.uniform(size=(n, p)) < true).astype(np.float64)
    probs = np.clip(true[None, :] + rng.normal(0.0, 0.05, size=(n, p)), 0.0, 1.0)
    return probs, labels


def test_per_stratum_coverage_within_band() -> None:
    rng = np.random.default_rng(0)
    p = 60
    strata = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=np.int64)
    true = rng.uniform(0.05, 0.6, size=p)
    cal_p, cal_y = _draw(rng, true, 600)
    test_p, test_y = _draw(rng, true, 600)
    conformal = SplitConformal(0.15)
    conformal.calibrate(cal_p, cal_y, strata)
    coverage = conformal.per_stratum_coverage(test_p, test_y, strata)
    for stratum in range(N_STRATA):
        size = int(conformal.counts[stratum])
        assert coverage[stratum] >= coverage_lower_bound(0.15) - 0.08
        assert coverage[stratum] <= coverage_upper_bound(0.15, size) + 0.08
