from __future__ import annotations

import numpy as np


def fs_moe_capacity_gap(total_params: int, rare_params: int, rare_prevalence: float) -> float:
    capacity = np.sqrt(total_params / max(rare_params, 1))
    frequency = np.sqrt(rare_prevalence / max(1.0 - rare_prevalence, 1e-9))
    return float(min(1.0, capacity * frequency))


def rademacher_gap_bound(
    rare_params: int,
    rare_samples: int,
    lipschitz: float = 1.0,
    failure: float = 0.05,
    constant: float = 1.0,
) -> float:
    rademacher = constant * np.sqrt(
        rare_params * np.log(max(rare_samples, 2)) / max(rare_samples, 1)
    )
    concentration = np.sqrt(np.log(2.0 / failure) / (2.0 * max(rare_samples, 1)))
    return float(2.0 * lipschitz * rademacher + concentration)
