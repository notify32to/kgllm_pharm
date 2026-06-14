from __future__ import annotations


def coverage_lower_bound(alpha: float) -> float:
    return 1.0 - alpha


def coverage_upper_bound(alpha: float, stratum_size: int) -> float:
    return 1.0 - alpha + 1.0 / (stratum_size + 1)
