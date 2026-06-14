from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class LossTerms:
    total: torch.Tensor
    bce: torch.Tensor
    balance: torch.Tensor


def load_balance_aux(weights: torch.Tensor) -> torch.Tensor:
    importance = weights.mean(dim=0)
    n_experts = weights.shape[-1]
    aux: torch.Tensor = (importance.pow(2).sum()) * n_experts
    return aux


def _pos_weight(targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum(dim=0)
    negatives = targets.shape[0] - positives
    ratio: torch.Tensor = (negatives + 1.0) / (positives + 1.0)
    return ratio.clamp(max=200.0)


def composite_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    balance_lambda: float,
) -> LossTerms:
    bce: torch.Tensor = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=_pos_weight(targets)
    )
    balance = load_balance_aux(weights)
    total = bce + balance_lambda * balance
    return LossTerms(total=total, bce=bce, balance=balance)
