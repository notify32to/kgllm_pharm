from __future__ import annotations

import math
from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer, warmup_steps: int, total_steps: int, decay: float
) -> LambdaLR:
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / float(max(1, warmup_steps))
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return decay + (1.0 - decay) * cosine

    schedule: Callable[[int], float] = factor
    return LambdaLR(optimizer, lr_lambda=schedule)
