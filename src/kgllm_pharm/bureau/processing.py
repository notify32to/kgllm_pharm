from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from kgllm_pharm.lanes.channels import RouterOutput
from kgllm_pharm.lanes.levy import composite_loss
from kgllm_pharm.tariff.schedule import OptimConfig


@dataclass
class TrainTensors:
    features: torch.Tensor
    kg_embedding: torch.Tensor
    resident: torch.Tensor
    labels: torch.Tensor


class TrainableModel(Protocol):
    def forward(
        self, features: torch.Tensor, kg_embedding: torch.Tensor, resident: torch.Tensor
    ) -> RouterOutput: ...

    def parameter_groups(
        self, adapter_lr: float, lora_lr: float, weight_decay: float
    ) -> list[dict[str, Any]]: ...

    def train(self, mode: bool = ...) -> Any: ...


def _trainable_params(groups: list[dict[str, Any]]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for group in groups:
        params.extend(group["params"])
    return params


def fit(
    model: TrainableModel,
    data: TrainTensors,
    cfg: OptimConfig,
    max_steps: int | None = None,
    logger: logging.Logger | None = None,
) -> list[float]:
    groups = model.parameter_groups(cfg.adapter_lr, cfg.lora_lr, cfg.weight_decay)
    params = _trainable_params(groups)
    optimizer = torch.optim.AdamW(groups, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    model.train(True)

    n = data.labels.shape[0]
    history: list[float] = []
    step = 0
    generator = torch.Generator().manual_seed(0)
    for _epoch in range(cfg.epochs):
        order = torch.randperm(n, generator=generator)
        optimizer.zero_grad()
        accumulated = 0
        for start in range(0, n, cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            out = model.forward(
                data.features[batch], data.kg_embedding[batch], data.resident[batch]
            )
            terms = composite_loss(
                out.logits, data.labels[batch], out.weights, cfg.loss_balance_lambda
            )
            scaled = terms.total / cfg.grad_accum
            torch.autograd.backward(scaled)
            accumulated += 1
            if accumulated == cfg.grad_accum:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0
            history.append(float(terms.total.detach()))
            if logger is not None:
                logger.info("step=%d loss=%.4f bce=%.4f", step, history[-1], float(terms.bce))
            step += 1
            if max_steps is not None and step >= max_steps:
                return history
    return history
