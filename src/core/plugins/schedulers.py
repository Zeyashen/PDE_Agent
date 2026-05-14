"""LR Scheduler 注册表。

LLM 通过 training.scheduler_name 选择。
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
)


SCHEDULER_REGISTRY: Dict[str, Callable] = {}


def register_scheduler(name: str):
    def deco(fn):
        if name in SCHEDULER_REGISTRY:
            raise ValueError(f"Scheduler {name} already registered")
        SCHEDULER_REGISTRY[name] = fn
        return fn
    return deco


def get_scheduler(name: str) -> Callable:
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler: {name}. Available: {sorted(SCHEDULER_REGISTRY)}"
        )
    return SCHEDULER_REGISTRY[name]


# ===================== 注册 =====================
@register_scheduler("cosine")
def cosine(optimizer: Optimizer, epochs: int, **kwargs):
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=kwargs.get("eta_min", 1e-6))


@register_scheduler("step")
def step(optimizer: Optimizer, epochs: int, **kwargs):
    step_size = kwargs.get("step_size", max(epochs // 4, 1))
    gamma = kwargs.get("gamma", 0.5)
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


@register_scheduler("plateau")
def plateau(optimizer: Optimizer, epochs: int, **kwargs):
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=kwargs.get("factor", 0.5),
        patience=kwargs.get("patience", 5),
    )
