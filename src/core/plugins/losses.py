"""损失函数注册表。

LLM 通过 training.loss_name 在 yaml 选择，不能直接改这里的代码。
新增损失函数需手动添加到此文件。
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F


LOSS_REGISTRY: Dict[str, Callable] = {}


def register_loss(name: str):
    def deco(fn):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss {name} already registered")
        LOSS_REGISTRY[name] = fn
        return fn
    return deco


def get_loss(name: str) -> Callable:
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: {name}. Available: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[name]


def list_losses() -> list:
    return sorted(LOSS_REGISTRY.keys())


# ===================== 注册 =====================
@register_loss("relative_l2")
def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """逐样本相对 L2，与赛题评分公式同方向。"""
    diff = pred - target
    return (diff.norm(dim=-1) / (target.norm(dim=-1) + 1e-8)).mean()


@register_loss("mse")
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


@register_loss("huber")
def huber_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=1.0)


@register_loss("h1")
def h1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """H1 Sobolev loss：同时惩罚函数值和一阶导数。
    
    对激波更敏感，适合 Burgers 方程。
    pred/target: (B, 1, X) 或 (B, X)
    """
    # 统一形状到 (B, X)
    if pred.dim() == 3:
        pred   = pred[:, 0, :]
        target = target[:, 0, :]

    # L2 项
    diff = pred - target
    loss_l2 = (diff.norm(dim=-1) / (target.norm(dim=-1) + 1e-8)).mean()

    # 梯度项：有限差分近似一阶导数
    pred_dx   = pred[:, 1:] - pred[:, :-1]
    target_dx = target[:, 1:] - target[:, :-1]
    diff_dx   = pred_dx - target_dx
    loss_h1   = (diff_dx.norm(dim=-1) / (target_dx.norm(dim=-1) + 1e-8)).mean()

    return loss_l2 + 0.1 * loss_h1


@register_loss("h1_weighted")
def h1_weighted_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """H1 loss，加大梯度权重（适合激波更陡峭的情况）。"""
    if pred.dim() == 3:
        pred   = pred[:, 0, :]
        target = target[:, 0, :]

    diff = pred - target
    loss_l2 = (diff.norm(dim=-1) / (target.norm(dim=-1) + 1e-8)).mean()

    pred_dx   = pred[:, 1:] - pred[:, :-1]
    target_dx = target[:, 1:] - target[:, :-1]
    diff_dx   = pred_dx - target_dx
    loss_h1   = (diff_dx.norm(dim=-1) / (target_dx.norm(dim=-1) + 1e-8)).mean()

    return loss_l2 + 0.5 * loss_h1


@register_loss("freq")
def freq_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """频域 loss：直接在傅里叶空间监督，契合 FNO 学习机制。
    
    低频分量权重更高（乘以频率倒数），符合物理先验。
    """
    if pred.dim() == 3:
        pred   = pred[:, 0, :]
        target = target[:, 0, :]

    pred_ft   = torch.fft.rfft(pred,   norm="ortho")
    target_ft = torch.fft.rfft(target, norm="ortho")

    # 频率权重：低频更重要
    n_freq = pred_ft.shape[-1]
    freq_weight = 1.0 / (torch.arange(1, n_freq + 1,
                         device=pred.device).float() ** 0.5)

    diff_ft = (pred_ft - target_ft).abs() * freq_weight
    loss = (diff_ft.norm(dim=-1) / (target_ft.abs().norm(dim=-1) + 1e-8)).mean()
    return loss


@register_loss("h1_freq")
def h1_freq_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """H1 + 频域混合 loss。"""
    return 0.5 * h1_loss(pred, target) + 0.5 * freq_loss(pred, target)
