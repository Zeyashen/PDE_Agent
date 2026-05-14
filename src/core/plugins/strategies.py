"""训练策略注册表。

策略 = "如何从一个 batch 计算 loss"。

- vanilla:     单步预测，标准 supervised
- pushforward: 滚动 K 步，每步监督 + 权重衰减

LLM 通过 training.strategy_name 切换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


STRATEGY_REGISTRY: Dict[str, Callable] = {}


@dataclass
class StrategyContext:
    """传给策略函数的上下文。"""
    model: nn.Module
    inp: torch.Tensor              # (B, W+1, X)
    target: torch.Tensor           # (B, 1, X)
    sample_n: torch.Tensor         # (B,)
    sample_t: torch.Tensor         # (B,)
    dataset_data: object           # BurgersDataset — 用于取 future target
    window_size: int
    pushforward_steps: int = 0
    pushforward_weight: float = 0.5
    ics_weight: float = 20.0
    loss_fn: Callable = None


def register_strategy(name: str):
    def deco(fn):
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy {name} already registered")
        STRATEGY_REGISTRY[name] = fn
        return fn
    return deco


def get_strategy(name: str) -> Callable:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]


# ===================== 注册 =====================
@register_strategy("vanilla")
def vanilla_step(ctx: StrategyContext) -> torch.Tensor:
    """单步 supervised。"""
    pred = ctx.model(ctx.inp)
    return ctx.loss_fn(pred, ctx.target)


@register_strategy("pushforward")
def pushforward_step(ctx: StrategyContext) -> torch.Tensor:
    """Pushforward: 滚 K 步，每步监督。

    权重 0.5^(K-k-1)，越近权重越大。
    任何样本越过时间轴时停止整个 batch 的 pushforward。
    """
    pred = ctx.model(ctx.inp)
    loss = ctx.loss_fn(pred, ctx.target)

    if ctx.pushforward_steps <= 0:
        return loss

    T_max = ctx.dataset_data.data.shape[1]
    window_pf = ctx.inp.clone()
    pred_pf = pred
    device = ctx.inp.device

    for k in range(ctx.pushforward_steps):
        window_pf = torch.cat(
            [
                window_pf[:, 1 : ctx.window_size, :],
                pred_pf,
                window_pf[:, ctx.window_size :, :],
            ],
            dim=1,
        )

        future_t = ctx.sample_t + k + 1
        if (future_t >= T_max).any():
            break

        # 检查每个样本的 future_t 是否在有效范围内
        t_valid = ctx.dataset_data.t_valid if hasattr(ctx.dataset_data, 't_valid') else None
        valid_mask = torch.ones(len(future_t), dtype=torch.bool)
        if t_valid is not None:
            for i, (n, t) in enumerate(zip(ctx.sample_n.tolist(), future_t.tolist())):
                if t >= t_valid[n]:
                    valid_mask[i] = False
        if not valid_mask.all():
            break

        future_target = torch.stack(
            [
                ctx.dataset_data.data[n, t, :]
                for n, t in zip(
                    ctx.sample_n.tolist(),
                    future_t.tolist(),
                )
            ]
        ).unsqueeze(1).to(device)

        pred_pf = ctx.model(window_pf)
        step_w = ctx.pushforward_weight * (
            0.5 ** (ctx.pushforward_steps - k - 1)
        )
        loss = loss + step_w * ctx.loss_fn(pred_pf, future_target)

    return loss


@register_strategy("pi_deeponet")
def pi_deeponet_step(ctx: StrategyContext) -> torch.Tensor:
    """PI-DeepONet: loss_ics + loss_bcs + loss_res
    
    ctx.inp 在 DeepONet 模式下被重用为:
        ctx.inp[:, 0, :]  → u0 (B, X) 初始条件
        ctx.inp[:, 1:, :] → xt_data (B, N, 2) 数据查询坐标
    ctx.target → s_data (B, N) 真实解
    
    另外从 dataset_data 中随机采样 ics/bcs/res 点
    """
    device = ctx.inp.device
    B = ctx.inp.shape[0]

    # 解包输入
    u0    = ctx.inp[:, 0, :]          # (B, X)
    xt    = ctx.inp[:, 1:, :2]        # (B, N, 2) 查询坐标 (t_norm, x_norm)
    s_gt  = ctx.target[:, :, 0]       # (B, N) 真实解

    # ── 数据损失 ──────────────────────────────────────────────
    s_pred = ctx.model(u0, xt)        # (B, N)
    loss_data = ((s_pred - s_gt) ** 2).mean()

    # ── ICS 损失: t=0 处预测应等于 u0 ────────────────────────
    X = u0.shape[1]
    x_ics = torch.linspace(0, 1, X, device=device).unsqueeze(0).expand(B, -1)  # (B, X)
    t_ics = torch.zeros(B, X, device=device)
    xt_ics = torch.stack([t_ics, x_ics], dim=-1)   # (B, X, 2)
    s_ics_pred = ctx.model(u0, xt_ics)              # (B, X)
    loss_ics = ((s_ics_pred - u0) ** 2).mean()

    # ── BCS 损失: 周期边界 u(0,t) = u(1,t) ───────────────────
    N_bc = 50
    t_bc = torch.rand(B, N_bc, device=device)
    xt_bc1 = torch.stack([t_bc, torch.zeros_like(t_bc)], dim=-1)   # x=0
    xt_bc2 = torch.stack([t_bc, torch.ones_like(t_bc)], dim=-1)    # x=1
    s_bc1 = ctx.model(u0, xt_bc1)
    s_bc2 = ctx.model(u0, xt_bc2)
    loss_bcs = ((s_bc1 - s_bc2) ** 2).mean()

    # ── PDE 残差损失: u_t + u*u_x - nu*u_xx = 0 ──────────────
    nu = 0.001
    N_res = 100
    t_res = torch.rand(B, N_res, device=device, requires_grad=False)
    x_res = torch.rand(B, N_res, device=device, requires_grad=False)
    xt_res = torch.stack([t_res, x_res], dim=-1).requires_grad_(True)

    s_res = ctx.model(u0, xt_res)   # (B, N_res)

    # 对 xt_res 求梯度
    grads = torch.autograd.grad(
        s_res.sum(), xt_res, create_graph=True
    )[0]   # (B, N_res, 2)
    s_t = grads[..., 0]   # ∂u/∂t_norm, 需要乘以 t_scale
    s_x = grads[..., 1]   # ∂u/∂x

    # t 是归一化的，真实时间 = t_norm * t_scale，所以 ∂/∂t_real = ∂/∂t_norm / t_scale
    t_scale = ctx.dataset_data.t_scale if hasattr(ctx.dataset_data, 't_scale') else 2.0
    s_t_real = s_t / t_scale

    # 二阶导数
    s_xx = torch.autograd.grad(
        s_x.sum(), xt_res, create_graph=True
    )[0][..., 1]   # (B, N_res)

    residual = s_t_real + s_res * s_x - nu * s_xx
    loss_res = (residual ** 2).mean()

    # ── 总损失 ────────────────────────────────────────────────
    ics_w = ctx.ics_weight if ctx.ics_weight else 20.0
    loss = ics_w * loss_ics + loss_bcs + loss_res + loss_data
    return loss
