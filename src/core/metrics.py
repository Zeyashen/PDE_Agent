"""赛题评分公式。

总分 = 0.25*seg1 + 0.25*seg2 + 0.5*seg3，范围 0-100。

时间分段 (针对去掉前 10 步的 190 个预测步):
    Seg1: 索引 [10, 57)   长度 47   权重 25%
    Seg2: 索引 [57, 105)  长度 48   权重 25%
    Seg3: 索引 [105, 200) 长度 95   权重 50%

得分公式:
    Seg1 = 100 * exp(-20 * rel_mse)
    Seg2 = 100 * exp(-10 * rel_mse)
    Seg3 = max(Lorentzian, Frechet)
        Lorentzian = 100 / (1 + 10 * RMSE)
        Frechet    = 50 * exp(-FD²)
"""

from __future__ import annotations

from typing import Dict

import numpy as np


_EPS = 1e-6


def rel_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred, gt: (N, T_seg, X)。"""
    diff = pred - gt
    num = (diff ** 2).sum(axis=-1)
    den = (gt ** 2).sum(axis=-1) + _EPS
    per_sample = (num / den).mean(axis=1)
    per_sample = np.clip(per_sample, None, 5.0)
    return float(per_sample.mean())


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(((pred - gt) ** 2).mean()))


def frechet_proxy(pred: np.ndarray, gt: np.ndarray) -> float:
    """离散 Frechet 距离的轻量代理。"""
    per_sample = np.sqrt(((pred - gt) ** 2).mean(axis=(1, 2)))
    return float(per_sample.mean())


def compute_segment_scores(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """完整评分。pred/gt: (N, 200, X)。"""
    seg1_p, seg1_g = pred[:, 10:57, :], gt[:, 10:57, :]
    seg2_p, seg2_g = pred[:, 57:105, :], gt[:, 57:105, :]
    seg3_p, seg3_g = pred[:, 105:200, :], gt[:, 105:200, :]

    r1 = rel_mse(seg1_p, seg1_g)
    r2 = rel_mse(seg2_p, seg2_g)
    r3_rmse = rmse(seg3_p, seg3_g)
    r3_fd = frechet_proxy(seg3_p, seg3_g)

    s1 = 100.0 * np.exp(-20 * r1)
    s2 = 100.0 * np.exp(-10 * r2)
    lorentzian = 100.0 / (1 + 10 * r3_rmse)
    frechet = 50.0 * np.exp(-(r3_fd ** 2))
    s3 = max(lorentzian, frechet)

    total = 0.25 * s1 + 0.25 * s2 + 0.5 * s3

    return {
        "seg1_rel_mse": r1,
        "seg1_score": float(s1),
        "seg2_rel_mse": r2,
        "seg2_score": float(s2),
        "seg3_rmse": r3_rmse,
        "seg3_fd": r3_fd,
        "seg3_lorentzian": float(lorentzian),
        "seg3_frechet": float(frechet),
        "seg3_score": float(s3),
        "total_score": float(total),
    }
