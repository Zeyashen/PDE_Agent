

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch

from .config import ExperimentConfig
from .metrics import compute_segment_scores
from .model import load_ckpt


@dataclass
class PredictResult:
    status: str
    pred_path: Optional[Path]
    inference_time: float
    has_gt: bool
    metrics: Optional[Dict[str, float]] = None
    init_diff_max: float = 0.0
    error_msg: str = ""


@torch.no_grad()
def _deeponet_predict(
    model: torch.nn.Module,
    init_data: torch.Tensor,
    device: torch.device,
    total_steps: int,
    batch_size: int,
    t_scale: float = 10.0,
) -> torch.Tensor:
    """DeepONet 直接预测任意 (x,t) 坐标处的解。
    
    init_data: (N, T_init, X)
    返回: (N, total_steps, X)
    """
    N, T_init, X = init_data.shape
    u0 = init_data[:, 0, :].to(device)  # (N, X) 初始条件

    # 构建查询坐标网格
    t_steps = total_steps
    x_coords = torch.linspace(0, 1, X, device=device)       # (X,)
    t_coords = torch.linspace(0, 1, t_steps, device=device) # (T,) 归一化

    pred_full = torch.zeros(N, total_steps, X, device=device)
    model.eval()

    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            u0_b = u0[s:e]  # (b, X)
            b = e - s

            # 对每个时间步预测所有空间点
            for t_idx in range(total_steps):
                t_norm = t_coords[t_idx].item()
                # xt: (b, X, 2)
                xt = torch.stack([
                    torch.full((b, X), t_norm, device=device),
                    x_coords.unsqueeze(0).expand(b, -1)
                ], dim=-1)  # (b, X, 2)
                s_pred = model(u0_b, xt)  # (b, X)
                pred_full[s:e, t_idx, :] = s_pred

    # 只用真实初始条件（前10步），不覆盖预测部分
    init_steps = min(10, T_init)
    pred_full[:, :init_steps, :] = init_data[:, :init_steps, :].to(device)
    return pred_full


def _autoregressive_predict(
    model: torch.nn.Module,
    init_data: torch.Tensor,
    device: torch.device,
    total_steps: int,
    window_size: int,
    use_mean_correction: bool,
    batch_size: int,
) -> torch.Tensor:
    """真实轴自回归。

    init_data: (N, T_init, X)
        T_init=10   提交模式
        T_init=200  本地评估模式
    返回 (N, total_steps, X)。
    """
    N, T_init, X = init_data.shape
    if T_init < window_size:
        raise ValueError(
            f"init_data 时间步 {T_init} < window_size {window_size}"
        )

    grid = torch.linspace(0, 1, X, device=device).view(1, 1, X)
    pred_full = torch.zeros(N, total_steps, X, device=device)

    fill = min(T_init, total_steps)
    pred_full[:, :fill, :] = init_data[:, :fill, :].to(device)

    model.eval()
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        b = e - s
        grid_b = grid.expand(b, -1, -1)

        window = init_data[s:e, :window_size, :].to(device).clone()
        win_mean_ref = window.mean(dim=[1, 2], keepdim=True)

        for t in range(window_size, total_steps):
            inp = torch.cat([window, grid_b], dim=1)
            out = model(inp)[:, 0:1, :]

            if use_mean_correction:
                pred_mean = out.mean(dim=-1, keepdim=True)
                out = out - pred_mean + win_mean_ref

            pred_full[s:e, t, :] = out[:, 0, :]
            window = torch.cat([window[:, 1:, :], out], dim=1)

    pred_full[:, :window_size, :] = init_data[:, :window_size, :].to(device)
    return pred_full


def predict(
    cfg: ExperimentConfig,
    ckpt_path: Path,
    out_dir: Path,
    logger: logging.Logger,
) -> PredictResult:
    """跑推理，写 hdf5 和 (可选) metrics.json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "task1_pred.hdf5"
    metrics_path = out_dir / "metrics.json"

    if not cfg.inference.input_path:
        input_path = Path(cfg.data.test_hdf5_path)
    else:
        input_path = Path(cfg.inference.input_path)
    if not input_path.exists():
        return PredictResult(
            status="failed", pred_path=None, inference_time=0,
            has_gt=False, error_msg=f"input_path 不存在: {input_path}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info(f"[predict] exp_id={cfg.exp_id}")
    logger.info(f"[predict] ckpt={ckpt_path}")
    logger.info(f"[predict] input={input_path}")
    logger.info(f"[predict] device={device}")

    try:
        model, meta = load_ckpt(Path(ckpt_path), device)
    except Exception as e:
        return PredictResult(
            status="failed", pred_path=None, inference_time=0,
            has_gt=False, error_msg=f"加载 ckpt 失败: {e}",
        )
    logger.info(
        f"[predict] arch={meta.get('architecture')}  "
        f"trained_val_loss={meta.get('val_loss')}"
    )

    with h5py.File(input_path, "r") as f:
        keys = list(f.keys())
        data_key = "tensor" if "tensor" in keys else [
            k for k in keys if len(f[k].shape) == 3
        ][0]
        val_np_all = f[data_key][:].astype(np.float32)
    # 只用后20条（未参与训练）做本地评估，前80条是训练数据
    val_np = val_np_all[80:] if val_np_all.shape[0] == 100 else val_np_all

    if val_np.shape[-1] == 1024 and cfg.data.reduced_resolution > 1:
        val_np = val_np[:, :, ::cfg.data.reduced_resolution]

    N, T_init, X = val_np.shape
    has_gt = T_init >= 200
    logger.info(
        f"[predict] input shape={val_np.shape}  "
        f"mode={'本地评估' if has_gt else '提交'}"
    )

    window_size = cfg.model.window_size
    if T_init < window_size:
        return PredictResult(
            status="failed", pred_path=None, inference_time=0,
            has_gt=has_gt, error_msg=(
                f"Input time step {T_init} < window_size {window_size}"
            ),
        )

    val_tensor = torch.from_numpy(val_np)
    t0 = time.time()
    try:
        if cfg.model.architecture == "DeepONet1d":
            pred_tensor = _deeponet_predict(
                model,
                init_data=val_tensor,
                device=device,
                total_steps=cfg.inference.total_steps,
                batch_size=cfg.inference.infer_batch,
                t_scale=10.0,
            )
        else:
            pred_tensor = _autoregressive_predict(
                model,
                init_data=val_tensor,
                device=device,
                total_steps=cfg.inference.total_steps,
                window_size=window_size,
            use_mean_correction=cfg.inference.use_mean_correction,
            batch_size=cfg.inference.infer_batch,
        )
    except Exception as e:
        return PredictResult(
            status="failed", pred_path=None, inference_time=time.time()-t0,
            has_gt=has_gt, error_msg=f"自回归失败: {e}",
        )
    infer_time = time.time() - t0
    logger.info(f"[predict] inference {infer_time:.2f}s /120s")
    if infer_time > 120:
        logger.error(f"[predict] not in limitation.")

    pred_np = pred_tensor.cpu().numpy()
    init_diff = float(np.abs(
        pred_np[:, :window_size, :] - val_np[:, :window_size, :]
    ).max())
    logger.info(f"[predict] 前 {window_size} 步最大误差: {init_diff:.2e} (上限 1e-3)")
    if init_diff > 1e-3:
        logger.warning("[predict]  前 window 步超容差！")

    expected = (N, cfg.inference.total_steps, X)
    if pred_np.shape != expected:
        return PredictResult(
            status="failed", pred_path=None, inference_time=infer_time,
            has_gt=has_gt, error_msg=(
                f"pred shape {pred_np.shape} != expected {expected}"
            ),
        )

    metrics: Optional[Dict[str, float]] = None
    if has_gt:
        metrics = compute_segment_scores(pred_np, val_np[:, :200, :])
        logger.info("-" * 40)
        logger.info(
            f"Seg1 [10:57]   rel_mse={metrics['seg1_rel_mse']:.6f}  "
            f"score={metrics['seg1_score']:.2f}"
        )
        logger.info(
            f"Seg2 [57:105]  rel_mse={metrics['seg2_rel_mse']:.6f}  "
            f"score={metrics['seg2_score']:.2f}"
        )
        logger.info(
            f"Seg3 [105:200] rmse={metrics['seg3_rmse']:.6f}  "
            f"lor={metrics['seg3_lorentzian']:.2f}  "
            f"frech={metrics['seg3_frechet']:.2f}  "
            f" {metrics['seg3_score']:.2f}"
        )
        logger.info(f"total_score: {metrics['total_score']:.4f} / 100")
        logger.info("-" * 40)
        metrics_path.write_text(json.dumps(metrics, indent=2))

    with h5py.File(pred_path, "w") as f:
        f.create_dataset("tensor", data=pred_np)
    logger.info(f"[predict] 写出 {pred_path}  shape={pred_np.shape}")
    logger.info("=" * 60)

    return PredictResult(
        status="completed",
        pred_path=pred_path,
        inference_time=infer_time,
        has_gt=has_gt,
        metrics=metrics,
        init_diff_max=init_diff,
    )
