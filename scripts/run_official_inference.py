#!/usr/bin/env python3
"""
官方权重纯推理脚本，不做任何训练，只跑验证集得分。
"""
import sys
import re
import json
import time
import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]   # guandi_agent/
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ── 模型（内联，避免 import 路径问题）────────────────────────────────

class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_ch * out_ch)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.weights1.shape[1], x.size(-1)//2+1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights1)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    def __init__(self, modes=12, width=20, in_ch=11, out_ch=1, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.fc0   = nn.Linear(in_ch, width)
        self.convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.ws    = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.fc1   = nn.Linear(width, 128)
        self.fc2   = nn.Linear(128, out_ch)

    def forward(self, x):
        x = self.fc0(x.permute(0,2,1)).permute(0,2,1)
        for i in range(self.n_layers):
            x = self.convs[i](x) + self.ws[i](x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
        x = F.gelu(self.fc1(x.permute(0,2,1)))
        return self.fc2(x).permute(0,2,1)


def load_official_ckpt(ckpt_path, device):
    """官方 key 映射: conv0->convs.0, w0->ws.0"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt.get("model_state_dict", ckpt)
    new_sd = {}
    for k, v in sd.items():
        k2 = re.sub(r'^conv(\d+)\.', lambda m: f'convs.{m.group(1)}.', k)
        k2 = re.sub(r'^w(\d+)\.',    lambda m: f'ws.{m.group(1)}.',    k2)
        new_sd[k2] = v
    out_ch = new_sd['fc2.weight'].shape[0]
    model  = FNO1d(modes=12, width=20, in_ch=11, out_ch=out_ch, n_layers=4).to(device)
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    return model


# ── 推理 ─────────────────────────────────────────────────────────────

def autoregressive_predict(model, init_data, device, n_pred=190, use_mean_correction=True):
    N, T0, X = init_data.shape
    grid   = torch.linspace(0, 1, X, device=device).view(1,1,X).expand(N,-1,-1)
    result = torch.zeros(N, T0+n_pred, X, device=device)
    result[:, :T0, :] = init_data
    window = init_data.clone()
    # 记录初始窗口的均值作为参考
    win_mean_ref = window.mean(dim=[1,2], keepdim=True)  # (N,1,1)
    with torch.no_grad():
        for t in range(n_pred):
            inp  = torch.cat([window, grid], dim=1)
            out  = model(inp)[:, 0:1, :]   # (N,1,X)
            if use_mean_correction:
                pred_mean = out.mean(dim=-1, keepdim=True)  # (N,1,1)
                out = out - pred_mean + win_mean_ref
            pred = out[:, 0, :]
            result[:, T0+t, :] = pred
            window = torch.cat([window[:, 1:, :], pred.unsqueeze(1)], dim=1)
    return result


# ── 评分 ─────────────────────────────────────────────────────────────

def _rel_mse(pp, gg):
    eps = 1e-6
    num = ((pp-gg)**2).sum(axis=-1)
    den = (gg**2).sum(axis=-1) + eps
    return float(np.clip(num/den, 0.0, 5.0).mean(axis=1).mean())

def _rmse(pp, gg):
    return float(np.sqrt(((pp-gg)**2).mean()))

def _frechet(P, Q):
    return float(np.linalg.norm(P-Q, axis=-1).max(axis=1).mean())

def eval_scores(pred, gt):
    p = pred[:, 10:, :];  g = gt[:, 10:, :]
    r1 = _rel_mse(p[:,:47,:],  g[:,:47,:]);  s1 = 100.0*np.exp(-20.0*r1)
    r2 = _rel_mse(p[:,47:95,:],g[:,47:95,:]); s2 = 100.0*np.exp(-10.0*r2)
    r3   = _rmse(p[:,95:,:], g[:,95:,:])
    lore = 100.0/(1.0+10.0*r3)
    fd   = _frechet(p[:,95:,:], g[:,95:,:])
    frec = 50.0*np.exp(-(fd**2))
    s3   = max(lore, frec)
    return {
        'seg1_score': float(s1), 'seg1_rel_mse': r1,
        'seg2_score': float(s2), 'seg2_rel_mse': r2,
        'seg3_score': float(s3), 'seg3_rmse': r3,
        'seg3_lorentzian': float(lore), 'seg3_frechet': float(frec),
        'seg3_winner': 'lorentzian' if lore>=frec else 'frechet',
        'total_score': float(0.25*s1+0.25*s2+0.5*s3),
        'n_val_samples': int(pred.shape[0]),
    }


# ── 主函数 ───────────────────────────────────────────────────────────

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR/"official_inference.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("official_infer")

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir   = ROOT / "runs" / "_official" / "v0_official"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"官方权重推理  ckpt={args.ckpt}  device={device}")

    model  = load_official_ckpt(args.ckpt, device)
    n_p    = sum(p.numel() for p in model.parameters())
    logger.info(f"参数量: {n_p:,}  (modes=12 width=20 n_layers=4)")

    with h5py.File(args.val_path, 'r') as f:
        keys     = list(f.keys())
        data_key = 'tensor' if 'tensor' in keys else keys[0]
        val_all  = f[data_key][:].astype(np.float32)   # (100, 200, 256)

    val_eval = val_all[args.val_train_n:]
    logger.info(f"验证集: 第{args.val_train_n}-99号，共{val_eval.shape[0]}条")

    val_init = torch.tensor(val_eval[:, :10, :], dtype=torch.float32).to(device)
    t0       = time.time()
    val_pred = autoregressive_predict(model, val_init, device, n_pred=190, use_mean_correction=True)
    elapsed  = time.time() - t0
    logger.info(f"推理耗时: {elapsed:.2f}s")

    pred_np = val_pred.cpu().numpy()
    scores  = eval_scores(pred_np, val_eval)

    logger.info("-" * 40)
    logger.info(f"Seg1  {scores['seg1_score']:.2f}  Seg2  {scores['seg2_score']:.2f}  "
                f"Seg3  {scores['seg3_score']:.2f}  Total  {scores['total_score']:.4f}")
    logger.info("-" * 40)

    result = {
        "exp_id": "v0_official",
        "ckpt_path": args.ckpt,
        "inference_time": elapsed,
        "metrics": scores,
    }
    score_file = out_dir / "scores.json"
    with open(score_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"结果写入: {score_file}")
    logger.info("=" * 60)
    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=str(
        ROOT.parent / 'checkpoints/1D_Burgers/1D_Burgers_Sols_Nu0.001_FNO.pt'))
    parser.add_argument('--val_path', default=str(
        ROOT.parent / 'data_and_sample_submission/train_val_test_init/task1_val.hdf5'))
    parser.add_argument('--val_train_n', type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    scores = main(parse_args())
    print(f"\n官方基线总分: {scores['total_score']:.4f} / 100")
