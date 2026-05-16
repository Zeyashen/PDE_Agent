"""
predict.py — DeepONet Task-1 推理与评分

支持两种模型：
  DeepONet (范式A): 直接查询任意时刻，无自回归
  FNO baseline:    自回归推理，用于初始标尺对比
"""

import argparse
import csv
import sys
import time
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT    = Path(__file__).parent.parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))


def setup_logger(log_path):
    logger = logging.getLogger("predict")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ============================================================
# 模型加载（自动检测 DeepONet 或 FNO）
# ============================================================
class _SpectralConv1d(torch.nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        self.weights1 = torch.nn.Parameter(
            (1/(in_ch*out_ch)) *
            torch.rand(in_ch, out_ch, modes, dtype=torch.cfloat))
    def forward(self, x):
        B = x.shape[0]
        xf = torch.fft.rfft(x)
        out = torch.zeros(B, self.weights1.shape[1], x.shape[-1]//2+1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", xf[:, :, :self.modes], self.weights1)
        return torch.fft.irfft(out, n=x.shape[-1])

class _FNO1d(torch.nn.Module):
    def __init__(self, modes, width, in_ch=11, out_ch=1, n_layers=4):
        super().__init__()
        self.fc0   = torch.nn.Linear(in_ch, width)
        self.convs = torch.nn.ModuleList(
            [_SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.ws    = torch.nn.ModuleList(
            [torch.nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.fc1   = torch.nn.Linear(width, 128)
        self.fc2   = torch.nn.Linear(128, out_ch)
        self.n_layers = n_layers
    def forward(self, x):
        import torch.nn.functional as F
        x = self.fc0(x.permute(0,2,1)).permute(0,2,1)
        for i,(c,w) in enumerate(zip(self.convs, self.ws)):
            x = c(x) + w(x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
        x = F.gelu(self.fc1(x.permute(0,2,1)))
        return self.fc2(x).permute(0,2,1)


def load_model(ckpt_path, device, logger):
    import re as _re
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    keys       = list(state_dict.keys())
    is_official = any(_re.match(r"^conv\d+\.", k) for k in keys)
    is_our_fno  = ("args" in ckpt and "modes" in ckpt.get("args", {}))

    if is_official or is_our_fno:
        if is_official:
            new_sd = {}
            for k, v in state_dict.items():
                k2 = _re.sub(r"^conv(\d+)\.", lambda m: f"convs.{m.group(1)}.", k)
                k2 = _re.sub(r"^w(\d+)\.",    lambda m: f"ws.{m.group(1)}.",    k2)
                new_sd[k2] = v
            modes    = new_sd["convs.0.weights1"].shape[-1]
            width    = new_sd["fc0.weight"].shape[0]
            n_layers = sum(1 for k in new_sd if _re.match(r"^convs\.\d+\.weights1$", k))
            sd = new_sd
        else:
            saved    = ckpt["args"]
            modes    = saved["modes"]
            width    = saved["width"]
            n_layers = saved["n_layers"]
            sd = state_dict
        model = _FNO1d(modes, width, n_layers=n_layers).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        logger.info(f"FNO: modes={modes} width={width} n_layers={n_layers} [仅用于评分对齐]")
        return model, "fno"

    from model import build_model
    cfg   = ckpt.get("cfg", {})
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"DeepONet: branch={cfg.get('branch_type','?')} "
                f"latent={cfg.get('latent_dim','?')} "
                f"val_loss={ckpt.get('val_loss',0):.6f}")
    return model, "deeponet"


def deeponet_predict_full(model, init_data, device, t_steps=190, x_points=256):
    """DeepONet 范式A：直接查询全量时空点，分批处理防OOM。"""
    N = init_data.shape[0]
    result = np.zeros((N, 10 + t_steps, x_points), dtype=np.float32)
    result[:, :10, :] = init_data.cpu().numpy()

    t_norm = torch.linspace(0, 1, t_steps, device=device)
    x_norm = torch.linspace(0, 1, x_points, device=device)
    tt, xx = torch.meshgrid(t_norm, x_norm, indexing='ij')
    coords_all = torch.stack([tt.flatten(), xx.flatten()], dim=-1)
    coords_all = coords_all.unsqueeze(0).expand(N, -1, -1)

    total_q    = t_steps * x_points
    batch_q    = x_points * 50   # 每次处理50个时间步
    pred_flat  = torch.zeros(N, total_q, device=device)

    with torch.no_grad():
        for start in range(0, total_q, batch_q):
            end = min(start + batch_q, total_q)
            pred_flat[:, start:end] = model(init_data, coords_all[:, start:end, :])

    result[:, 10:, :] = pred_flat.cpu().numpy().reshape(N, t_steps, x_points)
    return result


def fno_predict_full(model, init_data, device, n_pred=190):
    """FNO 自回归推理。"""
    N, T_init, X = init_data.shape
    grid   = torch.linspace(0, 1, X, device=device).view(1, 1, X).expand(N, -1, -1)
    result = torch.zeros(N, T_init + n_pred, X, device=device)
    result[:, :T_init, :] = init_data
    window = init_data.clone()

    with torch.no_grad():
        for t in range(n_pred):
            inp  = torch.cat([window, grid], dim=1)
            pred = model(inp)[:, 0, :]
            result[:, T_init + t, :] = pred
            window = torch.cat([window[:, 1:, :], pred.unsqueeze(1)], dim=1)

    return result.cpu().numpy()


# ============================================================
# 评分函数
# ============================================================
def _rel_mse(pp, gg):
    eps = 1e-6
    num = ((pp - gg)**2).sum(axis=-1)
    den = (gg**2).sum(axis=-1) + eps
    return float(np.clip(num/den, 0, 5.0).mean(axis=1).mean())


def _rmse(pp, gg):
    return float(np.sqrt(((pp-gg)**2).mean()))


def _frechet_aligned(P, Q):
    return float(np.linalg.norm(P - Q, axis=-1).max(axis=1).mean())


def eval_scores(pred, gt):
    p = pred[:, 10:, :]
    g = gt[:,  10:, :]

    r1 = _rel_mse(p[:, :47,  :], g[:, :47,  :])
    s1 = 100.0 * np.exp(-20.0 * r1)
    r2 = _rel_mse(p[:, 47:95,:], g[:, 47:95,:])
    s2 = 100.0 * np.exp(-10.0 * r2)

    seg3_p = p[:, 95:, :]
    seg3_g = g[:, 95:, :]
    r3     = _rmse(seg3_p, seg3_g)
    loren  = 100.0 / (1.0 + 10.0 * r3)
    fd     = _frechet_aligned(seg3_p, seg3_g)
    frech  = 50.0 * np.exp(-(fd**2))
    s3     = max(loren, frech)
    winner = 'lorentzian' if loren >= frech else 'frechet'
    total  = 0.25*s1 + 0.25*s2 + 0.5*s3

    return {
        'seg1_rel_mse': r1, 'seg1_score': float(s1),
        'seg2_rel_mse': r2, 'seg2_score': float(s2),
        'seg3_rmse':    r3, 'seg3_fd':    fd,
        'seg3_lorentzian': float(loren),
        'seg3_frechet':    float(frech),
        'seg3_winner':     winner,
        'seg3_score':      float(s3),
        'total_score':     float(total),
        'n_val_samples':   int(pred.shape[0]),
    }


def compute_error_timeseries(pred, gt):
    p   = pred[:, 10:, :]
    g   = gt[:,  10:, :]
    eps = 1e-6
    num = ((p - g)**2).sum(axis=-1)
    den = (g**2).sum(axis=-1) + eps
    rel = np.clip(num/den, 0, 5.0)
    step_err = rel.mean(axis=0).tolist()
    diffs    = np.diff(step_err)
    infl     = int(np.argmax(diffs)) + 10 if len(diffs) > 0 else 10
    t2s      = float(np.mean(step_err[:40]))
    t10s     = float(np.mean(step_err[40:]))
    return {
        'error_by_step':       [round(v, 6) for v in step_err],
        'inflection_step':     infl,
        't2s_mean_error':      round(t2s, 6),
        't10s_mean_error':     round(t10s, 6),
        'extrapolation_ratio': round(t10s / (t2s + 1e-9), 2),
    }


# ============================================================
# 主函数
# ============================================================
def predict(args):
    out_dir  = ROOT / "repos" / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "task1_predict.log"
    time_csv = out_dir / "task1_time.csv"

    logger = setup_logger(log_path)
    logger.info("=" * 60)
    logger.info(f"Task 1 推理  ckpt={args.ckpt}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")

    model, model_type = load_model(args.ckpt, device, logger)

    metrics    = {'total_score': 0.0}
    diag       = {}
    infer_time = 0.0

    # ── 验证集评分 ────────────────────────────────────────────
    val_path = Path(args.val_path)
    if val_path.exists():
        with h5py.File(val_path, 'r') as f:
            keys    = list(f.keys())
            val_all = f['tensor' if 'tensor' in keys else keys[0]][:]

        val_eval = val_all[args.val_train_n:]
        val_init = torch.tensor(val_eval[:, :10, :],
                                dtype=torch.float32).to(device)
        logger.info(f"验证集: 第{args.val_train_n}-99号样本（{val_eval.shape[0]}条）"
                    f" 模型类型: {model_type}")

        t0 = time.time()
        if model_type == 'deeponet':
            val_pred = deeponet_predict_full(model, val_init, device)
        else:
            val_pred = fno_predict_full(model, val_init, device)
        infer_time = time.time() - t0

        metrics = eval_scores(val_pred, val_eval)
        diag    = compute_error_timeseries(val_pred, val_eval)

        logger.info("-" * 40)
        logger.info(f"样本数: {metrics['n_val_samples']} "
                    f"(官方用1000条，存在统计偏差)")
        logger.info(f"Seg1 Rel-MSE={metrics['seg1_rel_mse']:.4f}  "
                    f"Score={metrics['seg1_score']:.2f}")
        logger.info(f"Seg2 Rel-MSE={metrics['seg2_rel_mse']:.4f}  "
                    f"Score={metrics['seg2_score']:.2f}")
        logger.info(f"Seg3 RMSE={metrics['seg3_rmse']:.4f}  "
                    f"FD={metrics['seg3_fd']:.4f}  "
                    f"Lorentzian={metrics['seg3_lorentzian']:.2f}  "
                    f"Frechet={metrics['seg3_frechet']:.2f}  "
                    f"-> {metrics['seg3_winner']}={metrics['seg3_score']:.2f}")
        logger.info(f"预测得分: {metrics['total_score']:.4f} / 100")
        logger.info(f"诊断: 训练覆盖区误差={diag['t2s_mean_error']:.4f}  "
                    f"外推区误差={diag['t10s_mean_error']:.4f}  "
                    f"外推/覆盖比={diag['extrapolation_ratio']:.1f}x  "
                    f"拐点步={diag['inflection_step']}")
        logger.info("-" * 40)
        metrics.update(diag)
    else:
        logger.warning(f"验证集不存在: {val_path}")

    if args.val_only:
        return None, infer_time, metrics

    # ── 测试集推理（生成提交文件）────────────────────────────
    test_path = Path(args.test_path)
    if not test_path.exists():
        logger.warning(f"测试集不存在: {test_path}")
        return None, infer_time, metrics

    with h5py.File(test_path, 'r') as f:
        keys    = list(f.keys())
        test_np = f['tensor' if 'tensor' in keys else keys[0]][:]

    N         = test_np.shape[0]
    test_init = torch.tensor(test_np, dtype=torch.float32).to(device)

    t0 = time.time()
    if model_type == 'deeponet':
        test_pred = deeponet_predict_full(model, test_init, device)
    else:
        test_pred = fno_predict_full(model, test_init, device)
    infer_time = time.time() - t0

    logger.info(f"测试集推理完成 {infer_time:.2f}s ({N}样本)")

    diff   = np.abs(test_pred[:, :10, :] - test_np).max()
    status = "pass" if diff < 1e-3 else "FAIL"
    logger.info(f"前10步一致性: 最大误差={diff:.2e} [{status}]")

    pred_path = out_dir / "task1_pred.hdf5"
    with h5py.File(pred_path, 'w') as f:
        f.create_dataset('tensor', data=test_pred.astype(np.float32))
    logger.info(f"预测结果: {pred_path}  shape={test_pred.shape}")

    # 更新计时文件（保留已有 train_time，更新 inference_time）
    train_time = 0.0
    if time_csv.exists():
        with open(time_csv, 'r') as f:
            for row in csv.DictReader(f):
                train_time = float(row.get('train_time', 0))
    with open(time_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['train_time', 'inference_time'])
        w.writerow([train_time, infer_time])
    logger.info(f"计时: train={train_time:.1f}s  infer={infer_time:.2f}s")
    logger.info("=" * 60)
    return pred_path, infer_time, metrics


def parse_args():
    root = Path(__file__).parent.parent.parent
    DATA = root / "data_and_sample_submission" / "train_val_test_init"
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',        required=True)
    parser.add_argument('--val_path',    default=str(DATA / 'task1_val.hdf5'))
    parser.add_argument('--test_path',   default=str(DATA / 'task1_test.hdf5'))
    parser.add_argument('--val_train_n', type=int, default=80)
    parser.add_argument('--val_only',    action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred_path, infer_time, metrics = predict(args)
    print(f"\n{'='*50}")
    print(f"总分: {metrics.get('total_score',0):.4f} / 100")
    print(f"Seg1={metrics.get('seg1_score',0):.2f}  "
          f"Seg2={metrics.get('seg2_score',0):.2f}  "
          f"Seg3={metrics.get('seg3_score',0):.2f}")
    print(f"推理耗时: {infer_time:.2f}s")
    print(f"{'='*50}")