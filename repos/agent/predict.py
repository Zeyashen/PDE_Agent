"""
Task 1 推理脚本

验证集评分（Agent内部监控）：
  使用 task1_val.hdf5 的后20条（第80-99号样本），含完整200步GT。
  输出 Seg1/2/3 得分 + 误差时序曲线（供 orchestrator 诊断）。

测试集推理（生成提交文件）：
  使用 task1_test.hdf5（1000条，只有前10步初值），自回归推理190步。
  输出 task1_pred.hdf5 (1000, 200, 256)。

用法:
    python predict.py              # 两者都做
    python predict.py --val_only   # 只做验证集评分
"""

import argparse
import csv
import sys
import time
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import FNO1d

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


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


def load_model(ckpt_path, device, logger):
    import re
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 检测是否是官方 PDEBench checkpoint（key 命名为 conv0/w0）
    state_dict = ckpt.get('model_state_dict', ckpt)
    keys = list(state_dict.keys())
    is_official = any(re.match(r'^conv\d+\.', k) for k in keys)

    if is_official:
        # 官方 key 重映射：conv0 → convs.0，w0 → ws.0
        new_sd = {}
        for k, v in state_dict.items():
            k2 = re.sub(r'^conv(\d+)\.', lambda m: f'convs.{m.group(1)}.', k)
            k2 = re.sub(r'^w(\d+)\.',    lambda m: f'ws.{m.group(1)}.',    k2)
            new_sd[k2] = v
        state_dict = new_sd
        # 从 state_dict 反推架构
        modes = state_dict['convs.0.weights1'].shape[-1]
        width = state_dict['fc0.weight'].shape[0]
        n_layers = sum(1 for k in state_dict if re.match(r'^convs\.\d+\.weights1$', k))
        logger.info(f"官方 checkpoint 检测到，架构反推: modes={modes} width={width} n_layers={n_layers}")
    else:
        # 我们自己训练的 checkpoint，从 args 读取架构
        saved    = ckpt.get('args', {})
        modes    = saved.get('modes',    16)
        width    = saved.get('width',    64)
        n_layers = saved.get('n_layers',  4)

    model = FNO1d(
        modes=modes, width=width,
        in_channels=11, out_channels=1,
        n_layers=n_layers,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    val_loss = ckpt.get('val_loss', 0) if not is_official else float('nan')
    logger.info(
        f"模型: modes={modes} width={width} n_layers={n_layers} "
        f"val_loss={val_loss:.6f}" if not is_official else
        f"模型: modes={modes} width={width} n_layers={n_layers} [官方checkpoint]"
    )
    return model


def autoregressive_predict(model, init_data, device, n_pred=190):
    """
    init_data: (N, 10, 256)
    返回:      (N, 200, 256)
    """
    N, T_init, X = init_data.shape
    grid   = torch.linspace(0, 1, X, device=device).view(1, 1, X).expand(N, -1, -1)
    result = torch.zeros(N, T_init + n_pred, X, device=device)
    result[:, :T_init, :] = init_data
    window = init_data.clone()

    with torch.no_grad():
        for t in range(n_pred):
            inp  = torch.cat([window, grid], dim=1)   # (N, 11, 256)
            pred = model(inp)[:, 0, :]                # (N, 256)
            result[:, T_init + t, :] = pred
            window = torch.cat([window[:, 1:, :], pred.unsqueeze(1)], dim=1)

    return result


# ============================================================
# 评分函数（完整实现官方公式）
# ============================================================

def _rel_mse(pp, gg):
    """
    逐样本逐时间步 rel_t = sum(pred-gt)^2 / sum(gt^2)，
    时间均值（单样本上限5.0），再样本均值。
    pp, gg: (N, T, X)
    """
    eps = 1e-6
    num = (( pp - gg)**2).sum(axis=-1)
    den = (gg**2).sum(axis=-1) + eps
    return float(np.clip(num / den, 0.0, 5.0).mean(axis=1).mean())


def _rmse(pp, gg):
    return float(np.sqrt(((pp - gg)**2).mean()))


def _frechet_aligned(P, Q):
    """
    对时间对齐的等长序列，Discrete Frechet Distance = max_t ||P_t - Q_t||_2
    这是精确等式（适用于 autoregressive 推理的对齐输出）。
    P, Q: (N, T, X)  返回: 样本均值 FD
    """
    pointwise = np.linalg.norm(P - Q, axis=-1)   # (N, T)
    return float(pointwise.max(axis=1).mean())


def eval_scores(pred, gt):
    """
    官方评分公式（完整版）。
    pred, gt: (N, 200, 256)
    评分仅针对步 10-199（去掉前10步初始条件）。

    Seg1(10-57,  47步, 25%): 100 * exp(-20 * Rel-MSE)
    Seg2(57-105, 48步, 25%): 100 * exp(-10 * Rel-MSE)
    Seg3(105-200,95步, 50%): max(Lorentzian, Frechet)
        Lorentzian = 100 / (1 + 10 * RMSE)
        Frechet    = 50  * exp(-FD^2)
    total = 0.25*s1 + 0.25*s2 + 0.5*s3
    """
    p = pred[:, 10:, :]   # (N, 190, 256)
    g = gt[:,  10:, :]

    # Seg1
    r1 = _rel_mse(p[:, :47,  :], g[:, :47,  :])
    s1 = 100.0 * np.exp(-20.0 * r1)

    # Seg2
    r2 = _rel_mse(p[:, 47:95,:], g[:, 47:95,:])
    s2 = 100.0 * np.exp(-10.0 * r2)

    # Seg3
    seg3_p     = p[:, 95:, :]
    seg3_g     = g[:, 95:, :]
    r3         = _rmse(seg3_p, seg3_g)
    lorentzian = 100.0 / (1.0 + 10.0 * r3)
    fd         = _frechet_aligned(seg3_p, seg3_g)
    frechet    = 50.0 * np.exp(-(fd**2))
    s3         = max(lorentzian, frechet)
    winner     = 'lorentzian' if lorentzian >= frechet else 'frechet'

    total = 0.25*s1 + 0.25*s2 + 0.5*s3

    return {
        'seg1_rel_mse': r1, 'seg1_score': float(s1),
        'seg2_rel_mse': r2, 'seg2_score': float(s2),
        'seg3_rmse':    r3,
        'seg3_fd':      fd,
        'seg3_lorentzian': float(lorentzian),
        'seg3_frechet':    float(frechet),
        'seg3_winner':     winner,
        'seg3_score':   float(s3),
        'total_score':  float(total),
        'n_val_samples': int(pred.shape[0]),
    }


def compute_error_timeseries(pred, gt):
    """
    计算逐步误差曲线，供 orchestrator 诊断用。
    pred, gt: (N, 200, 256)
    返回:
        error_by_step  : list[float], 长度190，每步平均 Rel-MSE
        inflection_step: int，误差增长率最大的步（拐点）
        t2s_mean_error : float，训练覆盖区(步10-50)均值误差
        t10s_mean_error: float，外推区(步50-200)均值误差
    """
    p = pred[:, 10:, :]   # (N, 190, X)
    g = gt[:,  10:, :]
    eps = 1e-6
    num  = ((p - g)**2).sum(axis=-1)          # (N, 190)
    den  = (g**2).sum(axis=-1) + eps
    rel  = np.clip(num / den, 0, 5.0)
    step_err = rel.mean(axis=0).tolist()      # (190,)

    # 拐点：一阶差分最大处
    diffs = np.diff(step_err)
    inflection = int(np.argmax(diffs)) + 1 if len(diffs) > 0 else 0

    # 训练覆盖区 vs 外推区
    # 训练集覆盖 t<1.95s，dt=0.05 → 前39步（步10~48相对于原始步）
    # 在190步（步10-200）中，前~40步对应训练覆盖区
    t2s_mean  = float(np.mean(step_err[:40]))   # 步10-50，t<2s
    t10s_mean = float(np.mean(step_err[40:]))   # 步50-200，t=2~10s

    return {
        'error_by_step':   [round(v, 6) for v in step_err],
        'inflection_step': inflection + 10,   # 转成绝对步编号
        't2s_mean_error':  round(t2s_mean,  6),
        't10s_mean_error': round(t10s_mean, 6),
        'extrapolation_ratio': round(t10s_mean / (t2s_mean + 1e-9), 2),
    }


# ============================================================
# 主函数
# ============================================================

def predict(args):
    root      = Path(__file__).parent.parent.parent
    ckpt_path = Path(args.ckpt)
    out_dir   = root / "repos" / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path  = LOG_DIR / "task1_predict.log"
    time_csv  = out_dir / "task1_time.csv"

    logger = setup_logger(log_path)
    logger.info("=" * 60)
    logger.info("Task 1 推理开始")
    logger.info(f"Checkpoint: {ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")

    model = load_model(ckpt_path, device, logger)

    metrics    = {'total_score': 0.0}
    diag       = {}
    infer_time = 0.0

    # ── 1. 验证集评分（val后20条，步80-99）──────────────────
    val_path = Path(args.val_path)
    if val_path.exists():
        with h5py.File(val_path, 'r') as f:
            keys     = list(f.keys())
            data_key = 'tensor' if 'tensor' in keys else keys[0]
            val_all  = f[data_key][:]   # (100, 200, 256)

        # 后20条用于评分（前80条已被训练集使用）
        val_eval = val_all[args.val_train_n:]   # (20, 200, 256)
        logger.info(
            f"验证集: 使用第{args.val_train_n}-99号样本（共{val_eval.shape[0]}条）"
        )

        val_init = torch.tensor(val_eval[:, :10, :], dtype=torch.float32).to(device)
        t0       = time.time()
        val_pred = autoregressive_predict(model, val_init, device, n_pred=190)
        infer_time_val = time.time() - t0

        val_pred_np = val_pred.cpu().numpy()
        metrics     = eval_scores(val_pred_np, val_eval)
        diag        = compute_error_timeseries(val_pred_np, val_eval)

        logger.info("-" * 40)
        logger.info(
            f"注意：本地评分基于{metrics['n_val_samples']}条样本，"
            f"官方用1000条，存在统计偏差"
        )
        logger.info(f"Seg1 Rel-MSE={metrics['seg1_rel_mse']:.4f}  Score={metrics['seg1_score']:.2f}")
        logger.info(f"Seg2 Rel-MSE={metrics['seg2_rel_mse']:.4f}  Score={metrics['seg2_score']:.2f}")
        logger.info(
            f"Seg3 RMSE={metrics['seg3_rmse']:.4f}  FD={metrics['seg3_fd']:.4f}  "
            f"Lorentzian={metrics['seg3_lorentzian']:.2f}  "
            f"Frechet={metrics['seg3_frechet']:.2f}  "
            f"-> {metrics['seg3_winner']}={metrics['seg3_score']:.2f}"
        )
        logger.info(f"预测得分: {metrics['total_score']:.4f} / 100")
        logger.info(
            f"诊断: 训练覆盖区误差={diag['t2s_mean_error']:.4f}  "
            f"外推区误差={diag['t10s_mean_error']:.4f}  "
            f"外推/覆盖比={diag['extrapolation_ratio']:.1f}x  "
            f"拐点步={diag['inflection_step']}"
        )
        logger.info("-" * 40)

        # 把诊断信息合并到 metrics（供 orchestrator 读取）
        metrics.update(diag)
        infer_time = infer_time_val
    else:
        logger.warning(f"验证集不存在: {val_path}")

    if args.val_only:
        return None, infer_time, metrics

    # ── 2. 测试集推理（生成提交文件）───────────────────────
    test_path = Path(args.test_path)
    if not test_path.exists():
        logger.warning(f"测试集不存在: {test_path}，跳过")
        return None, infer_time, metrics

    with h5py.File(test_path, 'r') as f:
        keys     = list(f.keys())
        data_key = 'tensor' if 'tensor' in keys else keys[0]
        test_np  = f[data_key][:]   # (1000, 10, 256)

    N = test_np.shape[0]
    logger.info(f"测试集: {test_np.shape}")

    test_init = torch.tensor(test_np, dtype=torch.float32).to(device)
    t0        = time.time()
    test_pred = autoregressive_predict(model, test_init, device, n_pred=190)
    infer_time = time.time() - t0
    logger.info(f"推理完成，耗时: {infer_time:.2f}s ({N}样本)")

    pred_np = test_pred.cpu().numpy()   # (1000, 200, 256)

    # 前10步一致性检查（官方容差 1e-3）
    diff   = np.abs(pred_np[:, :10, :] - test_np).max()
    status = "pass" if diff < 1e-3 else "FAIL"
    logger.info(f"前10步一致性: 最大误差={diff:.2e} [{status}]")

    pred_path = out_dir / "task1_pred.hdf5"
    with h5py.File(pred_path, 'w') as f:
        f.create_dataset('tensor', data=pred_np)
    logger.info(f"预测结果: {pred_path}  shape={pred_np.shape}")

    # 更新计时文件
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
    parser.add_argument('--ckpt', default=str(
        root / 'checkpoints/task1_trained/best_model.pt'))
    parser.add_argument('--val_path',     default=str(DATA / 'task1_val.hdf5'))
    parser.add_argument('--test_path',    default=str(DATA / 'task1_test.hdf5'))
    parser.add_argument('--val_train_n',  type=int, default=80,
                        help='与训练时一致：前N条给训练，后(100-N)条用于评分')
    parser.add_argument('--val_only',     action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred_path, infer_time, metrics = predict(args)
    print(f"\n{'='*40}")
    print(f"预测得分: {metrics.get('total_score', 0):.4f} / 100")
    if 'seg3_winner' in metrics:
        print(f"Seg3 {metrics['seg3_winner']}={metrics['seg3_score']:.2f} "
              f"(L={metrics['seg3_lorentzian']:.2f} F={metrics['seg3_frechet']:.2f})")
    if 'extrapolation_ratio' in metrics:
        print(f"外推误差比: {metrics['extrapolation_ratio']:.1f}x "
              f"(拐点步={metrics.get('inflection_step','?')})")
    print(f"{'='*40}")