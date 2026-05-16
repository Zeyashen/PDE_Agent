"""
predict_t2.py — Task 2 推理与评分

评分逻辑与 Task 1 完全一致（同一套 eval_scores）。
推理：DeepONet 范式A，直接查询 t=0.65~10.60s 的190步。

注意：
  - val 有200个预测步 (t=0.65~10.60s)，评分取后190步
    (与Task1保持一致：去掉第一个预测步)
  - test 输出 (1000, 200, 256)：前10步=初始条件，后190步=预测
"""

import argparse, csv, sys, time, logging
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT    = Path(__file__).parent.parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model
from dataset_t2 import (Task2ValDataset, _time_normalize,
                         T_INIT_START, T_PRED_END, T_NORM_RANGE)


def setup_logger(log_path):
    logger = logging.getLogger("predict_t2")
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
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt.get('cfg', {})
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info(f"Task2 模型: branch={cfg.get('branch_type')} "
                f"depth={cfg.get('branch_depth')} "
                f"width={cfg.get('branch_width')} "
                f"latent={cfg.get('latent_dim')} "
                f"val_loss={ckpt.get('val_loss',0):.6f}")
    return model


# ============================================================
# 评分函数（与 Task 1 完全相同）
# ============================================================
def _rel_mse(pp, gg):
    eps = 1e-6
    num = ((pp - gg)**2).sum(axis=-1)
    den = (gg**2).sum(axis=-1) + eps
    return float(np.clip(num/den, 0, 5.0).mean(axis=1).mean())


def _rmse(pp, gg):
    return float(np.sqrt(((pp - gg)**2).mean()))


def _frechet_aligned(P, Q):
    return float(np.linalg.norm(P - Q, axis=-1).max(axis=1).mean())


def eval_scores(pred, gt):
    """
    pred, gt: (N, 200, 256)
    评分步 10-199（去掉前10步初始条件）
    """
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
        'seg1_score': float(s1), 'seg2_score': float(s2),
        'seg3_score': float(s3), 'total_score': float(total),
        'seg3_winner': winner,
        'n_val_samples': int(pred.shape[0]),
    }


# ============================================================
# 推理：DeepONet 范式A
# ============================================================
def deeponet_predict(model, init_data, device,
                     t_query_abs, x_points=256):
    """
    init_data  : (N, 10, 256) 初始条件
    t_query_abs: 需要查询的绝对时间坐标数组
    返回       : (N, len(t_query), 256)
    """
    N      = init_data.shape[0]
    T_q    = len(t_query_abs)
    result = np.zeros((N, T_q, x_points), dtype=np.float32)

    t_norm = torch.tensor(_time_normalize(t_query_abs),
                          dtype=torch.float32, device=device)
    x_norm = torch.linspace(0, 1, x_points, device=device)
    tt, xx = torch.meshgrid(t_norm, x_norm, indexing='ij')
    coords_all = torch.stack([tt.flatten(), xx.flatten()], dim=-1)
    coords_all = coords_all.unsqueeze(0).expand(N, -1, -1)

    total_q  = T_q * x_points
    batch_q  = x_points * 50

    with torch.no_grad():
        pred_flat = torch.zeros(N, total_q, device=device)
        for start in range(0, total_q, batch_q):
            end = min(start + batch_q, total_q)
            pred_flat[:, start:end] = model(
                init_data, coords_all[:, start:end, :])

    result = pred_flat.cpu().numpy().reshape(N, T_q, x_points)
    return result


def predict(args):
    out_dir  = ROOT / "repos" / "submission"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "task2_predict.log"
    time_csv = out_dir / "task2_time.csv"

    logger = setup_logger(log_path)
    logger.info("=" * 60)
    logger.info(f"Task 2 推理 (DeepONet 范式A)")
    logger.info(f"Checkpoint: {args.ckpt}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(args.ckpt, device, logger)

    metrics    = {'total_score': 0.0}
    infer_time = 0.0

    # ── Val 评分（后20条）────────────────────────────────────
    val_path = Path(args.val_path)
    if val_path.exists():
        with h5py.File(val_path, 'r') as f:
            val_all = f['tensor'][:]      # (100, 210, 256)
            val_t   = f['t_coordinate'][:]

        val_eval = val_all[args.val_train_n:]   # (20, 210, 256)
        val_init = torch.tensor(val_eval[:, :10, :],
                                dtype=torch.float32).to(device)
        logger.info(f"验证集: 第{args.val_train_n}-99号样本（{val_eval.shape[0]}条）")

        # 预测时间：val 的第10步之后（t=0.65~10.60s，200步）
        t_query = val_t[10:]   # (200,)

        t0 = time.time()
        pred_future = deeponet_predict(model, val_init, device, t_query)
        infer_time  = time.time() - t0
        logger.info(f"推理完成 {infer_time:.2f}s")

        # 拼接：前10步初始条件 + 200步预测 = (N, 210, 256)
        # 评分只取后200步中的190步（去掉第一个预测步与Task1保持一致）
        # 构建 (N, 200, 256) 供 eval_scores 使用
        # 前10步填初始条件，后190步从pred_future[1:]取（跳过第一个预测步）
        val_for_score = np.zeros((val_eval.shape[0], 200, 256), dtype=np.float32)
        val_for_score[:, :10, :] = val_eval[:, :10, :]
        val_for_score[:, 10:, :] = pred_future[:, 10:10+190, :]  # 跳过第0个预测步

        gt_for_score  = np.zeros_like(val_for_score)
        gt_for_score[:, :10, :] = val_eval[:, :10, :]
        gt_for_score[:, 10:, :] = val_eval[:, 10+1:10+191, :]    # 对应的GT

        metrics = eval_scores(val_for_score, gt_for_score)

        logger.info("-" * 40)
        logger.info(f"样本数: {metrics['n_val_samples']} (官方用1000条)")
        logger.info(f"Seg1 Score={metrics['seg1_score']:.2f}")
        logger.info(f"Seg2 Score={metrics['seg2_score']:.2f}")
        logger.info(f"Seg3 Score={metrics['seg3_score']:.2f} "
                    f"({metrics['seg3_winner']})")
        logger.info(f"预测得分: {metrics['total_score']:.4f} / 100")
        logger.info("-" * 40)
    else:
        logger.warning(f"验证集不存在: {val_path}")

    if args.val_only:
        return None, infer_time, metrics

    # ── 测试集推理 ────────────────────────────────────────────
    test_path = Path(args.test_path)
    if not test_path.exists():
        logger.warning(f"测试集不存在: {test_path}")
        return None, infer_time, metrics

    with h5py.File(test_path, 'r') as f:
        test_np = f['tensor'][:]    # (1000, 10, 256)
        test_t  = f['t_coordinate'][:]

    N = test_np.shape[0]
    logger.info(f"测试集: {test_np.shape}")
    test_init = torch.tensor(test_np, dtype=torch.float32).to(device)

    # 查询时间：t=0.65~10.60s，步长0.05，共190步
    t_query_test = np.arange(0.65, 10.61, 0.05)[:190]

    t0 = time.time()
    pred_future = deeponet_predict(model, test_init, device, t_query_test)
    infer_time  = time.time() - t0
    logger.info(f"推理完成 {infer_time:.2f}s ({N}样本)")

    # 构建提交文件：(1000, 200, 256)
    # 前10步 = 初始条件，后190步 = 模型预测
    pred_submit = np.zeros((N, 200, 256), dtype=np.float32)
    pred_submit[:, :10, :]  = test_np
    pred_submit[:, 10:, :]  = pred_future

    # 验证前10步一致性
    diff   = np.abs(pred_submit[:, :10, :] - test_np).max()
    status = "pass" if diff < 1e-3 else "FAIL"
    logger.info(f"前10步一致性: 最大误差={diff:.2e} [{status}]")

    pred_path = out_dir / "task2_pred.hdf5"
    with h5py.File(pred_path, 'w') as f:
        f.create_dataset('tensor', data=pred_submit)
    logger.info(f"预测结果: {pred_path}  shape={pred_submit.shape}")

    # 更新 task2_time.csv
    with open(time_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['train_time', 'inference_time'])
        w.writerow([0, infer_time])  # Task2 训练时间不计入评测
    logger.info(f"推理时间: {infer_time:.2f}s")
    logger.info("=" * 60)
    return pred_path, infer_time, metrics


def parse_args():
    root = ROOT
    DATA = root / "data_and_sample_submission" / "train_val_test_init"
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',        required=True)
    parser.add_argument('--val_path',    default=str(DATA / 'task2_val.h5'))
    parser.add_argument('--test_path',   default=str(DATA / 'task2_test.h5'))
    parser.add_argument('--val_train_n', type=int, default=80)
    parser.add_argument('--val_only',    action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred_path, infer_time, metrics = predict(args)
    print(f"\n{'='*50}")
    print(f"Task2 总分: {metrics.get('total_score',0):.4f} / 100")
    print(f"Seg1={metrics.get('seg1_score',0):.2f}  "
          f"Seg2={metrics.get('seg2_score',0):.2f}  "
          f"Seg3={metrics.get('seg3_score',0):.2f}")
    print(f"推理耗时: {infer_time:.2f}s")
    print(f"{'='*50}")