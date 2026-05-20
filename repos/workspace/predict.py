import argparse
import time
import numpy as np
import torch
import h5py

from model import DeepONet
from dataset import load_eval_data, load_test_data

PRED_PATH = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/submission/task1_pred.hdf5"


def eval_scores(pred, gt):
    """计算 Seg1/Seg2/Seg3 得分。pred, gt: (N, 200, 256)"""
    p = pred[:, 10:, :]
    g = gt[:,  10:, :]

    def rel_mse(a, b):
        num = ((a - b) ** 2).sum(-1)
        den = (b ** 2).sum(-1) + 1e-6
        return float(np.clip(num / den, 0, 5).mean())

    r1 = rel_mse(p[:, :47],   g[:, :47])
    r2 = rel_mse(p[:, 47:95], g[:, 47:95])
    s1 = 100 * np.exp(-20 * r1)
    s2 = 100 * np.exp(-10 * r2)

    seg3_p, seg3_g = p[:, 95:], g[:, 95:]
    rmse  = float(np.sqrt(((seg3_p - seg3_g) ** 2).mean()))
    fd    = float(np.linalg.norm(seg3_p - seg3_g, axis=-1).max(axis=1).mean())
    s3    = max(100 / (1 + 10 * rmse), 50 * np.exp(-fd ** 2))
    total = 0.25 * s1 + 0.25 * s2 + 0.5 * s3

    print(f"Seg1 Score={s1:.2f}")
    print(f"Seg2 Score={s2:.2f}")
    print(f"Seg3 Score={s3:.2f}")
    print(f"预测得分: {total:.4f} / 100")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     required=True)
    parser.add_argument("--val_only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.ckpt, map_location=device, weights_only=False)
    model  = DeepONet(
        latent_dim   = ckpt.get("latent_dim",   128),
        branch_type  = ckpt.get("branch_type",  "cnn"),
        branch_depth = ckpt.get("branch_depth", 4),
        branch_width = ckpt.get("branch_width", 128),
        trunk_depth  = ckpt.get("trunk_depth",  4),
        trunk_width  = ckpt.get("trunk_width",  256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"模型加载成功 val_loss={ckpt.get('val_loss', 'N/A')}")

    val_gt   = load_eval_data()
    val_init = torch.tensor(val_gt[:, :10, :], dtype=torch.float32).to(device)

    t0         = time.time()
    val_pred   = model.predict_full(val_init)
    infer_time = time.time() - t0

    result = np.zeros((20, 200, 256), dtype=np.float32)
    result[:, :10, :] = val_gt[:, :10, :]
    result[:, 10:, :] = val_pred
    eval_scores(result, val_gt)
    print(f"推理完成 {infer_time:.2f}s")

    if args.val_only:
        return

    test_data = load_test_data()
    test_init = torch.tensor(test_data, dtype=torch.float32).to(device)

    t0         = time.time()
    test_pred  = model.predict_full(test_init)
    infer_time = time.time() - t0
    print(f"推理完成 {infer_time:.2f}s (1000样本)")

    pred_submit = np.zeros((1000, 200, 256), dtype=np.float32)
    pred_submit[:, :10, :] = test_data
    pred_submit[:, 10:, :] = test_pred

    with h5py.File(PRED_PATH, "w") as f:
        f.create_dataset("tensor", data=pred_submit)
    print(f"预测结果: {PRED_PATH}")


if __name__ == "__main__":
    main()
