import h5py
import numpy as np


def calculate_rel_mse(pred, gt):
    """
    官方 Rel-MSE 实现：

    1. 逐样本、逐时间步：
        rel_t = Σ(pred-gt)^2 / Σ(gt^2)

    2. 对时间维取平均

    3. 单样本截断 max=5.0

    4. 对 batch 取平均
    """

    eps = 1e-6

    # [B, T, D]
    diff = pred - gt

    # [B, T]
    numerator = np.sum(diff ** 2, axis=-1)
    denominator = np.sum(gt ** 2, axis=-1) + eps

    rel_t = numerator / denominator

    # 每个样本先对时间平均
    # [B]
    rel_sample = np.mean(rel_t, axis=1)

    # 官方单样本上限截断
    rel_sample = np.clip(rel_sample, a_min=None, a_max=5.0)

    # batch平均
    rel_mse = np.mean(rel_sample)

    return rel_mse


def calculate_rmse(pred, gt):
    """
    第三段 Lorentzian 使用 RMSE
    """

    mse = np.mean((pred - gt) ** 2)
    rmse = np.sqrt(mse)

    return rmse


def evaluate_prediction_score(pred_file, gt_file):
    """
    只计算：
        分段预测得分（0~100）

    不包含：
        - Task1 时间得分
        - Task2 总分
        - Frechet Distance

    第三段暂时仅使用 Lorentzian。
    """

    print("📊 开始计算预测精度得分...")

    # ----------------------------
    # 读取预测
    # ----------------------------
    with h5py.File(pred_file, "r") as f:
        pred_data = f["tensor"][:]

    # ----------------------------
    # 读取GT
    # ----------------------------
    with h5py.File(gt_file, "r") as f:
        keys = list(f.keys())

        if "tensor" in keys:
            gt_key = "tensor"
        else:
            gt_key = [k for k in keys if len(f[k].shape) == 3][0]

        gt_data = f[gt_key][:]

    # ----------------------------
    # Shape检查
    # ----------------------------
    if pred_data.shape != gt_data.shape:
        raise ValueError(
            f"❌ Shape不匹配: pred={pred_data.shape}, gt={gt_data.shape}"
        )

    print(f"✅ 数据加载成功: {pred_data.shape}")

    # ==========================================================
    # 官方规则：
    # 去掉前10步
    #
    # 190步:
    #   段1: 47
    #   段2: 48
    #   段3: 95
    # ==========================================================

    # 第1段
    pred_seg1 = pred_data[:, 10:57, :]
    gt_seg1 = gt_data[:, 10:57, :]

    # 第2段
    pred_seg2 = pred_data[:, 57:105, :]
    gt_seg2 = gt_data[:, 57:105, :]

    # 第3段
    pred_seg3 = pred_data[:, 105:200, :]
    gt_seg3 = gt_data[:, 105:200, :]

    # ==========================================================
    # Segment 1
    # ==========================================================

    rel_mse1 = calculate_rel_mse(pred_seg1, gt_seg1)

    score1 = 100 * np.exp(-20 * rel_mse1)

    # ==========================================================
    # Segment 2
    # ==========================================================

    rel_mse2 = calculate_rel_mse(pred_seg2, gt_seg2)

    score2 = 100 * np.exp(-10 * rel_mse2)

    # ==========================================================
    # Segment 3
    # 暂时只实现 Lorentzian
    # ==========================================================

    rmse3 = calculate_rmse(pred_seg3, gt_seg3)

    lorentzian_score = 100 / (1 + 10 * rmse3)

    score3 = lorentzian_score

    # ==========================================================
    # 最终预测得分
    # ==========================================================

    prediction_score = (
        0.25 * score1
        + 0.25 * score2
        + 0.5 * score3
    )

    # ==========================================================
    # 输出
    # ==========================================================

    print("\n================ 评测结果 ================")

    print(f"Segment 1 Rel-MSE : {rel_mse1:.6f}")
    print(f"Segment 1 Score   : {score1:.4f}")

    print()

    print(f"Segment 2 Rel-MSE : {rel_mse2:.6f}")
    print(f"Segment 2 Score   : {score2:.4f}")

    print()

    print(f"Segment 3 RMSE    : {rmse3:.6f}")
    print(f"Lorentzian Score  : {score3:.4f}")

    print()

    print(f"🔥 Prediction Score: {prediction_score:.4f} / 100")

    print("==========================================\n")

    return prediction_score


if __name__ == "__main__":

    pred_path = "repos/submission/task1_pred.hdf5"
    gt_path = "data_and_sample_submission/train_val_test_init/task1_val.hdf5"

    evaluate_prediction_score(pred_path, gt_path)

# import h5py
# import numpy as np
# import argparse

# def calculate_rel_mse(pred, gt):
#     """
#     计算相对均方误差 (Rel-MSE)
#     公式: Σ(pred-gt)² / Σ(gt²)
#     """
#     # 避免分母为0的极其微小的偏置
#     eps = 1e-6
#     diff = pred - gt
    
#     numerator = np.sum(diff**2, axis=-1)  # 对空间维度求和
#     denominator = np.sum(gt**2, axis=-1) + eps
    
#     rel_mse = numerator / denominator
    
#     # 官方规则：单样本上限截断为 5.0
#     rel_mse = np.clip(rel_mse, a_min=None, a_max=5.0)
    
#     # 对时间步和样本取均值
#     return np.mean(rel_mse)

# def evaluate_task1(pred_file, gt_file):
#     print("📊 开始执行官方标准评测...")
    
#     # 1. 读取我们的预测结果 (我们在 train.py 里明确写了叫 'tensor')
#     with h5py.File(pred_file, 'r') as f:
#         pred_data = f['tensor'][:]
        
#     # 2. 智能读取官方真实数据 (绕过坐标轴，直击 3D 核心)
#     with h5py.File(gt_file, 'r') as f:
#         keys = list(f.keys())
#         gt_key = 'tensor' if 'tensor' in keys else [k for k in keys if len(f[k].shape) == 3][0]
#         gt_data = f[gt_key][:]
        
#     # (保留你原本代码里的这行及后续内容)
#     if pred_data.shape != gt_data.shape:
#         raise ValueError(f"❌ 维度不匹配! 预测: {pred_data.shape}, 真实: {gt_data.shape}")
        
#     print(f"✅ 数据加载成功，Shape: {pred_data.shape}")

#     # 官方规则陷阱 1：前 10 步不计入算分
#     # 官方规则陷阱 2：分段计算 (总共 190 步，分为 47, 48, 95 步)
#     # 索引偏移 10: 段1 (10:57), 段2 (57:105), 段3 (105:200)
    
#     pred_seg1 = pred_data[:, 10:57, :]
#     gt_seg1 = gt_data[:, 10:57, :]
    
#     pred_seg2 = pred_data[:, 57:105, :]
#     gt_seg2 = gt_data[:, 57:105, :]
    
#     pred_seg3 = pred_data[:, 105:200, :]
#     gt_seg3 = gt_data[:, 105:200, :]

#     # 计算各段 Rel-MSE
#     mse1 = calculate_rel_mse(pred_seg1, gt_seg1)
#     mse2 = calculate_rel_mse(pred_seg2, gt_seg2)
#     mse3 = calculate_rel_mse(pred_seg3, gt_seg3)

#     # 计算最终得分 (这里简化处理了段3，如果要求严格的 Lorentzian/Frechet 可在此扩展)
#     score1 = 100 * np.exp(-20 * mse1)
#     score2 = 100 * np.exp(-10 * mse2)
#     score3 = 100 * np.exp(-10 * mse3) # 暂用 Rel-MSE 替代演示

#     total_score = 0.25 * score1 + 0.25 * score2 + 0.5 * score3

#     print(f"--- 评测报告 ---")
#     print(f"段1 (0-47)   Rel-MSE: {mse1:.4f} -> 得分: {score1:.2f}")
#     print(f"段2 (47-95)  Rel-MSE: {mse2:.4f} -> 得分: {score2:.2f}")
#     print(f"段3 (95-190) Rel-MSE: {mse3:.4f} -> 得分: {score3:.2f}")
#     print(f"🔥 Task 1 综合预测得分: {total_score:.2f} / 100")
    
#     return total_score

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pred", type=str, required=True, help="Agent跑出来的预测结果hdf5")
#     parser.add_argument("--gt", type=str, required=True, help="官方验证集task1_val.hdf5")
#     # 将原来的 args = parser.parse_args() 替换为下面这段：
#     args = parser.parse_args([
#         '--pred', 'repos/submission/task1_pred.hdf5',
#         '--gt', 'data_and_sample_submission/train_val_test_init/task1_val.hdf5'
#     ])
    
#     evaluate_task1(args.pred, args.gt)