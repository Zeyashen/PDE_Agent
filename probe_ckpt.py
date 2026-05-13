# diagnose_ckpt.py —— 运行这个，把输出贴给我
import torch
import h5py
import numpy as np

ckpt_path = 'checkpoints/1D_Burgers/1D_Burgers_Sols_Nu0.001_FNO.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt.get('model_state_dict', ckpt)

print("=== Checkpoint 所有权重 shape ===")
for k, v in state_dict.items():
    print(f"  {k}: {v.shape}")

print("\n=== 关键信息 ===")
print(f"fc0 输入维度 (期望通道数): {state_dict['fc0.weight'].shape}")
print(f"fc2 输出维度 (out_channels): {state_dict['fc2.weight'].shape[0]}")
print(f"conv0.weights1 shape: {state_dict['conv0.weights1'].shape}")

# 检查数据维度
val_path = 'data_and_sample_submission/train_val_test_init/task1_val.hdf5'
with h5py.File(val_path, 'r') as f:
    keys = list(f.keys())
    print(f"\n=== 验证数据 ===")
    for k in keys:
        print(f"  key='{k}', shape={f[k].shape}, dtype={f[k].dtype}")