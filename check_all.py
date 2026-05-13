import h5py

# 把你要查的 4 个文件路径全放进列表
files_to_check = [
    "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5",
    "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_test.hdf5",
    "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/submission/task1_pred.hdf5",
    "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_val.hdf5"
]

print("📡 [全景雷达] 正在扫描 HDF5 数据集...\n")

for file_path in files_to_check:
    # 只提取文件名，方便查看
    file_name = file_path.split('/')[-1]
    print(f"========================================")
    print(f"📂 文件: {file_name}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            print(f"  🔑 Keys: {keys}")
            for key in keys:
                shape = f[key].shape
                dtype = f[key].dtype
                print(f"  📐 '{key}' -> Shape: {shape}, Dtype: {dtype}")
    except FileNotFoundError:
        print("  ❌ 找不到该文件！(请检查路径)")
    except Exception as e:
        print(f"  ❌ 读取出错: {e}")
    print("\n")



import h5py
import numpy as np

# 我们只查这三个有时间坐标的文件（去掉了你生成的预测文件，因为它没有 t-coordinate）
files_to_check = [
    "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5",
    "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_test.hdf5",
    "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_val.hdf5"
]

print("⏱️ [时间刻度侦察] 正在提取物理时间步长 (Delta t)...\n")

for file_path in files_to_check:
    file_name = file_path.split('/')[-1]
    print(f"========================================")
    print(f"📂 {file_name}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            if 't-coordinate' in f.keys():
                t_data = f['t-coordinate'][:]
                
                # 打印前 5 个具体的时间刻度
                print(f"  🕒 前 5 个时间点: {t_data[:5]}")
                
                # 计算物理步长 Delta t (取相邻时间点的差值的平均数)
                if len(t_data) > 1:
                    diffs = np.diff(t_data)
                    dt = np.mean(diffs)
                    print(f"  📏 物理时间步长 (Delta t): {dt:.6f}")
                    print(f"  ⏱️ 物理时间跨度: t={t_data[0]:.4f} 到 t={t_data[-1]:.4f}")
                else:
                    print("  ⚠️ 时间点不足，无法计算 Delta t")
            else:
                print("  ❌ 未找到 't-coordinate'")
    except Exception as e:
        print(f"  ❌ 读取出错: {e}")
    print()



import h5py
import numpy as np

files_to_check = [
    "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5",
    "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_test.hdf5",
    "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_val.hdf5"
]

print("📏 [空间刻度侦察] 正在提取物理空间步长 (Delta x)...\n")

for file_path in files_to_check:
    file_name = file_path.split('/')[-1]
    print(f"========================================")
    print(f"📂 {file_name}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            if 'x-coordinate' in f.keys():
                x_data = f['x-coordinate'][:]
                
                print(f"  📍 空间网格点数: {len(x_data)}")
                print(f"  📏 前 5 个坐标点: {x_data[:5]}")
                
                if len(x_data) > 1:
                    diffs = np.diff(x_data)
                    dx = np.mean(diffs)
                    print(f"  📐 物理空间步长 (Delta x): {dx:.6f}")
                    print(f"  📏 物理空间跨度: x={x_data[0]:.4f} 到 x={x_data[-1]:.4f}")
                else:
                    print("  ⚠️ 空间点不足，无法计算 Delta x")
            else:
                print("  ❌ 未找到 'x-coordinate'")
    except Exception as e:
        print(f"  ❌ 读取出错: {e}")
    print()

import h5py
import numpy as np
f = h5py.File('/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5', 'r')
t = f['t-coordinate'][:]
print("前5个t:", t[:5])   # 应该是 [0, 0.01, 0.02, 0.03, 0.04]
print("t[0]:", t[0])      # 是否是 t=0