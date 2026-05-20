# 数据格式说明

## Task 1

### 训练数据
路径: /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5
Key: 'tensor'
原始 shape: (10000, 201, 1024)
读取方式（下采样）:
  data = f['tensor'][:, 1::5, ::4]  # → (10000, 40, 256)，跳过t=0与val/test时间对齐
  t范围: 0~1.95s，dt=0.05s（下采样后）
  Nu 固定: 0.001

### 验证数据
路径: /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_val.hdf5
Key: 'tensor'
shape: (100, 200, 256)
t范围: 0.01~9.96s，dt=0.05s
使用策略:
  前80条 → 可混入训练（覆盖长时动力学，解决分布外推问题）
  后20条 → 本地评分（Agent 的 reward 信号）

### 测试数据
路径: /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_test.hdf5
Key: 'tensor'
shape: (1000, 10, 256)
t范围: 0.01~0.46s（只有初始10步）
用途: 最终提交预测结果

### 提交格式
文件: submission/task1_pred.hdf5
Key: 'tensor'
shape: (1000, 200, 256)
要求: 前10步必须与 task1_test.hdf5 完全一致（容差 1e-3）
     步10-199 为预测结果（190步）

## Task 2

### 训练数据（3个part）
路径: /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/
  task2_part0_train.h5
  task2_part1_train.h5
  task2_part2_train.h5
Keys: 'tensor', 'nu', 't_coordinate', 'x_coordinate'
shape: (1000, 320, 256) × 3 = 3000条
t范围: 0.00~15.95s，dt=0.05s
Nu范围: 0.0001~0.01（连续分布，每条样本有对应Nu值）
初始条件对应: 第3~12步（t=0.15~0.60s，索引3:13）

### 验证数据
路径: .../task2_val.h5
Keys: 'tensor', 'nu', 't_coordinate'
shape: (100, 210, 256)
t范围: 0.15~10.60s，dt=0.05s
前10步(t=0.15~0.60s) = 初始条件
后200步(t=0.65~10.60s) = 预测目标
使用策略: 前80条可混入训练，后20条用于本地评分

### 测试数据
路径: .../task2_test.h5
Keys: 'tensor', 't_coordinate'（无Nu标签）
shape: (1000, 10, 256)
t范围: 0.15~0.60s（只有初始10步）

### 提交格式
文件: submission/task2_pred.hdf5
Key: 'tensor'
shape: (1000, 200, 256)
要求: 前10步与 task2_test.h5 完全一致（容差 1e-3）
     步10-199 为预测结果

## 空间坐标
x范围: 0.0005~0.9966（256个点，近似均匀分布在[0,1]）
x_norm = linspace(0, 1, 256) 作为归一化坐标

## 时间归一化
Task 1: t_norm = t / 9.96（以预测终点为1）
Task 2: t_norm = (t - 0.15) / (10.60 - 0.15)（以初始时刻为0）
