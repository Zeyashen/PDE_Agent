"""
dataset_t2.py — Task 2 数据集

Task 2 数据格式：
  训练: (1000, 320, 256) x3个part, t=0.00~15.95s, dt=0.05
        第3~12步 (t=0.15~0.60s) 对应 val/test 的初始10步
        第13步以后 (t=0.65~15.95s) 作为预测目标

  val:  (100, 210, 256), t=0.15~10.60s, dt=0.05
        前10步 (t=0.15~0.60s) = 初始条件
        后200步 (t=0.65~10.60s) = 200步，但只评后190步

  test: (1000, 10, 256), t=0.15~0.60s
        只有初始条件，无GT

策略：
  - 不使用 Nu 作为模型输入（test 没有 Nu 标签）
  - branch net 从初始10步隐式推断物理参数
  - 时间坐标归一化：t_norm = (t - 0.15) / (10.60 - 0.15)
    使得初始时刻=0，预测终点=1
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset


T_INIT_START = 0.15   # 初始条件起始时间
T_PRED_END   = 10.60  # 预测终止时间（val 最后一步）
T_NORM_RANGE = T_PRED_END - T_INIT_START  # = 10.45s，归一化范围


def _read_h5(path, key='tensor'):
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        k    = key if key in keys else keys[0]
        return f[k][:]


def _time_normalize(t_abs):
    """将绝对时间归一化到 [0, 1]，以 T_INIT_START 为零点。"""
    return (t_abs - T_INIT_START) / T_NORM_RANGE


class Task2TrainDataset(Dataset):
    """
    Task 2 训练 Dataset。

    从训练数据里切出：
      u0      : 第3~12步 (t=0.15~0.60s, 索引3:13)  → (10, 256)
      coords  : 从第13步开始随机采样 n_query 个时空点
      u_gt    : 对应时空点的真实值

    参数
    ----
    data    : (N, 320, 256) numpy array
    t_coord : (320,) 时间坐标
    T_in    : 初始条件步数（默认10）
    t_init_idx : 初始条件在训练数据中的起始索引（默认3，对应t=0.15）
    n_query : 每个样本的查询点数（训练时子采样）
    """

    def __init__(self, data, t_coord, T_in=10, t_init_idx=3, n_query=2048):
        super().__init__()
        self.data       = torch.from_numpy(data.astype(np.float32))
        self.T_in       = T_in
        self.t_init_idx = t_init_idx
        self.n_query    = n_query

        N, T, X     = self.data.shape
        self.N      = N
        self.X      = X

        # 初始条件步索引：t_init_idx 到 t_init_idx+T_in
        self.init_end = t_init_idx + T_in  # = 13

        # 预测目标：init_end 到末尾
        self.pred_t  = t_coord[self.init_end:]   # (307,) 绝对时间
        self.T_pred  = len(self.pred_t)

        # 归一化时间和空间坐标
        t_norm = torch.tensor(
            _time_normalize(self.pred_t), dtype=torch.float32)  # (T_pred,)
        x_norm = torch.linspace(0, 1, X)                        # (X,)

        tt, xx = torch.meshgrid(t_norm, x_norm, indexing='ij')
        self.full_coords = torch.stack(
            [tt.flatten(), xx.flatten()], dim=-1)               # (T_pred*X, 2)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        traj = self.data[idx]   # (320, 256)

        # 初始条件
        u0 = traj[self.t_init_idx: self.t_init_idx + self.T_in, :]  # (10, 256)

        # 预测目标（随机子采样）
        u_future = traj[self.init_end:, :]   # (T_pred, 256)
        total    = self.T_pred * self.X
        idx_q    = torch.randperm(total)[:self.n_query]
        coords   = self.full_coords[idx_q]   # (n_query, 2)
        u_gt     = u_future.flatten()[idx_q] # (n_query,)

        return u0, coords, u_gt


class Task2ValDataset(Dataset):
    """
    Task 2 Val/Eval Dataset。

    val 数据格式：(100, 210, 256), t=0.15~10.60s
      前10步 = 初始条件（t=0.15~0.60s）
      后200步 = 预测目标（t=0.65~10.60s）
      评分只看后190步（去掉第一个预测步，与Task1一致）
    """

    def __init__(self, data, t_coord, T_in=10, n_query=2048,
                 full_query=False):
        super().__init__()
        self.data       = torch.from_numpy(data.astype(np.float32))
        self.T_in       = T_in
        self.n_query    = n_query
        self.full_query = full_query

        N, T, X    = self.data.shape
        self.N     = N
        self.X     = X
        self.T_pred = T - T_in  # = 200

        # 归一化坐标（从第T_in步以后）
        t_norm = torch.tensor(
            _time_normalize(t_coord[T_in:]), dtype=torch.float32)
        x_norm = torch.linspace(0, 1, X)
        tt, xx = torch.meshgrid(t_norm, x_norm, indexing='ij')
        self.full_coords = torch.stack(
            [tt.flatten(), xx.flatten()], dim=-1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        traj     = self.data[idx]
        u0       = traj[:self.T_in, :]
        u_future = traj[self.T_in:, :]

        if self.full_query:
            coords = self.full_coords
            u_gt   = u_future.flatten()
        else:
            total  = self.T_pred * self.X
            idx_q  = torch.randperm(total)[:self.n_query]
            coords = self.full_coords[idx_q]
            u_gt   = u_future.flatten()[idx_q]

        return u0, coords, u_gt


def make_task2_dataloaders(
    data_dir,
    val_path,
    batch_size   = 32,
    n_query      = 2048,
    num_workers  = 4,
    val_train_n  = 80,
    distributed  = False,
    rank         = 0,
    world_size   = 1,
):
    """
    返回 train_loader, val_loader (内部验证用)。
    评分用的 eval_loader 由 predict_t2.py 单独构建。
    """
    from torch.utils.data.distributed import DistributedSampler

    data_dir = Path(data_dir)

    # 加载3个训练 part
    all_data   = []
    all_t      = None
    for i in range(3):
        fpath = data_dir / f"task2_part{i}_train.h5"
        with h5py.File(fpath, 'r') as f:
            all_data.append(f['tensor'][:])
            if all_t is None:
                all_t = f['t_coordinate'][:]
    train_all = np.concatenate(all_data, axis=0)  # (3000, 320, 256)

    # 加载 val
    with h5py.File(val_path, 'r') as f:
        val_data = f['tensor'][:]    # (100, 210, 256)
        val_t    = f['t_coordinate'][:]

    # 训练集：来源A（训练数据）+ 来源B（val前80条）
    ds_a = Task2TrainDataset(train_all, all_t, n_query=n_query)
    ds_b = Task2ValDataset(val_data[:val_train_n], val_t, n_query=n_query)

    train_ds = ConcatDataset([ds_a, ds_b])

    # 内部 val（训练数据最后10%，仅用于早停监控）
    n_internal = int(len(train_all) * 0.1)
    val_ds = Task2TrainDataset(
        train_all[-n_internal:], all_t, n_query=n_query)

    if distributed:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=DistributedSampler(train_ds, world_size, rank, shuffle=True),
            num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            sampler=DistributedSampler(val_ds, world_size, rank, shuffle=False),
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    if rank == 0:
        print(f"[Task2 Dataset] train={len(train_ds)} "
              f"(3000训练+val前{val_train_n}条) "
              f"internal_val={len(val_ds)}")
        print(f"[Task2 Dataset] T_pred(train)={ds_a.T_pred} "
              f"T_pred(val)={ds_b.T_pred} n_query={n_query}")

    return train_loader, val_loader