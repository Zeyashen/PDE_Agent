"""
dataset.py — DeepONet 范式A 数据集

数据混合策略：
  来源A: 1D_Burgers_Sols_Nu0.001.hdf5（10000条，40步，覆盖 t=0~1.95s）
  来源B: task1_val.hdf5 前80条（200步，覆盖 t=0~9.96s）← 关键！
  val_mix_ratio 控制来源B的采样权重（默认2.0，加大对长时动力学的关注）

分析发现：
  exp_001(mlp): Seg1≈0, Seg3=55 → 短期差，长期还行
  exp_002(cnn): Seg1≈1, Seg3=76 → 同样短期极差
  根本原因：来源A只覆盖 t=0~1.95s，模型没见过 t<0.5s 的高精度短期动力学
  → 适当降低 val_mix_ratio，让模型多学来源A的短期数据
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler


def _read_tensor(path: str) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        key  = 'tensor' if 'tensor' in keys else keys[0]
        return f[key][:]


class DeepONetDataset(Dataset):
    """
    单数据源 DeepONet 范式A Dataset。
    每个样本返回 (u0, coords, u_gt)。
    """

    def __init__(self, data, T_in=10, n_query=2048,
                 t_start=0.01, t_end=9.96, full_query=False):
        super().__init__()
        self.data       = torch.from_numpy(data.astype(np.float32))
        self.T_in       = T_in
        self.n_query    = n_query
        self.full_query = full_query

        N, T, X    = self.data.shape
        self.N     = N
        self.T     = T
        self.X     = X
        self.T_pred = T - T_in

        self.t_norm = torch.linspace(0, 1, self.T_pred)
        self.x_norm = torch.linspace(0, 1, X)
        tt, xx = torch.meshgrid(self.t_norm, self.x_norm, indexing='ij')
        self.full_coords = torch.stack([tt.flatten(), xx.flatten()], dim=-1)

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


class MixedBurgersDataset(Dataset):
    """
    混合数据集：原始训练集 + val前80条。
    """

    def __init__(self, train_hdf5, val_hdf5, split='train',
                 T_in=10, n_query=2048, val_train_n=80,
                 train_ratio=0.9, reduced_t=5, reduced_x=4,
                 n_samples=None):
        super().__init__()

        val_all = _read_tensor(val_hdf5).astype(np.float32)

        if split == 'val_eval':
            eval_data   = val_all[val_train_n:]
            self._inner = DeepONetDataset(eval_data, T_in=T_in,
                                          n_query=n_query)
            self.T_pred = 190
            self.X      = 256
            return

        raw  = _read_tensor(train_hdf5).astype(np.float32)
        T_r  = 200 // reduced_t
        raw  = raw[:, ::reduced_t, :][:, :T_r, :]
        raw  = raw[:, :, ::reduced_x]

        N = raw.shape[0]
        if n_samples:
            N   = min(n_samples, N)
            raw = raw[:N]
        n_train   = int(N * train_ratio)

        if split == 'internal':
            val_raw     = raw[n_train:]
            self._inner = DeepONetDataset(val_raw, T_in=T_in,
                                          n_query=n_query)
            self.T_pred = 30
            self.X      = 256
            return

        train_raw  = raw[:n_train]
        self._ds_a = DeepONetDataset(train_raw, T_in=T_in, n_query=n_query)
        self._ds_b = DeepONetDataset(val_all[:val_train_n], T_in=T_in,
                                     n_query=n_query)
        self._concat = ConcatDataset([self._ds_a, self._ds_b])
        self.T_pred  = 190
        self.X       = 256
        self.N       = len(self._concat)
        self._len_a  = len(self._ds_a)
        self._len_b  = len(self._ds_b)

    def __len__(self):
        if hasattr(self, '_inner'):
            return len(self._inner)
        return len(self._concat)

    def __getitem__(self, idx):
        if hasattr(self, '_inner'):
            return self._inner[idx]
        return self._concat[idx]

    def get_sample_weights(self, val_mix_ratio=2.0):
        """
        返回每个样本的采样权重。
        val_mix_ratio: 来源B(val前80条)相对来源A的采样倍率。
          = 1.0: 等权采样
          > 1.0: 更多采样 val 数据（改善 Seg2/3）
          < 1.0: 更少采样 val 数据（改善 Seg1）
        """
        weights_a = [1.0] * self._len_a
        weights_b = [val_mix_ratio] * self._len_b
        return weights_a + weights_b


def make_dataloaders(train_hdf5, val_hdf5, batch_size=32,
                     n_query=2048, n_samples=None, num_workers=4,
                     val_train_n=80, train_ratio=0.9,
                     val_mix_ratio=2.0,
                     distributed=False, rank=0, world_size=1):
    """
    val_mix_ratio: 控制 val 前80条在训练中的采样权重。
      分析显示 Seg1≈0 是因为短期数据不足，建议从 1.0 开始调。
    """
    from torch.utils.data.distributed import DistributedSampler

    common = dict(train_hdf5=train_hdf5, val_hdf5=val_hdf5,
                  T_in=10, n_query=n_query, val_train_n=val_train_n,
                  train_ratio=train_ratio)
    if n_samples:
        common['n_samples'] = n_samples

    train_ds = MixedBurgersDataset(**common, split='train')
    val_ds   = MixedBurgersDataset(**common, split='internal')

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
        # 用 WeightedRandomSampler 控制来源A和来源B的采样比例
        weights = train_ds.get_sample_weights(val_mix_ratio)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    if rank == 0:
        print(f"[Dataset] task=1 train={len(train_ds)} "
              f"internal_val={len(val_ds)} "
              f"n_query={n_query} val_mix_ratio={val_mix_ratio}")
    return train_loader, val_loader