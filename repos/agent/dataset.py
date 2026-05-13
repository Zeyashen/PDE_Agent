"""
dataset.py — PDE Agent Task-1 数据集

设计说明
--------
训练数据覆盖问题：
  原始训练集 1D_Burgers_Sols_Nu0.001.hdf5 下采样后只覆盖 t=0~1.95s，
  而推理要求预测到 t=9.96s（覆盖率仅 20%）。
  这是 Seg3 低分的根本原因，不是模型容量不足。

解决方案：
  将 task1_val.hdf5 的前80条样本（覆盖 t=0~9.96s）混入训练集，
  使模型见过完整推理范围的动力学行为。
  后20条保留用于 Agent 本地评分（eval_split）。

Pushforward 支持：
  __getitem__ 返回 (inp, targets)，targets 包含连续 rollout_k 步的 GT，
  供 train.py 做多步 rollout loss（pushforward trick）。

数据源：
  来源A: 1D_Burgers_Sols_Nu0.001.hdf5 (10000, 201, 1024)
         下采样 → (N, 40, 256)，window=10，每条30对
  来源B: task1_val.hdf5 前80条 (80, 200, 256)
         直接用，window=10，每条190对（覆盖 t<10s）
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# ============================================================
# 辅助：从 hdf5 读取 tensor
# ============================================================
def _read_tensor(path: str) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        key  = 'tensor' if 'tensor' in keys else keys[0]
        return f[key][:]


# ============================================================
# 单数据源 Dataset
# ============================================================
class _SingleSourceDataset(Dataset):
    """
    从一个 (N, T, X) 的 numpy 数组构建 (inp, targets) 对。

    inp     : (rollout_k + 1, X) = (window_size + k, X)... 不对，见下
    实际上：
      inp     : (window_size + 1, X)  最后一行是空间坐标
      targets : (rollout_k, 1, X)     连续 rollout_k 步的 GT

    为了支持 pushforward，每个样本需要 window_size + rollout_k 步的数据，
    因此 start_t 的范围是 [window_size, T - rollout_k]。
    """

    def __init__(
        self,
        data:        np.ndarray,   # (N, T, X) float32
        window_size: int = 10,
        rollout_k:   int = 3,
    ):
        super().__init__()
        self.data        = torch.from_numpy(data.astype(np.float32))
        self.window_size = window_size
        self.rollout_k   = rollout_k

        N, T, X = self.data.shape
        self.X   = X

        # 空间坐标，shape (1, X)，每个样本共享
        self.grid = torch.linspace(0, 1, X).unsqueeze(0)

        # 构建 (sample_idx, start_t) 索引
        # start_t: 预测的第一步索引，需要保证 start_t + rollout_k - 1 < T
        self.index_pairs = []
        for n in range(N):
            for t in range(window_size, T - rollout_k + 1):
                self.index_pairs.append((n, t))

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        n, t = self.index_pairs[idx]

        # 历史窗口: [t-window_size, t)，共 window_size 步
        history = self.data[n, t - self.window_size: t, :]   # (window_size, X)

        # 目标: 连续 rollout_k 步
        targets = self.data[n, t: t + self.rollout_k, :].unsqueeze(1)
        # shape: (rollout_k, 1, X)

        # 输入: history + 空间坐标
        inp = torch.cat([history, self.grid], dim=0)          # (window_size+1, X)

        return inp, targets


# ============================================================
# 公开 Dataset：BurgersDataset（兼容旧接口 + 新混合逻辑）
# ============================================================
class BurgersDataset(Dataset):
    """
    混合数据集：原始训练集 + val前80条（覆盖长时动力学）。

    参数
    ----
    train_hdf5_path : 原始训练集路径（10000, 201, 1024）
    val_hdf5_path   : task1_val 路径（100, 200, 256）
    val_train_n     : val 中用于训练的样本数（默认80，后20用于评分）
    reduced_resolution_t : 时间下采样倍率（默认5）
    reduced_resolution   : 空间下采样倍率（默认4）
    window_size     : 历史窗口（默认10）
    rollout_k       : pushforward 步数（默认3）
    n_samples       : 原始训练集最多用多少样本（None=全部9000训练样本）
    split           : 'train' | 'val_eval'
                      train    → 原始训练集90% + val前80条
                      val_eval → val后20条，用于 Agent 评分
    train_ratio     : 原始训练集的训练/内部val分割（默认0.9）
    """

    def __init__(
        self,
        train_hdf5_path:      str,
        val_hdf5_path:        str,
        val_train_n:          int   = 80,
        reduced_resolution_t: int   = 5,
        reduced_resolution:   int   = 4,
        window_size:          int   = 10,
        rollout_k:            int   = 3,
        n_samples:            int   = None,
        split:                str   = 'train',
        train_ratio:          float = 0.9,
    ):
        super().__init__()

        self.window_size = window_size
        self.rollout_k   = rollout_k

        if split == 'val_eval':
            # ── 只用 val 后20条做本地评分 ──────────────────────
            val_data = _read_tensor(val_hdf5_path).astype(np.float32)
            # val_data: (100, 200, 256)，已下采样，直接使用
            eval_data   = val_data[val_train_n:]   # (20, 200, 256)
            self._inner = _SingleSourceDataset(eval_data, window_size, rollout_k)
            self.T_r    = 200
            self.X_r    = 256
            self.N      = len(eval_data)
            if hasattr(self._inner, 'data'):
                self.data = self._inner.data
            return

        # ── 训练模式：混合两个数据源 ──────────────────────────
        # --- 来源A: 原始训练集下采样 ---
        raw = _read_tensor(train_hdf5_path).astype(np.float32)
        # raw: (10000, 201, 1024)

        # 时间下采样：取 t=0,5,10,...195 共40步（覆盖 t=0~1.95s）
        T_r = 200 // reduced_resolution_t   # = 40
        raw = raw[:, ::reduced_resolution_t, :]  # (10000, 41, 1024)
        raw = raw[:, :T_r, :]                    # (10000, 40, 1024)，截断到40步

        # 空间下采样
        raw = raw[:, :, ::reduced_resolution]    # (10000, 40, 256)

        N_total = raw.shape[0]
        if n_samples is not None:
            N_total = min(n_samples, N_total)
            raw = raw[:N_total]

        # 训练/内部val分割（内部val不用于评分，仅供 train.py 监控 val_loss）
        n_train = int(N_total * train_ratio)
        train_raw = raw[:n_train]   # (9000, 40, 256)

        self.T_r = T_r
        self.X_r = 256

        ds_a = _SingleSourceDataset(train_raw, window_size, rollout_k)

        # --- 来源B: val前80条（覆盖 t=0~9.96s，关键！）---
        val_data  = _read_tensor(val_hdf5_path).astype(np.float32)
        # val_data: (100, 200, 256)，已下采样，dt=0.05，t=0.01~9.96
        train_val = val_data[:val_train_n]   # (80, 200, 256)

        ds_b = _SingleSourceDataset(train_val, window_size, rollout_k)

        # 合并
        self._concat = ConcatDataset([ds_a, ds_b])
        self.N = len(self._concat)

        if hasattr(ds_a, 'data'):
            # 暴露给外部（如 orchestrator 读取配置）
            self.data = ds_a.data

    def __len__(self):
        if hasattr(self, '_inner'):
            return len(self._inner)
        return len(self._concat)

    def __getitem__(self, idx):
        if hasattr(self, '_inner'):
            return self._inner[idx]
        return self._concat[idx]


# ============================================================
# 内部验证 Dataset（用于 train.py 的 val_loss 监控）
# ============================================================
class BurgersInternalValDataset(Dataset):
    """
    从原始训练集切出10%做内部 val，供 train.py 计算 val_loss/早停。
    注意：这不是评分用的 val，只是训练过程的监控信号。
    """

    def __init__(
        self,
        train_hdf5_path:      str,
        reduced_resolution_t: int   = 5,
        reduced_resolution:   int   = 4,
        window_size:          int   = 10,
        rollout_k:            int   = 3,
        n_samples:            int   = None,
        train_ratio:          float = 0.9,
    ):
        super().__init__()
        raw = _read_tensor(train_hdf5_path).astype(np.float32)
        T_r = 200 // reduced_resolution_t
        raw = raw[:, ::reduced_resolution_t, :][:, :T_r, :]
        raw = raw[:, :, ::reduced_resolution]

        N_total = raw.shape[0]
        if n_samples is not None:
            N_total = min(n_samples, N_total)
            raw = raw[:N_total]

        n_train  = int(N_total * train_ratio)
        val_data = raw[n_train:]   # (1000, 40, 256)

        self._inner = _SingleSourceDataset(val_data, window_size, rollout_k)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]


# ============================================================
# make_dataloaders：统一入口
# ============================================================
def make_dataloaders(
    train_hdf5_path:      str,
    val_hdf5_path:        str,
    batch_size:           int   = 64,
    n_samples:            int   = None,
    num_workers:          int   = 4,
    reduced_resolution_t: int   = 5,
    reduced_resolution:   int   = 4,
    window_size:          int   = 10,
    rollout_k:            int   = 3,
    val_train_n:          int   = 80,
    train_ratio:          float = 0.9,
    distributed:          bool  = False,
    rank:                 int   = 0,
    world_size:           int   = 1,
):
    """
    返回:
        train_loader : 混合数据集（原始训练集 + val前80条）
        val_loader   : 原始训练集内部 val（仅用于 val_loss 监控/早停）

    注意：val_loader 不是评分用的 val，评分逻辑在 predict.py 里。
    """
    from torch.utils.data.distributed import DistributedSampler

    common = dict(
        reduced_resolution_t=reduced_resolution_t,
        reduced_resolution=reduced_resolution,
        window_size=window_size,
        rollout_k=rollout_k,
        n_samples=n_samples,
        train_ratio=train_ratio,
    )

    train_ds = BurgersDataset(
        train_hdf5_path=train_hdf5_path,
        val_hdf5_path=val_hdf5_path,
        val_train_n=val_train_n,
        split='train',
        **common,
    )
    val_ds = BurgersInternalValDataset(
        train_hdf5_path=train_hdf5_path,
        **common,
    )

    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    if rank == 0:
        print(f"[Dataset] train={len(train_ds):,} pairs "
              f"(原始训练集 + val前{val_train_n}条)")
        print(f"[Dataset] internal_val={len(val_ds):,} pairs (仅用于早停监控)")
        print(f"[Dataset] T_r={train_ds.T_r}, X_r={train_ds.X_r}, "
              f"window={window_size}, rollout_k={rollout_k}")
        print(f"[Dataset] distributed={distributed}, world_size={world_size}")

    return train_loader, val_loader