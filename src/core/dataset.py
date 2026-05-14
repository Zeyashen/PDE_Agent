import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple


class BurgersDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int, t_valid: np.ndarray = None):
        """
        data    : (N, T_max, X)
        t_valid : (N,) 每个样本的有效时间步数，默认全部有效
        """
        super().__init__()
        self.window_size = window_size
        self.data = torch.from_numpy(data.astype(np.float32))
        self.N, self.T_max, self._X_r = self.data.shape
        self.grid = torch.linspace(0, 1, self.X_r).unsqueeze(0)

        # 每个样本的有效时间步数
        if t_valid is None:
            self.t_valid = np.full(self.N, self.T_max, dtype=np.int64)
        else:
            self.t_valid = t_valid

        # index_pairs 只在各自有效范围内滑动
        self.index_pairs = [
            (n, t)
            for n in range(self.N)
            for t in range(window_size, int(self.t_valid[n]))
        ]

    @property
    def T_r(self):
        return self.T_max

    @property
    def X_r(self):
        return self._X_r

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        n, t = self.index_pairs[idx]
        history = self.data[n, t - self.window_size: t, :]
        target  = self.data[n, t, :].unsqueeze(0)
        inp     = torch.cat([history, self.grid], dim=0)
        return inp, target, n, t


def make_dataloaders(
    data_cfg,
    model_cfg,
    batch_size: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    val_train_n: int = 80,
) -> Tuple[DataLoader, DataLoader]:
    window_size = model_cfg.window_size

    # ── 1. 原始训练集（40步）────────────────────────────────────────
    with h5py.File(data_cfg.hdf5_path, "r") as f:
        keys = list(f.keys())
        data_key = "tensor" if "tensor" in keys else keys[0]
        raw = f[data_key][:data_cfg.n_samples].astype(np.float32)

    t_stride = data_cfg.reduced_resolution_t  # 5
    x_stride = data_cfg.reduced_resolution    # 4
    raw = raw[:, 1:201:t_stride, ::x_stride]  # (N, 40, 256)
    N_raw, T_raw, X = raw.shape

    # ── 2. val 前80条（200步）────────────────────────────────────────
    with h5py.File(data_cfg.val_hdf5_path, "r") as f:
        keys = list(f.keys())
        data_key = "tensor" if "tensor" in keys else keys[0]
        val_all = f[data_key][:].astype(np.float32)  # (100, 200, 256)

    val_train_data = val_all[:val_train_n]   # (80, 200, 256)
    N_val, T_val, _ = val_train_data.shape

    # ── 3. 合并为一个 data tensor，用 t_valid 记录各自有效步数 ──────
    T_max = max(T_raw, T_val)  # 200

    combined = np.zeros((N_raw + N_val, T_max, X), dtype=np.float32)
    combined[:N_raw, :T_raw, :] = raw             # 前N_raw条，只有前40步有效
    combined[N_raw:, :T_val, :] = val_train_data  # 后80条，200步都有效

    t_valid = np.concatenate([
        np.full(N_raw, T_raw, dtype=np.int64),   # 原始训练集有效步=40
        np.full(N_val, T_val, dtype=np.int64),   # val数据有效步=200
    ])

    train_ds = BurgersDataset(combined, window_size, t_valid=t_valid)

    # ── 4. 验证集（后20条）──────────────────────────────────────────
    val_eval_data = val_all[val_train_n:]    # (20, 200, 256)
    val_ds = BurgersDataset(val_eval_data, window_size)

    # ── 5. DataLoader ────────────────────────────────────────────────
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=data_cfg.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, sampler=val_sampler,
            num_workers=data_cfg.num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=data_cfg.num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=data_cfg.num_workers
        )

    return train_loader, val_loader


class DeepONetDataset(Dataset):
    """PI-DeepONet 数据集。

    每个样本返回:
        inp:    (W+1+N, X) — 第0行是u0，后N行是xt查询坐标(t_norm,x_norm,0,...,0)
        target: (N, X)    — 真实解，每行是s(x,t)在X个点的值（实际只用前1列）
        n, t:   dummy（兼容trainer）
    """
    def __init__(self, data: np.ndarray, n_query: int = 200,
                 t_scale: float = 2.0):
        super().__init__()
        self.data    = torch.from_numpy(data.astype(np.float32))
        self.N, self.T, self.X = data.shape
        self.n_query = n_query
        self.t_scale = t_scale
        self.t_valid = np.full(self.N, self.T, dtype=np.int64)

        self.t_coords = np.linspace(0, t_scale, self.T)
        self.x_coords = np.linspace(0, 1, self.X)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        sol = self.data[idx]     # (T, X)
        u0  = sol[0, :]         # (X,)

        # 随机采样查询点
        t_idx = np.random.randint(0, self.T, self.n_query)
        x_idx = np.random.randint(0, self.X, self.n_query)

        t_norm = (self.t_coords[t_idx] / self.t_scale).astype(np.float32)
        x_norm = self.x_coords[x_idx].astype(np.float32)

        # xt 编码进 inp 的后 n_query 行，每行前两列是 (t_norm, x_norm)
        xt = np.zeros((self.n_query, self.X), dtype=np.float32)
        xt[:, 0] = t_norm
        xt[:, 1] = x_norm

        # inp: (1+n_query, X)
        inp = np.vstack([u0.numpy(), xt])

        # target: (n_query, X), 只有第0列有效
        s = sol[t_idx, x_idx].numpy()     # (n_query,)
        target = np.zeros((self.n_query, self.X), dtype=np.float32)
        target[:, 0] = s

        return (
            torch.from_numpy(inp),
            torch.from_numpy(target),
            torch.tensor(idx),
            torch.tensor(0),
        )


def make_deeponet_dataloaders(
    data_cfg,
    model_cfg,
    batch_size: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    val_train_n: int = 80,
):
    """为 DeepONet 构建 DataLoader。"""
    # 加载训练数据
    with h5py.File(data_cfg.hdf5_path, 'r') as f:
        keys = list(f.keys())
        data_key = "tensor" if "tensor" in keys else keys[0]
        raw = f[data_key][:data_cfg.n_samples].astype(np.float32)
    raw = raw[:, 1:201:data_cfg.reduced_resolution_t, ::data_cfg.reduced_resolution]

    # 加载 val 前80条
    with h5py.File(data_cfg.val_hdf5_path, 'r') as f:
        val_all = f['tensor'][:].astype(np.float32)
    val_train = val_all[:val_train_n]
    val_eval  = val_all[val_train_n:]

    # 分别建 dataset（步数不同不能直接合并）
    # raw: 40步(0~2s), val_train: 200步(0~10s)
    train_ds_raw = DeepONetDataset(raw,       n_query=200, t_scale=2.0)
    train_ds_val = DeepONetDataset(val_train, n_query=200, t_scale=10.0)
    from torch.utils.data import ConcatDataset
    train_ds = ConcatDataset([train_ds_raw, train_ds_val])
    # ConcatDataset 没有 t_scale 属性，给它加上
    train_ds.t_scale = 2.0
    val_ds = DeepONetDataset(val_eval, n_query=200, t_scale=10.0)

    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_loader = DataLoader(train_ds, batch_size=batch_size,
            sampler=DistributedSampler(train_ds, world_size, rank, shuffle=True),
            num_workers=data_cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
            sampler=DistributedSampler(val_ds, world_size, rank, shuffle=False),
            num_workers=data_cfg.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
            shuffle=True, num_workers=data_cfg.num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
            shuffle=False, num_workers=data_cfg.num_workers)

    return train_loader, val_loader
