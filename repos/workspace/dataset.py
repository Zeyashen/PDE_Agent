"""
dataset.py — DeepONet Task-1 框架层数据集

这是框架层文件，由人工维护，不提交到 code/。
Agent 通过 train.py 调用以下接口。

对外接口：
  load_train_data(...)   → (train_raw, val_mix)  训练原料数据
  load_eval_data()       → numpy (20, 200, 256)  评分用 val 后20条
  load_test_data()       → numpy (1000, 10, 256) 测试集初始条件
  make_dataloaders(...)  → (train_loader, val_loader)
  DeepONetDataset        → Dataset 类，可直接使用

数据混合策略（解决训练集只覆盖 t=0~1.95s 的问题）：
  来源A: 原始训练集  短时数据 t=0.01~2.01s（跳过t=0，与val/test对齐）
  来源B: val 前N条   长时数据 t=0~9.96s  ← 关键！覆盖长时动力学
  通过 n_val_mix 和 val_mix_ratio 控制长时数据的数量和权重
"""

import h5py
import numpy as np
import torch
from torch.utils.data import (Dataset, DataLoader,
                               ConcatDataset, WeightedRandomSampler)
from torch.utils.data.distributed import DistributedSampler

# ── 数据路径（硬编码，框架层保证正确）────────────────────────
TRAIN_HDF5 = ("/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/"
              "data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5")
VAL_HDF5   = ("/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/"
              "data_and_sample_submission/train_val_test_init/task1_val.hdf5")
TEST_HDF5  = ("/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/"
              "data_and_sample_submission/train_val_test_init/task1_test.hdf5")


# ============================================================
# Dataset 类
# ============================================================
class DeepONetDataset(Dataset):
    """
    DeepONet 范式A Dataset。

    每个样本返回:
      u0    : (T_in, X)      初始条件
      coords: (n_query, 2)   随机采样的时空查询坐标，归一化到 [0,1]
      u_gt  : (n_query,)     对应查询点的真实解
    """

    def __init__(self, data: np.ndarray, T_in: int = 10,
                 n_query: int = 2048):
        super().__init__()
        self.data    = torch.from_numpy(data.astype(np.float32))
        self.T_in    = T_in
        self.n_query = n_query

        N, T, X      = self.data.shape
        self.T_pred  = T - T_in

        t_norm = torch.linspace(0, 1, self.T_pred)
        x_norm = torch.linspace(0, 1, X)
        tt, xx = torch.meshgrid(t_norm, x_norm, indexing='ij')
        self.full_coords = torch.stack(
            [tt.flatten(), xx.flatten()], dim=-1)  # (T_pred*X, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj     = self.data[idx]
        u0       = traj[:self.T_in, :]
        u_future = traj[self.T_in:, :]
        total    = self.T_pred * u0.shape[-1]
        idx_q    = torch.randperm(total)[:self.n_query]
        coords   = self.full_coords[idx_q]
        u_gt     = u_future.flatten()[idx_q]
        return u0, coords, u_gt


# ============================================================
# 数据加载函数
# ============================================================
def load_train_data(
    n_samples:    int = None,
    n_val_mix:    int = 80,
    downsample_t: int = 5,
    downsample_x: int = 4,
) -> tuple:
    """
    加载训练原料数据，供 Agent 自行控制混合比例。

    参数
    ----
    n_samples    : 原始训练集取多少条（None = 全部10000条）
    n_val_mix    : val 数据混入多少条（从头取，0 = 不混入）
                   增大 → 更多长时数据 → 改善 Seg2/3
                   减小 → 更多短时数据 → 改善 Seg1
    downsample_t : 时间下采样倍数（默认5，跳过t=0后得到40步 t=0.01~2.01s，与val/test对齐）
    downsample_x : 空间下采样倍数（默认4，得到256空间点）

    返回
    ----
    train_raw : numpy (N, 40, 256)  float32  原始训练集（短时）
    val_mix   : numpy (M, 200, 256) float32  val 前M条（长时）
                M=0 时返回 shape=(0, 200, 256) 空数组

    使用示例
    --------
    # 默认配置
    train_raw, val_mix = load_train_data()

    # 增大长时比例
    train_raw, val_mix = load_train_data(n_val_mix=100)

    # smoke test：只用少量数据
    train_raw, val_mix = load_train_data(n_samples=200, n_val_mix=10)
    """
    with h5py.File(TRAIN_HDF5, 'r') as f:
        raw = f['tensor'][:n_samples, 1::downsample_t, ::downsample_x]  # 跳过t=0，从t=0.01开始与val/test对齐
    train_raw = raw.astype(np.float32)

    if n_val_mix > 0:
        with h5py.File(VAL_HDF5, 'r') as f:
            val_mix = f['tensor'][:n_val_mix].astype(np.float32)
    else:
        val_mix = np.zeros((0, 200, 256), dtype=np.float32)

    return train_raw, val_mix


def load_eval_data() -> np.ndarray:
    """加载 val 后20条用于评分。返回 (20, 200, 256) float32。"""
    with h5py.File(VAL_HDF5, 'r') as f:
        return f['tensor'][80:].astype(np.float32)


def load_test_data() -> np.ndarray:
    """加载测试集初始条件。返回 (1000, 10, 256) float32。"""
    with h5py.File(TEST_HDF5, 'r') as f:
        return f['tensor'][:].astype(np.float32)


def make_dataloaders(
    batch_size:    int   = 32,
    n_query:       int   = 2048,
    val_mix_ratio: float = 2.0,
    n_val_mix:     int   = 80,
    train_ratio:   float = 0.9,
    n_samples:     int   = None,
    num_workers:   int   = 4,
    distributed:   bool  = False,
    rank:          int   = 0,
    world_size:    int   = 1,
) -> tuple:
    """
    构建训练和验证 DataLoader。

    参数
    ----
    batch_size    : 每卡 batch size
    n_query       : 每样本查询点数
    val_mix_ratio : 来源B（长时数据）的采样权重倍率
                    > 1.0: 更多长时 → 改善 Seg2/3
                    < 1.0: 更多短时 → 改善 Seg1
    n_val_mix     : val 混入条数（默认80）
    train_ratio   : 原始训练集中用于训练的比例（其余做内部 val）
    n_samples     : 原始训练集取多少条（None=全部）
    distributed   : DDP 时设 True
    rank          : DDP rank
    world_size    : DDP world_size

    返回
    ----
    train_loader, val_loader
    每个 batch: (u0, coords, u_gt)
      u0:    (B, 10, 256)
      coords:(B, n_query, 2)  值域 [0,1]
      u_gt:  (B, n_query)
    """
    train_raw, val_mix = load_train_data(
        n_samples=n_samples, n_val_mix=n_val_mix)

    N       = len(train_raw)
    n_train = int(N * train_ratio)

    ds_a = DeepONetDataset(train_raw[:n_train], T_in=10, n_query=n_query)
    ds_v = DeepONetDataset(train_raw[n_train:], T_in=10, n_query=n_query)

    if len(val_mix) > 0:
        ds_b     = DeepONetDataset(val_mix, T_in=10, n_query=n_query)
        train_ds = ConcatDataset([ds_a, ds_b])
        weights  = [1.0] * len(ds_a) + [val_mix_ratio] * len(ds_b)
        sampler  = WeightedRandomSampler(weights, len(weights), replacement=True)
    else:
        train_ds = ds_a
        sampler  = None

    if distributed:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=DistributedSampler(train_ds, world_size, rank, shuffle=True),
            num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(
            ds_v, batch_size=batch_size,
            sampler=DistributedSampler(ds_v, world_size, rank, shuffle=False),
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler, shuffle=(sampler is None),
            num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(
            ds_v, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

    if rank == 0:
        print(f"[Dataset] train={len(train_ds)} "
              f"(来源A:{len(ds_a)}"
              + (f" + 来源B:{len(ds_b)} val_mix_ratio={val_mix_ratio})"
                 if len(val_mix) > 0 else ")")
              )
        print(f"[Dataset] internal_val={len(ds_v)}")

    return train_loader, val_loader


# ============================================================
# Unittest
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("dataset.py unittest")
    print("=" * 50)
    errors = []

    def check(name, fn):
        try:
            fn()
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            errors.append((name, str(e)))

    # Test 1: load_train_data 默认
    print("\n[1] load_train_data(n_samples=10, n_val_mix=5)")
    def t1():
        tr, vm = load_train_data(n_samples=10, n_val_mix=5)
        assert tr.shape == (10, 40, 256), f"train_raw shape: {tr.shape}"
        assert vm.shape == (5, 200, 256), f"val_mix shape: {vm.shape}"
        assert tr.dtype == np.float32,    f"train dtype: {tr.dtype}"
        assert vm.dtype == np.float32,    f"val dtype: {vm.dtype}"
        assert not np.isnan(tr).any(),    "train_raw 含 NaN"
        assert not np.isnan(vm).any(),    "val_mix 含 NaN"
    check("load_train_data shape/dtype/NaN", t1)

    # Test 2: load_train_data n_val_mix=0
    print("\n[2] load_train_data(n_val_mix=0)")
    def t2():
        tr, vm = load_train_data(n_samples=5, n_val_mix=0)
        assert tr.shape == (5, 40, 256),  f"shape: {tr.shape}"
        assert vm.shape == (0, 200, 256), f"val_mix shape: {vm.shape}"
    check("load_train_data n_val_mix=0", t2)

    # Test 3: load_eval_data
    print("\n[3] load_eval_data()")
    def t3():
        ev = load_eval_data()
        assert ev.shape == (20, 200, 256), f"shape: {ev.shape}"
        assert ev.dtype == np.float32,     f"dtype: {ev.dtype}"
        assert not np.isnan(ev).any(),     "含 NaN"
    check("load_eval_data", t3)

    # Test 4: load_test_data
    print("\n[4] load_test_data()")
    def t4():
        te = load_test_data()
        assert te.shape == (1000, 10, 256), f"shape: {te.shape}"
        assert te.dtype == np.float32,      f"dtype: {te.dtype}"
    check("load_test_data", t4)

    # Test 5: DeepONetDataset
    print("\n[5] DeepONetDataset")
    def t5():
        tr, _ = load_train_data(n_samples=5, n_val_mix=0)
        ds    = DeepONetDataset(tr, T_in=10, n_query=128)
        assert len(ds) == 5
        u0, coords, u_gt = ds[0]
        assert u0.shape     == (10, 256), f"u0: {u0.shape}"
        assert coords.shape == (128, 2),  f"coords: {coords.shape}"
        assert u_gt.shape   == (128,),    f"u_gt: {u_gt.shape}"
        assert float(coords.min()) >= 0.0 and float(coords.max()) <= 1.0
    check("DeepONetDataset shape/range", t5)

    # Test 6: make_dataloaders
    print("\n[6] make_dataloaders(batch_size=4, n_query=128, n_samples=50)")
    def t6():
        tl, vl = make_dataloaders(
            batch_size=4, n_query=128, n_samples=50, n_val_mix=5)
        u0, coords, u_gt = next(iter(tl))
        assert u0.shape     == (4, 10, 256), f"u0: {u0.shape}"
        assert coords.shape == (4, 128, 2),  f"coords: {coords.shape}"
        assert u_gt.shape   == (4, 128),     f"u_gt: {u_gt.shape}"
        u0v, _, _ = next(iter(vl))
        assert u0v.shape == (4, 10, 256),    f"val u0: {u0v.shape}"
    check("make_dataloaders batch shape", t6)

    print("\n" + "=" * 50)
    if errors:
        print(f"❌ {len(errors)} 个测试失败:")
        for name, err in errors:
            print(f"   {name}: {err}")
    else:
        print("✅ 所有测试通过")
    print("=" * 50)