"""
dataset.py — DeepONet Task-2 框架层数据集

这是框架层文件，由人工维护，不提交到 code/。
Agent 通过 train.py 调用以下接口。

对外接口：
  load_train_data(...)   → (train_raw, val_mix, train_nu, val_nu)
  load_eval_data()       → (numpy (20, 210, 256), nu (20,))
  load_test_data()       → numpy (1000, 10, 256)
  make_dataloaders(...)  → (train_loader, val_loader)
  DeepONetDataset        → Dataset 类

Task2 数据说明：
  训练集：3个part文件，各 (1000, 320, 256)，共3000条
          时间范围 t=0~15.95s，dt=0.05s
          初始条件：索引 3:13（t=0.15~0.60s），与val/test对齐
          含 nu 标签（每条样本的粘性系数）

  验证集：(100, 210, 256)，t=0.15~10.60s
          前10步 = 初始条件，后200步 = 预测目标
          含 nu 标签

  测试集：(1000, 10, 256)，t=0.15~0.60s（仅初始条件，无nu标签）

数据混合策略：
  来源A: 原始训练集（3000条），覆盖 t=0.15~15.95s
  来源B: val 前N条，覆盖 t=0.15~10.60s
  通过 n_val_mix 和 val_mix_ratio 控制混合比例
"""

import h5py
import numpy as np
import torch
from torch.utils.data import (Dataset, DataLoader,
                               ConcatDataset, WeightedRandomSampler)
from torch.utils.data.distributed import DistributedSampler

# ── 数据路径 ──────────────────────────────────────────────────
_BASE  = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent"
_INIT  = f"{_BASE}/data_and_sample_submission/train_val_test_init"

TRAIN_HDF5_LIST = [
    f"{_INIT}/task2_part0_train.h5",
    f"{_INIT}/task2_part1_train.h5",
    f"{_INIT}/task2_part2_train.h5",
]
VAL_HDF5  = f"{_INIT}/task2_val.h5"
TEST_HDF5 = f"{_INIT}/task2_test.h5"

# 初始条件在训练集时间轴上的索引（t=0.15~0.60s 对应索引 3:13）
T_INIT_START = 3
T_INIT_END   = 13  # 不含
T_IN         = T_INIT_END - T_INIT_START  # = 10


# ============================================================
# Dataset 类
# ============================================================
class DeepONetDataset(Dataset):
    """
    DeepONet 范式A Dataset for Task2。

    每个样本返回:
      u0    : (T_in, X)      初始条件（10步）
      coords: (n_query, 2)   随机采样的时空查询坐标，归一化到 [0,1]
      u_gt  : (n_query,)     对应查询点的真实解

    nu 标签可选，用于训练时条件化（推理时不使用）。
    """

    def __init__(self, data: np.ndarray, n_query: int = 2048,
                 nu: np.ndarray = None):
        """
        data  : (N, T, 256)，其中前 T_in 步为初始条件，其余为预测目标
        nu    : (N,) 可选，粘性系数标签
        """
        super().__init__()
        self.data    = torch.from_numpy(data.astype(np.float32))
        self.n_query = n_query
        self.nu      = torch.from_numpy(nu.astype(np.float32)) \
                       if nu is not None else None

        N, T, X      = self.data.shape
        self.T_pred  = T - T_IN

        t_norm = torch.linspace(0, 1, self.T_pred)
        x_norm = torch.linspace(0, 1, X)
        tt, xx = torch.meshgrid(t_norm, x_norm, indexing='ij')
        self.full_coords = torch.stack(
            [tt.flatten(), xx.flatten()], dim=-1)  # (T_pred*X, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj     = self.data[idx]
        u0       = traj[:T_IN, :]
        u_future = traj[T_IN:, :]
        total    = self.T_pred * u0.shape[-1]
        idx_q    = torch.randperm(total)[:self.n_query]
        coords   = self.full_coords[idx_q]
        u_gt     = u_future.flatten()[idx_q]
        # 始终返回4个值，nu不存在时返回0（模型会忽略）
        nu = self.nu[idx] if self.nu is not None else torch.zeros(1)
        return u0, coords, u_gt, nu


# ============================================================
# 数据加载函数
# ============================================================
def load_train_data(
    n_samples:    int = None,
    n_val_mix:    int = 80,
) -> tuple:
    """
    加载训练原料数据。

    参数
    ----
    n_samples  : 每个 part 文件取多少条（None = 全部1000条）
    n_val_mix  : val 数据混入多少条（从头取，0 = 不混入）

    返回
    ----
    train_raw : numpy (N, 200, 256)  float32
                训练集，从初始条件开始共210步（10步初始+200步预测），
                取自训练集索引 T_INIT_START:T_INIT_START+210
    val_mix   : numpy (M, 210, 256)  float32  val 前M条
    train_nu  : numpy (N,)           float32  训练集 nu 标签
    val_nu    : numpy (M,)           float32  val nu 标签（M=0时为空）
    """
    tensors = []
    nus     = []
    for path in TRAIN_HDF5_LIST:
        with h5py.File(path, 'r') as f:
            t = f['tensor'][:n_samples,
                            T_INIT_START:T_INIT_START + 210,
                            :]  # (N, 210, 256)
            nu = f['nu'][:n_samples]
            tensors.append(t.astype(np.float32))
            nus.append(nu.astype(np.float32))

    train_raw = np.concatenate(tensors, axis=0)
    train_nu  = np.concatenate(nus,     axis=0)

    if n_val_mix > 0:
        with h5py.File(VAL_HDF5, 'r') as f:
            val_mix = f['tensor'][:n_val_mix].astype(np.float32)
            val_nu  = f['nu'][:n_val_mix].astype(np.float32)
    else:
        val_mix = np.zeros((0, 210, 256), dtype=np.float32)
        val_nu  = np.zeros((0,),          dtype=np.float32)

    return train_raw, val_mix, train_nu, val_nu


def load_eval_data() -> tuple:
    """
    加载 val 后20条用于评分。
    返回 (tensor (20, 210, 256), nu (20,))
    """
    with h5py.File(VAL_HDF5, 'r') as f:
        tensor = f['tensor'][80:].astype(np.float32)
        nu     = f['nu'][80:].astype(np.float32)
    return tensor, nu


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

    返回
    ----
    train_loader, val_loader
    每个 batch: (u0, coords, u_gt, nu)
      u0:    (B, 10, 256)
      coords:(B, n_query, 2)
      u_gt:  (B, n_query)
      nu:    (B,)  粘性系数，无标签时为 0
    """
    train_raw, val_mix, train_nu, val_nu = load_train_data(
        n_samples=n_samples, n_val_mix=n_val_mix)

    N       = len(train_raw)
    n_train = int(N * train_ratio)

    ds_a = DeepONetDataset(train_raw[:n_train], n_query=n_query,
                           nu=train_nu[:n_train])
    ds_v = DeepONetDataset(train_raw[n_train:], n_query=n_query,
                           nu=train_nu[n_train:])

    if len(val_mix) > 0:
        ds_b     = DeepONetDataset(val_mix, n_query=n_query, nu=val_nu)
        train_ds = ConcatDataset([ds_a, ds_b])
        weights  = [1.0] * len(ds_a) + [val_mix_ratio] * len(ds_b)
        sampler  = WeightedRandomSampler(weights, len(weights),
                                         replacement=True)
    else:
        train_ds = ds_a
        sampler  = None

    if distributed:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=DistributedSampler(train_ds, world_size, rank,
                                       shuffle=True),
            num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(
            ds_v, batch_size=batch_size,
            sampler=DistributedSampler(ds_v, world_size, rank,
                                       shuffle=False),
            num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler, shuffle=(sampler is None),
            num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(
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
    print("dataset_task2.py unittest")
    print("=" * 50)
    errors = []

    def check(name, fn):
        try:
            fn()
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            errors.append((name, str(e)))

    print("\n[1] load_train_data(n_samples=5, n_val_mix=3)")
    def t1():
        tr, vm, tnu, vnu = load_train_data(n_samples=5, n_val_mix=3)
        assert tr.shape  == (15, 210, 256), f"train shape: {tr.shape}"
        assert vm.shape  == (3,  210, 256), f"val_mix shape: {vm.shape}"
        assert tnu.shape == (15,),          f"train_nu shape: {tnu.shape}"
        assert vnu.shape == (3,),           f"val_nu shape: {vnu.shape}"
        assert tr.dtype  == np.float32
        assert not np.isnan(tr).any()
    check("load_train_data shape/dtype/NaN", t1)

    print("\n[2] load_train_data(n_val_mix=0)")
    def t2():
        tr, vm, tnu, vnu = load_train_data(n_samples=5, n_val_mix=0)
        assert vm.shape == (0, 210, 256)
        assert vnu.shape == (0,)
    check("load_train_data n_val_mix=0", t2)

    print("\n[3] load_eval_data()")
    def t3():
        ev, nu = load_eval_data()
        assert ev.shape == (20, 210, 256), f"shape: {ev.shape}"
        assert nu.shape == (20,),          f"nu shape: {nu.shape}"
        assert ev.dtype == np.float32
    check("load_eval_data", t3)

    print("\n[4] load_test_data()")
    def t4():
        te = load_test_data()
        assert te.shape == (1000, 10, 256), f"shape: {te.shape}"
        assert te.dtype == np.float32
    check("load_test_data", t4)

    print("\n[5] DeepONetDataset")
    def t5():
        tr, _, tnu, _ = load_train_data(n_samples=5, n_val_mix=0)
        ds = DeepONetDataset(tr, n_query=128, nu=tnu)
        assert len(ds) == 15
        u0, coords, u_gt, nu = ds[0]
        assert u0.shape     == (10, 256), f"u0: {u0.shape}"
        assert coords.shape == (128, 2),  f"coords: {coords.shape}"
        assert u_gt.shape   == (128,),    f"u_gt: {u_gt.shape}"
        assert nu.shape     == (),        f"nu shape: {nu.shape}"
        assert float(coords.min()) >= 0.0 and float(coords.max()) <= 1.0
    check("DeepONetDataset shape/range", t5)

    print("\n[6] make_dataloaders")
    def t6():
        tl, vl = make_dataloaders(
            batch_size=4, n_query=128, n_samples=5, n_val_mix=3)
        u0, coords, u_gt, nu = next(iter(tl))
        assert u0.shape     == (4, 10, 256)
        assert coords.shape == (4, 128, 2)
        assert u_gt.shape   == (4, 128)
        assert nu.shape     == (4,),  f"nu shape: {nu.shape}"
    check("make_dataloaders batch shape", t6)

    print("\n" + "=" * 50)
    if errors:
        print(f"❌ {len(errors)} 个测试失败:")
        for name, err in errors:
            print(f"   {name}: {err}")
    else:
        print("✅ 所有测试通过")
    print("=" * 50)