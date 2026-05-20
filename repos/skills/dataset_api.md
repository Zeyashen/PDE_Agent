# dataset.py 接口规范

此文件由框架层维护，Agent 不需要也不能修改。
Agent 在 train.py 和 predict.py 中直接 import 使用。

## 导入方式

```python
from dataset import (
    DeepONetDataset,
    load_train_data,
    load_eval_data,
    load_test_data,
    make_dataloaders,
)
```

## 数据说明

| 数据集 | 时间范围 | 步数 | shape | 用途 |
|---|---|---|---|---|
| 训练集（下采样后） | t=0.01~2.01s | 40步 | (N, 40, 256) | 短时训练 |
| val 前80条 | t=0.01~9.96s | 200步 | (80, 200, 256) | 长时训练（关键） |
| val 后20条 | t=0.01~9.96s | 200步 | (20, 200, 256) | 本地评分 |
| test | t=0.01~0.46s | 10步 | (1000, 10, 256) | 提交预测 |

**重要**：所有数据集的时间从 t=0.01 开始（不含 t=0），已对齐。

## 接口详情

### load_train_data()

```python
train_raw, val_mix = load_train_data(
    n_samples=None,   # 训练集取多少条，None=全部10000条
    n_val_mix=80,     # val 长时数据混入多少条（0=不混入）
    downsample_t=5,   # 时间下采样倍数
    downsample_x=4,   # 空间下采样倍数
)
# train_raw: numpy (N, 40, 256) float32  短时数据 t=0.01~2.01s
# val_mix:   numpy (M, 200, 256) float32 长时数据 t=0.01~9.96s
#            M=0 时返回 shape=(0, 200, 256) 空数组
```

**调整长时数据比例**：
```python
# 增大长时比例（改善 Seg2/3）
train_raw, val_mix = load_train_data(n_val_mix=100)

# smoke test：最小数据量
train_raw, val_mix = load_train_data(n_samples=200, n_val_mix=10)
```

### load_eval_data()

```python
val_gt = load_eval_data()
# 返回: numpy (20, 200, 256) float32
# 用途: 本地评分，计算 Seg1/2/3，不能用于训练
```

### load_test_data()

```python
test_data = load_test_data()
# 返回: numpy (1000, 10, 256) float32
# 用途: 生成最终提交文件 task1_pred.hdf5
```

### make_dataloaders()

```python
train_loader, val_loader = make_dataloaders(
    batch_size=32,        # 每卡 batch size
    n_query=2048,         # 每样本查询点数
    val_mix_ratio=2.0,    # 长时数据采样权重（>1.0 → 更多长时）
    n_val_mix=80,         # val 长时数据混入条数
    n_samples=None,       # 训练集取多少条
    distributed=False,    # DDP 时设 True
    rank=0,
    world_size=1,
)
# 每个 batch 返回三元组:
#   u0:    (B, 10, 256)    初始10步
#   coords:(B, n_query, 2) 归一化时空坐标 [0,1]
#   u_gt:  (B, n_query)    真实解
```

### DeepONetDataset

```python
ds = DeepONetDataset(
    data,          # numpy (N, T, 256)
    T_in=10,       # 初始条件步数
    n_query=2048,  # 查询点数
)
# len(ds) == N
# ds[i] → (u0, coords, u_gt)
#   u0:    (T_in, 256)
#   coords:(n_query, 2)  值域 [0,1]
#   u_gt:  (n_query,)
```

## 调优建议

| 目标 | 调整方式 |
|---|---|
| 改善 Seg1（短期精度） | 减小 val_mix_ratio 或 n_val_mix |
| 改善 Seg2/3（长期精度） | 增大 val_mix_ratio 或 n_val_mix |
| 加快 smoke test | n_samples=200, n_val_mix=10, n_query=128 |
| 完整训练 | n_samples=None, n_val_mix=80, n_query=2048 |