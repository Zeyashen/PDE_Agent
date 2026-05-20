# DeepONet 设计指南

## 核心思想：范式A（直接算子映射）

不用自回归（FNO 每步预测再滚动），直接映射：
  初始条件 u(x, t=0~0.5s) → 任意时刻 u(x, t)

优势：无自回归误差累积，190步预测误差不叠加。

## 标准架构

```
Branch net: 编码初始10步 → latent vector  (B, p)
  输入: (B, 10, 256)
  输出: (B, latent_dim)

Trunk net:  编码查询坐标 (t_norm, x_norm) → basis  (B, N_q, p)
  输入: (B, N_query, 2)
  输出: (B, N_query, latent_dim)

输出: sum(branch × trunk, dim=-1) + bias = u(x,t)
  (B, latent_dim) × (B, N_q, latent_dim) → (B, N_q)
```

## Branch Net 变体

### MLP Branch（最简单）
```python
# 输入 flatten: (B, 10*256=2560) → MLP → (B, latent_dim)
layers = [Linear(2560, width), Act()] + \
         [Linear(width, width), Act()] * (depth-1) + \
         [Linear(width, latent_dim)]
```

### CNN Branch（推荐，Task1最优）
```python
# 输入: (B, 10, 256) 视为10通道1D信号
# Conv1d(10, width, kernel=5) → ... → GlobalAvgPool → Linear(width, latent_dim)
```

### FNO Encoder Branch（频域特征）
```python
# 用 SpectralConv1d 提取频域特征，然后 GlobalAvgPool → Linear
```

## Trunk Net（含 Fourier 特征嵌入）

```python
class TrunkNet(nn.Module):
    def __init__(self, latent_dim, depth, width, fourier_features=64):
        # 随机频率矩阵（固定不训练）
        self.register_buffer('B_mat',
            torch.randn(2, fourier_features) * 10.0)
        # in_dim = 2 * fourier_features（sin+cos）
        in_dim = 2 * fourier_features
        # MLP: in_dim → width → ... → latent_dim

    def forward(self, coords):
        # coords: (B, N_q, 2)
        proj = coords @ self.B_mat   # (B, N_q, ff_dim)
        ff   = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(ff)           # (B, N_q, latent_dim)
```

Fourier 特征嵌入对捕捉 shock 结构的高频成分至关重要。

## 训练策略

```python
# 训练：子采样查询点（省内存）
idx    = torch.randperm(T_pred * X)[:n_query]
coords = full_coords[idx]   # (n_query, 2)
u_gt   = u_future.flatten()[idx]

# 损失函数
def relative_l2_loss(pred, target):
    return (pred - target).norm() / (target.norm() + 1e-8)

# 推理：全量查询（分批防OOM）
for start in range(0, T*X, batch_size):
    pred_flat[:, start:end] = model(u0, coords[start:end])
```

## 数据混合策略（Task 1）

训练数据只覆盖 t=0~1.95s，但预测需到 t=9.96s（覆盖率仅20%）。
解决方案：将 task1_val 前80条（覆盖0~9.96s）混入训练。

```python
# val_mix_ratio 控制 val 数据的采样权重
# < 1.0: 更多短期数据 → 改善 Seg1
# > 1.0: 更多长期数据 → 改善 Seg3
weights = [1.0] * len(train_data) + [val_mix_ratio] * len(val_train_data)
sampler = WeightedRandomSampler(weights, len(weights))
```

## 关键超参数

| 参数 | 含义 | 推荐范围 |
|------|------|----------|
| branch_type | MLP/CNN/FNO_encoder | CNN 最优 |
| branch_depth | Branch 层数 | 4~8 |
| branch_width | Branch 宽度 | 128~512 |
| trunk_depth | Trunk 层数 | 4~6 |
| trunk_width | Trunk 宽度 | 128~256 |
| latent_dim | 共享表示维度 | 64~256 |
| fourier_features | Fourier 嵌入维度 | 64~128 |
| n_query | 训练时查询点数 | 2048~4096 |
| val_mix_ratio | Val 数据采样权重 | 0.3~2.0 |

## 设计建议（基于理论分析和文献先验）

### Branch Net 容量
- depth=2, width=64 容量严重不足，无法表达 Burgers shock 结构
- 推荐起点：depth=4, width=128, latent_dim=128
- 更大模型：depth=6, width=256, latent_dim=256（Task2 多Nu泛化时考虑）

### Seg1 得分优化
- Seg1 评分公式 exp(-20×RelMSE) 对误差极敏感
- RelMSE=0.05 才能得到37分，RelMSE=0.02 才能得67分
- 主要影响因素：branch net 的特征提取能力
- val_mix_ratio < 1.0 可以让模型更多学习短期动力学

### Fourier 特征嵌入
- Burgers 方程在 shock 形成后有强高频成分
- fourier_features=64~128 对捕捉这些高频特征至关重要
- 不加 Fourier 嵌入的 trunk net 会显著降低 Seg1 和 Seg2 精度

### 推理时间控制
- DeepONet 范式A对 1000 个样本推理约 0.3~1s，远低于 2 分钟限制
- 分批处理查询点（batch_size=50个时间步×256空间点）防止 OOM
