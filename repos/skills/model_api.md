# model.py 接口规范

此文件由框架层维护，Agent 不需要也不能修改。
Agent 在 train.py 和 predict.py 中直接 import 使用。

## 导入方式

```python
from model import DeepONet, count_params
```

## DeepONet 架构说明

DeepONet 范式A：直接算子映射，无自回归误差累积。
- Branch net：编码初始10步 → latent vector (B, latent_dim)
- Trunk net：编码查询坐标 (t,x) → basis vectors (B, N_q, latent_dim)
- 输出：dot(branch, trunk) + bias = u(x,t)

## 构造函数

```python
model = DeepONet(
    T_in=10,              # 初始条件步数（固定10）
    X=256,                # 空间点数（固定256）
    latent_dim=128,       # 共享表示维度（可调）
    branch_type="cnn",    # "cnn" | "mlp" | "fno_encoder"
    branch_depth=4,       # Branch 层数（可调）
    branch_width=128,     # Branch 宽度（可调）
    trunk_depth=4,        # Trunk 层数（可调）
    trunk_width=256,      # Trunk 宽度（可调）
    activation="gelu",    # "gelu" | "tanh" | "silu"
    ff_dim=64,            # Fourier 特征维度（可调）
)
```

## Branch Net 选择建议

| branch_type | 特点 | 适用场景 |
|---|---|---|
| `"cnn"` | 1D卷积保留空间结构，全局平均池化 | 推荐，性能最好 |
| `"mlp"` | flatten后全连接，最简单 | 基线对比 |
| `"fno_encoder"` | Fourier谱卷积，频域特征 | 高频 shock 结构 |

## 前向接口（训练时用）

```python
# 输入
u0     = torch.tensor(...)  # (B, 10, 256)  float32  cuda
coords = torch.tensor(...)  # (B, N_q, 2)   float32  cuda  值域[0,1]

# 前向
pred = model(u0, coords)    # (B, N_q)      float32  cuda
```

## 推理接口（predict.py 里用）

```python
model.eval()
u0   = torch.tensor(val_gt[:, :10, :], dtype=torch.float32).to(device)
pred = model.predict_full(
    u0,
    t_steps=190,    # 预测时间步数（默认190）
    x_points=256,   # 空间点数（默认256）
)
# 返回: numpy (B, t_steps, x_points)  float32  CPU
# 内部分批处理（每批50步×256点）防 OOM
```

## Checkpoint 规范

**保存**（train.py 必须按此格式）：
```python
torch.save({
    "epoch":            epoch,
    "model_state_dict": raw_model.state_dict(),  # raw_model = DDP 解包后
    "val_loss":         val_loss,                # float
}, CKPT_PATH)
```

**加载**（predict.py 使用）：
```python
ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
model = DeepONet()   # 参数与训练时一致
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"val_loss={ckpt['val_loss']}")
```

## 参数量参考

默认配置（branch_depth=4, branch_width=128, latent_dim=128, trunk_width=256）：

| branch_type | 参数量 |
|---|---|
| cnn | ~3.5M |
| mlp | ~8M |
| fno_encoder | ~3M |

## 调优建议

| 目标 | 调整方式 |
|---|---|
| 提升 Seg1（短期精度） | 增大 branch_depth/branch_width/latent_dim |
| 提升高频特征捕捉 | 增大 ff_dim 或改用 fno_encoder |
| 减少显存 | 减小 trunk_width 或 latent_dim |
| 加快训练 | 减小 branch_depth，latent_dim=64 |

## 常见错误

```python
# ❌ 错误：类名不存在
from model import Agent
from model import BurgersNet

# ✅ 正确
from model import DeepONet

# ❌ 错误：参数名不对
DeepONet(input_dim=100, output_dim=190)

# ✅ 正确：使用默认值即可
DeepONet()
DeepONet(branch_type="cnn", latent_dim=256)

# ❌ 错误：非法 branch_type 会抛出 ValueError
DeepONet(branch_type="transformer")
```