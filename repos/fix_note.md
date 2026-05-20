# 待修复：Task1 predict.py 模型加载问题

## 问题描述
`predict.py` 加载模型时用硬编码默认参数 `DeepONet()`，
Agent 改过超参（如 `branch_width=256`）后会报 shape mismatch：
```
size mismatch for branch.convs.0.weight: copying a param with shape 
torch.Size([256, 10, 5]) from checkpoint, the shape in current model 
is torch.Size([128, 10, 5])
```

## 影响文件
`/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/agent/predict.py`

## 修复方法
从 checkpoint 里读取模型配置重建模型，替换以下代码：

**改前：**
```python
model  = DeepONet().to(device)
model.load_state_dict(ckpt["model_state_dict"])
```

**改后：**
```python
model  = DeepONet(
    branch_type  = ckpt.get("branch_type",  "cnn"),
    latent_dim   = ckpt.get("latent_dim",   128),
    branch_depth = ckpt.get("branch_depth", 4),
    branch_width = ckpt.get("branch_width", 128),
    trunk_depth  = ckpt.get("trunk_depth",  4),
    trunk_width  = ckpt.get("trunk_width",  256),
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
```

## 备注
- Task2 的 predict.py 已修复
- checkpoint 里已经保存了这些 key（由 ddp_runner.py 的 save_checkpoint 负责）
- Task1 workspace/predict.py 也需要同步修复