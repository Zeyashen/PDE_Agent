# ddp_runner.py 接口规范

框架层文件，Agent 在 train.py 里直接 import 使用。
封装了所有 DDP 样板代码，Agent 无需了解底层细节。

## 导入方式

```python
from ddp_runner import (
    DDPContext,
    sync_early_stop,
    save_checkpoint,
    load_checkpoint,
    make_logger,
)
```

---

## DDPContext

DDP 初始化和管理，单卡时自动退化为普通模式。

```python
ctx = DDPContext()
```

### 属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `ctx.rank` | int | 当前进程 rank（单卡=0） |
| `ctx.local_rank` | int | 当前 GPU 编号（单卡=0） |
| `ctx.world_size` | int | 总进程数（单卡=1） |
| `ctx.is_main` | bool | 是否为 rank 0 |
| `ctx.device` | torch.device | 当前设备 |

### 方法

```python
# 把模型移到 device 并包装为 DDP
model = ctx.wrap(model)

# 取出原始模型（保存 checkpoint 前用）
raw_model = ctx.unwrap(model)

# 给 DistributedSampler 设置 epoch（训练循环开头调用）
ctx.set_epoch(train_loader, epoch)

# 清理 DDP 进程组（训练结束后调用）
ctx.cleanup()
```

---

## sync_early_stop

跨进程同步早停信号，rank 0 判断，广播给所有进程。

```python
should_stop = sync_early_stop(
    no_improve,       # 连续未改善的 epoch 数
    patience,         # 早停耐心值
    ctx.is_main,      # 是否为 rank 0
    ctx.world_size,   # 总进程数
)
# 返回 True 表示应该停止
```

---

## save_checkpoint / load_checkpoint

```python
# 保存（只在 rank 0 调用）
if ctx.is_main and val_loss < best_val_loss:
    save_checkpoint(
        ctx.unwrap(model),   # 原始模型（已 unwrap）
        epoch=epoch,
        val_loss=val_loss,
        ckpt_path=None,      # None = 默认路径 workspace/checkpoints/best.pt
        extra={"lr": 0.001}, # 可选：额外信息写入 checkpoint
    )

# 加载
ckpt = load_checkpoint(
    ckpt_path=None,   # None = 默认路径 workspace/checkpoints/best.pt
    device=ctx.device,
)
if ckpt is not None:
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["val_loss"]
```

**checkpoint 格式**：
```python
{
    "epoch":            int,
    "model_state_dict": OrderedDict,
    "val_loss":         float,
    # extra 里的字段也会保存进来
}
```

**默认路径**：`/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/workspace/checkpoints/best.pt`

---

## make_logger

创建训练 logger，rank 0 写文件+stdout，其他进程只写 WARNING。

```python
logger = make_logger(
    name="train",           # logger 名称
    log_filename="train.log", # 日志文件名（写到 logs/ 目录）
    is_main=ctx.is_main,    # 是否为 rank 0
)

logger.info(f"Epoch {epoch}: val={val_loss:.6f}")
```

---

## 完整 train.py 模板

```python
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import DeepONet
from dataset import make_dataloaders
from ddp_runner import (
    DDPContext, sync_early_stop,
    save_checkpoint, load_checkpoint, make_logger,
)

CKPT_PATH = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/workspace/checkpoints/best.pt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--n_query",      type=int,   default=2048)
    parser.add_argument("--val_mix_ratio",type=float, default=2.0)
    parser.add_argument("--n_samples",    type=int,   default=None)
    args = parser.parse_args()

    # ── 初始化 DDP ────────────────────────────────────────────
    ctx    = DDPContext()
    logger = make_logger("train", "train.log", ctx.is_main)

    # ── 数据 ─────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(
        batch_size=args.batch_size,
        n_query=args.n_query,
        val_mix_ratio=args.val_mix_ratio,
        n_samples=args.n_samples,
        distributed=(ctx.world_size > 1),
        rank=ctx.rank,
        world_size=ctx.world_size,
    )

    # ── 模型 ─────────────────────────────────────────────────
    model     = DeepONet(branch_type="cnn", latent_dim=128)
    model     = ctx.wrap(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

    best_val_loss = float("inf")
    no_improve    = 0

    # ── 训练循环 ──────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        ctx.set_epoch(train_loader, epoch)

        model.train()
        train_loss = 0.0
        for u0, coords, u_gt in train_loader:
            u0, coords, u_gt = (u0.to(ctx.device),
                                coords.to(ctx.device),
                                u_gt.to(ctx.device))
            optimizer.zero_grad()
            pred = model(u0, coords)
            # ===== AGENT_LOSS_BEGIN =====
            loss = (pred - u_gt).norm() / (u_gt.norm() + 1e-8)
            # ===== AGENT_LOSS_END =====
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # 验证（只在 rank 0）
        if ctx.is_main:
            ctx.unwrap(model).eval()
            val_loss = 0.0
            with torch.no_grad():
                for u0, coords, u_gt in val_loader:
                    u0, coords, u_gt = (u0.to(ctx.device),
                                        coords.to(ctx.device),
                                        u_gt.to(ctx.device))
                    pred      = ctx.unwrap(model)(u0, coords)
                    val_loss += ((pred-u_gt).norm()/(u_gt.norm()+1e-8)).item()
            val_loss /= max(len(val_loader), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                save_checkpoint(ctx.unwrap(model), epoch, val_loss)
                marker = " <- best"
            else:
                no_improve += 1
                marker = f" (no improve {no_improve}/{args.patience})"

            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                logger.info(f"Epoch {epoch:4d}/{args.epochs} "
                            f"train={train_loss:.6f} val={val_loss:.6f}{marker}")

        scheduler.step()

        # 早停
        if sync_early_stop(no_improve, args.patience, ctx.is_main, ctx.world_size):
            if ctx.is_main:
                logger.info(f"早停: {args.patience} epoch 无改善")
            break

    if ctx.is_main:
        logger.info(f"训练完成 best_val_loss={best_val_loss:.6f}")
        logger.info(f"Candidate: {CKPT_PATH}")

    ctx.cleanup()

if __name__ == "__main__":
    main()
```

## Agent 可修改的部分

`train.py` 中 Agent 主要修改以下部分：

| 位置 | 说明 |
|---|---|
| `argparse` 参数默认值 | 超参数调整 |
| `DeepONet(...)` 构造参数 | 模型架构选择 |
| `AGENT_LOSS_BEGIN/END` 之间 | 损失函数设计 |
| `make_dataloaders` 参数 | 数据配置 |

其余（DDP 初始化、早停同步、checkpoint 保存）均由框架函数处理，不需要修改。