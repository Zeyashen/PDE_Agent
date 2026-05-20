# 完整工作流指南

Agent 的工作范围：只需编写 train.py 和 predict.py。
其余文件由框架层提供，直接 import 使用即可。


# 工作阶段建议（软引导，不强制）

Agent 的工作分为以下阶段。每完成一个阶段，建议用 todo_write 更新任务状态。

## Phase 0: 准备代码（5-10分钟）
目标：写好 train.py 和 predict.py
退出条件：workspace/ 下存在 train.py 和 predict.py
建议工具调用：
  1. read_file("skills/workflow.md")          # 获取完整模板（含 train.py/predict.py）
  2. write_file("train.py", ...)             # 写训练脚本
  3. write_file("predict.py", ...)           # 写推理脚本

## Phase 1: 单 batch 测试（2-3分钟）
目标：快速验证代码无明显 bug，不浪费时间在错的方向
退出条件：用极小配置（batch_size=2, epochs=1, n_query=128）训练成功
建议工具调用：
  1. run_training("train.py", extra_args="--epochs 1 --batch_size 2 --n_query 128", n_gpus=1)
  说明：单卡、单 epoch、小 batch，几十秒内必须出 val_loss

## Phase 2: 快速基线（5-10分钟）
目标：用合理配置训练 10 epoch，得到第一个基线分数
退出条件：拿到 Seg1/Seg2/Seg3 真实得分
建议工具调用：
  1. run_training("train.py", epochs=10, n_gpus=4)
  2. quick_eval("checkpoints/best.pt")
  3. full_eval("checkpoints/best.pt")

## Phase 3: 诊断（每次评分后必做）
目标：识别瓶颈，决定下一步改进方向
建议分析角度：
  - Seg1 低（<30）：短期精度差，原因通常是 branch net容量不足，建议加深加宽
  - Seg2 低（<50）：中期累积误差，建议增大 val_mix_ratio
  - Seg3 低（<60）：长期发散，建议更大 latent_dim 或更多 Fourier features
  - val_loss 在 epoch 5 之前就停止下降：lr 过大或模型容量不足
  - 训练损失 NaN：lr 过大或没有梯度裁剪

## Phase 4: 主体训练（剩余时间）
目标：基于诊断结果改进模型，得到最优分数
建议工作流（每轮迭代）：
  1. edit_file 修改 train.py 中的超参或损失函数
  2. run_training（epochs 30-50，4 GPU）
  3. quick_eval 看 val_loss 是否改善
  4. 改善则 full_eval 验证真实得分，否则回到步骤 1

## Phase 5: 最终提交（5分钟）
目标：用最优 checkpoint 生成测试集预测
建议工具调用：
  1. full_eval("checkpoints/best.pt", val_only=False)  # 生成 task1_pred.hdf5

## 通用原则

每轮只做 1-3 件最重要的事，不要一次想做完所有阶段。
LLM 每轮 observation 会显示当前 Phase 提示和建议工具调用。
状态判断逻辑由 orchestrator 自动完成，你只需按提示执行。

## 时间预算

Task 1（55分钟总预算）：
  Phase 0+1+2: 约 15-20 分钟（写代码+单batch+10epoch基线）
  Phase 3+4:   约 30 分钟（2-3 轮诊断+改进）
  Phase 5:     约 5 分钟（最终提交）
  缓冲:        5 分钟

Task 2（600分钟总预算）：
  Phase 0+1:   约 10 分钟
  Phase 2:     约 30 分钟（更大数据集，10 epoch 也较慢）
  Phase 3+4:   约 500 分钟（多轮充分迭代）
  Phase 5:     约 10 分钟
---

## 框架层文件（已提供，不需要写）

| 文件 | 提供内容 | 详细文档 |
|---|---|---|
| `model.py` | DeepONet 模型（三种 branch type） | skills/model_api.md |
| `dataset.py` | 数据加载函数 | skills/dataset_api.md |
| `ddp_runner.py` | DDP 样板封装 | skills/ddp_runner_api.md |
| `orchestrator.py` | Agent 主循环（框架层，不用管） | — |

---

## Agent 需要编写的文件

### train.py — 训练脚本

```python
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# 框架层提供，直接 import
from model import DeepONet
from dataset import make_dataloaders
from ddp_runner import (
    DDPContext, sync_early_stop,
    save_checkpoint, make_logger,
)

CKPT_PATH = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/workspace/checkpoints/best.pt"

def main():
    parser = argparse.ArgumentParser()
    # ── Agent 可调整的超参数 ──────────────────────────────────
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--n_query",       type=int,   default=2048)
    parser.add_argument("--val_mix_ratio", type=float, default=2.0)
    parser.add_argument("--n_val_mix",     type=int,   default=80)
    parser.add_argument("--n_samples",     type=int,   default=None)
    # 模型架构
    parser.add_argument("--branch_type",  default="cnn")
    parser.add_argument("--latent_dim",   type=int, default=128)
    parser.add_argument("--branch_depth", type=int, default=4)
    parser.add_argument("--branch_width", type=int, default=128)
    parser.add_argument("--trunk_depth",  type=int, default=4)
    parser.add_argument("--trunk_width",  type=int, default=256)
    args = parser.parse_args()

    # ── DDP 初始化 ────────────────────────────────────────────
    ctx    = DDPContext()
    logger = make_logger("train", "train.log", ctx.is_main)
    if ctx.is_main:
        logger.info(f"训练配置: {vars(args)}")

    # ── 数据加载 ──────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(
        batch_size=args.batch_size,
        n_query=args.n_query,
        val_mix_ratio=args.val_mix_ratio,
        n_val_mix=args.n_val_mix,
        n_samples=args.n_samples,
        distributed=(ctx.world_size > 1),
        rank=ctx.rank,
        world_size=ctx.world_size,
    )

    # ── 模型构建 ──────────────────────────────────────────────
    model = DeepONet(
        branch_type=args.branch_type,
        latent_dim=args.latent_dim,
        branch_depth=args.branch_depth,
        branch_width=args.branch_width,
        trunk_depth=args.trunk_depth,
        trunk_width=args.trunk_width,
    )
    model     = ctx.wrap(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.lr * 0.01)

    best_val_loss = float("inf")
    no_improve    = 0

    # ── 训练循环 ──────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        ctx.set_epoch(train_loader, epoch)

        model.train()
        train_loss = 0.0
        for u0, coords, u_gt in train_loader:
            u0     = u0.to(ctx.device)
            coords = coords.to(ctx.device)
            u_gt   = u_gt.to(ctx.device)

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

        # ── 验证（只在 rank 0） ───────────────────────────────
        if ctx.is_main:
            raw = ctx.unwrap(model)
            raw.eval()
            val_loss = 0.0
            with torch.no_grad():
                for u0, coords, u_gt in val_loader:
                    u0     = u0.to(ctx.device)
                    coords = coords.to(ctx.device)
                    u_gt   = u_gt.to(ctx.device)
                    pred   = raw(u0, coords)
                    val_loss += ((pred - u_gt).norm() /
                                 (u_gt.norm() + 1e-8)).item()
            val_loss /= max(len(val_loader), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                save_checkpoint(raw, epoch, val_loss, extra={
                    "latent_dim":   args.latent_dim,
                    "branch_type":  args.branch_type,
                    "branch_depth": args.branch_depth,
                    "branch_width": args.branch_width,
                    "trunk_depth":  args.trunk_depth,
                    "trunk_width":  args.trunk_width,
})
                marker = " <- best"
            else:
                no_improve += 1
                marker = f" (no improve {no_improve}/{args.patience})"

            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                logger.info(f"Epoch {epoch:4d}/{args.epochs} "
                            f"train={train_loss:.6f} "
                            f"val={val_loss:.6f}{marker}")

        scheduler.step()

        # ── 早停 ─────────────────────────────────────────────
        if sync_early_stop(no_improve, args.patience,
                           ctx.is_main, ctx.world_size):
            if ctx.is_main:
                logger.info(f"早停: {args.patience} epoch 无改善")
            break

    if ctx.is_main:
        logger.info(f"训练完成 best_val_loss={best_val_loss:.6f}")
        logger.info(f"Candidate: {CKPT_PATH}")

    ctx.cleanup()

if __name__ == "__main__":
    main()