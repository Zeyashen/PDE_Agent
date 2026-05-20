import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import DeepONet
from dataset import make_dataloaders
from ddp_runner import (
    DDPContext, sync_early_stop,
    save_checkpoint, make_logger,
)

CKPT_PATH = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/workspace/checkpoints/best.pt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--n_query",       type=int,   default=2048)
    parser.add_argument("--val_mix_ratio", type=float, default=2.0)
    parser.add_argument("--n_val_mix",     type=int,   default=80)
    parser.add_argument("--n_samples",     type=int,   default=None)
    parser.add_argument("--branch_type",   default="cnn")
    parser.add_argument("--latent_dim",    type=int,   default=128)
    parser.add_argument("--branch_depth",  type=int,   default=4)
    parser.add_argument("--branch_width",  type=int,   default=256)
    parser.add_argument("--trunk_depth",   type=int,   default=4)
    parser.add_argument("--trunk_width",   type=int,   default=256)
    args = parser.parse_args()

    ctx    = DDPContext()
    logger = make_logger("train", "train.log", ctx.is_main)
    if ctx.is_main:
        logger.info(f"训练配置: {vars(args)}")

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
