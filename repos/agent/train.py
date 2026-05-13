"""
Task 1 训练脚本 — FNO on 1D Burgers

核心改动
--------
1. Pushforward trick (k=3)
2. Candidate 机制（不直接覆盖 best_model.pt）
3. 数据混合（原始训练集 + val前80条）
4. P0-2 修复：resume 时架构参数强制沿用 saved
5. Physics loss dt 修正：dt=0.05
6. 代码注入 marker：AGENT_LOSS_BEGIN/END
"""

import argparse
import csv
import os
import sys
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from model import FNO1d, load_official_checkpoint
from dataset import make_dataloaders


# ===================== DDP =====================
def init_ddp():
    if "RANK" not in os.environ:
        return 0, 0, 1, True
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, (rank == 0)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ===================== 日志 =====================
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(log_path, is_main):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    if is_main:
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if not is_main:
        logger.setLevel(logging.WARNING)
    return logger


# ===================== 损失函数 =====================
def relative_l2_loss(pred, target):
    """相对 L2 损失。pred/target: (B, 1, X)"""
    diff = pred - target
    return (diff.norm(dim=-1) / (target.norm(dim=-1) + 1e-8)).mean()


def physics_loss_burgers(pred, u_prev, nu=0.001, dx=1/256, dt=0.05):
    """
    Burgers 物理残差损失。
    dt=0.05：val/test 数据颗粒度（不是原始训练集的 0.01）
    """
    u_x  = (torch.roll(pred, -1, -1) - torch.roll(pred,  1, -1)) / (2 * dx)
    u_xx = (torch.roll(pred, -1, -1) - 2*pred + torch.roll(pred, 1, -1)) / (dx**2)
    u_t  = (pred - u_prev) / dt
    return ((u_t + pred * u_x - nu * u_xx) ** 2).mean()


# ===================== 评估 =====================
def compute_rel_mse(pred, gt):
    eps  = 1e-6
    diff = pred - gt
    num  = (diff**2).sum(dim=-1)
    den  = (gt**2).sum(dim=-1) + eps
    return (num / den).mean().item()


# ===================== 主训练函数 =====================
def train(args):
    root      = Path(__file__).parent.parent.parent
    ckpt_dir  = root / "checkpoints" / "task1_trained"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path      = ckpt_dir / "best_model.pt"
    candidate_path = ckpt_dir / "best_model_candidate.pt"
    log_path       = LOG_DIR / "task1_train.log"
    time_csv       = root / "repos" / "submission" / "task1_time.csv"

    rank, local_rank, world_size, is_main = init_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(log_path, is_main)

    if is_main:
        logger.info("=" * 60)
        logger.info("Task 1 训练开始")
        logger.info(f"配置: {vars(args)}")
        logger.info(f"DDP: world_size={world_size}, rank={rank}, device={device}")
        logger.info(f"Pushforward k={args.pushforward_k}")

    # ---------- 数据 ----------
    t0 = time.time()
    train_loader, val_loader = make_dataloaders(
        train_hdf5_path      = args.data_path,
        val_hdf5_path        = args.val_path,
        batch_size           = args.batch_size,
        n_samples            = args.n_samples,
        num_workers          = args.num_workers,
        reduced_resolution_t = 5,
        reduced_resolution   = 4,
        window_size          = 10,
        rollout_k            = args.pushforward_k,
        val_train_n          = args.val_train_n,
        train_ratio          = 0.9,
        distributed          = (world_size > 1),
        rank                 = rank,
        world_size           = world_size,
    )
    if is_main:
        logger.info(f"数据加载完成，耗时 {time.time()-t0:.1f}s")

    # ---------- 模型 ----------
    start_epoch   = 1
    best_val_loss = float('inf')

    if args.resume and ckpt_path.exists():
        if is_main:
            logger.info(f"断点续训：加载 {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved = ckpt.get("args", {})
        for k in ("modes", "width", "n_layers"):
            if k in saved and saved[k] != getattr(args, k):
                if is_main:
                    logger.warning(
                        f"resume: 忽略 CLI --{k}={getattr(args,k)}，沿用 saved {k}={saved[k]}"
                    )
                setattr(args, k, saved[k])
        model = FNO1d(
            modes=args.modes, width=args.width,
            in_channels=11, out_channels=1, n_layers=args.n_layers,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        last_epoch    = ckpt.get("epoch", 1)
        start_epoch   = last_epoch + 1
        best_val_loss = ckpt.get("val_loss", float('inf'))
        if start_epoch > args.epochs:
            if is_main:
                logger.info(f"已达 epoch {last_epoch}，额外再训 {args.epochs} epochs")
            args.epochs = last_epoch + args.epochs
        if is_main:
            logger.info(f"从 epoch {start_epoch} 继续，val_loss={best_val_loss:.6f}")

    elif args.finetune and args.official_ckpt:
        if is_main:
            logger.info(f"微调模式：{args.official_ckpt}")
        model = load_official_checkpoint(args.official_ckpt, device=device)
        for attr in ('modes', 'width', 'n_layers'):
            if hasattr(model, attr):
                setattr(args, attr, getattr(model, attr))
        if is_main:
            logger.info(f"finetune 架构: modes={args.modes} width={args.width} n_layers={args.n_layers}")

    else:
        if is_main:
            logger.info(f"冷启动: modes={args.modes} width={args.width} n_layers={args.n_layers}")
        model = FNO1d(
            modes=args.modes, width=args.width,
            in_channels=11, out_channels=1, n_layers=args.n_layers,
        ).to(device)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数量: {n_params:,}")

    # ---------- 优化器 ----------
    remaining = args.epochs - (start_epoch - 1)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(remaining, 1), eta_min=args.lr*0.01)

    if args.resume and ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            pass

    # ---------- 训练循环 ----------
    X_r        = 256
    grid       = torch.linspace(0, 1, X_r).view(1, 1, X_r).to(device)
    no_improve = 0
    train_start = time.time()

    if is_main:
        logger.info(f"训练 {args.epochs} epochs（从 {start_epoch}），patience={args.patience}")
        logger.info("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        if world_size > 1 and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        n_batches  = 0

        for inp, targets in train_loader:
            # inp    : (B, window_size+1, X)  最后行是空间坐标
            # targets: (B, rollout_k, 1, X)
            inp     = inp.to(device)
            targets = targets.to(device)
            B       = inp.shape[0]
            history = inp[:, :-1, :]            # (B, 10, X)
            grid_b  = grid.expand(B, -1, -1)    # (B, 1, X)

            optimizer.zero_grad()

            # ===== AGENT_LOSS_BEGIN =====
            # Pushforward trick：unroll rollout_k 步，累积相对 L2 loss
            # LLM 可用 custom_loss(model, history, grid_b, targets, args) 替换此段
            k      = targets.shape[1]
            loss   = torch.tensor(0.0, device=device)
            window = history.clone()

            for step in range(k):
                model_inp = torch.cat([window, grid_b], dim=1)  # (B, 11, X)
                pred      = model(model_inp)                     # (B, 1, X)
                tgt_step  = targets[:, step, :, :]              # (B, 1, X)
                step_loss = relative_l2_loss(pred, tgt_step)
                if args.physics_loss > 0:
                    u_prev    = window[:, -1:, :]
                    step_loss = step_loss + args.physics_loss * physics_loss_burgers(pred, u_prev)
                loss   = loss + step_loss
                window = torch.cat([window[:, 1:, :], pred.detach()], dim=1)

            loss = loss / k
            # ===== AGENT_LOSS_END =====

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        train_loss /= max(n_batches, 1)

        # --- 验证（主进程）---
        if is_main:
            raw_model = model.module if world_size > 1 else model
            raw_model.eval()
            val_loss = val_rel_mse = 0.0
            n_val    = 0
            with torch.no_grad():
                for inp, targets in val_loader:
                    inp     = inp.to(device)
                    targets = targets.to(device)
                    B       = inp.shape[0]
                    history = inp[:, :-1, :]
                    grid_b  = grid.expand(B, -1, -1)
                    pred    = raw_model(torch.cat([history, grid_b], dim=1))
                    tgt     = targets[:, 0, :, :]
                    val_loss    += relative_l2_loss(pred, tgt).item()
                    val_rel_mse += compute_rel_mse(pred, tgt)
                    n_val       += 1
            val_loss    /= max(n_val, 1)
            val_rel_mse /= max(n_val, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss':             val_loss,
                    'val_rel_mse':          val_rel_mse,
                    'args':                 vars(args),
                }, candidate_path)
                marker = " <- best (candidate)"
            else:
                no_improve += 1
                marker     = f" (no improve {no_improve}/{args.patience})"

            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                elapsed = time.time() - train_start
                logger.info(
                    f"Epoch {epoch:4d}/{args.epochs} | "
                    f"train={train_loss:.6f} | val={val_loss:.6f} | "
                    f"rel_mse={val_rel_mse:.6f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"elapsed={elapsed:.0f}s{marker}"
                )

        scheduler.step()

        if world_size > 1:
            flag = torch.tensor(
                1 if (is_main and no_improve >= args.patience) else 0
            ).cuda()
            dist.broadcast(flag, src=0)
            dist.barrier()
            if flag.item() == 1:
                if is_main:
                    logger.info(f"早停触发：连续 {args.patience} epoch 无改善")
                break
        else:
            if is_main and no_improve >= args.patience:
                logger.info(f"早停触发：连续 {args.patience} epoch 无改善")
                break

    # ---------- 收尾 ----------
    train_time = time.time() - train_start
    if is_main:
        logger.info("=" * 60)
        logger.info(f"训练完成！耗时: {train_time:.1f}s ({train_time/60:.1f}min)")
        logger.info(f"本轮最优 val_loss: {best_val_loss:.6f}")
        if candidate_path.exists():
            logger.info(f"Candidate -> {candidate_path}")
            logger.info("orchestrator 将用评分决定是否 promote")
        else:
            logger.warning("本轮未产出 candidate（val_loss 全程未改善）")
        time_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(time_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['train_time', 'inference_time'])
            w.writerow([train_time, 0])

    cleanup_ddp()
    return ckpt_path, train_time, log_path


# ===================== 参数解析 =====================
def parse_args():
    parser = argparse.ArgumentParser()
    root = Path(__file__).parent.parent.parent
    parser.add_argument('--data_path', default=str(
        root / 'data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5'))
    parser.add_argument('--val_path', default=str(
        root / 'data_and_sample_submission/train_val_test_init/task1_val.hdf5'))
    parser.add_argument('--n_samples',     type=int,   default=None)
    parser.add_argument('--val_train_n',   type=int,   default=80)
    parser.add_argument('--modes',         type=int,   default=16)
    parser.add_argument('--width',         type=int,   default=64)
    parser.add_argument('--n_layers',      type=int,   default=4)
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--patience',      type=int,   default=15)
    parser.add_argument('--pushforward_k', type=int,   default=3)
    parser.add_argument('--physics_loss',  type=float, default=0.0)
    parser.add_argument('--finetune',      action='store_true')
    parser.add_argument('--resume',        action='store_true')
    parser.add_argument('--official_ckpt', default=str(
        root / 'checkpoints/1D_Burgers/1D_Burgers_Sols_Nu0.001_FNO.pt'))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path, train_time, log_path = train(args)
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"耗时: {train_time/60:.1f} min")