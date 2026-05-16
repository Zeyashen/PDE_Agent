"""
train_t2.py — Task 2 训练脚本

复用 Task 1 的 DeepONet 模型结构，只改数据加载部分。
直接用 Task 1 最优 checkpoint (CNN, depth=4, width=128) 作为热启动。
"""

import argparse, csv, os, sys, time, logging, sqlite3
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist

ROOT     = Path(__file__).parent.parent.parent
LOG_DIR  = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints" / "task2_trained"
LOG_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, count_params
from dataset_t2 import make_task2_dataloaders

DATA_DIR = ROOT / "data_and_sample_submission" / "train_val_test_init"


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


def setup_logger(log_path, is_main):
    logger = logging.getLogger("train_t2")
    logger.handlers.clear()
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


def relative_l2_loss(pred, target):
    return (pred - target).norm() / (target.norm() + 1e-8)


def train(args):
    rank, local_rank, world_size, is_main = init_ddp()
    device   = torch.device(f'cuda:{local_rank}'
                            if torch.cuda.is_available() else 'cpu')
    log_path = LOG_DIR / "task2_train.log"
    logger   = setup_logger(log_path, is_main)

    exp_id         = args.exp_id
    ckpt_candidate = CKPT_DIR / f"{exp_id}_candidate.pt"

    if is_main:
        logger.info("=" * 60)
        logger.info(f"Task2 训练 exp_id={exp_id}")
        logger.info(f"配置: {vars(args)}")

    # 数据
    train_loader, val_loader = make_task2_dataloaders(
        data_dir    = str(DATA_DIR),
        val_path    = str(DATA_DIR / "task2_val.h5"),
        batch_size  = args.batch_size,
        n_query     = args.n_query,
        num_workers = args.num_workers,
        val_train_n = 80,
        distributed = (world_size > 1),
        rank        = rank,
        world_size  = world_size,
    )

    # 模型配置
    cfg = {
        "T_in": 10, "X": 256,
        "latent_dim":       args.latent_dim,
        "branch_type":      args.branch_type,
        "branch_depth":     args.branch_depth,
        "branch_width":     args.branch_width,
        "trunk_depth":      args.trunk_depth,
        "trunk_width":      args.trunk_width,
        "activation":       args.activation,
        "fourier_features": args.fourier_features,
    }
    model         = build_model(cfg).to(device)
    start_epoch   = 1
    best_val_loss = float('inf')

    # 热启动：从 Task1 最优 checkpoint 加载权重
    if args.task1_ckpt and Path(args.task1_ckpt).exists():
        if is_main:
            logger.info(f"从 Task1 checkpoint 热启动: {args.task1_ckpt}")
        ckpt = torch.load(args.task1_ckpt, map_location=device,
                          weights_only=False)
        saved_cfg = ckpt.get('cfg', {})
        # 只有架构一致才能加载权重
        if (saved_cfg.get('branch_type') == args.branch_type and
            saved_cfg.get('branch_depth') == args.branch_depth and
            saved_cfg.get('branch_width') == args.branch_width and
            saved_cfg.get('latent_dim') == args.latent_dim):
            model.load_state_dict(ckpt['model_state_dict'])
            if is_main:
                logger.info("Task1 权重加载成功（架构匹配）")
        else:
            if is_main:
                logger.warning(
                    f"架构不匹配，冷启动。"
                    f"Task1: {saved_cfg.get('branch_type')} "
                    f"d{saved_cfg.get('branch_depth')}w{saved_cfg.get('branch_width')} "
                    f"→ Task2: {args.branch_type} "
                    f"d{args.branch_depth}w{args.branch_width}"
                )
    elif args.resume and ckpt_candidate.exists():
        if is_main:
            logger.info(f"断点续训: {ckpt_candidate}")
        ckpt = torch.load(ckpt_candidate, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch   = ckpt.get('epoch', 1) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        if start_epoch > args.epochs:
            args.epochs = ckpt.get('epoch', 1) + args.epochs
    else:
        if is_main:
            logger.info(f"冷启动: {args.branch_type} latent={args.latent_dim} "
                        f"params={count_params(model):,}")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)

    remaining = args.epochs - (start_epoch - 1)
    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(remaining, 1),
                                  eta_min=args.lr * 0.01)

    no_improve  = 0
    train_start = time.time()

    if is_main:
        logger.info(f"训练 {args.epochs} epochs（从 {start_epoch}），"
                    f"patience={args.patience}")

    for epoch in range(start_epoch, args.epochs + 1):
        if world_size > 1 and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        train_loss, n_batches = 0.0, 0

        for u0, coords, u_gt in train_loader:
            u0, coords, u_gt = (u0.to(device), coords.to(device),
                                u_gt.to(device))
            optimizer.zero_grad()

            # ===== AGENT_LOSS_BEGIN =====
            pred = model(u0, coords)
            loss = relative_l2_loss(pred, u_gt)
            # ===== AGENT_LOSS_END =====

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1

        train_loss /= max(n_batches, 1)

        if is_main:
            raw_model = model.module if world_size > 1 else model
            raw_model.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for u0, coords, u_gt in val_loader:
                    u0, coords, u_gt = (u0.to(device), coords.to(device),
                                        u_gt.to(device))
                    pred      = raw_model(u0, coords)
                    val_loss += relative_l2_loss(pred, u_gt).item()
                    n_val    += 1
            val_loss /= max(n_val, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'val_loss':         val_loss,
                    'cfg':              cfg,
                    'args':             vars(args),
                    'task':             'task2',
                }, ckpt_candidate)
                marker = " <- best"
            else:
                no_improve += 1
                marker     = f" (no improve {no_improve}/{args.patience})"

            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                elapsed = time.time() - train_start
                logger.info(
                    f"Epoch {epoch:4d}/{args.epochs} | "
                    f"train={train_loss:.6f} | val={val_loss:.6f} | "
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
                    logger.info(f"早停: {args.patience} epoch 无改善")
                break
        else:
            if is_main and no_improve >= args.patience:
                logger.info(f"早停: {args.patience} epoch 无改善")
                break

    train_time = time.time() - train_start
    if is_main:
        logger.info(f"完成 exp={exp_id} 耗时={train_time:.0f}s "
                    f"best_val={best_val_loss:.6f}")
        if ckpt_candidate.exists():
            logger.info(f"Candidate: {ckpt_candidate}")
        else:
            logger.warning("未产出 candidate")

    cleanup_ddp()
    return ckpt_candidate, train_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id',    required=True)
    parser.add_argument('--resume',    action='store_true')
    parser.add_argument('--task1_ckpt', default=None,
                        help='Task1 最优 checkpoint 路径，用于热启动')

    # 模型结构（A类）
    parser.add_argument('--branch_type',      default='cnn',
                        choices=['mlp', 'cnn', 'fno_encoder'])
    parser.add_argument('--branch_depth',     type=int, default=4)
    parser.add_argument('--branch_width',     type=int, default=128)
    parser.add_argument('--trunk_depth',      type=int, default=4)
    parser.add_argument('--trunk_width',      type=int, default=128)
    parser.add_argument('--latent_dim',       type=int, default=128)
    parser.add_argument('--activation',       default='gelu')
    parser.add_argument('--fourier_features', type=int, default=64)

    # 训练参数（B类）
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--n_query',      type=int,   default=2048)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience',     type=int,   default=15)
    parser.add_argument('--num_workers',  type=int,   default=4)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)