"""训练主循环。

入口: train(cfg, out_dir, logger) -> TrainResult

设计:
    1. 所有超参/路径/策略从 ExperimentConfig 来
    2. 损失/调度器/策略从 plugins 注册表取
    3. DDP-aware: 仅 rank 0 写 log/ckpt
    4. 返回 TrainResult 由调用方决定后续动作
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import ExperimentConfig
from .dataset import make_dataloaders, make_deeponet_dataloaders
from .model import build_model, load_ckpt
from .plugins.losses import get_loss
from .plugins.schedulers import get_scheduler
from .plugins.strategies import get_strategy, StrategyContext


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_rel_mse: float
    lr: float
    elapsed: float


@dataclass
class TrainResult:
    status: str                                 # completed | early_stopped | diverged | failed
    ckpt_path: Optional[Path]
    train_time: float
    best_val_loss: float
    best_epoch: int
    history: List[EpochRecord] = field(default_factory=list)
    error_msg: str = ""


def _init_ddp() -> tuple[int, int, int, bool]:
    if "RANK" not in os.environ:
        return 0, 0, 1, True
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, (rank == 0)


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def _compute_val_metrics(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_rel = 0.0
    n = 0
    eps = 1e-6
    # 判断是否为 DeepONet
    raw_model = model.module if hasattr(model, 'module') else model
    is_deeponet = raw_model.__class__.__name__ == 'DeepONet1d'

    for inp, target, _, _ in val_loader:
        inp    = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if is_deeponet:
            u0   = inp[:, 0, :]
            xt   = inp[:, 1:, :2]
            s_gt = target[:, :, 0]
            pred = model(u0, xt)
            loss = ((pred - s_gt) ** 2).mean()
            total_loss += loss.item()
            num = ((pred - s_gt) ** 2).sum(dim=-1)
            den = (s_gt ** 2).sum(dim=-1) + eps
            total_rel += (num / den).mean().item()
        else:
            pred = model(inp)
            total_loss += loss_fn(pred, target).item()
            diff = pred - target
            num = (diff ** 2).sum(dim=-1)
            den = (target ** 2).sum(dim=-1) + eps
            total_rel += (num / den).mean().item()
        n += 1
    return total_loss / max(n, 1), total_rel / max(n, 1)


def train(
    cfg: ExperimentConfig,
    out_dir: Path,
    logger: logging.Logger,
) -> TrainResult:
    """跑一个实验。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"
    history_path = out_dir / "history.jsonl"

    rank, local_rank, world_size, is_main = _init_ddp()
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    if is_main:
        logger.info("=" * 60)
        logger.info(f"[train] exp_id={cfg.exp_id}  out_dir={out_dir}")
        logger.info(f"[train] device={device}  DDP world={world_size}")
        logger.info(
            f"[train] strategy={cfg.training.strategy_name}  "
            f"loss={cfg.training.loss_name}  "
            f"scheduler={cfg.training.scheduler_name}"
        )
        logger.info(f"[train] init_from={cfg.training.init_from}")

    # ---------- 数据 ----------
    t0 = time.time()
    if cfg.model.architecture == "DeepONet1d":
        cfg.training.strategy_name = "pi_deeponet"
        train_loader, val_loader = make_deeponet_dataloaders(
            cfg.data,
            cfg.model,
            batch_size=cfg.training.batch_size,
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size,
        )
    else:
        train_loader, val_loader = make_dataloaders(
            cfg.data,
            cfg.model,
            batch_size=cfg.training.batch_size,
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size,
        )
    train_ds = train_loader.dataset
    if is_main:
        t_r = getattr(train_ds, 'T_r', getattr(train_ds, 'T', '?'))
        x_r = getattr(train_ds, 'X_r', getattr(train_ds, 'X', '?'))
        logger.info(
            f"[data] T_r={t_r}  X_r={x_r}  "
            f"train={len(train_ds)}  loaded in {time.time()-t0:.1f}s"
        )

    # ---------- 模型 ----------
    start_epoch = 1
    if cfg.training.init_from == "scratch":
        model = build_model(cfg.model).to(device)
        if is_main:
            n_params = sum(p.numel() for p in model.parameters())
            logger.info(
                f"[model] scratch  params={n_params:,}  "
                f"arch={cfg.model.architecture}  "
                f"modes={cfg.model.modes}  width={cfg.model.width}"
            )
    else:
        if not cfg.training.init_ckpt:
            return TrainResult(
                status="failed", ckpt_path=None, train_time=0,
                best_val_loss=float("inf"), best_epoch=0,
                error_msg=f"init_from={cfg.training.init_from} 但 init_ckpt 为空",
            )
        ckpt_src = Path(cfg.training.init_ckpt)
        if not ckpt_src.exists():
            return TrainResult(
                status="failed", ckpt_path=None, train_time=0,
                best_val_loss=float("inf"), best_epoch=0,
                error_msg=f"init_ckpt 不存在: {ckpt_src}",
            )
        model, meta = load_ckpt(ckpt_src, device)
        if cfg.training.init_from == "resume":
            resumed_from_epoch = (meta.get("epoch") or 0)
            start_epoch = resumed_from_epoch + 1
            # resume 时 epochs 表示额外训练轮数，end_epoch 为绝对终止轮次
            cfg.training.epochs = resumed_from_epoch + cfg.training.epochs
        if is_main:
            logger.info(
                f"[model] {cfg.training.init_from} from {ckpt_src.name}  "
                f"arch={meta.get('architecture')}  "
                f"prev_val_loss={meta.get('val_loss')}  "
                f"start_epoch={start_epoch}"
            )

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    raw_model = model.module if world_size > 1 else model

    # ---------- 损失 / 优化 / 调度 ----------
    loss_fn = get_loss(cfg.training.loss_name)
    optimizer = Adam(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler_factory = get_scheduler(cfg.training.scheduler_name)
    scheduler = scheduler_factory(
        optimizer,
        epochs=cfg.training.epochs,
        **cfg.training.scheduler_kwargs,
    )
    is_plateau = isinstance(scheduler, ReduceLROnPlateau)
    strategy_fn = get_strategy(cfg.training.strategy_name)

    # ---------- 训练循环 ----------
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    diverged = False
    train_start = time.time()
    history: List[EpochRecord] = []

    if is_main:
        logger.info(
            f"[loop] epochs={cfg.training.epochs}  "
            f"patience={cfg.training.patience}  "
            f"start_epoch={start_epoch}"
        )
        history_path.write_text("")

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        if world_size > 1 and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # --- train ---
        model.train()
        running = 0.0
        n_batch = 0

        for inp, target, sample_n, sample_t in train_loader:
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            ctx = StrategyContext(
                model=model,
                inp=inp,
                target=target,
                sample_n=sample_n,
                sample_t=sample_t,
                dataset_data=train_ds,
                window_size=cfg.model.window_size,
                pushforward_steps=cfg.training.pushforward_steps,
                pushforward_weight=cfg.training.pushforward_weight,
                ics_weight=cfg.training.ics_weight,
                loss_fn=loss_fn,
            )
            try:
                loss = strategy_fn(ctx)
            except Exception as e:
                if is_main:
                    logger.error(f"[loop] strategy 抛异常 @ epoch {epoch}: {e}")
                return TrainResult(
                    status="failed",
                    ckpt_path=ckpt_path if ckpt_path.exists() else None,
                    train_time=time.time() - train_start,
                    best_val_loss=best_val,
                    best_epoch=best_epoch,
                    history=history,
                    error_msg=str(e),
                )

            if not torch.isfinite(loss):
                if is_main:
                    logger.error(f"[loop] loss=NaN/Inf @ epoch {epoch}")
                diverged = True
                break

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip)
            optimizer.step()

            running += loss.item()
            n_batch += 1

        if diverged:
            break

        train_loss = running / max(n_batch, 1)

        val_loss, val_rel = _compute_val_metrics(
            raw_model, val_loader, loss_fn, device
        )

        if is_plateau:
            scheduler.step(val_loss)
        else:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - train_start
        rec = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_rel_mse=val_rel,
            lr=cur_lr,
            elapsed=elapsed,
        )
        history.append(rec)

        improved = val_loss < best_val
        if is_main:
            if improved:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_rel_mse": val_rel,
                    "args": asdict(cfg),
                }, ckpt_path)
                marker = "  best"
            else:
                no_improve += 1
                marker = f"  (no_improve {no_improve}/{cfg.training.patience})"

            if epoch == start_epoch or epoch % 5 == 0 or epoch == cfg.training.epochs:
                logger.info(
                    f"Epoch {epoch:4d}/{cfg.training.epochs} | "
                    f"train={train_loss:.6f} | "
                    f"val={val_loss:.6f} | "
                    f"rel_mse={val_rel:.6f} | "
                    f"lr={cur_lr:.2e} | "
                    f"{elapsed:.0f}s{marker}"
                )

            with open(history_path, "a") as f:
                f.write(json.dumps(asdict(rec)) + "\n")

        if world_size > 1:
            stop = torch.tensor(
                1 if (is_main and no_improve >= cfg.training.patience) else 0,
                device=device,
            )
            dist.broadcast(stop, src=0)
            dist.barrier()
            if stop.item() == 1:
                if is_main:
                    logger.info(f"[loop] early stop: no_improve {no_improve}")
                break
        else:
            if no_improve >= cfg.training.patience:
                logger.info(f"[loop] early stop: no_improve {no_improve}")
                break

    train_time = time.time() - train_start

    if diverged:
        status = "diverged"
    elif no_improve >= cfg.training.patience:
        status = "early_stopped"
    else:
        status = "completed"

    if is_main:
        logger.info("=" * 60)
        logger.info(
            f"[train] {status} | total={train_time:.1f}s "
            f"({train_time/60:.1f}min) | best@epoch{best_epoch} "
            f"val={best_val:.6f}"
        )
        logger.info(f"[train] ckpt: {ckpt_path}")
        # 保存训练摘要供 agent 读取
        import json as _json
        train_result = {
            "status": status,
            "best_epoch": best_epoch,
            "n_epochs_run": len(history),
            "best_val_loss": best_val if best_val != float("inf") else None,
            "train_time": train_time,
        }
        (out_dir / "train_result.json").write_text(
            _json.dumps(train_result, indent=2)
        )

    _cleanup_ddp()
    return TrainResult(
        status=status,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
        train_time=train_time,
        best_val_loss=best_val,
        best_epoch=best_epoch,
        history=history,
    )
