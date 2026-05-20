"""
ddp_runner.py — DDP 样板封装

框架层文件，由人工维护，不提交到 code/。
Agent 在 train.py 里 import 使用，无需了解 DDP 底层细节。

提供：
  DDPContext        DDP 初始化/包装/清理
  sync_early_stop   跨进程早停同步
  save_checkpoint   保存 checkpoint
  load_checkpoint   加载 checkpoint
  make_logger       训练日志工厂
"""

import os
import logging
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

PROJECT_DIR = Path("/mimer/NOBACKUP/groups/phy_geo/PDE_Agent")
WORKSPACE   = PROJECT_DIR / "repos" / "workspace"
LOG_DIR     = PROJECT_DIR / "logs"
CKPT_PATH   = WORKSPACE / "checkpoints" / "best.pt"


# ============================================================
# DDP 上下文
# ============================================================
class DDPContext:
    """
    封装 DDP 初始化和清理。单卡时自动退化为普通模式。

    用法：
        ctx = DDPContext()
        model = ctx.wrap(model)
        for epoch in range(epochs):
            ctx.set_epoch(train_loader, epoch)
            ...
            if sync_early_stop(no_improve, patience, ctx.is_main, ctx.world_size):
                break
        ctx.cleanup()

    属性：
        ctx.rank        当前进程 rank（单卡时为 0）
        ctx.local_rank  当前 GPU 编号（单卡时为 0）
        ctx.world_size  总进程数（单卡时为 1）
        ctx.is_main     是否为 rank 0
        ctx.device      当前 cuda 设备
    """

    def __init__(self):
        if "RANK" not in os.environ:
            # 单卡模式
            self.rank       = 0
            self.local_rank = 0
            self.world_size = 1
            self.is_main    = True
        else:
            dist.init_process_group(backend="nccl")
            self.rank       = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = dist.get_world_size()
            self.is_main    = (self.rank == 0)
            torch.cuda.set_device(self.local_rank)

        self.device = torch.device(
            f"cuda:{self.local_rank}"
            if torch.cuda.is_available() else "cpu"
        )

    def wrap(self, model: nn.Module) -> nn.Module:
        """把模型移到 device 并包装为 DDP（单卡时只移到 device）。"""
        model = model.to(self.device)
        if self.world_size > 1:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        return model

    def unwrap(self, model: nn.Module) -> nn.Module:
        """取出 DDP 内部的原始模型（用于保存 checkpoint）。"""
        return model.module if self.world_size > 1 else model

    def set_epoch(self, loader, epoch: int):
        """给 DistributedSampler 设置 epoch，保证每轮 shuffle 不同。"""
        if self.world_size > 1 and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

    def cleanup(self):
        """清理 DDP 进程组（训练结束后调用）。"""
        if dist.is_initialized():
            dist.destroy_process_group()


# ============================================================
# 早停同步
# ============================================================
def sync_early_stop(no_improve: int, patience: int,
                    is_main: bool, world_size: int) -> bool:
    """
    跨进程同步早停信号。rank 0 判断，广播给所有进程。
    返回 True 表示应该停止训练。

    用法：
        if sync_early_stop(no_improve, args.patience, ctx.is_main, ctx.world_size):
            break
    """
    if world_size == 1:
        return is_main and no_improve >= patience

    flag = torch.tensor(
        1 if (is_main and no_improve >= patience) else 0
    ).cuda()
    dist.broadcast(flag, src=0)
    dist.barrier()
    return bool(flag.item() == 1)


# ============================================================
# Checkpoint 保存和加载
# ============================================================
def save_checkpoint(
    model,
    epoch:     int,
    val_loss:  float,
    ckpt_path: str = None,
    extra:     dict = None,
) -> Path:
    """
    保存 checkpoint（只在 rank 0 调用）。
    默认保存到 workspace/checkpoints/best.pt。

    用法：
        if ctx.is_main and val_loss < best_val_loss:
            save_checkpoint(ctx.unwrap(model), epoch, val_loss)
            print(f"best saved, val_loss={val_loss:.6f}")

    参数：
        model:     原始模型（已 unwrap）
        epoch:     当前 epoch
        val_loss:  验证损失
        ckpt_path: 自定义保存路径（None 则用默认路径）
        extra:     额外信息写入 checkpoint（如超参数 dict）

    返回：
        保存路径（Path 对象）
    """
    path = Path(ckpt_path) if ckpt_path else CKPT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "val_loss":         val_loss,
    }
    if extra:
        state.update(extra)

    torch.save(state, path)
    return path


def load_checkpoint(ckpt_path: str = None, device=None) -> dict:
    """
    加载 checkpoint，返回 dict 或 None（文件不存在时）。

    用法：
        ckpt = load_checkpoint()
        if ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["val_loss"]

    参数：
        ckpt_path: checkpoint 路径（None 则用默认路径）
        device:    加载到的设备（None 则用 cpu）
    """
    path = Path(ckpt_path) if ckpt_path else CKPT_PATH
    if not path.exists():
        return None
    device = device or "cpu"
    return torch.load(path, map_location=device, weights_only=False)


# ============================================================
# Logger 工厂
# ============================================================
def make_logger(name: str = "train",
                log_filename: str = "train.log",
                is_main: bool = True) -> logging.Logger:
    """
    创建训练 logger。rank 0 同时写文件和 stdout，其他进程只写 WARNING+。

    用法：
        logger = make_logger("train", "train.log", ctx.is_main)
        logger.info(f"Epoch {epoch}: val_loss={val_loss:.6f}")
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if is_main:
        LOG_DIR.mkdir(exist_ok=True)
        fh = logging.FileHandler(
            LOG_DIR / log_filename, mode="a", encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if not is_main:
        logger.setLevel(logging.WARNING)

    return logger


# ============================================================
# Unittest
# ============================================================
if __name__ == "__main__":
    import tempfile, shutil
    print("=" * 50)
    print("ddp_runner.py unittest")
    print("=" * 50)

    errors = []

    def check(name, fn):
        try:
            fn()
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            errors.append((name, str(e)))

    # Test 1: DDPContext 单卡模式
    print("\n[1] DDPContext 单卡模式")
    def t1():
        ctx = DDPContext()
        assert ctx.rank       == 0,    f"rank={ctx.rank}"
        assert ctx.local_rank == 0,    f"local_rank={ctx.local_rank}"
        assert ctx.world_size == 1,    f"world_size={ctx.world_size}"
        assert ctx.is_main    == True, f"is_main={ctx.is_main}"
        assert str(ctx.device).startswith("cuda") or str(ctx.device) == "cpu"
        ctx.cleanup()
    check("DDPContext 属性正确（单卡）", t1)

    # Test 2: DDPContext.wrap 和 unwrap
    print("\n[2] DDPContext.wrap / unwrap")
    def t2():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from model import DeepONet
        ctx   = DDPContext()
        model = DeepONet()
        wrapped   = ctx.wrap(model)
        unwrapped = ctx.unwrap(wrapped)
        # 单卡时 wrap 不加 DDP
        assert not isinstance(wrapped, nn.parallel.DistributedDataParallel)
        assert str(next(unwrapped.parameters()).device) == str(ctx.device)
        ctx.cleanup()
    check("wrap/unwrap 单卡", t2)

    # Test 3: sync_early_stop 单卡
    print("\n[3] sync_early_stop 单卡")
    def t3():
        assert sync_early_stop(10, 5,  True,  1) == True,  "超过 patience 应停止"
        assert sync_early_stop(3,  5,  True,  1) == False, "未超过 patience 不停止"
        assert sync_early_stop(10, 5,  False, 1) == False, "非 main rank 不停止"
    check("sync_early_stop 逻辑正确", t3)

    # Test 4: save_checkpoint / load_checkpoint
    print("\n[4] save_checkpoint / load_checkpoint")
    def t4():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from model import DeepONet
        model = DeepONet()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_best.pt"

            # 保存
            saved = save_checkpoint(
                model, epoch=5, val_loss=0.042,
                ckpt_path=str(ckpt_path),
                extra={"branch_type": "cnn"}
            )
            assert saved == ckpt_path,          f"返回路径不对: {saved}"
            assert ckpt_path.exists(),           "checkpoint 文件不存在"

            # 加载
            ckpt = load_checkpoint(str(ckpt_path))
            assert ckpt is not None,             "加载结果为 None"
            assert ckpt["epoch"]    == 5,        f"epoch={ckpt['epoch']}"
            assert ckpt["val_loss"] == 0.042,    f"val_loss={ckpt['val_loss']}"
            assert ckpt["branch_type"] == "cnn", f"extra 未保存"

            # 加载模型权重
            model2 = DeepONet()
            model2.load_state_dict(ckpt["model_state_dict"])

        # 不存在时返回 None
        result = load_checkpoint("/nonexistent/path/best.pt")
        assert result is None, "不存在的文件应返回 None"
    check("save/load checkpoint 完整流程", t4)

    # Test 5: make_logger
    print("\n[5] make_logger")
    def t5():
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = make_logger("test_logger", "test_train.log", is_main=True)
            assert logger is not None
            assert logger.level == logging.INFO
            # 非 main rank 应为 WARNING
            logger2 = make_logger("test_logger2", "test2.log", is_main=False)
            assert logger2.level == logging.WARNING
    check("make_logger rank0/非rank0", t5)

    # Test 6: DDPContext.set_epoch 无 sampler 时不报错
    print("\n[6] set_epoch 兼容性")
    def t6():
        from torch.utils.data import DataLoader, TensorDataset
        ctx = DDPContext()
        ds  = TensorDataset(torch.randn(10, 3))
        loader = DataLoader(ds, batch_size=2, shuffle=True)
        # 没有 set_epoch 的 sampler，不应该报错
        ctx.set_epoch(loader, epoch=1)
        ctx.cleanup()
    check("set_epoch 无 DistributedSampler 不报错", t6)

    print("\n" + "=" * 50)
    if errors:
        print(f"❌ {len(errors)} 个测试失败:")
        for name, err in errors:
            print(f"   {name}: {err}")
    else:
        print("✅ 所有测试通过")
    print("=" * 50)