"""命令行入口。

用法:
    python -m core.cli train   --config conf/experiments/v1_baseline.yaml
    python -m core.cli predict --config conf/experiments/v1_baseline.yaml --ckpt path.pt

这是 core 层唯一带 argparse 的文件。agent 层通过 subprocess 调用此 CLI。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config, ExperimentConfig
from .train import train as run_train
from .predict import predict as run_predict


def _setup_logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _resolve_out_dir(cfg: ExperimentConfig, override: str | None) -> Path:
    if override:
        return Path(override)
    if cfg.out_dir:
        return Path(cfg.out_dir)
    raise ValueError("out_dir 未指定: 请在 yaml 里设 out_dir 或通过 --out_dir 覆盖")


def cmd_train(args):
    cfg = load_config(args.config)
    out_dir = _resolve_out_dir(cfg, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger("train", out_dir / "train.log")
    logger.info(f"[cli] config={args.config}")

    result = run_train(cfg, out_dir, logger)
    summary = {
        "status": result.status,
        "ckpt_path": str(result.ckpt_path) if result.ckpt_path else None,
        "train_time": result.train_time,
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "n_epochs_run": len(result.history),
        "error_msg": result.error_msg,
    }
    (out_dir / "train_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if result.status in ("completed", "early_stopped") else 1


def cmd_predict(args):
    cfg = load_config(args.config)
    out_dir = _resolve_out_dir(cfg, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger("predict", out_dir / "predict.log")

    ckpt_path = Path(args.ckpt) if args.ckpt else (out_dir / "best_model.pt")
    if not ckpt_path.exists():
        logger.error(f"ckpt 不存在: {ckpt_path}")
        return 2

    if args.input:
        cfg.inference.input_path = args.input

    result = run_predict(cfg, ckpt_path, out_dir, logger)
    summary = {
        "status": result.status,
        "pred_path": str(result.pred_path) if result.pred_path else None,
        "inference_time": result.inference_time,
        "has_gt": result.has_gt,
        "metrics": result.metrics,
        "init_diff_max": result.init_diff_max,
        "error_msg": result.error_msg,
    }
    (out_dir / "predict_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if result.status == "completed" else 1


def main():
    p = argparse.ArgumentParser("core.cli")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="训练实验")
    t.add_argument("--config", required=True)
    t.add_argument("--out_dir", default=None, help="覆盖 cfg.out_dir")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="推理 + 评估")
    pr.add_argument("--config", required=True)
    pr.add_argument("--ckpt", default=None,
                    help="覆盖 ckpt 路径，默认 {out_dir}/best_model.pt")
    pr.add_argument("--input", default=None,
                    help="覆盖 cfg.inference.input_path")
    pr.add_argument("--out_dir", default=None)
    pr.set_defaults(func=cmd_predict)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
