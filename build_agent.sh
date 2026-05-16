#!/usr/bin/env bash
# ============================================================
# guandi_agent 一键搭建脚本
# 用法:
#   bash setup_guandi_agent.sh
# 默认目标路径:
#   /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/guandi_agent
# 可通过环境变量覆盖:
#   ROOT=/path/to/guandi_agent bash setup_guandi_agent.sh
# ============================================================

set -euo pipefail

ROOT="${ROOT:-/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/guandi_agent}"

echo "目标工程根: $ROOT"
read -p "确认在此路径搭建? [y/N] " ans
[[ "$ans" =~ ^[Yy]$ ]] || { echo "取消"; exit 1; }

# ---------- 创建目录骨架 ----------
mkdir -p "$ROOT"/{conf/experiments,runs/_shared,scripts,tests}
mkdir -p "$ROOT"/src/{agent,submission}
mkdir -p "$ROOT"/src/core/plugins

cd "$ROOT"

# ---------- 创建空的 __init__.py 让 Python 把它们当作 package ----------
touch src/__init__.py
touch src/agent/__init__.py
touch src/submission/__init__.py
touch src/core/__init__.py
touch src/core/plugins/__init__.py
touch tests/__init__.py

echo "✓ 目录骨架已创建"

# ============================================================
# 文件 1/14: pyproject.toml
# ============================================================
cat > pyproject.toml << 'PYPROJECT_EOF'
[project]
name = "guandi_agent"
version = "0.1.0"
description = "PDE Neural Operator Research Agent for AI4S CNS Challenge"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
PYPROJECT_EOF

# ============================================================
# 文件 2/14: src/core/config.py
# ============================================================
cat > src/core/config.py << 'CONFIG_EOF'
"""配置数据类与加载/保存接口。

设计原则:
    1. 一个实验 = 一个 ExperimentConfig 实例
    2. 所有路径、超参、策略选择都在 dataclass 中显式声明
    3. YAML 是唯一可信源；agent 通过生成新 yaml 提案实验
    4. 加载时做类型校验，字段写错立刻报错

注意:
    本模块不依赖 torch，只用标准库 + pyyaml，方便测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ===================== 子配置 =====================
@dataclass
class ModelConfig:
    """模型架构配置。"""
    architecture: str = "FNO1d"
    modes: int = 16
    width: int = 64
    n_layers: int = 4
    window_size: int = 10


@dataclass
class DataConfig:
    """数据加载配置。"""
    hdf5_path: str = ""
    n_samples: Optional[int] = 8000
    reduced_resolution_t: int = 1
    reduced_resolution: int = 4
    train_ratio: float = 0.9
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """训练超参与策略。"""
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    grad_clip: float = 1.0

    loss_name: str = "relative_l2"
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    scheduler_name: str = "cosine"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    strategy_name: str = "pushforward"
    pushforward_steps: int = 3
    pushforward_weight: float = 0.5

    init_from: str = "scratch"
    init_ckpt: Optional[str] = None


@dataclass
class InferenceConfig:
    """推理配置。"""
    use_mean_correction: bool = True
    infer_batch: int = 256
    input_path: str = ""
    total_steps: int = 200


@dataclass
class ExperimentConfig:
    """单个实验的完整配置 (LLM 生成此结构的 yaml 即为提案)。"""
    exp_id: str = ""
    parent_exp_id: Optional[str] = None
    rationale: str = ""
    description: str = ""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    out_dir: str = ""


# ===================== 加载/保存 =====================
_NESTED_FIELDS = {
    "ExperimentConfig": {
        "model": ModelConfig,
        "data": DataConfig,
        "training": TrainingConfig,
        "inference": InferenceConfig,
    },
}


def load_config(path: Path | str) -> ExperimentConfig:
    """从 yaml 加载并校验。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _dict_to_experiment(raw)
    _validate(cfg)
    return cfg


def save_config(cfg: ExperimentConfig, path: Path | str) -> None:
    """保存为 yaml。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)


def diff_configs(
    base: ExperimentConfig, new: ExperimentConfig
) -> Dict[str, Tuple[Any, Any]]:
    """返回 new 相对 base 改动的字段。"""
    out: Dict[str, Tuple[Any, Any]] = {}
    _diff_recursive(asdict(base), asdict(new), "", out)
    return out


# ===================== 内部 =====================
def _dict_to_experiment(data: dict) -> ExperimentConfig:
    valid = {f.name for f in fields(ExperimentConfig)}
    unknown = set(data.keys()) - valid
    if unknown:
        raise ValueError(
            f"ExperimentConfig 有未知字段: {unknown}。"
            f"合法字段: {sorted(valid)}"
        )

    kwargs: Dict[str, Any] = {}
    for f in fields(ExperimentConfig):
        if f.name not in data:
            continue
        val = data[f.name]
        if f.name in _NESTED_FIELDS["ExperimentConfig"]:
            sub_cls = _NESTED_FIELDS["ExperimentConfig"][f.name]
            kwargs[f.name] = _build_dataclass(val or {}, sub_cls)
        else:
            kwargs[f.name] = val
    return ExperimentConfig(**kwargs)


def _build_dataclass(data: dict, cls):
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    valid = {f.name for f in fields(cls)}
    unknown = set(data.keys()) - valid
    if unknown:
        raise ValueError(
            f"{cls.__name__} 有未知字段: {unknown}。合法: {sorted(valid)}"
        )
    return cls(**{k: v for k, v in data.items() if k in valid})


def _diff_recursive(a: dict, b: dict, prefix: str, out: dict):
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        va, vb = a.get(k), b.get(k)
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(va, dict) and isinstance(vb, dict):
            _diff_recursive(va, vb, full, out)
        elif va != vb:
            out[full] = (va, vb)


def _validate(cfg: ExperimentConfig) -> None:
    errors: List[str] = []

    if cfg.model.architecture not in ("FNO1d", "ClassicFNO1d"):
        errors.append(f"model.architecture={cfg.model.architecture} 不合法")
    if cfg.model.modes < 1:
        errors.append(f"model.modes={cfg.model.modes} 必须 >= 1")
    if cfg.model.width < 1:
        errors.append(f"model.width={cfg.model.width} 必须 >= 1")
    if cfg.model.window_size < 1:
        errors.append(f"model.window_size={cfg.model.window_size} 必须 >= 1")

    if not cfg.data.hdf5_path:
        errors.append("data.hdf5_path 不能为空")
    if cfg.data.reduced_resolution_t not in (1, 5):
        errors.append(
            f"data.reduced_resolution_t={cfg.data.reduced_resolution_t} "
            f"目前只支持 1 (真实轴) 或 5 (reduced)"
        )

    if cfg.training.init_from not in ("scratch", "resume", "finetune"):
        errors.append(f"training.init_from={cfg.training.init_from} 不合法")
    if cfg.training.init_from != "scratch" and not cfg.training.init_ckpt:
        errors.append(
            f"init_from={cfg.training.init_from} 时必须指定 init_ckpt"
        )
    if cfg.training.pushforward_steps < 0:
        errors.append(
            f"pushforward_steps={cfg.training.pushforward_steps} 不能为负"
        )
    if cfg.training.strategy_name not in ("vanilla", "pushforward"):
        errors.append(
            f"strategy_name={cfg.training.strategy_name} 不在白名单中"
        )

    if cfg.inference.total_steps != 200:
        errors.append(
            f"inference.total_steps={cfg.inference.total_steps} != 200，"
            f"提交时会被 shape 检查拒绝"
        )

    if errors:
        msg = "配置校验失败:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)
CONFIG_EOF

# ============================================================
# 文件 3/14: src/core/dataset.py
# ============================================================
cat > src/core/dataset.py << 'DATASET_EOF'
"""PDEBench 1D Burgers 数据集。

【时间轴策略】策略 B —— 真实 200 步轴 (reduced_resolution_t=1)
    原始 (N, 200, 1024) → 空间下采样 → (N, 200, 256)
    模型在真实轴上学 1 步预测器，自回归 190 步。
    输出与赛题要求的 (N, 200, 256) 完全对齐，无需插值。

【可选】reduced_resolution_t=5 时真实做时间下采样到 40 步 (策略 A)，
    但 predict 必须配套写插值，初版不启用。

每个样本生成 (T_r - window_size) 个训练对：
    input:  (window_size + 1, X_r) — window_size 个历史步 + 1 个 grid 通道
    target: (1, X_r)                — 下一个时间步
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import DataConfig, ModelConfig


class BurgersDataset(Dataset):
    def __init__(
        self,
        data_cfg: DataConfig,
        window_size: int,
        split: str = "train",
    ):
        super().__init__()
        self.window_size = window_size
        self.cfg = data_cfg

        # ---------- 读取 ----------
        with h5py.File(data_cfg.hdf5_path, "r") as f:
            keys = list(f.keys())
            data_key = "tensor" if "tensor" in keys else keys[0]
            data = f[data_key][:]
        data = data.astype(np.float32)

        # 时间裁剪 (训练数据有 201 步) + 时间下采样
        data = data[:, :200, :]
        if data_cfg.reduced_resolution_t > 1:
            data = data[:, :: data_cfg.reduced_resolution_t, :]

        # 空间下采样
        data = data[:, :, :: data_cfg.reduced_resolution]

        N, T_r, X_r = data.shape
        self.T_r = T_r
        self.X_r = X_r

        if data_cfg.n_samples is not None:
            data = data[: data_cfg.n_samples]
            N = data.shape[0]

        n_train = int(N * data_cfg.train_ratio)
        if split == "train":
            data = data[:n_train]
        elif split == "val":
            data = data[n_train:]
        else:
            raise ValueError(f"unknown split: {split}")

        self.data = torch.from_numpy(data)  # (N, T_r, X_r)
        self.N = self.data.shape[0]

        self.grid = torch.linspace(0, 1, X_r).unsqueeze(0)  # (1, X_r)

        self.index_pairs = [
            (n, t)
            for n in range(self.N)
            for t in range(window_size, T_r)
        ]

    def __len__(self) -> int:
        return len(self.index_pairs)

    def __getitem__(self, idx: int):
        n, t = self.index_pairs[idx]
        history = self.data[n, t - self.window_size: t, :]
        target = self.data[n, t, :].unsqueeze(0)
        inp = torch.cat([history, self.grid], dim=0)
        return inp, target, n, t


def make_dataloaders(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    batch_size: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = BurgersDataset(data_cfg, model_cfg.window_size, split="train")
    val_ds = BurgersDataset(data_cfg, model_cfg.window_size, split="val")

    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=data_cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, sampler=val_sampler,
            num_workers=data_cfg.num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=data_cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=data_cfg.num_workers, pin_memory=True,
        )

    return train_loader, val_loader
DATASET_EOF

# ============================================================
# 文件 4/14: src/core/model.py
# ============================================================
cat > src/core/model.py << 'MODEL_EOF'
"""FNO 模型定义 + 工厂函数。

- FNO1d         我们的标准 FNO (width/n_layers 可配)
- ClassicFNO1d  官方 PDEBench 结构 (4 层固定，用于加载官方 ckpt)

LLM 通过 model.architecture 字段切换，不能改这里的代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, inp, weights):
        return torch.einsum("bix,iox->box", inp, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    """通用 1D FNO，width/n_layers 可配。

    输入: (B, in_channels, X) — in_channels = window_size + 1
    输出: (B, out_channels, X) — out_channels=1 (预测下一时间步)
    """

    def __init__(
        self,
        modes: int,
        width: int,
        in_channels: int,
        out_channels: int,
        n_layers: int,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc0 = nn.Linear(in_channels, width)
        self.convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.ws = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)]
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


class ClassicFNO1d(nn.Module):
    """官方 PDEBench FNO 结构 (4 层固定，key 命名 conv0..conv3)。

    in_channels 固定为 11，out_channels 可调。
    用于加载官方 ckpt (modes=12, width=20)。
    """

    def __init__(self, modes: int = 12, width: int = 20, out_channels: int = 1):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(11, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


def build_model(cfg: ModelConfig) -> nn.Module:
    """根据 ModelConfig 构造模型。"""
    if cfg.architecture == "FNO1d":
        return FNO1d(
            modes=cfg.modes,
            width=cfg.width,
            in_channels=cfg.window_size + 1,
            out_channels=1,
            n_layers=cfg.n_layers,
        )
    elif cfg.architecture == "ClassicFNO1d":
        return ClassicFNO1d(
            modes=cfg.modes,
            width=cfg.width,
            out_channels=1,
        )
    else:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")


def detect_architecture(state_dict: dict) -> str:
    """从 state_dict key 推断结构。"""
    if any(k.startswith("conv0.") for k in state_dict):
        return "ClassicFNO1d"
    if any(k.startswith("convs.0.") for k in state_dict):
        return "FNO1d"
    raise ValueError(
        f"无法识别的 state_dict, top-level keys: {list(state_dict)[:5]}..."
    )


def load_ckpt(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    """通用 ckpt 加载。

    支持:
      1. 我们训练得到的 (含 'model_state_dict' + 'args')
      2. 官方 PDEBench (纯 state_dict)

    返回 (model, meta)，meta 含 epoch / val_loss / args / architecture。
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        args = ckpt.get("args", {})
        meta = {
            "epoch": ckpt.get("epoch"),
            "val_loss": ckpt.get("val_loss"),
            "args": args,
        }
    elif isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in list(ckpt.values())[:5]
    ):
        state_dict = ckpt
        args = {}
        meta = {"epoch": None, "val_loss": None, "args": {}}
    else:
        raise ValueError(f"无法识别的 ckpt 格式: {ckpt_path}")

    arch = detect_architecture(state_dict)

    if arch == "ClassicFNO1d":
        width = state_dict["fc0.weight"].shape[0]
        modes = state_dict["conv0.weights1"].shape[-1]
        out_channels = state_dict["fc2.weight"].shape[0]
        model = ClassicFNO1d(modes=modes, width=width, out_channels=out_channels)
    else:
        modes = state_dict["convs.0.weights1"].shape[-1]
        width = state_dict["fc0.weight"].shape[0]
        in_channels = state_dict["fc0.weight"].shape[1]
        n_layers = max(
            int(k.split(".")[1]) for k in state_dict if k.startswith("convs.")
        ) + 1
        model = FNO1d(
            modes=modes,
            width=width,
            in_channels=in_channels,
            out_channels=1,
            n_layers=n_layers,
        )

    model.load_state_dict(state_dict)
    model.to(device)
    meta["architecture"] = arch
    return model, meta
MODEL_EOF

# ============================================================
# 文件 5/14: src/core/metrics.py
# ============================================================
cat > src/core/metrics.py << 'METRICS_EOF'
"""赛题评分公式。

总分 = 0.25*seg1 + 0.25*seg2 + 0.5*seg3，范围 0-100。

时间分段 (针对去掉前 10 步的 190 个预测步):
    Seg1: 索引 [10, 57)   长度 47   权重 25%
    Seg2: 索引 [57, 105)  长度 48   权重 25%
    Seg3: 索引 [105, 200) 长度 95   权重 50%

得分公式:
    Seg1 = 100 * exp(-20 * rel_mse)
    Seg2 = 100 * exp(-10 * rel_mse)
    Seg3 = max(Lorentzian, Frechet)
        Lorentzian = 100 / (1 + 10 * RMSE)
        Frechet    = 50 * exp(-FD²)
"""

from __future__ import annotations

from typing import Dict

import numpy as np


_EPS = 1e-6


def rel_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred, gt: (N, T_seg, X)。"""
    diff = pred - gt
    num = (diff ** 2).sum(axis=-1)
    den = (gt ** 2).sum(axis=-1) + _EPS
    per_sample = (num / den).mean(axis=1)
    per_sample = np.clip(per_sample, None, 5.0)
    return float(per_sample.mean())


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(((pred - gt) ** 2).mean()))


def frechet_proxy(pred: np.ndarray, gt: np.ndarray) -> float:
    """离散 Frechet 距离的轻量代理。"""
    per_sample = np.sqrt(((pred - gt) ** 2).mean(axis=(1, 2)))
    return float(per_sample.mean())


def compute_segment_scores(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """完整评分。pred/gt: (N, 200, X)。"""
    seg1_p, seg1_g = pred[:, 10:57, :], gt[:, 10:57, :]
    seg2_p, seg2_g = pred[:, 57:105, :], gt[:, 57:105, :]
    seg3_p, seg3_g = pred[:, 105:200, :], gt[:, 105:200, :]

    r1 = rel_mse(seg1_p, seg1_g)
    r2 = rel_mse(seg2_p, seg2_g)
    r3_rmse = rmse(seg3_p, seg3_g)
    r3_fd = frechet_proxy(seg3_p, seg3_g)

    s1 = 100.0 * np.exp(-20 * r1)
    s2 = 100.0 * np.exp(-10 * r2)
    lorentzian = 100.0 / (1 + 10 * r3_rmse)
    frechet = 50.0 * np.exp(-(r3_fd ** 2))
    s3 = max(lorentzian, frechet)

    total = 0.25 * s1 + 0.25 * s2 + 0.5 * s3

    return {
        "seg1_rel_mse": r1,
        "seg1_score": float(s1),
        "seg2_rel_mse": r2,
        "seg2_score": float(s2),
        "seg3_rmse": r3_rmse,
        "seg3_fd": r3_fd,
        "seg3_lorentzian": float(lorentzian),
        "seg3_frechet": float(frechet),
        "seg3_score": float(s3),
        "total_score": float(total),
    }
METRICS_EOF

# ============================================================
# 文件 6/14: src/core/plugins/losses.py
# ============================================================
cat > src/core/plugins/losses.py << 'LOSSES_EOF'
"""损失函数注册表。

LLM 通过 training.loss_name 在 yaml 选择，不能直接改这里的代码。
新增损失函数需手动添加到此文件。
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F


LOSS_REGISTRY: Dict[str, Callable] = {}


def register_loss(name: str):
    def deco(fn):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss {name} already registered")
        LOSS_REGISTRY[name] = fn
        return fn
    return deco


def get_loss(name: str) -> Callable:
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: {name}. Available: {sorted(LOSS_REGISTRY)}"
        )
    return LOSS_REGISTRY[name]


def list_losses() -> list:
    return sorted(LOSS_REGISTRY.keys())


# ===================== 注册 =====================
@register_loss("relative_l2")
def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """逐样本相对 L2，与赛题评分公式同方向。"""
    diff = pred - target
    return (diff.norm(dim=-1) / (target.norm(dim=-1) + 1e-8)).mean()


@register_loss("mse")
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


@register_loss("huber")
def huber_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=1.0)
LOSSES_EOF

# ============================================================
# 文件 7/14: src/core/plugins/schedulers.py
# ============================================================
cat > src/core/plugins/schedulers.py << 'SCHEDULERS_EOF'
"""LR Scheduler 注册表。

LLM 通过 training.scheduler_name 选择。
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
)


SCHEDULER_REGISTRY: Dict[str, Callable] = {}


def register_scheduler(name: str):
    def deco(fn):
        if name in SCHEDULER_REGISTRY:
            raise ValueError(f"Scheduler {name} already registered")
        SCHEDULER_REGISTRY[name] = fn
        return fn
    return deco


def get_scheduler(name: str) -> Callable:
    if name not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler: {name}. Available: {sorted(SCHEDULER_REGISTRY)}"
        )
    return SCHEDULER_REGISTRY[name]


# ===================== 注册 =====================
@register_scheduler("cosine")
def cosine(optimizer: Optimizer, epochs: int, **kwargs):
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=kwargs.get("eta_min", 1e-6))


@register_scheduler("step")
def step(optimizer: Optimizer, epochs: int, **kwargs):
    step_size = kwargs.get("step_size", max(epochs // 4, 1))
    gamma = kwargs.get("gamma", 0.5)
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


@register_scheduler("plateau")
def plateau(optimizer: Optimizer, epochs: int, **kwargs):
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=kwargs.get("factor", 0.5),
        patience=kwargs.get("patience", 5),
    )
SCHEDULERS_EOF

# ============================================================
# 文件 8/14: src/core/plugins/strategies.py
# ============================================================
cat > src/core/plugins/strategies.py << 'STRATEGIES_EOF'
"""训练策略注册表。

策略 = "如何从一个 batch 计算 loss"。

- vanilla:     单步预测，标准 supervised
- pushforward: 滚动 K 步，每步监督 + 权重衰减

LLM 通过 training.strategy_name 切换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


STRATEGY_REGISTRY: Dict[str, Callable] = {}


@dataclass
class StrategyContext:
    """传给策略函数的上下文。"""
    model: nn.Module
    inp: torch.Tensor              # (B, W+1, X)
    target: torch.Tensor           # (B, 1, X)
    sample_n: torch.Tensor         # (B,)
    sample_t: torch.Tensor         # (B,)
    dataset_data: torch.Tensor     # (N_total, T_r, X_r) — 用于取 future target
    window_size: int
    pushforward_steps: int = 0
    pushforward_weight: float = 0.5
    loss_fn: Callable = None


def register_strategy(name: str):
    def deco(fn):
        if name in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy {name} already registered")
        STRATEGY_REGISTRY[name] = fn
        return fn
    return deco


def get_strategy(name: str) -> Callable:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]


# ===================== 注册 =====================
@register_strategy("vanilla")
def vanilla_step(ctx: StrategyContext) -> torch.Tensor:
    """单步 supervised。"""
    pred = ctx.model(ctx.inp)
    return ctx.loss_fn(pred, ctx.target)


@register_strategy("pushforward")
def pushforward_step(ctx: StrategyContext) -> torch.Tensor:
    """Pushforward: 滚 K 步，每步监督。

    权重 0.5^(K-k-1)，越近权重越大。
    任何样本越过时间轴时停止整个 batch 的 pushforward。
    """
    pred = ctx.model(ctx.inp)
    loss = ctx.loss_fn(pred, ctx.target)

    if ctx.pushforward_steps <= 0:
        return loss

    T_max = ctx.dataset_data.shape[1]
    window_pf = ctx.inp.clone()
    pred_pf = pred
    device = ctx.inp.device

    for k in range(ctx.pushforward_steps):
        window_pf = torch.cat(
            [
                window_pf[:, 1 : ctx.window_size, :],
                pred_pf,
                window_pf[:, ctx.window_size :, :],
            ],
            dim=1,
        )

        future_t = ctx.sample_t + k + 1
        if (future_t >= T_max).any():
            break

        future_target = torch.stack(
            [
                ctx.dataset_data[n, t, :]
                for n, t in zip(
                    ctx.sample_n.tolist(),
                    future_t.tolist(),
                )
            ]
        ).unsqueeze(1).to(device)

        pred_pf = ctx.model(window_pf)
        step_w = ctx.pushforward_weight * (
            0.5 ** (ctx.pushforward_steps - k - 1)
        )
        loss = loss + step_w * ctx.loss_fn(pred_pf, future_target)

    return loss
STRATEGIES_EOF

# ============================================================
# 文件 9/14: src/core/train.py
# ============================================================
cat > src/core/train.py << 'TRAIN_EOF'
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
from .dataset import make_dataloaders
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
    for inp, target, _, _ in val_loader:
        inp = inp.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
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
        logger.info(
            f"[data] T_r={train_ds.T_r}  X_r={train_ds.X_r}  "
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
            start_epoch = (meta.get("epoch") or 0) + 1
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
                dataset_data=train_ds.data,
                window_size=cfg.model.window_size,
                pushforward_steps=cfg.training.pushforward_steps,
                pushforward_weight=cfg.training.pushforward_weight,
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
                marker = " ← best"
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

    _cleanup_ddp()
    return TrainResult(
        status=status,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
        train_time=train_time,
        best_val_loss=best_val,
        best_epoch=best_epoch,
        history=history,
    )
TRAIN_EOF

# ============================================================
# 文件 10/14: src/core/predict.py
# ============================================================
cat > src/core/predict.py << 'PREDICT_EOF'
"""推理主循环。

入口: predict(cfg, ckpt_path, out_dir, logger) -> PredictResult

策略 B (真实 200 步轴):
    输入 input_path 可以是:
      - task1_test.hdf5: (N, 10, 256), 仅前 10 步初始条件 → 提交模式
      - task1_val.hdf5:  (N, 200, 256), 完整数据 → 本地评估模式
    自动识别，同份代码两用。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch

from .config import ExperimentConfig
from .metrics import compute_segment_scores
from .model import load_ckpt


@dataclass
class PredictResult:
    status: str
    pred_path: Optional[Path]
    inference_time: float
    has_gt: bool
    metrics: Optional[Dict[str, float]] = None
    init_diff_max: float = 0.0
    error_msg: str = ""


@torch.no_grad()
def _autoregressive_predict(
    model: torch.nn.Module,
    init_data: torch.Tensor,
    device: torch.device,
    total_steps: int,
    window_size: int,
    use_mean_correction: bool,
    batch_size: int,
) -> torch.Tensor:
    """真实轴自回归。

    init_data: (N, T_init, X)
        T_init=10  → 提交模式
        T_init=200 → 本地评估模式
    返回 (N, total_steps, X)。
    """
    N, T_init, X = init_data.shape
    if T_init < window_size:
        raise ValueError(
            f"init_data 时间步 {T_init} < window_size {window_size}"
        )

    grid = torch.linspace(0, 1, X, device=device).view(1, 1, X)
    pred_full = torch.zeros(N, total_steps, X, device=device)

    fill = min(T_init, total_steps)
    pred_full[:, :fill, :] = init_data[:, :fill, :].to(device)

    model.eval()
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        b = e - s
        grid_b = grid.expand(b, -1, -1)

        window = init_data[s:e, :window_size, :].to(device).clone()
        win_mean_ref = window.mean(dim=[1, 2], keepdim=True)

        for t in range(window_size, total_steps):
            inp = torch.cat([window, grid_b], dim=1)
            out = model(inp)[:, 0:1, :]

            if use_mean_correction:
                pred_mean = out.mean(dim=-1, keepdim=True)
                out = out - pred_mean + win_mean_ref

            pred_full[s:e, t, :] = out[:, 0, :]
            window = torch.cat([window[:, 1:, :], out], dim=1)

    pred_full[:, :window_size, :] = init_data[:, :window_size, :].to(device)
    return pred_full


def predict(
    cfg: ExperimentConfig,
    ckpt_path: Path,
    out_dir: Path,
    logger: logging.Logger,
) -> PredictResult:
    """跑推理，写 hdf5 和 (可选) metrics.json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "task1_pred.hdf5"
    metrics_path = out_dir / "metrics.json"

    input_path = Path(cfg.inference.input_path)
    if not input_path.exists():
        return PredictResult(
            status="failed", pred_path=None, inference_time=0,
            has_gt=False, error_msg=f"input_path 不存在: {input_path}",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info(f"[predict] exp_id={cfg.exp_id}")
    logger.info(f"[predict] ckpt={ckpt_path}")
    logger.info(f"[predict] input={input_path}")
    logger.info(f"[predict] device={device}")

    try:
        model, meta = load_ckpt(Path(ckpt_path), device)
    except Exception as e:
        return PredictResult(
            status="failed", pred_path=None, inference_time=0,
            has_gt=False, error_msg=f"加载 ckpt 失败: {e}",
        )
    logger.info(
        f"[predict] arch={meta.get('architecture')}  "
        f"trained_val_loss={meta.get('val_loss')}"
    )

    with h5py.File(input_path, "r") as f:
        keys = list(f.keys())
        data_key = "tensor" if "tensor" in keys else [
            k for k in keys if len(f[k].shape) == 3
        ][0]
        val_np = f[data_key][:].astype(np.float32)
    N, T_init, X = val_np.shape
    has_gt = T_init >= 200
    logger.info(
        f"[predict] input shape={val_np.shape}  "
        f"mode={'本地评估' if has_gt else '提交'}"
    )

    window_size = cfg.model.window_size
    if T_init < window_size:
        return PredictResult(
            status="failed", pred_path=None, inference_time=0,
            has_gt=has_gt, error_msg=(
                f"输入时间步 {T_init} < window_size {window_size}"
            ),
        )

    val_tensor = torch.from_numpy(val_np)
    t0 = time.time()
    try:
        pred_tensor = _autoregressive_predict(
            model,
            init_data=val_tensor,
            device=device,
            total_steps=cfg.inference.total_steps,
            window_size=window_size,
            use_mean_correction=cfg.inference.use_mean_correction,
            batch_size=cfg.inference.infer_batch,
        )
    except Exception as e:
        return PredictResult(
            status="failed", pred_path=None, inference_time=time.time()-t0,
            has_gt=has_gt, error_msg=f"自回归失败: {e}",
        )
    infer_time = time.time() - t0
    logger.info(f"[predict] 推理 {infer_time:.2f}s (赛题上限 120s)")
    if infer_time > 120:
        logger.error(f"[predict] 超过 2 分钟！该任务会得 0 分")

    pred_np = pred_tensor.cpu().numpy()
    init_diff = float(np.abs(
        pred_np[:, :window_size, :] - val_np[:, :window_size, :]
    ).max())
    logger.info(f"[predict] 前 {window_size} 步最大误差: {init_diff:.2e} (上限 1e-3)")
    if init_diff > 1e-3:
        logger.warning("[predict] ⚠️  前 window 步超容差！")

    expected = (N, cfg.inference.total_steps, X)
    if pred_np.shape != expected:
        return PredictResult(
            status="failed", pred_path=None, inference_time=infer_time,
            has_gt=has_gt, error_msg=(
                f"pred shape {pred_np.shape} != expected {expected}"
            ),
        )

    metrics: Optional[Dict[str, float]] = None
    if has_gt:
        metrics = compute_segment_scores(pred_np, val_np[:, :200, :])
        logger.info("-" * 40)
        logger.info(
            f"Seg1 [10:57]   rel_mse={metrics['seg1_rel_mse']:.6f}  "
            f"score={metrics['seg1_score']:.2f}"
        )
        logger.info(
            f"Seg2 [57:105]  rel_mse={metrics['seg2_rel_mse']:.6f}  "
            f"score={metrics['seg2_score']:.2f}"
        )
        logger.info(
            f"Seg3 [105:200] rmse={metrics['seg3_rmse']:.6f}  "
            f"lor={metrics['seg3_lorentzian']:.2f}  "
            f"frech={metrics['seg3_frechet']:.2f}  "
            f"→ {metrics['seg3_score']:.2f}"
        )
        logger.info(f"🔥 total_score: {metrics['total_score']:.4f} / 100")
        logger.info("-" * 40)
        metrics_path.write_text(json.dumps(metrics, indent=2))

    with h5py.File(pred_path, "w") as f:
        f.create_dataset("tensor", data=pred_np)
    logger.info(f"[predict] 写出 {pred_path}  shape={pred_np.shape}")
    logger.info("=" * 60)

    return PredictResult(
        status="completed",
        pred_path=pred_path,
        inference_time=infer_time,
        has_gt=has_gt,
        metrics=metrics,
        init_diff_max=init_diff,
    )
PREDICT_EOF

# ============================================================
# 文件 11/14: src/core/cli.py
# ============================================================
cat > src/core/cli.py << 'CLI_EOF'
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
CLI_EOF

# ============================================================
# 文件 12/14: conf/experiments/v1_baseline.yaml
# ============================================================
cat > conf/experiments/v1_baseline.yaml << 'YAML_EOF'
# v1_baseline.yaml
# 策略 B: 真实 200 步时间轴 + FNO1d + pushforward + mean_correction
# 目的: 跑通 core 链路，作为后续 agent 提案的 baseline

exp_id: v1_baseline
parent_exp_id: null
rationale: |
  初始 baseline. 采用真实 200 步时间轴 (策略 B), 自训 FNO1d (width=64 比官方 20 大),
  pushforward_steps=3 缓解长时累积误差, mean_correction 修正质量漂移.
description: 真实轴 FNO + pushforward + mean_correction
out_dir: /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/guandi_agent/runs/_manual/v1_baseline

model:
  architecture: FNO1d
  modes: 16
  width: 64
  n_layers: 4
  window_size: 10

data:
  hdf5_path: /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5
  n_samples: 8000
  reduced_resolution_t: 1
  reduced_resolution: 4
  train_ratio: 0.9
  num_workers: 4

training:
  epochs: 50
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  patience: 15
  grad_clip: 1.0
  loss_name: relative_l2
  loss_kwargs: {}
  scheduler_name: cosine
  scheduler_kwargs:
    eta_min: 0.00001
  strategy_name: pushforward
  pushforward_steps: 3
  pushforward_weight: 0.5
  init_from: scratch
  init_ckpt: null

inference:
  use_mean_correction: true
  infer_batch: 256
  input_path: /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_val.hdf5
  total_steps: 200
YAML_EOF

# ============================================================
# 文件 13/14: tests/test_smoke.py
# ============================================================
cat > tests/test_smoke.py << 'SMOKE_EOF'
"""Smoke test: 用合成数据跑通 train → predict 全链路。

跑法 (在 Alvis 容器内):
    cd guandi_agent
    PYTHONPATH=src python -m pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from core.config import (
    ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, InferenceConfig,
    load_config, save_config, diff_configs, _dict_to_experiment,
)
from core.dataset import make_dataloaders
from core.model import build_model, load_ckpt, FNO1d, ClassicFNO1d
from core.metrics import compute_segment_scores
from core.train import train as run_train
from core.predict import predict as run_predict


@pytest.fixture
def synthetic_data(tmp_path):
    rng = np.random.default_rng(42)
    N_train = 100
    train_data = rng.standard_normal((N_train, 201, 1024)).astype(np.float32) * 0.1
    x = np.linspace(0, 1, 1024)
    for i in range(N_train):
        for t in range(201):
            train_data[i, t, :] += 0.5 * np.sin(2 * np.pi * (x - 0.01 * t))
    train_path = tmp_path / "train.hdf5"
    with h5py.File(train_path, "w") as f:
        f.create_dataset("tensor", data=train_data)

    N_val = 20
    val_data = rng.standard_normal((N_val, 200, 1024)).astype(np.float32) * 0.1
    for i in range(N_val):
        for t in range(200):
            val_data[i, t, :] += 0.5 * np.sin(2 * np.pi * (x - 0.01 * t))
    val_path = tmp_path / "val.hdf5"
    with h5py.File(val_path, "w") as f:
        f.create_dataset("tensor", data=val_data)
    return train_path, val_path


@pytest.fixture
def small_config(synthetic_data, tmp_path):
    train_path, val_path = synthetic_data
    out_dir = tmp_path / "exp_smoke"
    cfg = ExperimentConfig(
        exp_id="smoke",
        description="smoke test",
        out_dir=str(out_dir),
        model=ModelConfig(
            architecture="FNO1d",
            modes=4, width=8, n_layers=2, window_size=5,
        ),
        data=DataConfig(
            hdf5_path=str(train_path),
            n_samples=20,
            reduced_resolution_t=1,
            reduced_resolution=16,
            train_ratio=0.8,
            num_workers=0,
        ),
        training=TrainingConfig(
            epochs=2,
            batch_size=8,
            lr=1e-3,
            patience=10,
            strategy_name="pushforward",
            pushforward_steps=2,
            pushforward_weight=0.3,
            scheduler_name="cosine",
        ),
        inference=InferenceConfig(
            use_mean_correction=True,
            infer_batch=8,
            input_path=str(val_path),
            total_steps=200,
        ),
    )
    return cfg, out_dir


class TestConfig:
    def test_save_load_roundtrip(self, small_config, tmp_path):
        cfg, _ = small_config
        path = tmp_path / "cfg.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert asdict(loaded) == asdict(cfg)

    def test_unknown_field_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("foo_bar_unknown: 1\n")
        with pytest.raises(ValueError, match="未知字段"):
            load_config(path)

    def test_diff_configs(self, small_config):
        cfg, _ = small_config
        cfg2 = _dict_to_experiment(asdict(cfg))
        cfg2.training.lr = 5e-4
        cfg2.model.modes = 8
        diffs = diff_configs(cfg, cfg2)
        assert "training.lr" in diffs
        assert "model.modes" in diffs


class TestModel:
    def test_build_fno1d(self):
        cfg = ModelConfig(architecture="FNO1d", modes=4, width=8,
                          n_layers=2, window_size=5)
        m = build_model(cfg)
        assert isinstance(m, FNO1d)
        x = torch.randn(2, 6, 64)
        y = m(x)
        assert y.shape == (2, 1, 64)

    def test_build_classic(self):
        cfg = ModelConfig(architecture="ClassicFNO1d", modes=12, width=20,
                          n_layers=4, window_size=10)
        m = build_model(cfg)
        assert isinstance(m, ClassicFNO1d)
        x = torch.randn(2, 11, 64)
        y = m(x)
        assert y.shape == (2, 1, 64)


class TestMetrics:
    def test_perfect_prediction(self):
        rng = np.random.default_rng(0)
        gt = rng.standard_normal((3, 200, 64)).astype(np.float32)
        m = compute_segment_scores(gt.copy(), gt)
        assert m["seg1_score"] > 99
        assert m["total_score"] > 99


class TestEndToEnd:
    def test_train_predict_pipeline(self, small_config):
        cfg, out_dir = small_config
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("smoke")
        logger.setLevel(logging.INFO)

        tr = run_train(cfg, out_dir, logger)
        assert tr.status in ("completed", "early_stopped"), \
            f"train 失败: {tr.status}, {tr.error_msg}"
        assert tr.ckpt_path is not None and tr.ckpt_path.exists()

        pr = run_predict(cfg, tr.ckpt_path, out_dir, logger)
        assert pr.status == "completed", f"predict 失败: {pr.error_msg}"
        assert pr.has_gt
        assert pr.metrics is not None
        assert pr.init_diff_max < 1e-5

        with h5py.File(out_dir / "task1_pred.hdf5", "r") as f:
            data = f["tensor"][:]
        assert data.shape[1] == 200
        assert data.shape[2] == 1024 // cfg.data.reduced_resolution
SMOKE_EOF

# ============================================================
# 文件 14/14: README.md
# ============================================================
cat > README.md << 'README_EOF'
# guandi_agent

PDE Neural Operator Research Agent — AI4S CNS Challenge Task 1 (1D Burgers).

## 目录

```
guandi_agent/
├── conf/experiments/         实验配置 (yaml)
├── src/core/                 干科研代码 (不依赖 LLM)
├── src/agent/                LLM 调度层 (Phase 3 实现)
├── src/submission/           打包提交 (Phase 4 实现)
├── runs/                     训练产物
├── scripts/                  运维脚本
└── tests/                    单元/冒烟测试
```

## Phase 1 用法 (无 agent)

```bash
# 容器内
PYTHONPATH=src python -m pytest tests/test_smoke.py -v    # smoke test
PYTHONPATH=src python -m core.cli train --config conf/experiments/v1_baseline.yaml
PYTHONPATH=src python -m core.cli predict --config conf/experiments/v1_baseline.yaml
```

## 设计原则

- LLM 只能改 yaml，不能改 .py 源码
- 每个实验独立目录，永不覆盖
- 三轨日志: jsonl (机器) + SQLite (查询) + markdown (评委)
- 详见 architecture.md
README_EOF

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================"
echo "✅ 工程骨架已搭建完成: $ROOT"
echo "============================================"
echo ""
echo "文件清单:"
find "$ROOT" -type f -not -path '*/\.*' | sort | sed "s|$ROOT/|  |"
echo ""
echo "下一步:"
echo "  cd $ROOT"
echo "  # 1. AST 语法检查 (无需 torch)"
echo "  python -c \"import ast; [ast.parse(open(p).read()) for p in __import__('pathlib').Path('src').rglob('*.py')]; print('OK')\""
echo ""
echo "  # 2. 进入 Alvis 容器后跑 smoke test"
echo "  PYTHONPATH=src python -m pytest tests/test_smoke.py -v"
echo ""
echo "  # 3. 真实数据 baseline 训练 (~25 分钟)"
echo "  PYTHONPATH=src python -m core.cli train --config conf/experiments/v1_baseline.yaml"