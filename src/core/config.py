"""配置数据类与加载/保存接口。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ===================== 子配置（必须在 ExperimentConfig 前定义）=====================

@dataclass
class ModelConfig:
    architecture: str = "FNO1d"
    modes: int = 16
    width: int = 64
    n_layers: int = 4
    window_size: int = 10
    # DeepONet 专用参数
    p_dim: int = 128
    hidden_dim: int = 128
    branch_layers: int = 6
    trunk_layers: int = 6


@dataclass
class DataConfig:
    hdf5_path: str = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data/1D/Burgers/Train/1D_Burgers_Sols_Nu0.001.hdf5"
    val_hdf5_path: str = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_val.hdf5"
    test_hdf5_path: str = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/data_and_sample_submission/train_val_test_init/task1_test.hdf5"
    n_samples: Optional[int] = 2000
    reduced_resolution_t: int = 5
    reduced_resolution: int = 4
    train_ratio: float = 1.0
    num_workers: int = 4


@dataclass
class TrainingConfig:
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
    ics_weight: float = 20.0  # PI-DeepONet ICS loss 权重
    init_from: str = "scratch"
    init_ckpt: Optional[str] = None


@dataclass
class InferenceConfig:
    use_mean_correction: bool = True
    infer_batch: int = 256
    input_path: str = ""
    total_steps: int = 200


# ===================== 顶层配置 =====================

@dataclass
class ExperimentConfig:
    exp_id: str = ""
    parent_exp_id: Optional[str] = None
    rationale: str = ""
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    out_dir: str = ""


# ===================== 内部映射 =====================

_NESTED_FIELDS = {
    "ExperimentConfig": {
        "model":     ModelConfig,
        "data":      DataConfig,
        "training":  TrainingConfig,
        "inference": InferenceConfig,
    },
}


# ===================== 加载/保存 =====================

def load_config(path: Path | str) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _dict_to_experiment(raw)
    _validate(cfg)
    return cfg


def save_config(cfg: ExperimentConfig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)


def diff_configs(
    base: ExperimentConfig, new: ExperimentConfig
) -> Dict[str, Tuple[Any, Any]]:
    out: Dict[str, Tuple[Any, Any]] = {}
    _diff_recursive(asdict(base), asdict(new), "", out)
    return out


# ===================== 内部函数 =====================

def _dict_to_experiment(data: dict) -> ExperimentConfig:
    valid = {f.name for f in fields(ExperimentConfig)}
    unknown = set(data.keys()) - valid
    if unknown:
        raise ValueError(f"ExperimentConfig 有未知字段: {unknown}。合法字段: {sorted(valid)}")
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
        raise ValueError(f"{cls.__name__} 有未知字段: {unknown}。合法: {sorted(valid)}")
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

    if cfg.model.architecture not in ("FNO1d", "ClassicFNO1d", "DeepONet1d"):
        errors.append(f"model.architecture={cfg.model.architecture} 不合法")
    if cfg.model.modes < 1:
        errors.append(f"model.modes={cfg.model.modes} 必须 >= 1")
    if cfg.model.width < 1:
        errors.append(f"model.width={cfg.model.width} 必须 >= 1")
    if cfg.model.window_size < 1:
        errors.append(f"model.window_size={cfg.model.window_size} 必须 >= 1")

    if not cfg.data.hdf5_path:
        errors.append("data.hdf5_path 不能为空")
    if cfg.data.reduced_resolution_t != 5:
        errors.append(f"data.reduced_resolution_t 必须为 5，当前={cfg.data.reduced_resolution_t}")
    if cfg.data.reduced_resolution != 4:
        errors.append(f"data.reduced_resolution 必须为 4，当前={cfg.data.reduced_resolution}")
    if not cfg.data.val_hdf5_path:
        errors.append("必须指定 data.val_hdf5_path")
    if not cfg.data.test_hdf5_path:
        errors.append("必须指定 data.test_hdf5_path")

    if cfg.training.init_from not in ("scratch", "resume", "finetune"):
        errors.append(f"training.init_from={cfg.training.init_from} 不合法")
    if cfg.training.init_from != "scratch" and not cfg.training.init_ckpt:
        errors.append(f"init_from={cfg.training.init_from} 时必须指定 init_ckpt")
    if cfg.training.pushforward_steps < 0:
        errors.append(f"pushforward_steps 不能为负")
    if cfg.training.strategy_name not in ("vanilla", "pushforward", "pi_deeponet"):
        errors.append(f"strategy_name={cfg.training.strategy_name} 不在白名单")

    if not cfg.inference.use_mean_correction and cfg.model.architecture != "DeepONet1d":
        errors.append("强烈建议开启 inference.use_mean_correction")
    if cfg.inference.total_steps != 200:
        errors.append(f"inference.total_steps 必须为 200，当前={cfg.inference.total_steps}")

    if errors:
        raise ValueError("配置校验失败:\n" + "\n".join(f"  - {e}" for e in errors))
