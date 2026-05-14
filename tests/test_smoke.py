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
