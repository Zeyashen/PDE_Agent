import subprocess
import json
import os
import numpy as np
from pathlib import Path
from core.config import load_config, save_config

ROOT = Path(__file__).resolve().parents[2]   # guandi_agent/
DATA_ROOT = ROOT.parent                          # PDE_Agent/

class AgentTools:
    def __init__(self, work_dir: Path, python_bin: str, tracker):
        self.work_dir   = work_dir
        self.python_bin = python_bin
        self.tracker    = tracker

    def run_official_inference(self, exp_id: str = "v0_official") -> dict:
        """加载官方权重纯推理，幂等（已存在则读缓存）。"""
        existing = self.tracker.get_experiment(exp_id)
        if existing and existing.get('status') == 'completed':
            print(f"原checkpoint已存在，跳过推理。", flush=True)
            return {
                'seg1_score':  existing.get('seg1_score', 0),
                'seg2_score':  existing.get('seg2_score', 0),
                'seg3_score':  existing.get('seg3_score', 0),
                'total_score': existing.get('total_score', 0),
            }

        script = ROOT / "scripts" / "run_official_inference.py"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"

        print("正在用官方权重推理...", flush=True)
        result = subprocess.run(
            [self.python_bin, str(script)],
            capture_output=True, text=True, timeout=1800, env=env
        )
        if result.returncode != 0:
            raise RuntimeError(f"官方推理失败:\n{result.stderr[-2000:]}")

        score_file = ROOT / "runs" / "_official" / exp_id / "scores.json"
        with open(score_file) as f:
            data = json.load(f)

        scores  = data["metrics"]
        metrics = {
            'seg1_score':     scores['seg1_score'],
            'seg2_score':     scores['seg2_score'],
            'seg3_score':     scores['seg3_score'],
            'total_score':    scores['total_score'],
            'inference_time': data.get('inference_time', 0),
        }

        self.tracker.register_experiment(
            exp_id=exp_id,
            config_path=str(score_file),
            parent_id=None,
            rationale="官方预训练权重，仅推理，不参与训练迭代"
        )
        self.tracker.update_metrics(exp_id, metrics, status="completed")
        print(f"The reference score: {scores['total_score']:.4f}", flush=True)
        return metrics

    def propose_and_run(self, new_exp_id: str, parent_id: str,
                        changes: dict, rationale: str):
        """基于父实验提出改动，DDP 4卡训练 + 推理。"""
        parent_entry = self.tracker.get_experiment(parent_id)
        if not parent_entry:
            return f"Error: Parent experiment '{parent_id}' not found in database."

        # v0_official 没有 YAML 配置，用 default.yaml 里的 baseline_config 作为模板
        if parent_id == "v0_official":
            import yaml
            _default_yaml = ROOT / "conf" / "default.yaml"
            with open(_default_yaml) as _f:
                _default_cfg = yaml.safe_load(_f)
            _baseline = _default_cfg.get("paths", {}).get(
                "baseline_config",
                str(ROOT / "conf" / "experiments" / "v1_baseline.yaml")
            )
            parent_config_path = Path(_baseline)
        else:
            parent_config_path = Path(parent_entry['config_path'])
            if not parent_config_path.is_absolute():
                parent_config_path = self.work_dir / parent_config_path

        if not parent_config_path.exists():
            return f"Error: Config file missing at {parent_config_path.absolute()}"

        new_dir         = self.work_dir / "runs" / new_exp_id
        new_dir.mkdir(parents=True, exist_ok=True)
        new_config_path = new_dir / "config.yaml"

        print(f"Loading config from: {parent_config_path}", flush=True)
        cfg = load_config(parent_config_path)
        cfg.exp_id        = new_exp_id
        cfg.parent_exp_id = parent_id
        cfg.rationale     = rationale
        cfg.out_dir       = str(new_dir)

        # ── 硬性约束保护 ──────────────────────────────────────
        arch = cfg.model.architecture

        # 锁死不可修改的参数
        changes.pop("window_size", None)
        changes.pop("architecture", None)
        changes.pop("strategy_name", None)

        # epochs 约束
        if "epochs" in changes:
            if arch == "DeepONet1d":
                changes["epochs"] = min(int(changes["epochs"]), 200)
            else:
                changes["epochs"] = min(int(changes["epochs"]), 100)

        # patience 约束
        if "patience" in changes:
            changes["patience"] = max(int(changes["patience"]), 20)

        # lr 约束
        if "lr" in changes:
            changes["lr"] = float(np.clip(float(changes["lr"]), 1e-5, 1e-2))

        # weight_decay 约束
        if "weight_decay" in changes:
            changes["weight_decay"] = float(np.clip(float(changes["weight_decay"]), 1e-6, 1e-2))

        # loss_name 约束：只允许已注册的 loss
        VALID_LOSSES = {"relative_l2", "mse", "huber", "h1", "h1_weighted", "freq", "h1_freq"}
        if "loss_name" in changes:
            if changes["loss_name"] not in VALID_LOSSES:
                print(f"️ loss_name={changes['loss_name']} 不合法，回退到 relative_l2")
                changes["loss_name"] = "relative_l2"

        # batch_size 约束
        if "batch_size" in changes:
            changes["batch_size"] = int(np.clip(int(changes["batch_size"]), 16, 512))

        # DeepONet 专用约束
        if arch == "DeepONet1d":
            if "hidden_dim" in changes:
                changes["hidden_dim"] = int(np.clip(int(changes["hidden_dim"]), 32, 512))
            if "p_dim" in changes:
                changes["p_dim"] = int(np.clip(int(changes["p_dim"]), 32, 512))
            if "branch_layers" in changes:
                changes["branch_layers"] = int(np.clip(int(changes["branch_layers"]), 2, 10))
            if "trunk_layers" in changes:
                changes["trunk_layers"] = int(np.clip(int(changes["trunk_layers"]), 2, 10))
            if "ics_weight" in changes:
                changes["ics_weight"] = float(np.clip(float(changes["ics_weight"]), 1.0, 100.0))

        # FNO 专用约束
        if arch == "FNO1d":
            if "modes" in changes:
                changes["modes"] = int(np.clip(int(changes["modes"]), 4, 128))
            if "width" in changes:
                changes["width"] = int(np.clip(int(changes["width"]), 16, 256))
            if "n_layers" in changes:
                changes["n_layers"] = int(np.clip(int(changes["n_layers"]), 2, 8))
            if "pushforward_steps" in changes:
                changes["pushforward_steps"] = int(np.clip(int(changes["pushforward_steps"]), 1, 10))
            if "pushforward_weight" in changes:
                changes["pushforward_weight"] = float(np.clip(float(changes["pushforward_weight"]), 0.1, 1.0))

        # 判断是否改了拓扑参数，决定 scratch 还是 resume
        TOPOLOGY_PARAMS = {"modes", "width", "n_layers", "p_dim", "hidden_dim", "branch_layers", "trunk_layers"}
        # window_size 固定为10，不允许修改
        changes.pop("window_size", None)
        # architecture 由人工决定，agent 不能修改
        changes.pop("architecture", None)
        if set(changes.keys()) & TOPOLOGY_PARAMS:
            # 改了拓扑，必须从头训
            cfg.training.init_from = "scratch"
            cfg.training.init_ckpt = None
            print(f"检测到拓扑参数变化，init_from=scratch", flush=True)
        else:
            # 只改训练参数，resume 父实验 ckpt
            parent_ckpt = Path(parent_config_path).parent / "best_model.pt"
            if parent_ckpt.exists():
                cfg.training.init_from = "resume"
                cfg.training.init_ckpt = str(parent_ckpt)
                print(f"只改训练参数，init_from=resume: {parent_ckpt}", flush=True)
            else:
                cfg.training.init_from = "scratch"
                cfg.training.init_ckpt = None
                print(f"父实验 ckpt 不存在，回退到 scratch", flush=True)

        for key, value in changes.items():
            if hasattr(cfg.training, key):
                setattr(cfg.training, key, value)
            elif hasattr(cfg.model, key):
                setattr(cfg.model, key, value)

        save_config(cfg, new_config_path)
        self.tracker.register_experiment(
            new_exp_id, new_config_path.absolute(), parent_id, rationale)

        env = os.environ.copy()
        env["PYTHONPATH"]           = str(ROOT / "src")
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

        try:
            print(f"[DDP] 训练: {new_exp_id}...", flush=True)
            subprocess.run([
                self.python_bin, "-m", "torch.distributed.run",
                "--nproc_per_node=4", "--master_port=29501",
                "-m", "core.cli", "train", "--config", str(new_config_path)
            ], check=True, env=env)

            print(f"[PREDICT] predicting...", flush=True)
            val_input = str(DATA_ROOT / "data_and_sample_submission" / "train_val_test_init" / "task1_val.hdf5")
            subprocess.run([
                self.python_bin, "-m", "core.cli", "predict",
                "--config", str(new_config_path),
                "--input", val_input,
            ], check=True, env=env)

            result_json = new_dir / "predict_result.json"
            if not result_json.exists():
                return f"Error: {new_exp_id} 完成但未找到 predict_result.json"

            with open(result_json) as f:
                res = json.load(f)

            metrics = res.get("metrics") or {}
            self.tracker.update_metrics(new_exp_id, metrics)
            total = metrics.get('total_score', 0) if metrics else 0

            # 读取训练摘要，告知 LLM 训练状态
            train_summary = ""
            try:
                train_result_json = new_dir / "train_result.json"
                if train_result_json.exists():
                    with open(train_result_json) as _f:
                        tr = json.load(_f)
                    best_epoch = tr.get('best_epoch', '?')
                    n_epochs   = tr.get('n_epochs_run', '?')
                    status     = tr.get('status', '?')
                    best_val   = tr.get('best_val_loss', float('inf'))
                    train_summary = (f"\n training abstract: status={status} "
                                     f"best_epoch={best_epoch}/{n_epochs} "
                                     f"best_val_loss={best_val:.6f}")
                    if status == 'early_stopped':
                        train_summary += "Early Stopped"
            except Exception:
                pass

            return (f"Success: {new_exp_id} 完成。"
                    f"Total Score: {total:.4f}"
                    f"{train_summary}")

        except subprocess.CalledProcessError as e:
            self.tracker.update_metrics(new_exp_id, {}, status="failed")
            return f"Error: 子进程退出码 {e.returncode}"
        except Exception as e:
            self.tracker.update_metrics(new_exp_id, {}, status="failed")
            return f"Error: {str(e)}"
