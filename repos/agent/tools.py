"""
Agent 工具集 — PDE 科研工作流

Agent 可以调用的所有工具函数。
每个工具返回 {"success": bool, "result": str} 格式。
"""

import os
import re
import sys
import csv
import json
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/mimer/NOBACKUP/groups/phy_geo/PDE_Agent")
AGENT_DIR   = PROJECT_DIR / "repos" / "agent"
LOG_DIR     = PROJECT_DIR / "logs"
CKPT_DIR    = PROJECT_DIR / "checkpoints" / "task1_trained"
SUBMISSION  = PROJECT_DIR / "repos" / "submission"

SIF  = PROJECT_DIR / "env" / "pytorch.sif"
VENV = PROJECT_DIR / "env" / "pde_venv" / "bin" / "activate"


# ============================================================
# 内部工具：用容器执行 Python
# ============================================================
VENV_ACTIVATE = str(PROJECT_DIR / "env" / "pde_venv" / "bin" / "activate")

def _run_in_container(cmd: str, timeout: int = 3600) -> dict:
    """直接执行命令（已在容器内，无需再套 apptainer）。"""
    try:
        import copy
        env = copy.copy(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        # 在命令前加source activate，确保torchrun在PATH里
        full_cmd = f"source {VENV_ACTIVATE} && {cmd}"
        proc = subprocess.run(
            full_cmd, shell=True, capture_output=True,
            text=True, timeout=timeout,
            cwd=str(PROJECT_DIR), env=env,
            executable="/bin/bash"
        )
        output = proc.stdout + proc.stderr
        success = proc.returncode == 0
        return {"success": success, "result": output.strip()}
    except subprocess.TimeoutExpired:
        return {"success": False, "result": f"超时（>{timeout}s）"}
    except Exception as e:
        return {"success": False, "result": str(e)}


# ============================================================
# 工具1: 读取文件
# ============================================================
def read_file(path: str) -> dict:
    """
    读取文件内容。
    path: 相对于 PROJECT_DIR 的路径，或绝对路径。
    """
    try:
        p = Path(path) if Path(path).is_absolute() else PROJECT_DIR / path
        content = p.read_text(encoding="utf-8")
        # 超长文件只返回前200行
        lines = content.splitlines()
        if len(lines) > 200:
            content = "\n".join(lines[:200]) + f"\n... (共{len(lines)}行，已截断)"
        return {"success": True, "result": content}
    except Exception as e:
        return {"success": False, "result": str(e)}


# ============================================================
# 工具2: 写入文件
# ============================================================
def write_file(path: str, content: str, backup: bool = True) -> dict:
    """
    写入文件内容。写入前自动备份原文件。
    path: 相对于 PROJECT_DIR 的路径，或绝对路径。
    """
    try:
        p = Path(path) if Path(path).is_absolute() else PROJECT_DIR / path
        p.parent.mkdir(parents=True, exist_ok=True)

        # 备份原文件
        if backup and p.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = p.with_suffix(f".bak_{ts}{p.suffix}")
            shutil.copy2(p, backup_path)

        p.write_text(content, encoding="utf-8")
        return {"success": True, "result": f"已写入 {p}"}
    except Exception as e:
        return {"success": False, "result": str(e)}


# ============================================================
# 工具3: 读取训练日志
# ============================================================
def read_training_log(max_lines: int = 100) -> dict:
    """
    读取最新的训练日志，返回关键信息。
    """
    log_path = LOG_DIR / "task1_train.log"
    if not log_path.exists():
        return {"success": False, "result": "训练日志不存在，可能还未训练"}

    lines = log_path.read_text(encoding="utf-8").splitlines()
    # 只返回最后 max_lines 行
    recent = lines[-max_lines:] if len(lines) > max_lines else lines

    # 解析关键指标
    epochs, train_losses, val_losses, val_rel_mses = [], [], [], []
    for line in lines:
        m = re.search(
            r"Epoch\s+(\d+)/\d+.*train_loss=([\d.]+).*val_loss=([\d.]+).*val_rel_mse=([\d.]+)",
            line
        )
        if m:
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))
            val_rel_mses.append(float(m.group(4)))

    summary = ""
    if epochs:
        summary = (
            f"\n[日志摘要]\n"
            f"  训练轮数: {epochs[-1]}\n"
            f"  最终 train_loss: {train_losses[-1]:.6f}\n"
            f"  最终 val_loss:   {val_losses[-1]:.6f}\n"
            f"  最终 val_rel_mse:{val_rel_mses[-1]:.6f}\n"
            f"  最优 val_loss:   {min(val_losses):.6f} (Epoch {epochs[val_losses.index(min(val_losses))]})\n"
        )

    return {
        "success": True,
        "result": summary + "\n[最近日志]\n" + "\n".join(recent)
    }


# ============================================================
# 工具4: 读取推理评测结果
# ============================================================
def read_eval_metrics() -> dict:
    """
    读取最新的推理日志，提取评测得分。
    """
    log_path = LOG_DIR / "task1_predict.log"
    if not log_path.exists():
        return {"success": False, "result": "推理日志不存在，请先运行 predict.py"}

    content = log_path.read_text(encoding="utf-8")

    # 提取关键数字
    metrics = {}
    patterns = {
        "seg1_rel_mse": r"Seg1 Rel-MSE=([\d.]+)",
        "seg1_score":   r"Seg1.*Score=([\d.]+)",
        "seg2_rel_mse": r"Seg2 Rel-MSE=([\d.]+)",
        "seg2_score":   r"Seg2.*Score=([\d.]+)",
        "seg3_rmse":    r"Seg3 RMSE\s+=\s*([\d.]+)",
        "seg3_score":   r"Seg3.*Score=([\d.]+)",
        "total_score":  r"预测得分[^\d]*([\d.]+)",
        "infer_time":   r"推理完成，耗时:\s*([\d.]+)s",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, content)
        if matches:
            metrics[key] = float(matches[-1])  # 取最后一个，确保是最新结果

    if not metrics:
        return {"success": False, "result": "无法解析评测指标，日志格式可能有问题"}

    result = (
        f"[当前评测结果]\n"
        f"  Seg1 Rel-MSE={metrics.get('seg1_rel_mse','N/A'):.4f}  Score={metrics.get('seg1_score','N/A'):.2f}\n"
        f"  Seg2 Rel-MSE={metrics.get('seg2_rel_mse','N/A'):.4f}  Score={metrics.get('seg2_score','N/A'):.2f}\n"
        f"  Seg3 RMSE   ={metrics.get('seg3_rmse','N/A'):.4f}  Score={metrics.get('seg3_score','N/A'):.2f}\n"
        f"  总分: {metrics.get('total_score','N/A'):.4f} / 100\n"
        f"  推理耗时: {metrics.get('infer_time','N/A'):.2f}s\n"
    )
    return {"success": True, "result": result, "metrics": metrics}


# ============================================================
# 工具5: 运行训练
# ============================================================
def run_training(
    epochs=50, batch_size=64, modes=16, width=64, lr=1e-3,
    extra_args="",
    gpu_ids=os.environ.get("TRAIN_GPUS", "0,1,2,3"),
):
    """
    启动 DDP 训练。
    - 4卡 A40：torchrun，随机端口避免多轮冲突
    - 单卡：直接 python
    A40 NCCL 推荐设置全部内置，不需要外部配置。
    """
    import random
    n_gpus = len(gpu_ids.split(","))

    if n_gpus > 1:
        master_port = random.randint(29500, 30500)
        launcher = (
            f"torchrun "
            f"--nproc_per_node={n_gpus} "
            f"--master_addr=localhost "
            f"--master_port={master_port} "
            f"--rdzv_backend=c10d "
            f"--rdzv_endpoint=localhost:{master_port}"
        )
    else:
        master_port = None
        launcher    = "python"

    # A40 多卡 DDP 稳定性设置
    nccl_env = (
        "NCCL_P2P_DISABLE=1 "       # 禁用 P2P，A40 驱动兼容性更好
        "NCCL_IB_DISABLE=1 "        # 单节点不需要 InfiniBand
        "TORCH_NCCL_BLOCKING_WAIT=1 "  # 超时报错而不是卡死
    )

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_ids} {nccl_env}"
        f"{launcher} repos/agent/train.py "
        f"--epochs {epochs} "
        f"--batch_size {batch_size} "
        f"--modes {modes} "
        f"--width {width} "
        f"--lr {lr} "
        f"{extra_args}"
    )
    port_info = f"port={master_port}" if master_port else "单卡"
    print(f"[tools] 启动训练 ({n_gpus}卡 {port_info}): {cmd[:120]}...")

    timeout = (epochs * 120) + 600
    return _run_in_container(cmd, timeout=timeout)


# ============================================================
# 工具6: 运行推理评测
# ============================================================
def run_inference(gpu_ids: str = os.environ.get("TRAIN_GPUS", "2,3")) -> dict:
    """运行推理并生成提交文件。"""
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_ids} python repos/agent/predict.py"
    print(f"[tools] 启动推理: {cmd}")
    return _run_in_container(cmd, timeout=300)


# ============================================================
# 工具7: 追加科研日志
# ============================================================
# 记录日志起始时间（用于计算 elapsed_seconds）
_LOG_START_TIME = {}

def append_research_log(content: str, log_name: str = "task1_logs.log",
                        tool_calls: str = None) -> dict:
    """
    向科研日志追加一行内容。
    content 已经是组装好的 JSON 字符串（由 orchestrator._log() 负责组装）。
    tool_calls 参数保留但忽略（兼容旧接口）。
    """
    log_path = SUBMISSION / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")
        return {"success": True, "result": f"已追加到 {log_path}"}
    except Exception as e:
        return {"success": False, "result": str(e)}

def call_tool(tool_name: str, kwargs: dict) -> dict:
    """统一工具调用入口。"""
    if tool_name not in TOOL_MAP:
        return {"success": False, "result": f"未知工具: {tool_name}"}
    try:
        return TOOL_MAP[tool_name](**kwargs)
    except Exception as e:
        return {"success": False, "result": f"工具执行异常: {e}"}