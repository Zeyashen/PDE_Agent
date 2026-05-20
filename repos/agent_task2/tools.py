"""
tools.py — Agent 工具实现

所有工具在 apptainer 容器内的 pde_venv 环境中执行。
工具返回结构化结果（方案B），供 orchestrator 解析和记录。
"""

import os
import re
import csv
import json
import time
import random
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# ── 路径常量 ────────────────────────────────────────────────
PROJECT_DIR  = Path("/mimer/NOBACKUP/groups/phy_geo/PDE_Agent")
WORKSPACE    = PROJECT_DIR / "repos" / "workspace"
SKILLS_DIR   = PROJECT_DIR / "repos" / "skills_task2"
SUBMISSION   = PROJECT_DIR / "repos" / "submission"
LOG_DIR      = PROJECT_DIR / "logs"
SIF          = "/apps/containers/vLLM/vllm-0.19.1.sif"
VENV         = PROJECT_DIR / "env" / "pde_venv" / "bin" / "activate"

WORKSPACE.mkdir(parents=True, exist_ok=True)
(WORKSPACE / ".history").mkdir(exist_ok=True)
SUBMISSION.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# ============================================================
# 内部：执行命令（在 apptainer 容器内）
# ============================================================
def _exec(cmd: str, timeout: int = 600, cwd: str = None) -> dict:
    """执行命令（已在容器内运行，直接执行，不需要再套 apptainer）。"""
    cwd = cwd or str(PROJECT_DIR)
    # 已在容器内，直接 cd 到目标目录执行命令
    full_cmd = f"cd {cwd} && {cmd}"
    t0 = time.time()
    try:
        proc = subprocess.run(
            full_cmd, shell=True, capture_output=True,
            text=True, timeout=timeout, executable="/bin/bash"
        )
        elapsed = time.time() - t0
        return {
            "stdout":    proc.stdout.strip(),
            "stderr":    proc.stderr.strip(),
            "returncode": proc.returncode,
            "elapsed":   round(elapsed, 2),
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "", "stderr": f"超时（>{timeout}s）",
            "returncode": -1, "elapsed": timeout,
        }
    except Exception as e:
        return {
            "stdout": "", "stderr": str(e),
            "returncode": -1, "elapsed": time.time() - t0,
        }


# ============================================================
# 文件操作工具
# ============================================================
def write_file(path: str, content: str) -> dict:
    """
    写入文件到 workspace/。
    自动备份旧版本到 workspace/.history/。
    path: 文件名或相对路径，如 "model.py" 或 "checkpoints/best.pt"
          不要包含 "workspace/" 前缀
    """
    # 去掉 LLM 可能传入的 workspace/ 前缀
    clean_path = path
    if clean_path.startswith("workspace/"):
        clean_path = clean_path[len("workspace/"):]

    target = WORKSPACE / clean_path
    target.parent.mkdir(parents=True, exist_ok=True)

    # 备份旧版本（version = 第几次写入该文件）
    hist_dir  = WORKSPACE / ".history"
    hist_dir.mkdir(exist_ok=True)
    hist_name = clean_path.replace("/", "_")
    existing  = list(hist_dir.glob(f"{hist_name}.v*"))
    if target.exists():
        # 文件已存在：备份当前版本，新版本号 = 已备份数 + 2
        shutil.copy2(target, hist_dir / f"{hist_name}.v{len(existing)}")
        version = len(existing) + 2
    else:
        # 文件首次创建
        version = 1

    # 验证 Python 文件内容合法性
    if clean_path.endswith(".py") and content.strip():
        lines = content.strip().splitlines()
        if len(lines) < 30:
            return {
                "success": False,
                "error": f"写入失败：train.py 只有 {len(lines)} 行，太短，不是完整代码。workflow.md 里的模板超过100行，请完整复制。"
            }
        if not any(kw in content for kw in ["import", "def ", "class "]):
            return {
                "success": False,
                "error": "写入失败：content 不是有效的 Python 代码，必须包含 import/def/class。请从 workflow.md 的代码块中复制完整代码。"
            }

    target.write_text(content, encoding="utf-8")

    # 同步到 submission/code/
    code_target = SUBMISSION / "code" / clean_path
    code_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, code_target)

    lines = content.count("\n") + 1
    return {
        "success": True,
        "path":    f"workspace/{clean_path}",
        "lines":   lines,
        "version": version,
    }


def read_file(path: str) -> dict:
    """
    读取文件内容。
    支持 workspace/ 和 skills/ 路径。
    path: 如 "model.py" 或 "skills/scoring.md"
    """
    # 确定实际路径
    if path.startswith("skills/"):
        target = PROJECT_DIR / "repos" / path
    elif path.startswith("workspace/"):
        # workspace/model.py → WORKSPACE/model.py
        clean = path[len("workspace/"):]
        target = WORKSPACE / clean
    else:
        # 裸文件名，先找 workspace，再找 skills
        target = WORKSPACE / path
        if not target.exists():
            target = PROJECT_DIR / "repos" / "skills" / path

    if not target.exists():
        return {"success": False, "error": f"文件不存在: {path}"}

    content = target.read_text(encoding="utf-8")
    return {
        "success": True,
        "path":    str(target.relative_to(PROJECT_DIR / "repos")),
        "content": content,
        "lines":   content.count("\n") + 1,
    }


def edit_file(path: str, old_str: str, new_str: str) -> dict:
    """
    精准替换文件中的唯一字符串。
    old_str 必须在文件中唯一出现一次。
    """
    # 去掉可能的 workspace/ 前缀
    clean_path = path
    if clean_path.startswith("workspace/"):
        clean_path = clean_path[len("workspace/"):]

    target = WORKSPACE / clean_path
    if not target.exists():
        return {"success": False, "error": f"文件不存在: workspace/{clean_path}"}

    content = target.read_text(encoding="utf-8")
    count   = content.count(old_str)

    if count == 0:
        return {"success": False, "error": "old_str 在文件中未找到"}
    if count > 1:
        return {"success": False,
                "error": f"old_str 在文件中出现 {count} 次，必须唯一"}

    # 备份
    hist_dir  = WORKSPACE / ".history"
    hist_dir.mkdir(exist_ok=True)
    hist_name = clean_path.replace("/", "_")
    existing  = list(hist_dir.glob(f"{hist_name}.v*"))
    version   = len(existing) + 1
    shutil.copy2(target, hist_dir / f"{hist_name}.v{version-1}")

    new_content = content.replace(old_str, new_str, 1)
    target.write_text(new_content, encoding="utf-8")

    # 同步到 submission/code/
    code_target = SUBMISSION / "code" / clean_path
    code_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, code_target)

    old_lines = old_str.count("\n") + 1
    new_lines = new_str.count("\n") + 1
    return {
        "success":       True,
        "path":          f"workspace/{clean_path}",
        "lines_changed": new_lines - old_lines,
        "version":       version,
    }


def list_files(dir: str = "workspace") -> dict:
    """列出目录内容（文件名、大小、修改时间）。"""
    if dir == "workspace":
        target = WORKSPACE
    elif dir == "skills":
        target = SKILLS_DIR
    else:
        target = PROJECT_DIR / "repos" / dir

    if not target.exists():
        return {"success": False, "error": f"目录不存在: {dir}"}

    files = []
    for f in sorted(target.iterdir()):
        if f.name.startswith(".") or f.name == "code":
            continue
        if f.is_file():
            files.append({
                "name":     f.name,
                "size_kb":  round(f.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(
                    f.stat().st_mtime).strftime("%m-%d %H:%M"),
            })

    return {"success": True, "dir": dir, "files": files}


# ============================================================
# 执行工具
# ============================================================
def run_bash(command: str, timeout: int = 300) -> dict:
    """
    在容器内执行任意 bash 命令。
    返回 stdout、stderr、returncode、elapsed。
    """
    result = _exec(command, timeout=timeout)
    return result


def run_training(
    script:     str = "train.py",
    extra_args: str = "",
    timeout:    int = 3600,
    n_gpus:     int = 4,
    epochs:     int = None,
) -> dict:
    """
    封装 DDP 训练，自动处理 torchrun、NCCL、随机端口。
    训练日志实时写入 logs/train_live.log，可用 read_training_log 监控。
    
    script:     文件名，如 "train.py"（不要加 workspace/ 前缀）
    extra_args: 额外命令行参数，如 "--epochs 30 --lr 0.001"
    epochs:     如果指定，自动追加 --epochs N 到 extra_args
    n_gpus:     使用的 GPU 数量
    """
    port = random.randint(29500, 30500)

    PYTHON_BIN = str(PROJECT_DIR / "env" / "pde_venv" / "bin" / "python")
    if n_gpus > 1:
        launcher = (
            f"CUDA_VISIBLE_DEVICES=0,1,2,3 "
            f"NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 "
            f"{PYTHON_BIN} -m torch.distributed.run "
            f"--nproc_per_node={n_gpus} "
            f"--master_addr=localhost --master_port={port} "
            f"--rdzv_backend=c10d --rdzv_endpoint=localhost:{port}"
        )
    else:
        launcher = "CUDA_VISIBLE_DEVICES=0 /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/env/pde_venv/bin/python"

    # 去掉 workspace/ 前缀
    clean_script = script
    if clean_script.startswith("workspace/"):
        clean_script = clean_script[len("workspace/"):]
    script_path = WORKSPACE / clean_script

    # epochs 参数
    if epochs is not None:
        extra_args = f"--epochs {epochs} {extra_args}".strip()

    # 实时日志路径
    live_log = LOG_DIR / "train_live.log"
    live_log.write_text("", encoding="utf-8")  # 清空

    # 用 tee 实时写日志，同时捕获输出
    cmd = f"{launcher} {script_path} {extra_args} 2>&1 | tee {live_log}"
    result = _exec(cmd, timeout=timeout, cwd=str(WORKSPACE))

    stdout  = result["stdout"] + result["stderr"]
    # 也读 live_log 确保完整
    if live_log.exists():
        stdout = live_log.read_text(encoding="utf-8")

    parsed  = _parse_training_output(stdout)
    success = result["returncode"] == 0 and parsed.get("val_loss") is not None

    log_lines = stdout.splitlines()
    if not success:
        error_lines = [l for l in log_lines
                       if any(k in l for k in
                              ["Error","Exception","Traceback","error","FAILED"])]
        log_summary = "\n".join(log_lines[-30:])
        if error_lines:
            log_summary = ("关键错误:\n" + "\n".join(error_lines[-8:])
                           + "\n\n最后20行:\n" + "\n".join(log_lines[-20:]))
    else:
        log_summary = "\n".join(log_lines[-20:])

    return {
        "success":         success,
        "returncode":      result["returncode"],
        "val_loss":        parsed.get("val_loss"),
        "best_epoch":      parsed.get("best_epoch"),
        "elapsed_seconds": result["elapsed"],
        "ckpt_path":       parsed.get("ckpt_path",
                           str(WORKSPACE / "checkpoints" / "task2_best.pt")),
        "log_tail":        log_summary,
        "error":           (result["stderr"] or stdout)[-500:] if not success else "",
        "live_log_path":   str(live_log),
    }


# def quick_eval(
#     ckpt_path: str,
#     script:    str = "predict.py",
#     timeout:   int = 60,
# ) -> dict:
#     """
#     快速评估：只读取 checkpoint 里保存的 val_loss，秒级返回。
#     用于训练完成后快速判断模型是否有改善，不需要跑完整推理。
#     workflow: run_training → quick_eval → (如果 val_loss 好) → full_eval
#     """
#     clean_ckpt = ckpt_path
#     if clean_ckpt.startswith("workspace/"):
#         clean_ckpt = clean_ckpt[len("workspace/"):]
#     ckpt_abs = WORKSPACE / clean_ckpt

#     if not ckpt_abs.exists():
#         return {"success": False,
#                 "error": f"Checkpoint 不存在: {ckpt_abs}",
#                 "val_loss": None}

#     # 更简单的方式：直接写一个临时 Python 脚本
#     tmp_script = LOG_DIR / "_quick_eval_tmp.py"
#     tmp_script.write_text(
#         f"import torch\n"
#         f"c = torch.load('{ckpt_abs}', map_location='cpu', weights_only=False)\n"
#         f"print(f'val_loss={{c.get(\"val_loss\", \"N/A\")}}')\n"
#         f"print(f'epoch={{c.get(\"epoch\", \"N/A\")}}')\n",
#         encoding="utf-8"
#     )
#     result = _exec(f"CUDA_VISIBLE_DEVICES=0 /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/env/pde_venv/bin/python {tmp_script}", timeout=timeout)
#     stdout = result["stdout"].strip()
#     try:
#         tmp_script.unlink()
#     except Exception:
#         pass

#     val_loss = None
#     import re as _re
#     m = _re.search(r"val_loss=([\d.eE+-]+)", stdout)
#     if m:
#         try:
#             val_loss = float(m.group(1))
#         except:
#             pass

#     return {
#         "success":  val_loss is not None,
#         "val_loss": val_loss,
#         "ckpt_path": str(ckpt_abs),
#         "stdout":   stdout,
#         "note": "quick_eval 只看 val_loss，用 full_eval 获取 Seg1/2/3 真实得分",
#     }


def full_eval(
    ckpt_path: str,
    script:    str = "predict.py",
    val_only:  bool = True,
    timeout:   int = 300,
) -> dict:
    """
    完整评估：运行推理并计算 Seg1/2/3 真实竞赛得分。
    比 quick_eval 慢（需要跑完整推理），但给出真实分数。
    建议：先 quick_eval 确认 val_loss OK，再 full_eval 确认真实得分。
    """
    if ckpt_path in ("ckpt_path", "", None):
        ckpt_path = "checkpoints/task2_best.pt"
    if script in ("script", "", None):
        script = "predict.py"
    clean_script = script
    if clean_script.startswith("workspace/"):
        clean_script = clean_script[len("workspace/"):]
    script_path = WORKSPACE / clean_script

    clean_ckpt = ckpt_path
    if clean_ckpt.startswith("workspace/"):
        clean_ckpt = clean_ckpt[len("workspace/"):]
    ckpt_abs = WORKSPACE / clean_ckpt

    flag   = "--val_only" if val_only else ""
    PYTHON_BIN = str(PROJECT_DIR / "env" / "pde_venv" / "bin" / "python")
    cmd = f"CUDA_VISIBLE_DEVICES=0 {PYTHON_BIN} {script_path} --ckpt {ckpt_abs} {flag}"
    result = _exec(cmd, timeout=timeout, cwd=str(WORKSPACE))

    stdout  = result["stdout"] + result["stderr"]
    parsed  = _parse_eval_output(stdout)
    success = result["returncode"] == 0 and parsed.get("total_score") is not None

    total     = parsed.get("total_score", 0.0)
    precision = round(total * 0.75, 4)
    infer_t   = parsed.get("infer_time", 0.0)
    infer_min = infer_t / 60
    if infer_min <= 0:    infer_score = 40.0
    elif infer_min < 2:   infer_score = round(40*(1-infer_min/2), 2)
    else:                 infer_score = 0.0

    return {
        "success":         success,
        "total_score":     total,
        "seg1_score":      parsed.get("seg1_score",  0.0),
        "seg2_score":      parsed.get("seg2_score",  0.0),
        "seg3_score":      parsed.get("seg3_score",  0.0),
        "precision_score": precision,
        "infer_time":      infer_t,
        "infer_score":     infer_score,
        "log_tail":        "\n".join(stdout.splitlines()[-15:]),
        "error":           stdout[-500:] if not success else "",
    }


def run_evaluation(ckpt_path, script="predict.py",
                   val_only=True, timeout=300):
    """full_eval 的别名，向后兼容。"""
    return full_eval(ckpt_path, script, val_only, timeout)


def get_score_history() -> dict:
    """返回所有迭代的实验记录（从 experiment_log.md 读取）。"""
    log_path = SKILLS_DIR / "experiment_log.md"
    if not log_path.exists():
        return {"success": True, "history": [], "count": 0}
    content = log_path.read_text(encoding="utf-8")
    return {
        "success": True,
        "content": content,
        "count":   content.count("## 迭代#"),
    }


def check_remaining_time(start_time: float, budget_minutes: float) -> dict:
    """返回时间预算状态。"""
    elapsed_s   = time.time() - start_time
    elapsed_min = elapsed_s / 60
    remaining   = budget_minutes - elapsed_min

    if elapsed_min <= 60:
        time_score = 35
    elif elapsed_min <= 120:
        time_score = 25
    elif elapsed_min <= 300:
        time_score = 20
    elif elapsed_min <= 500:
        time_score = 10
    else:
        time_score = 0

    return {
        "elapsed_minutes":        round(elapsed_min, 1),
        "remaining_minutes":      round(remaining, 1),
        "budget_minutes":         budget_minutes,
        "time_score_prediction":  time_score,
        "should_stop":            remaining < 8,
    }


# ============================================================
# 辅助：日志追加
# ============================================================
def append_research_log(content: str, log_name: str = "task1_logs.log") -> None:
    """追加一条 JSONL 记录到科研日志。"""
    log_path = SUBMISSION / log_name
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")


def append_experiment_log(iteration: int, summary: dict) -> None:
    """把实验结论追加到 skills/experiment_log.md。"""
    log_path = SKILLS_DIR / "experiment_log.md"
    ts       = datetime.now().strftime("%Y-%m-%d %H:%M")
    block    = (
        f"\n## 迭代#{iteration} | {ts}\n"
        f"模型: {summary.get('model_desc', '未知')}\n"
        f"分数: 总分={summary.get('total_score', 0):.2f} "
        f"Seg1={summary.get('seg1', 0):.1f} "
        f"Seg2={summary.get('seg2', 0):.1f} "
        f"Seg3={summary.get('seg3', 0):.1f}\n"
        f"训练耗时: {summary.get('train_time', 0):.0f}s  "
        f"推理耗时: {summary.get('infer_time', 0):.2f}s\n"
        f"结论: {summary.get('conclusion', '无')}\n"
        f"下一步: {summary.get('next_step', '待定')}\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(block)


# ============================================================
# 内部：解析训练和评分输出
# ============================================================
def _parse_training_output(text: str) -> dict:
    result = {}
    # val_loss
    m = re.findall(r"val[=:\s]+([\d.]+)", text)
    if m:
        result["val_loss"] = float(m[-1])
    # best_epoch
    m = re.findall(r"Epoch\s+(\d+).*?<-\s*best", text)
    if m:
        result["best_epoch"] = int(m[-1])
    # ckpt_path
    m = re.search(r"Candidate.*?:\s*(\S+\.pt)", text)
    if m:
        result["ckpt_path"] = m.group(1)
    else:
        result["ckpt_path"] = "workspace/checkpoints/task2_best.pt"
    return result


def _parse_eval_output(text: str) -> dict:
    result = {}
    patterns = {
        "seg1_score":  r"Seg1.*?Score=([\d.]+)",
        "seg2_score":  r"Seg2.*?Score=([\d.]+)",
        "seg3_score":  r"Seg3.*?Score=([\d.]+)",
        "total_score": r"预测得分:\s*([\d.]+)",
        "infer_time":  r"推理完成\s+([\d.]+)s",
    }
    for key, pat in patterns.items():
        m = re.findall(pat, text)
        if m:
            result[key] = float(m[-1])
    return result


# ============================================================
# 工具分发（orchestrator 调用）
# ============================================================

# ============================================================
# 新增工具：todo_write / todo_read / run_bash_async / check_process
# ============================================================

def todo_write(todos: list) -> dict:
    """
    写入任务列表到 workspace/todos.json，作为 Agent 的工作记忆。
    每个 todo 包含 content, status, priority 三个字段。
    status: "pending" | "in_progress" | "completed" | "failed"
    """
    todos_file = WORKSPACE / "todos.json"
    if not isinstance(todos, list):
        return {"success": False, "error": "todos 必须是列表"}

    # 验证每个 todo 的格式，兼容字符串列表和字典列表
    valid_status = {"pending", "in_progress", "completed", "failed"}
    cleaned = []
    for i, t in enumerate(todos):
        if isinstance(t, str):
            # 兼容字符串格式：["任务1", "任务2"]
            t = {"content": t, "status": "pending", "priority": "normal"}
        if not isinstance(t, dict):
            continue  # 跳过无法解析的元素，不报错
        content  = t.get("content", "").strip()
        status   = t.get("status", "pending")
        priority = t.get("priority", "normal")
        if not content:
            continue
        if status not in valid_status:
            status = "pending"
        cleaned.append({
            "id":       i + 1,
            "content":  content,
            "status":   status,
            "priority": priority,
        })

    todos_file.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    n_done = sum(1 for t in cleaned if t["status"] == "completed")
    n_total = len(cleaned)
    return {
        "success": True,
        "saved_count": n_total,
        "completed":   n_done,
        "summary": f"已保存 {n_total} 个任务，完成 {n_done}/{n_total}",
    }


def todo_read() -> dict:
    """读取当前任务列表。"""
    todos_file = WORKSPACE / "todos.json"
    if not todos_file.exists():
        return {
            "success": True,
            "todos": [],
            "count": 0,
            "note": "todos.json 不存在，建议先用 todo_write 创建任务列表",
        }
    try:
        todos = json.loads(todos_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON 解析失败: {e}"}

    n_total = len(todos)
    n_done  = sum(1 for t in todos if t.get("status") == "completed")
    n_failed = sum(1 for t in todos if t.get("status") == "failed")
    n_inprogress = sum(1 for t in todos if t.get("status") == "in_progress")
    return {
        "success":     True,
        "todos":       todos,
        "count":       n_total,
        "completed":   n_done,
        "failed":      n_failed,
        "in_progress": n_inprogress,
        "pending":     n_total - n_done - n_failed - n_inprogress,
    }


# 后台进程注册表（pid → log_file）
_BG_PROCESSES = {}


def run_bash_async(command: str, log_file: str = None) -> dict:
    """
    后台执行命令，立即返回 pid。
    适用于长时间训练（>10分钟），Agent 可以用 check_process 后续查看进度。

    注意：只允许 1 个后台训练同时运行，启动新的会先检查旧的。
    log_file: 日志文件路径（相对 LOG_DIR），如不指定则自动生成
    """
    # 检查是否已有后台任务在跑
    alive = []
    for pid, lf in list(_BG_PROCESSES.items()):
        try:
            os.kill(pid, 0)  # 检查进程是否存在
            alive.append(pid)
        except OSError:
            _BG_PROCESSES.pop(pid, None)  # 已结束，清理

    if alive:
        return {
            "success": False,
            "error": f"已有后台进程 {alive} 在运行，请先 check_process 确认状态",
        }

    if not log_file:
        log_file = f"bg_task_{int(time.time())}.log"
    log_path = LOG_DIR / log_file
    log_path.write_text("", encoding="utf-8")

    # nohup 后台执行
    full_cmd = (
        f"nohup bash -c 'cd {PROJECT_DIR} && {command}' "
        f"> {log_path} 2>&1 & echo $!"
    )
    try:
        proc = subprocess.run(
            full_cmd, shell=True, capture_output=True,
            text=True, timeout=10, executable="/bin/bash",
        )
        pid = int(proc.stdout.strip())
        _BG_PROCESSES[pid] = str(log_path)
        return {
            "success":  True,
            "pid":      pid,
            "log_file": str(log_path),
            "command":  command[:200],
            "note":     "用 check_process(pid) 查看进度",
        }
    except Exception as e:
        return {"success": False, "error": f"启动失败: {e}"}


def check_process(pid: int, tail_n: int = 30) -> dict:
    """
    检查后台进程状态，返回是否在运行 + 日志末尾 N 行。
    pid: run_bash_async 返回的进程 ID
    tail_n: 返回日志的最后 N 行
    """
    log_file = _BG_PROCESSES.get(pid)
    if not log_file:
        # 试图通过 ps 命令检查
        log_file = None

    # 检查进程是否存活
    try:
        os.kill(int(pid), 0)
        alive = True
    except (OSError, ValueError):
        alive = False
        _BG_PROCESSES.pop(int(pid), None)

    result = {
        "pid":       pid,
        "running":   alive,
        "status":    "running" if alive else "finished",
    }

    if log_file and Path(log_file).exists():
        lines = Path(log_file).read_text(encoding="utf-8").splitlines()
        result["log_tail"]   = "\n".join(lines[-tail_n:])
        result["total_lines"] = len(lines)
        result["log_file"]   = log_file
    else:
        result["log_tail"] = "（日志文件未找到）"

    # 尝试解析训练进度
    if log_file:
        try:
            content = Path(log_file).read_text(encoding="utf-8")
            parsed  = _parse_training_output(content)
            if parsed.get("val_loss") is not None:
                result["current_val_loss"] = parsed["val_loss"]
            if parsed.get("best_epoch") is not None:
                result["best_epoch"] = parsed["best_epoch"]
        except Exception:
            pass

    return result



def smoke_test(script: str = "train.py", timeout: int = 90) -> dict:
    """
    单 batch 快速测试：用最小配置实际跑一次训练循环。
    epochs=1, batch_size=2, n_query=128, 单卡。
    通常 30-60 秒内完成，验证代码能否端到端运行。

    返回:
      success: 是否成功
      val_loss: 如果成功，返回 val_loss（约略值）
      error: 失败时的关键错误信息
      elapsed: 耗时
    """
    clean_script = script
    if clean_script.startswith("workspace/"):
        clean_script = clean_script[len("workspace/"):]
    script_path = WORKSPACE / clean_script

    if not script_path.exists():
        return {"success": False, "error": f"脚本不存在: {script_path}"}

    cmd = (f"CUDA_VISIBLE_DEVICES=0 /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/env/pde_venv/bin/python {script_path} "
       f"--epochs 1 --batch_size 2 --n_query 128 --n_samples 4")
    result = _exec(cmd, timeout=timeout)
    stdout = result["stdout"] + result["stderr"]
    parsed = _parse_training_output(stdout)
    success = (result["returncode"] == 0 and
               parsed.get("val_loss") is not None)

    log_lines = stdout.splitlines()
    if not success:
        # 提取最关键的错误信息（Error/Traceback 行）
        error_lines = [l for l in log_lines
                       if any(k in l for k in
                              ["Error", "Exception", "Traceback",
                               "error", "ModuleNotFound", "AttributeError",
                               "TypeError", "KeyError", "RuntimeError"])]
        err_summary = "\n".join(error_lines[-10:]) if error_lines else ""
        err_summary += "\n--- 最后20行 ---\n" + "\n".join(log_lines[-20:])
    else:
        err_summary = ""

    return {
        "success":  success,
        "val_loss": parsed.get("val_loss"),
        "elapsed":  result["elapsed"],
        "error":    err_summary[:1000],
        "log_tail": "\n".join(log_lines[-20:]),
    }

def read_training_log(max_lines: int = 50) -> dict:
    """读取实时训练日志（训练进行中可调用，查看当前 epoch 进度）。"""
    live_log = LOG_DIR / "train_live.log"
    if not live_log.exists():
        return {"success": False, "error": "训练日志不存在，训练可能尚未开始"}
    lines  = live_log.read_text(encoding="utf-8").splitlines()
    recent = lines[-max_lines:] if len(lines) > max_lines else lines
    return {
        "success":     True,
        "total_lines": len(lines),
        "content":     "\n".join(recent),
    }


TOOL_MAP = {
    "write_file":   write_file,
    "read_file":    read_file,
    "edit_file":    edit_file,
    "list_files":   list_files,
    "run_bash":     run_bash,
    "run_bash_async": run_bash_async,
    "check_process":  check_process,
    "run_training": run_training,
    "smoke_test":   smoke_test,
    # "quick_eval":   quick_eval,
    "full_eval":    full_eval,
    "run_evaluation": run_evaluation,
    "read_training_log":    read_training_log,
    "todo_write":          todo_write,
    "todo_read":           todo_read,
    "get_score_history":    get_score_history,
    "check_remaining_time": check_remaining_time,
}


def execute_tool(tool_name: str, args: dict,
                 start_time: float = None,
                 budget_minutes: float = 55) -> dict:
    """统一工具执行入口。"""
    if tool_name not in TOOL_MAP:
        return {"success": False, "error": f"未知工具: {tool_name}"}

    # check_remaining_time 需要注入运行时参数
    if tool_name == "check_remaining_time":
        args["start_time"]      = start_time or time.time()
        args["budget_minutes"]  = budget_minutes

    try:
        fn     = TOOL_MAP[tool_name]
        result = fn(**args)
        return result if isinstance(result, dict) else {"result": result}
    except TypeError as e:
        return {"success": False, "error": f"参数错误: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# Unittest
# ============================================================
if __name__ == "__main__":
    import tempfile, time
    print("=" * 55)
    print("tools.py unittest")
    print("=" * 55)

    errors = []

    def check(name, fn):
        try:
            fn()
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            errors.append((name, str(e)))

    # ── 环境准备 ─────────────────────────────────────────────
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / ".history").mkdir(exist_ok=True)
    (SUBMISSION / "code").mkdir(parents=True, exist_ok=True)

    # Test 1: write_file
    print("\n[1] write_file")
    def t1a():
        r = write_file("_test_file.py", "# test v1\nprint('hello')")
        assert r["success"],          f"success=False: {r}"
        assert r["version"] >= 1,     f"version应>=1: {r['version']}"
        assert r["lines"] == 2,       f"lines={r['lines']}"
        assert (WORKSPACE / "_test_file.py").exists()
        assert (SUBMISSION / "code" / "_test_file.py").exists()
    check("write_file 新建文件", t1a)

    def t1b():
        # 清理历史文件，确保从干净状态开始
        import glob
        for f in (WORKSPACE / ".history").glob("_test_version.py.v*"):
            f.unlink()
        (WORKSPACE / "_test_version.py").unlink(missing_ok=True)
        # 连续写三次，验证版本单调递增
        r1 = write_file("_test_version.py", "# v1")
        r2 = write_file("_test_version.py", "# v2")
        r3 = write_file("_test_version.py", "# v3")
        assert r1["success"] and r2["success"] and r3["success"]
        assert r1["version"] == 1, f"v1应=1: {r1['version']}"
        assert r2["version"] == 2, f"v2应=2: {r2['version']}"
        assert r3["version"] == 3, f"v3应=3: {r3['version']}"
    check("write_file 覆盖+版本递增", t1b)

    def t1c():
        r = write_file("workspace/_test_file.py", "# v3")
        assert r["success"],          "带workspace/前缀应该被去掉"
        assert "workspace/workspace" not in r["path"]
    check("write_file 自动去除workspace/前缀", t1c)

    # Test 2: read_file
    print("\n[2] read_file")
    def t2a():
        r = read_file("_test_file.py")
        assert r["success"],           f"success=False: {r}"
        assert "v3" in r["content"],   f"内容不对: {r['content'][:50]}"
    check("read_file workspace文件", t2a)

    def t2b():
        r = read_file("skills/scoring.md")
        assert r["success"],           f"skills/scoring.md 不存在: {r}"
        assert len(r["content"]) > 0,  "内容为空"
    check("read_file skills/文件", t2b)

    def t2c():
        r = read_file("_nonexistent_file_xyz.py")
        assert not r["success"],       "不存在的文件应该返回 success=False"
        assert "error" in r
    check("read_file 不存在的文件返回错误", t2c)

    # Test 3: edit_file
    print("\n[3] edit_file")
    write_file("_test_edit.py", "x = 1\ny = 2\nz = 3\n")

    def t3a():
        r = edit_file("_test_edit.py", "y = 2", "y = 999")
        assert r["success"],  f"success=False: {r}"
        content = (WORKSPACE / "_test_edit.py").read_text()
        assert "y = 999" in content
        assert "y = 2" not in content
    check("edit_file 正常替换", t3a)

    def t3b():
        r = edit_file("_test_edit.py", "NOT_EXIST_STRING", "new")
        assert not r["success"],  "不存在的 old_str 应返回失败"
        assert "未找到" in r.get("error", "")
    check("edit_file old_str不存在返回错误", t3b)

    def t3c():
        write_file("_test_dup.py", "a = 1\na = 1\n")
        r = edit_file("_test_dup.py", "a = 1", "a = 2")
        assert not r["success"],  "重复 old_str 应返回失败"
        assert "次" in r.get("error", "")
    check("edit_file old_str不唯一返回错误", t3c)

    def t3d():
        r = edit_file("workspace/_test_edit.py", "z = 3", "z = 777")
        assert r["success"],  "带workspace/前缀应被正确处理"
        assert "workspace/workspace" not in r.get("path", "")
    check("edit_file 自动去除workspace/前缀", t3d)

    # Test 4: list_files
    print("\n[4] list_files")
    def t4a():
        r = list_files("workspace")
        assert r["success"],         f"success=False: {r}"
        names = [f["name"] for f in r["files"]]
        assert "_test_file.py" in names
    check("list_files workspace", t4a)

    def t4b():
        r = list_files("skills")
        assert r["success"],         f"success=False: {r}"
        names = [f["name"] for f in r["files"]]
        assert any("scoring" in n for n in names), f"找不到scoring.md: {names}"
    check("list_files skills", t4b)

    def t4c():
        r = list_files("nonexistent_dir_xyz")
        assert not r["success"],     "不存在目录应返回失败"
    check("list_files 不存在目录返回错误", t4c)

    # Test 5: todo_write / todo_read
    print("\n[5] todo_write / todo_read")
    def t5a():
        todos = [
            {"content": "写 train.py",  "status": "completed", "priority": "high"},
            {"content": "跑训练",        "status": "in_progress","priority": "high"},
            {"content": "评估结果",      "status": "pending",    "priority": "normal"},
        ]
        r = todo_write(todos)
        assert r["success"],              f"todo_write 失败: {r}"
        assert r["saved_count"] == 3,     f"saved_count={r['saved_count']}"
        assert r["completed"] == 1,       f"completed={r['completed']}"
    check("todo_write 写入3条任务", t5a)

    def t5b():
        r = todo_read()
        assert r["success"],              f"todo_read 失败: {r}"
        assert r["count"] == 3,           f"count={r['count']}"
        assert r["completed"] == 1,       f"completed={r['completed']}"
        assert r["in_progress"] == 1,     f"in_progress={r['in_progress']}"
        assert r["pending"] == 1,         f"pending={r['pending']}"
    check("todo_read 状态统计正确", t5b)

    def t5c():
        r = todo_write("not a list")
        assert not r["success"],          "非列表输入应返回失败"
    check("todo_write 非法输入返回错误", t5c)

    # Test 6: quick_eval checkpoint 不存在
    print("\n[6] quick_eval")
    def t6():
        r = quick_eval("checkpoints/nonexistent_xyz.pt")
        assert not r["success"],          "不存在的ckpt应返回失败"
        assert r["val_loss"] is None,     "val_loss应为None"
        assert "error" in r
    check("quick_eval ckpt不存在返回错误", t6)

    # Test 7: check_remaining_time 时间分档
    print("\n[7] check_remaining_time")
    def t7():
        start = time.time() - 30 * 60  # 模拟已用30分钟
        r = check_remaining_time(start, 55.0)
        assert r["success"] if "success" in r else True
        assert r["time_score_prediction"] == 35,  f"30min应得35分: {r}"
        assert abs(r["elapsed_minutes"] - 30) < 1

        start2 = time.time() - 65 * 60  # 模拟已用65分钟
        r2 = check_remaining_time(start2, 200.0)
        assert r2["time_score_prediction"] == 25,  f"65min应得25分: {r2}"

        start3 = time.time() - 150 * 60  # 模拟已用150分钟
        r3 = check_remaining_time(start3, 600.0)
        assert r3["time_score_prediction"] == 20,  f"150min应得20分: {r3}"
    check("check_remaining_time 时间分档正确", t7)

    # Test 8: smoke_test 脚本不存在
    print("\n[8] smoke_test")
    def t8():
        r = smoke_test("nonexistent_train_xyz.py", timeout=5)
        assert not r["success"],  "不存在脚本应返回失败"
    check("smoke_test 脚本不存在返回失败", t8)

    # Test 9: execute_tool 未知工具
    print("\n[9] execute_tool")
    def t9a():
        r = execute_tool("nonexistent_tool_xyz", {})
        assert not r["success"],  "未知工具应返回失败"
        assert "未知工具" in r.get("error", "")
    check("execute_tool 未知工具返回错误", t9a)

    def t9b():
        r = execute_tool("list_files", {"dir": "workspace"})
        assert r["success"],  f"list_files 通过execute_tool调用失败: {r}"
    check("execute_tool 正常调用 list_files", t9b)

    # 清理测试文件
    for f in ["_test_file.py", "_test_version.py", "_test_edit.py", "_test_dup.py"]:
        try: (WORKSPACE / f).unlink()
        except: pass

    print("\n" + "=" * 55)
    if errors:
        print(f"❌ {len(errors)} 个测试失败:")
        for name, err in errors:
            print(f"   {name}: {err}")
    else:
        print("✅ 所有测试通过")
    print("=" * 55)
