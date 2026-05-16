"""
tools.py — Agent 工具集

核心变化：
  1. propose_and_run 新增 val_mix_ratio 参数
  2. 推理后正确写入 inference_time 到 task1_time.csv
  3. 解析 predict 日志时补充推理时间
"""

import os, re, csv, json, time, random, sqlite3, subprocess
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/mimer/NOBACKUP/groups/phy_geo/PDE_Agent")
AGENT_DIR   = PROJECT_DIR / "repos" / "agent"
LOG_DIR     = PROJECT_DIR / "logs"
CKPT_DIR    = PROJECT_DIR / "checkpoints" / "task1_trained"
SUBMISSION  = PROJECT_DIR / "repos" / "submission"
DB_PATH     = AGENT_DIR / "registry.db"
VENV_ACTIVATE = str(PROJECT_DIR / "env" / "pde_venv" / "bin" / "activate")

A_CLASS_PARAMS = {
    "branch_type", "branch_depth", "branch_width",
    "trunk_depth", "trunk_width", "latent_dim",
    "activation", "fourier_features",
}


# ============================================================
# Registry DB
# ============================================================
def init_registry():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        exp_id       TEXT PRIMARY KEY,
        parent_id    TEXT,
        task         TEXT DEFAULT 'task1',
        status       TEXT DEFAULT 'pending',
        init_from    TEXT,
        branch_type      TEXT DEFAULT 'mlp',
        branch_depth     INTEGER DEFAULT 4,
        branch_width     INTEGER DEFAULT 256,
        trunk_depth      INTEGER DEFAULT 4,
        trunk_width      INTEGER DEFAULT 256,
        latent_dim       INTEGER DEFAULT 128,
        activation       TEXT DEFAULT 'tanh',
        fourier_features INTEGER DEFAULT 64,
        lr           REAL DEFAULT 0.001,
        weight_decay REAL DEFAULT 1e-4,
        epochs       INTEGER DEFAULT 50,
        batch_size   INTEGER DEFAULT 32,
        n_query      INTEGER DEFAULT 2048,
        val_mix_ratio REAL DEFAULT 1.0,
        data_weight    REAL DEFAULT 1.0,
        physics_weight REAL DEFAULT 0.0,
        seg1_score   REAL,
        seg2_score   REAL,
        seg3_score   REAL,
        total_score  REAL,
        val_loss     REAL,
        train_time   REAL,
        infer_time   REAL,
        hypothesis   TEXT,
        rationale    TEXT,
        conclusion   TEXT,
        inflection_step  INTEGER,
        extrap_ratio     REAL,
        created_at   TEXT
    )""")
    conn.commit()
    conn.close()


def get_experiment(exp_id):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row  = conn.execute("SELECT * FROM experiments WHERE exp_id=?",
                        (exp_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}


def insert_experiment(exp):
    conn = sqlite3.connect(str(DB_PATH))
    cols = ", ".join(exp.keys())
    qs   = ", ".join("?" for _ in exp)
    conn.execute(f"INSERT OR REPLACE INTO experiments ({cols}) VALUES ({qs})",
                 list(exp.values()))
    conn.commit()
    conn.close()


def update_experiment(exp_id, fields):
    conn = sqlite3.connect(str(DB_PATH))
    cols = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE experiments SET {cols} WHERE exp_id=?",
                 list(fields.values()) + [exp_id])
    conn.commit()
    conn.close()


def get_leaderboard(task='task1', limit=20):
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT exp_id, parent_id, status, init_from,
               branch_type, branch_depth, branch_width,
               trunk_depth, trunk_width, latent_dim,
               lr, epochs, val_mix_ratio,
               seg1_score, seg2_score, seg3_score, total_score,
               val_loss, train_time, infer_time,
               inflection_step, extrap_ratio,
               hypothesis, conclusion
        FROM experiments
        WHERE task=? AND status IN ('done','trained')
        ORDER BY total_score DESC NULLS LAST
        LIMIT ?
    """, (task, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_structures_tested(task='task1'):
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("""
        SELECT DISTINCT branch_type, branch_depth, branch_width,
                        trunk_depth, trunk_width, latent_dim
        FROM experiments WHERE task=? AND status='done'
    """, (task,)).fetchall()
    conn.close()
    return rows


# ============================================================
# 内部命令执行
# ============================================================
def _run_cmd(cmd, timeout=7200):
    try:
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        full_cmd = f"source {VENV_ACTIVATE} && {cmd}"
        proc = subprocess.run(
            full_cmd, shell=True, capture_output=True,
            text=True, timeout=timeout,
            cwd=str(PROJECT_DIR), env=env, executable="/bin/bash")
        return {
            "success": proc.returncode == 0,
            "result":  (proc.stdout + proc.stderr).strip(),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "result": f"超时（>{timeout}s）"}
    except Exception as e:
        return {"success": False, "result": str(e)}


# ============================================================
# 核心工具：propose_and_run
# ============================================================
def propose_and_run(exp_id, parent_id, changes, rationale,
                    hypothesis="",
                    gpu_ids=os.environ.get("TRAIN_GPUS", "0,1,2,3")):
    init_registry()

    # 继承配置
    all_params = A_CLASS_PARAMS | {
        "lr", "weight_decay", "epochs", "batch_size",
        "n_query", "val_mix_ratio", "data_weight", "physics_weight"
    }
    if parent_id and parent_id != "scratch":
        parent = get_experiment(parent_id)
        if not parent:
            return {"success": False, "result": f"parent {parent_id} 不存在"}
        cfg = {k: parent[k] for k in all_params if k in parent}
    else:
        cfg = {
            "branch_type": "mlp", "branch_depth": 4, "branch_width": 256,
            "trunk_depth":  4,    "trunk_width":  256, "latent_dim": 128,
            "activation": "tanh", "fourier_features": 64,
            "lr": 1e-3, "weight_decay": 1e-4, "epochs": 50,
            "batch_size": 32, "n_query": 2048, "val_mix_ratio": 1.0,
            "data_weight": 1.0, "physics_weight": 0.0,
        }
    cfg.update(changes)

    a_changed = bool(A_CLASS_PARAMS & set(changes.keys()))
    init_from = "scratch" if (a_changed or not parent_id
                              or parent_id == "scratch") else "resume"
    if init_from == "resume":
        if not (CKPT_DIR / f"{parent_id}_candidate.pt").exists():
            init_from = "scratch"

    print(f"[tools] {exp_id} parent={parent_id} init_from={init_from} "
          f"changes={list(changes.keys())}")

    insert_experiment({
        "exp_id": exp_id, "parent_id": parent_id, "task": "task1",
        "status": "running", "init_from": init_from,
        "hypothesis": hypothesis, "rationale": rationale,
        "created_at": datetime.utcnow().isoformat(),
        **{k: cfg.get(k) for k in all_params},
    })

    # 训练命令
    n_gpus = len(gpu_ids.split(","))
    if n_gpus > 1:
        port     = random.randint(29500, 30500)
        launcher = (f"torchrun --nproc_per_node={n_gpus} "
                    f"--master_addr=localhost --master_port={port} "
                    f"--rdzv_backend=c10d --rdzv_endpoint=localhost:{port}")
    else:
        launcher = "python"

    nccl = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 TORCH_NCCL_BLOCKING_WAIT=1 "
    resume_flag = (f"--resume --parent_id {parent_id}"
                   if init_from == "resume" else "")

    train_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_ids} {nccl}"
        f"{launcher} repos/agent/train.py "
        f"--exp_id {exp_id} "
        f"--branch_type {cfg['branch_type']} "
        f"--branch_depth {cfg['branch_depth']} "
        f"--branch_width {cfg['branch_width']} "
        f"--trunk_depth {cfg['trunk_depth']} "
        f"--trunk_width {cfg['trunk_width']} "
        f"--latent_dim {cfg['latent_dim']} "
        f"--activation {cfg['activation']} "
        f"--fourier_features {cfg['fourier_features']} "
        f"--lr {cfg['lr']} "
        f"--weight_decay {cfg['weight_decay']} "
        f"--epochs {cfg['epochs']} "
        f"--batch_size {cfg['batch_size']} "
        f"--n_query {cfg['n_query']} "
        f"--val_mix_ratio {cfg['val_mix_ratio']} "
        f"{resume_flag}"
    )
    print(f"[tools] {train_cmd[:120]}...")
    timeout = int(cfg['epochs']) * 120 + 600
    train_r = _run_cmd(train_cmd, timeout=timeout)

    if not train_r["success"]:
        update_experiment(exp_id, {"status": "failed"})
        return {"success": False,
                "result": f"训练失败: {train_r['result'][-300:]}",
                "exp_id": exp_id}

    ckpt_path = CKPT_DIR / f"{exp_id}_candidate.pt"
    if not ckpt_path.exists():
        update_experiment(exp_id, {"status": "failed"})
        return {"success": False,
                "result": "训练完成但无 candidate（val_loss 全程未改善）",
                "exp_id": exp_id}

    # 推理评测
    infer_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_ids.split(',')[0]} "
        f"python repos/agent/predict.py "
        f"--ckpt {ckpt_path} --val_only"
    )
    print(f"[tools] 推理评测...")
    t0       = time.time()
    infer_r  = _run_cmd(infer_cmd, timeout=300)
    infer_t  = time.time() - t0

    metrics = _parse_predict_log()

    # 写入 inference_time 到 task1_time.csv
    time_csv = SUBMISSION / "task1_time.csv"
    train_time_saved = 0.0
    if time_csv.exists():
        with open(time_csv, 'r') as f:
            for row in csv.DictReader(f):
                train_time_saved = float(row.get('train_time', 0))
    with open(time_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['train_time', 'inference_time'])
        w.writerow([train_time_saved, infer_t])

    update_experiment(exp_id, {
        "status":          "done",
        "seg1_score":      metrics.get("seg1_score",       0.0),
        "seg2_score":      metrics.get("seg2_score",       0.0),
        "seg3_score":      metrics.get("seg3_score",       0.0),
        "total_score":     metrics.get("total_score",      0.0),
        "inflection_step": int(metrics.get("inflection_step", -1)),
        "extrap_ratio":    metrics.get("extrapolation_ratio", -1.0),
        "infer_time":      infer_t,
    })

    return {
        "success":     True,
        "exp_id":      exp_id,
        "init_from":   init_from,
        "total_score": metrics.get("total_score", 0.0),
        "seg1_score":  metrics.get("seg1_score",  0.0),
        "seg2_score":  metrics.get("seg2_score",  0.0),
        "seg3_score":  metrics.get("seg3_score",  0.0),
        "infer_time":  infer_t,
        "metrics":     metrics,
    }


def _parse_predict_log():
    log_path = LOG_DIR / "task1_predict.log"
    if not log_path.exists():
        return {}
    content = log_path.read_text(encoding="utf-8")
    patterns = {
        "seg1_score":          r"Seg1.*Score=([\d.]+)",
        "seg2_score":          r"Seg2.*Score=([\d.]+)",
        "seg3_score":          r"-> \w+=([\d.]+)",
        "total_score":         r"预测得分:\s*([\d.]+)",
        "t2s_mean_error":      r"训练覆盖区误差=([\d.]+)",
        "t10s_mean_error":     r"外推区误差=([\d.]+)",
        "extrapolation_ratio": r"外推/覆盖比=([\d.]+)x",
        "inflection_step":     r"拐点步=(\d+)",
    }
    result = {}
    for key, pat in patterns.items():
        m = re.findall(pat, content)
        if m:
            result[key] = float(m[-1])
    return result


# ============================================================
# 辅助
# ============================================================
def read_training_log(max_lines=80):
    log_path = LOG_DIR / "task1_train.log"
    if not log_path.exists():
        return {"success": False, "result": "训练日志不存在"}
    lines  = log_path.read_text(encoding="utf-8").splitlines()
    recent = lines[-max_lines:] if len(lines) > max_lines else lines
    return {"success": True, "result": "\n".join(recent)}


def append_research_log(content, log_name="task1_logs.log"):
    log_path = SUBMISSION / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")
        return {"success": True}
    except Exception as e:
        return {"success": False, "result": str(e)}