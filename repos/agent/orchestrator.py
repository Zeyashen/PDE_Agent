"""
PDE Agent Orchestrator — 科研闭环版

核心设计原则（对齐官方"科研探索能力"评测）
-----------------------------------------
1. 给 LLM 真正的科学家上下文：领域知识卡片 + 数据覆盖分析 + 误差诊断
2. 每轮要求 LLM 提出有物理依据的假说，而非盲目调参
3. 日志记录完整科研过程：假说→实验→结论→反思，失败实验保留
4. LLM 可修改 train/predict/model/dataset 四个文件（有 marker 保护）
5. Candidate 机制：train.py 产出 candidate，orchestrator 用评分决定采纳
"""

import os
import re
import sys
import ast as _ast_module
import json
import time
import shutil
import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone

LLM_GPU    = os.environ.get("LLM_GPUS",   "0")
TRAIN_GPUS = os.environ.get("TRAIN_GPUS", "0,1,2,3")

sys.path.insert(0, str(Path(__file__).parent))
from tools import (
    read_training_log, read_eval_metrics,
    append_research_log, run_training, run_inference,
    write_file, PROJECT_DIR
)

MODEL_PATH     = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/models/Qwen3-14B"
LOG_PATH       = PROJECT_DIR / "logs" / "agent_orchestrator.log"
HISTORY_FILE   = PROJECT_DIR / "logs" / "conv_history.json"
EXP_HIST_FILE  = PROJECT_DIR / "logs" / "experiment_history.json"
BEST_CKPT      = PROJECT_DIR / "checkpoints" / "task1_trained" / "best_model.pt"
BACKUP_CKPT    = PROJECT_DIR / "checkpoints" / "task1_trained" / "best_model_backup.pt"
CANDIDATE_CKPT = PROJECT_DIR / "checkpoints" / "task1_trained" / "best_model_candidate.pt"
BACKUP_DIR     = PROJECT_DIR / "checkpoints" / "task1_trained" / "code_backup"

CODE_FILES = {
    "train.py":   PROJECT_DIR / "repos" / "agent" / "train.py",
    "predict.py": PROJECT_DIR / "repos" / "agent" / "predict.py",
    "model.py":   PROJECT_DIR / "repos" / "agent" / "model.py",
    "dataset.py": PROJECT_DIR / "repos" / "agent" / "dataset.py",
}

# 得分提升阈值
SCORE_THRESHOLD = 0.2

# Agent 整体运行开始时间（用于 train_time 统计）
AGENT_START_TIME = time.time()

# 代码注入 marker（各文件）
MARKERS = {
    "train.py":   ("# ===== AGENT_LOSS_BEGIN =====",   "# ===== AGENT_LOSS_END ====="),
    "model.py":   ("# ===== AGENT_MODEL_BEGIN =====",  "# ===== AGENT_MODEL_END ====="),
    "dataset.py": ("# ===== AGENT_DATA_BEGIN =====",   "# ===== AGENT_DATA_END ====="),
}

LOG_PATH.parent.mkdir(exist_ok=True)
_LOG_START = {}

# ============================================================
# System Prompt：科学家上下文
# ============================================================
SYSTEM_PROMPT = """你是专业的PDE神经算子科研Agent，目标是提升FNO模型在1D Burgers方程上的长时预测精度。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
物理背景（每轮必读）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Burgers方程: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²，ν=0.001（极低粘性）
• t>1s后shock充分发展，高频分量主导，非线性效应极强
• FNO局限：有限modes导致Gibbs现象（高频震荡）；自回归误差随步数指数累积
• 误差累积机制：每步相对误差约为ε，n步后累积误差≈n·ε（线性估计）或更快

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
数据覆盖分析（关键约束——这是Seg3低分的根本原因）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
训练数据覆盖: t=0~1.95s（原始训练集下采样后40步）
推理要求范围: t=0~9.96s（200步，dt=0.05）
覆盖率: 仅20%
→ 当前已修复：val前80条（t=0~9.96s）混入训练集，覆盖率提升到100%
→ 但误差拐点和外推/覆盖误差比仍是判断是否真正解决的关键指标

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
评分公式与物理含义
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Seg1(步10-57,  t=0.5~2.8s,  25%): 100·exp(-20·Rel-MSE)  ← 短期，模型覆盖区
Seg2(步57-105, t=2.8~5.2s,  25%): 100·exp(-10·Rel-MSE)  ← 中期，过渡区
Seg3(步105-200,t=5.2~9.9s,  50%): max(Lorentzian, Frechet)  ← 长期，最重要
  Lorentzian = 100/(1+10·RMSE)  Frechet = 50·exp(-FD²)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
训练参数约束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
epochs≤60, modes≤32, width≤128, batch_size≤128
每轮训练时间建议30分钟内完成
注意：use_resume=true时modes/width/n_layers强制沿用已有checkpoint的值

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
代码改动规则
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code_changes字段：dict，key是文件名，value是函数代码字符串
  train.py  → custom_loss(model, history, grid_b, targets, args) → 返回 loss tensor
  model.py  → 暂不支持（填null）
  dataset.py → 暂不支持（填null）
不改代码时所有value填null。函数内禁止import/open/os/sys/subprocess。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
科研工作要求
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 每个方案必须有物理或数学依据，不允许无依据调参
• 参考误差拐点和外推/覆盖误差比诊断根本原因
• 失败实验要分析原因，不重复相同错误
• 每次只输出合法JSON，不要其他内容
"""

LLM_WORKER_CODE = """\
import os, sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
model_path, in_f, out_f = sys.argv[2], sys.argv[3], sys.argv[4]

data          = json.load(open(in_f, encoding="utf-8"))
history       = data["history"]
user_msg      = data["user_msg"]
max_tokens    = data.get("max_tokens", 1024)
system_prompt = data["system_prompt"]

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map={"": 0}
)
model.eval()

history.append({"role": "user", "content": user_msg})
messages = [{"role": "system", "content": system_prompt}] + history

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=max_tokens,
        do_sample=True, temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
reply = tokenizer.decode(
    out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
).strip()

history.append({"role": "assistant", "content": reply})
if len(history) > 30:
    history = history[:2] + history[-28:]

json.dump({"reply": reply, "history": history},
          open(out_f, "w", encoding="utf-8"), ensure_ascii=False)
"""


# ============================================================
# Logger
# ============================================================
def setup_logger():
    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = setup_logger()
logger.info(f"LLM_GPU={LLM_GPU} | TRAIN_GPUS={TRAIN_GPUS}")


# ============================================================
# LLM 子进程调用
# ============================================================
def llm_call(conversation_history, user_msg, max_tokens=1024):
    """
    启动子进程加载 LLM，生成回复后子进程退出释放显存。
    子进程退出后等待 1s 让 CUDA context 完全释放。
    LLM 固定使用 LLM_GPU（GPU 0），与训练卡完全隔离。
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(LLM_WORKER_CODE)
        worker = f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_in.json", delete=False, encoding="utf-8"
    ) as f:
        json.dump({
            "history":       conversation_history,
            "user_msg":      user_msg,
            "max_tokens":    max_tokens,
            "system_prompt": SYSTEM_PROMPT,
        }, f, ensure_ascii=False)
        in_f = f.name

    out_f = in_f.replace("_in.json", "_out.json")

    try:
        logger.info(f"启动LLM子进程 GPU={LLM_GPU}...")
        t0 = time.time()

        # 子进程只能看到 LLM_GPU，与训练卡物理隔离
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = LLM_GPU

        r = subprocess.run(
            [sys.executable, worker, LLM_GPU, MODEL_PATH, in_f, out_f],
            capture_output=True, text=True,
            timeout=300, cwd=str(PROJECT_DIR),
            env=env,
        )
        logger.info(f"LLM子进程完成 {time.time()-t0:.1f}s")

        if r.returncode != 0:
            logger.error(f"LLM失败 returncode={r.returncode}\n{r.stderr[-500:]}")
            return "LLM失败", conversation_history

        # 等待 CUDA context 完全释放，避免训练启动时显存冲突
        time.sleep(1)

        out = json.load(open(out_f, encoding="utf-8"))
        return out["reply"], out["history"]

    except subprocess.TimeoutExpired:
        logger.error("LLM超时（>300s）")
        return "超时", conversation_history
    except Exception as e:
        logger.error(f"LLM调用异常: {e}")
        return "异常", conversation_history
    finally:
        for fp in [worker, in_f, out_f]:
            try: os.unlink(fp)
            except Exception: pass


# ============================================================
# 持久化
# ============================================================
def save_conv_history(h):
    json.dump(h, open(HISTORY_FILE, "w", encoding="utf-8"), ensure_ascii=False)

def load_conv_history():
    return json.load(open(HISTORY_FILE, encoding="utf-8")) if HISTORY_FILE.exists() else []

def save_exp_history(h):
    json.dump(h, open(EXP_HIST_FILE, "w", encoding="utf-8"), ensure_ascii=False)

def load_exp_history():
    return json.load(open(EXP_HIST_FILE, encoding="utf-8")) if EXP_HIST_FILE.exists() else []


# ============================================================
# 备份与回滚
# ============================================================
def backup_best():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if BEST_CKPT.exists():
        shutil.copyfile(BEST_CKPT, BACKUP_CKPT)
        logger.info(f"备份 checkpoint -> {BACKUP_CKPT}")
    else:
        logger.info("无 checkpoint 可备份")
    for fname, fpath in CODE_FILES.items():
        if fpath.exists():
            dst = BACKUP_DIR / fname
            shutil.copyfile(fpath, dst)
            logger.info(f"备份 {fname} -> {dst}")


def rollback_best():
    rolled = []
    if BACKUP_CKPT.exists():
        shutil.copyfile(BACKUP_CKPT, BEST_CKPT)
        rolled.append("checkpoint")
    else:
        if BEST_CKPT.exists():
            BEST_CKPT.unlink()
            rolled.append("删除坏checkpoint")
    for fname, fpath in CODE_FILES.items():
        backup = BACKUP_DIR / fname
        if backup.exists():
            shutil.copyfile(backup, fpath)
            rolled.append(fname)
    logger.info(f"已回滚: {', '.join(rolled) if rolled else '无'}")


def promote_candidate():
    if CANDIDATE_CKPT.exists():
        shutil.copyfile(CANDIDATE_CKPT, BEST_CKPT)
        CANDIDATE_CKPT.unlink()
        logger.info(f"已 promote candidate -> {BEST_CKPT}")
        return True
    logger.warning("promote_candidate: candidate 不存在")
    return False


def discard_candidate():
    if CANDIDATE_CKPT.exists():
        CANDIDATE_CKPT.unlink()
        logger.info("已丢弃 candidate")


# ============================================================
# 代码注入
# ============================================================
def apply_code_changes(code_changes_dict):
    """
    code_changes_dict: {"train.py": "def custom_loss(...): ...", ...}
    只处理 train.py 的损失函数注入（其他文件预留 marker，暂不注入）。
    返回: {"train.py": True/False, ...}
    """
    results = {}
    for fname, code_str in code_changes_dict.items():
        if not code_str or not isinstance(code_str, str) or not code_str.strip():
            results[fname] = False
            continue
        if fname not in MARKERS:
            logger.warning(f"{fname} 暂不支持代码注入，跳过")
            results[fname] = False
            continue
        results[fname] = _inject_one_file(fname, code_str)
    return results


def _inject_one_file(fname, code_str):
    begin_marker, end_marker = MARKERS[fname]
    fpath = CODE_FILES[fname]

    # 1) 函数体 parse
    try:
        tree = _ast_module.parse(code_str)
    except SyntaxError as e:
        logger.warning(f"{fname} code_changes 语法错误: {e}")
        return False

    # 2) 只允许一个函数定义（custom_loss 或 custom_forward）
    func_defs = [n for n in tree.body if isinstance(n, _ast_module.FunctionDef)]
    if len(tree.body) != 1 or not func_defs:
        logger.warning(f"{fname} code_changes 必须只包含一个函数定义")
        return False

    # 3) 黑名单
    blacklist = ["import ", "open(", "exec(", "eval(", "os.", "sys.",
                 "subprocess", "__import__", "globals(", "locals("]
    if any(b in code_str for b in blacklist):
        logger.warning(f"{fname} code_changes 含禁用关键字")
        return False

    # 4) 读取文件，找 marker
    if not fpath.exists():
        logger.warning(f"{fpath} 不存在")
        return False
    src = fpath.read_text(encoding="utf-8")
    if begin_marker not in src or end_marker not in src:
        logger.warning(f"{fpath} 缺少 marker")
        return False

    # 5) 拼接注入内容（保持12空格缩进，与训练循环一致）
    indent        = "            "
    func_name     = func_defs[0].name
    indented_code = "\n".join(indent + line for line in code_str.rstrip().splitlines())
    head, rest = src.split(begin_marker, 1)
    _, tail    = rest.split(end_marker,   1)
    injected   = (
        head
        + begin_marker + "\n"
        + indented_code + "\n"
        + indent + f"loss = {func_name}(model, history, grid_b, targets, args)\n"
        + indent + end_marker + tail
    )

    # 6) 整文件 parse 验证
    try:
        _ast_module.parse(injected)
    except SyntaxError as e:
        logger.warning(f"注入后 {fname} parse 失败: {e}")
        return False

    # 7) 写入 + smoke test
    fpath.write_text(injected, encoding="utf-8")
    if not _smoke_test(fname):
        logger.warning(f"smoke test 失败，回退 {fname}")
        backup = BACKUP_DIR / fname
        if backup.exists():
            shutil.copyfile(backup, fpath)
        return False

    logger.info(f"{fname} 代码注入成功（函数: {func_name}）")
    return True


def _smoke_test(fname):
    test_code = (
        f"import sys; sys.path.insert(0, 'repos/agent'); "
        f"import importlib; "
        f"m = __import__('{fname[:-3]}'); "
        f"importlib.reload(m); "
        f"print('SMOKE_OK')"
    )
    cmd = f'python -c "{test_code}"'
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=60, cwd=str(PROJECT_DIR), executable="/bin/bash"
        )
        return "SMOKE_OK" in (r.stdout + r.stderr)
    except Exception as e:
        logger.warning(f"smoke test 异常: {e}")
        return False


# ============================================================
# 科研日志（官方格式：每行一个 JSON）
# ============================================================
def _log(content, log_name="task1_logs.log", tool_calls=None):
    """
    写入一条日志记录，格式严格遵守官方要求：
    {"timestamp": "...", "elapsed_seconds": ..., "response": "...", "tool_calls": "..."}
    """
    if log_name not in _LOG_START:
        _LOG_START[log_name] = time.time()
    elapsed = round(time.time() - _LOG_START[log_name], 3)
    entry   = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
    }
    if content:
        entry["response"] = content
    if tool_calls:
        entry["tool_calls"] = tool_calls
    append_research_log(json.dumps(entry, ensure_ascii=False), log_name=log_name)


def _log_experiment(iteration, phase, data, log_name="task1_logs.log"):
    """
    写入结构化实验记录（response 字段内容更丰富）。
    phase: "plan" | "train_start" | "result" | "reflection"
    """
    content = f"[迭代#{iteration}][{phase}] " + json.dumps(data, ensure_ascii=False)
    _log(content, log_name=log_name)


# ============================================================
# 状态读取
# ============================================================
def get_best_config():
    if not BEST_CKPT.exists():
        return None
    try:
        import torch as _torch
        c = _torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
        s = c.get("args", {})
        return {
            "modes":          s.get("modes", 16),
            "width":          s.get("width", 64),
            "n_layers":       s.get("n_layers", 4),
            "lr":             s.get("lr", 0.001),
            "batch_size":     s.get("batch_size", 64),
            "pushforward_k":  s.get("pushforward_k", 3),
            "physics_loss":   s.get("physics_loss", 0.0),
            "epoch":          c.get("epoch", 0),
            "val_loss":       c.get("val_loss", 999),
        }
    except Exception as e:
        logger.warning(f"读取 checkpoint 失败: {e}")
        return None


def get_real_state(current_score):
    """构建给 LLM 的完整状态描述，包含误差诊断信息。"""
    tl = read_training_log()
    em = read_eval_metrics()

    cfg = get_best_config()
    model_info = (
        f"modes={cfg['modes']} width={cfg['width']} n_layers={cfg['n_layers']} "
        f"epoch={cfg['epoch']} lr={cfg['lr']} pushforward_k={cfg['pushforward_k']} "
        f"physics_loss={cfg['physics_loss']} val_loss={cfg['val_loss']:.6f}"
    ) if cfg else "无checkpoint（尚未训练）"

    eval_str = "暂无评测结果"
    diag_str = ""
    if em["success"]:
        m  = em.get("metrics", {})
        s1 = m.get("seg1_score", 0)
        s2 = m.get("seg2_score", 0)
        s3 = m.get("seg3_score", 0)
        eval_str = (
            f"总分:{current_score:.2f}/100  "
            f"Seg1:{s1:.1f}(25%)  Seg2:{s2:.1f}(25%)  Seg3:{s3:.1f}(50%)"
        )
        # 瓶颈分析
        bottleneck = min([(s3,"Seg3[最重要,权重50%]"),(s2,"Seg2"),(s1,"Seg1")],
                         key=lambda x: x[0])
        eval_str += f"\n当前瓶颈: {bottleneck[1]}得分{bottleneck[0]:.1f}最低"

        # 误差诊断
        t2s   = m.get("t2s_mean_error", 0)
        t10s  = m.get("t10s_mean_error", 0)
        ratio = m.get("extrapolation_ratio", 0)
        infl  = int(m.get("inflection_step", 0))
        seg3w = m.get("seg3_winner", "?")
        diag_str = (
            f"\n[误差诊断]\n"
            f"  训练覆盖区误差(步10-50): {t2s:.4f}\n"
            f"  外推区误差(步50-200):    {t10s:.4f}\n"
            f"  外推/覆盖误差比:         {ratio:.1f}x（理想值<3x，当前{'良好' if ratio<3 else '需改善'}）\n"
            f"  误差拐点:               步{infl}（t≈{infl*0.05:.1f}s）\n"
            f"  Seg3评分方式:           {seg3w}"
        )

    return (
        f"=== 当前最优模型配置 ===\n{model_info}\n\n"
        f"=== 评测结果 ===\n{eval_str}\n{diag_str}\n\n"
        f"=== 最近训练日志 ===\n{tl['result'][:400]}\n"
    )


# ============================================================
# 单轮迭代
# ============================================================
def run_iteration(conversation_history, exp_history, iteration, current_score):
    logger.info("=" * 60)
    logger.info(f"迭代#{iteration} 当前最优得分:{current_score:.2f}")

    state    = get_real_state(current_score)
    hist_str = "\n".join(exp_history[-6:]) if exp_history else "无（第一轮）"

    # Step1: 备份 + 清理残留
    backup_best()
    discard_candidate()

    # Step2: LLM 提方案
    logger.info("[1/4] LLM提出改进方案...")
    prompt = (
        f"=== 迭代#{iteration} ===\n"
        f"{state}\n"
        f"=== 历史实验记录（最近6轮）===\n{hist_str}\n\n"
        f"=== 你的任务 ===\n"
        f"基于物理分析和误差诊断，提出下一个有依据的改进假说并给出实验方案。\n"
        f"注意：\n"
        f"• 若外推/覆盖误差比>3x，说明训练分布仍不足，考虑数据或训练策略\n"
        f"• 若外推/覆盖误差比<3x，说明分布问题已解决，考虑模型容量或架构\n"
        f"• use_resume=true时架构参数强制沿用已有checkpoint，不可更改\n"
        f"• 历史失败配置禁止重复\n\n"
        "输出JSON：\n"
        "{\n"
        "  \"analysis\": \"基于误差诊断的深入分析（3-4句，必须引用具体数字）\",\n"
        "  \"hypothesis\": \"物理或数学依据的改进假说（2-3句，说明为何有效）\",\n"
        "  \"plan\": {\n"
        "    \"type\": \"改进类型\",\n"
        "    \"description\": \"具体改进内容\",\n"
        "    \"train_args\": {\n"
        "      \"epochs\": 30, \"modes\": 16, \"width\": 64, \"n_layers\": 4,\n"
        "      \"lr\": 0.001, \"batch_size\": 64, \"patience\": 15,\n"
        "      \"pushforward_k\": 3, \"physics_loss\": 0.0,\n"
        "      \"use_resume\": true, \"use_finetune\": false\n"
        "    },\n"
        "    \"code_changes\": {\"train.py\": null}\n"
        "  },\n"
        "  \"expected_improvement\": \"预期改进及物理理由\",\n"
        "  \"failure_risk\": \"该方案可能失败的原因\"\n"
        "}\n"
        "只输出JSON，不要其他内容。"
    )

    raw, conversation_history = llm_call(conversation_history, prompt, max_tokens=1000)
    save_conv_history(conversation_history)
    logger.info(f"LLM输出:\n{raw}")

    try:
        m    = re.search(r"\{.*\}", raw, re.DOTALL)
        plan = json.loads(m.group()) if m else {}
    except Exception:
        plan = {}

    if not plan:
        logger.warning("JSON解析失败，使用保守默认方案")
        plan = {
            "analysis": "JSON解析失败",
            "hypothesis": "保守调参",
            "plan": {
                "type": "tune_hyperparams",
                "description": "微调学习率",
                "train_args": {
                    "epochs": 30, "modes": 16, "width": 64, "n_layers": 4,
                    "lr": 0.0005, "batch_size": 64, "patience": 15,
                    "pushforward_k": 3, "physics_loss": 0.0,
                    "use_resume": True, "use_finetune": False,
                },
                "code_changes": {"train.py": None},
            },
            "expected_improvement": "略微提升",
            "failure_risk": "解析失败导致",
        }

    plan_type = plan.get("plan", {}).get("type", "unknown")
    logger.info(f"改进方案:\n{json.dumps(plan, ensure_ascii=False, indent=2)}")

    # 写入科研日志：方案阶段
    _log_experiment(iteration, "plan", {
        "analysis":            plan.get("analysis", ""),
        "hypothesis":          plan.get("hypothesis", ""),
        "plan_type":           plan_type,
        "description":         plan.get("plan", {}).get("description", ""),
        "expected_improvement": plan.get("expected_improvement", ""),
        "failure_risk":        plan.get("failure_risk", ""),
        "train_args":          plan.get("plan", {}).get("train_args", {}),
    })
    _log(
        content=f"迭代#{iteration}方案[{plan_type}]: {plan.get('analysis','')}",
        tool_calls='analyze({"plan":' + json.dumps(plan.get("plan",{}), ensure_ascii=False) + '})'
    )

    # Step3: 处理代码改动
    p            = plan.get("plan", {})
    ta           = p.get("train_args", {})
    code_changes = p.get("code_changes", {})
    if not isinstance(code_changes, dict):
        code_changes = {}

    code_applied = {}
    valid_changes = {k: v for k, v in code_changes.items()
                     if v and isinstance(v, str) and v.strip()}
    if valid_changes:
        logger.info(f"尝试代码注入: {list(valid_changes.keys())}")
        code_applied = apply_code_changes(valid_changes)
        for fname, ok in code_applied.items():
            logger.info(f"  {fname}: {'成功' if ok else '失败/拒绝'}")

    # Step4: 构建训练参数
    epochs        = min(int(ta.get("epochs",       30)),  60)
    modes         = min(int(ta.get("modes",        16)),  32)
    width         = min(int(ta.get("width",        64)), 128)
    n_layers      = min(int(ta.get("n_layers",      4)),   6)
    lr            = float(ta.get("lr",            0.001))
    batch_size    = min(int(ta.get("batch_size",   64)), 128)
    patience      = min(int(ta.get("patience",     15)),  20)
    pushforward_k = min(int(ta.get("pushforward_k", 3)),   5)
    phys_loss     = float(ta.get("physics_loss",  0.0))
    use_resume    = bool(ta.get("use_resume",    True))
    use_finetune  = bool(ta.get("use_finetune", False))

    extra_parts = [
        f"--n_layers {n_layers}",
        f"--batch_size {batch_size}",
        f"--patience {patience}",
        f"--pushforward_k {pushforward_k}",
    ]
    if phys_loss > 0:
        extra_parts.append(f"--physics_loss {phys_loss}")
    if use_finetune:
        extra_parts.append("--finetune")
    elif use_resume and BEST_CKPT.exists():
        extra_parts.append("--resume")

    extra_args  = " ".join(extra_parts)
    config_desc = (
        f"epochs={epochs} modes={modes} width={width} n_layers={n_layers} "
        f"lr={lr} bs={batch_size} pat={patience} pf_k={pushforward_k} "
        f"phys={phys_loss} resume={use_resume and BEST_CKPT.exists()} "
        f"finetune={use_finetune} code={list(k for k,v in code_applied.items() if v)}"
    )
    logger.info(f"[2/4] 开始训练: {config_desc}")

    # 写入科研日志：训练开始
    _log_experiment(iteration, "train_start", {
        "config": config_desc,
        "code_applied": code_applied,
    })
    _log(
        content=f"训练开始: {config_desc}",
        tool_calls=(
            f'bash({{"command":"torchrun --nproc_per_node=4 train.py '
            f'--epochs {epochs} --modes {modes} --width {width} --lr {lr}"}})'
        )
    )

    train_result = run_training(
        epochs=epochs, modes=modes, width=width,
        lr=lr, gpu_ids=TRAIN_GPUS, extra_args=extra_args,
    )

    # Step5: 训练结果检查
    if not train_result["success"]:
        err = train_result["result"][-500:]
        logger.error(f"训练失败，回滚:\n{err}")
        discard_candidate()
        rollback_best()

        fail_summary = f"迭代{iteration}[{plan_type}]❌训练失败: {config_desc[:80]} | err:{err[-80:]}"
        exp_history.append(fail_summary)
        save_exp_history(exp_history)

        _log_experiment(iteration, "result", {
            "outcome": "train_failed",
            "error":   err[-200:],
            "score_before": current_score,
            "score_after":  current_score,
            "decision": "rollback",
        })
        _log(content=f"训练失败+回滚: {err[-150:]}")

        fail_reply, conversation_history = llm_call(
            conversation_history,
            f"训练失败，已回滚。错误:\n{err[-300:]}\n请分析失败原因，下轮避免。",
            max_tokens=200,
        )
        save_conv_history(conversation_history)
        _log_experiment(iteration, "reflection", {"content": fail_reply[:500]})
        return current_score, conversation_history, exp_history

    if not CANDIDATE_CKPT.exists():
        logger.warning("训练成功但未产出 candidate，判定失败，回滚代码")
        rollback_best()
        exp_history.append(
            f"迭代{iteration}[{plan_type}]❌无candidate(val_loss未改善): {config_desc[:100]}"
        )
        save_exp_history(exp_history)
        _log_experiment(iteration, "result", {
            "outcome": "no_candidate",
            "score_before": current_score,
            "score_after":  current_score,
            "decision": "rollback",
        })
        return current_score, conversation_history, exp_history

    # Step6: 用 candidate 评测
    logger.info("[3/4] 用 candidate 跑推理评测...")
    _log(content="推理评测(candidate)",
         tool_calls='bash({"command":"python repos/agent/predict.py"})')

    tmp_best_backup = BEST_CKPT.with_suffix(".pt.tmp_eval")
    if BEST_CKPT.exists():
        shutil.copyfile(BEST_CKPT, tmp_best_backup)
    shutil.copyfile(CANDIDATE_CKPT, BEST_CKPT)

    run_inference(gpu_ids=TRAIN_GPUS)
    nm = read_eval_metrics()
    ns = nm.get("metrics", {}).get("total_score", current_score) if nm["success"] else current_score
    score_change = ns - current_score
    logger.info(f"迭代#{iteration}: {current_score:.2f} -> {ns:.2f} ({score_change:+.2f})")

    # Step7: promote 或 rollback
    if score_change >= SCORE_THRESHOLD:
        result_tag = f"✅提升{score_change:+.2f}"
        if tmp_best_backup.exists():
            tmp_best_backup.unlink()
        discard_candidate()
        exp_history.append(
            f"迭代{iteration}[{plan_type}]{result_tag}: "
            f"{current_score:.2f}->{ns:.2f} | {config_desc[:100]}"
        )
        decision = "promote"
    else:
        result_tag = f"❌无改善{score_change:+.2f}"
        if tmp_best_backup.exists():
            shutil.copyfile(tmp_best_backup, BEST_CKPT)
            tmp_best_backup.unlink()
        elif not BEST_CKPT.exists():
            BEST_CKPT.unlink(missing_ok=True)
        discard_candidate()
        rollback_best()
        exp_history.append(
            f"迭代{iteration}[{plan_type}]{result_tag}: "
            f"{current_score:.2f}->{ns:.2f}(回滚) | {config_desc[:100]}"
        )
        ns       = current_score
        decision = "rollback"

    save_exp_history(exp_history)

    # 写入科研日志：实验结果（包含完整评测指标）
    eval_metrics = nm.get("metrics", {}) if nm["success"] else {}
    _log_experiment(iteration, "result", {
        "outcome":     "success" if score_change >= SCORE_THRESHOLD else "no_improvement",
        "score_before": current_score,
        "score_after":  ns,
        "score_change": round(score_change, 4),
        "seg1_score":   eval_metrics.get("seg1_score", 0),
        "seg2_score":   eval_metrics.get("seg2_score", 0),
        "seg3_score":   eval_metrics.get("seg3_score", 0),
        "seg3_winner":  eval_metrics.get("seg3_winner", "?"),
        "t2s_error":    eval_metrics.get("t2s_mean_error", 0),
        "t10s_error":   eval_metrics.get("t10s_mean_error", 0),
        "extrap_ratio": eval_metrics.get("extrapolation_ratio", 0),
        "inflection":   eval_metrics.get("inflection_step", 0),
        "decision":     decision,
        "config":       config_desc,
    })
    _log(content=f"迭代#{iteration}完成 {current_score:.2f}->{ns:.2f} {result_tag}")

    # Step8: LLM 反思
    logger.info("[4/4] LLM反思...")
    tln = read_training_log()
    reflection, conversation_history = llm_call(
        conversation_history,
        (
            f"本轮实验完成。\n"
            f"方案: {config_desc[:200]}\n"
            f"假说: {plan.get('hypothesis','')}\n"
            f"结果: {current_score:.2f} -> {ns:.2f} ({score_change:+.2f}) {result_tag}\n"
            f"诊断: 外推/覆盖比={eval_metrics.get('extrapolation_ratio',0):.1f}x "
            f"拐点步={eval_metrics.get('inflection_step',0)}\n"
            f"{'已采纳新版本。' if score_change >= SCORE_THRESHOLD else '已回滚到最优版本。'}\n"
            f"训练日志:\n{tln['result'][:250]}\n\n"
            "请分析（≤300字）：\n"
            "1. 假说验证结果（成功/失败的物理原因）\n"
            "2. 误差诊断数据说明了什么\n"
            "3. 下一步最有价值的改进方向"
        ),
        max_tokens=500,
    )
    save_conv_history(conversation_history)

    _log_experiment(iteration, "reflection", {
        "content": reflection[:800],
        "hypothesis_validated": score_change >= SCORE_THRESHOLD,
    })
    logger.info(f"迭代#{iteration}完成 | 对话历史:{len(conversation_history)}条")
    return ns, conversation_history, exp_history


# ============================================================
# 主函数
# ============================================================
def main():
    import torch as _torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iterations", type=int,   default=5)
    parser.add_argument("--target_score",   type=float, default=90.0)
    parser.add_argument("--reset",          action="store_true")
    args = parser.parse_args()

    if args.reset:
        for f in [HISTORY_FILE, EXP_HIST_FILE, BACKUP_CKPT, CANDIDATE_CKPT]:
            try: Path(f).unlink()
            except Exception: pass
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        logger.info("历史记录和备份已清除")

    logger.info("=" * 60)
    logger.info(f"PDE Agent启动 | 迭代:{args.max_iterations} 目标:{args.target_score}")
    logger.info("=" * 60)
    _log(content=f"Agent启动 | 目标:{args.target_score}/100")

    conversation_history = load_conv_history()
    exp_history          = load_exp_history()
    logger.info(f"对话历史:{len(conversation_history)}条 | 实验历史:{len(exp_history)}条")

    # 获取初始得分
    if not BEST_CKPT.exists():
        logger.info("无checkpoint，得分设为0")
        current_score = 0.0
    else:
        logger.info("运行基线推理...")
        run_inference(gpu_ids=TRAIN_GPUS)
        em = read_eval_metrics()
        current_score = em["metrics"].get("total_score", 0.0) if em["success"] else 0.0
        logger.info(f"基线得分: {current_score:.2f}")

    _log(content=f"初始得分:{current_score:.2f}/100")

    # 首次运行介绍任务
    if not conversation_history:
        cfg     = get_best_config()
        cfg_str = json.dumps(cfg, ensure_ascii=False) if cfg else "无（尚未训练）"
        first_hint = (
            "重要提示：当前没有训练好的模型。\n"
            "第一轮建议：use_finetune=true（从官方FNO热启动，modes=12 width=20）"
            "或冷启动（use_resume=false, use_finetune=false，需epochs>=40）。\n"
            "注意：val前80条（覆盖t=0~10s）已混入训练集，这是解决Seg3低分的基础。\n"
            "第一个科研假说应该是验证'扩充训练数据时间覆盖是否改善Seg3'。"
            if not BEST_CKPT.exists() else
            "请基于当前模型配置和误差诊断提出改进假说。"
        )
        intro = (
            f"开始科研迭代，当前得分{current_score:.2f}/100，目标{args.target_score}/100。\n"
            f"当前配置: {cfg_str}\n"
            f"{first_hint}\n"
            f"得分提升>={SCORE_THRESHOLD}分才采纳，否则自动回滚。\n"
            f"请回复'明白'。"
        )
        _, conversation_history = llm_call(conversation_history, intro, max_tokens=50)
        save_conv_history(conversation_history)

    best_score = current_score

    for i in range(1, args.max_iterations + 1):
        ns, conversation_history, exp_history = run_iteration(
            conversation_history, exp_history, i, current_score
        )

        if ns > best_score:
            best_score = ns
            logger.info(f"最优更新: {best_score:.2f}")

        current_score = ns
        save_exp_history(exp_history)

        # 更新 task1_time.csv 的 train_time（含 Agent 思考时间）
        total_elapsed = time.time() - AGENT_START_TIME
        time_csv      = PROJECT_DIR / "repos" / "submission" / "task1_time.csv"
        if time_csv.exists():
            with open(time_csv, 'r') as f:
                rows = list(csv.DictReader(f))
            infer_time = float(rows[-1].get('inference_time', 0)) if rows else 0.0
        else:
            infer_time = 0.0
        with open(time_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['train_time', 'inference_time'])
            w.writerow([total_elapsed, infer_time])

        if current_score >= args.target_score:
            logger.info(f"已达目标 {args.target_score}，停止")
            break

    _log(content=f"Agent完成 | 最终:{best_score:.2f}/100 | 精度分:{best_score*0.75:.2f}/75")
    logger.info(f"Agent完成 | 最优:{best_score:.2f}/100")


if __name__ == "__main__":
    main()