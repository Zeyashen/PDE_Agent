"""
orchestrator.py — Agent 主循环

框架层（人工写，不提交到 code/）。
所有科研代码（model.py 等）由 Agent 生成，写到 workspace/。

流程：
  每轮 → 构建 observation → LLM 推理 → 解析 tool_calls → 执行工具
       → 记录到 log → 更新 experiment_log → 下一轮
"""

import os
import re
import sys
import csv
import json
import time
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# ── 路径 ────────────────────────────────────────────────────
PROJECT_DIR = Path("/mimer/NOBACKUP/groups/phy_geo/PDE_Agent")
AGENT_DIR   = PROJECT_DIR / "repos" / "agent_task2"
SKILLS_DIR  = PROJECT_DIR / "repos" / "skills_task2"
WORKSPACE   = PROJECT_DIR / "repos" / "workspace"
SUBMISSION  = PROJECT_DIR / "repos" / "submission"
LOG_DIR     = PROJECT_DIR / "logs" / "task2"

sys.path.insert(0, str(AGENT_DIR))
from tools import (
    execute_tool, append_research_log, append_experiment_log,
    WORKSPACE, SUBMISSION, SKILLS_DIR,
)

# ── LLM 配置 ─────────────────────────────────────────────────
LLM_GPU    = os.environ.get("LLM_GPUS", "0,1,2,3")
TRAIN_GPUS = os.environ.get("TRAIN_GPUS", "0,1,2,3")
SIF        = "/apps/containers/vLLM/vllm-0.19.1.sif"
VENV       = str(PROJECT_DIR / "env" / "pde_venv" / "bin" / "activate")

LOG_PATH         = LOG_DIR / "agent_orchestrator.log"
AGENT_START_TIME = time.time()
_CURRENT_LLM_ELAPSED = 0.0

LOG_PATH.parent.mkdir(exist_ok=True)
WORKSPACE.mkdir(parents=True, exist_ok=True)
(WORKSPACE / ".history").mkdir(exist_ok=True)
(SUBMISSION / "code").mkdir(parents=True, exist_ok=True)


# ============================================================
# System Prompt
# ============================================================
SYSTEM_PROMPT_TASK2 = """你是一个 AI 科学家，目标是为 1D Burgers 方程（多粘性系数 Nu 泛化）编写高精度预测代码，并通过实验持续优化。

你需要自主完成：训练模型 → 评估结果 → 分析瓶颈 → 改进代码 → 循环迭代。

## 任务特点（与 Task1 的关键区别）

- 训练数据包含多种 Nu 值（0.0001~0.01），每个样本有对应的 Nu 标签
- 测试时不提供 Nu 值，模型必须仅基于初始条件预测
- 训练时间不计入评测，但总时长必须 ≤ 12 小时
- 推理时间必须 < 2 分钟（否则该任务得 0 分）
- 禁止使用任何公开预训练权重，必须从头训练

## 核心科研问题

如何让模型在训练时利用 Nu 信息，但推理时仅用初始条件就能泛化到未知 Nu？

可探索的方向：
1. 忽略 Nu：直接用初始条件训练，让模型隐式学习 Nu 的影响
2. Nu 作为条件输入：训练时将 Nu 编码进 branch net，推理时用初始条件预测 Nu 再输入
3. Nu 作为辅助监督：多任务学习，同时预测轨迹和 Nu 值

## 可用工具

文件操作：
  read_file(path)                    读取文件（workspace/ 或 skills/ 路径）
  write_file(path, content)          写入文件到 workspace/（自动备份旧版本）
  edit_file(path, old_str, new_str)  精准替换文件中的唯一字符串（省token）
  list_files(dir)                    列出目录文件列表

执行：
  run_bash(command, timeout)                       同步执行 bash 命令
  run_bash_async(command, log_file)                ★ 后台执行长任务，立即返回 pid
  check_process(pid, tail_n)                       ★ 检查后台进程状态 + 日志末尾
  run_training(script, extra_args, epochs, n_gpus) 封装DDP训练（同步）
  smoke_test(script)                               ★ 单batch快速验证代码能否跑通（30-60秒）
  full_eval(ckpt_path)                             完整评估：Seg1/2/3 真实得分

工作记忆（重要）：
  todo_write(todos)                  ★ 写任务列表，避免遗忘
  todo_read()                        ★ 读当前任务状态

查询：
  read_training_log()                查看实时训练日志
  get_score_history()                查看所有实验记录
  check_remaining_time()             查看剩余时间预算

## Skill 文档（按需读取）

★★★ 第一轮必须读且只读这一个：
  skills/workflow.md  ← 包含完整模板、标准工作流、诊断建议

按需查阅（第二轮以后）：
  skills/ddp_runner_api.md  DDP 接口规范
  skills/tools_api.md       工具调用规范
  skills/scoring.md         评分规则和公式
  skills/deeponet.md        超参调优建议
  skills/data_format.md     Task2 数据路径和格式（含 Nu 标签说明）
  skills/experiment_log.md  历史实验记录

## 工作规范
- 第一轮：smoke_test("train.py") 验证初始代码，通过后直接训练
- 每次只改一个地方，验证通过再改下一个
- 修改代码后必须先 smoke_test 验证，通过后才能正式训练
- 推荐工作流：smoke_test → run_training → full_eval → 诊断 → 改进 → 循环
- model.py、dataset.py、ddp_runner.py 已由框架层提供，直接 import，不要自己写
- 只能写 train.py 和 predict.py
- 推理时间必须 < 2分钟（否则该任务0分）
- 总训练时间必须 ≤ 12 小时
- 禁止加载任何公开预训练权重

## 输出格式（严格遵守，违反会导致执行失败）
<think>
[简洁分析，100-200字，不要在think里写代码]
</think>
[
  {"tool": "工具名", "参数名": "参数值"},
  {"tool": "工具名", "参数名": "参数值"}
]

严格要求：
1. </think> 后紧跟 JSON 数组，不允许有其他文字
2. JSON 数组不能为空 []，至少包含1个工具调用
3. 代码必须通过 write_file 工具写入，不要在 think 里写代码
4. think 简洁，不超过200字"""

# LLM Worker 代码（子进程） 缓存重定位
LLM_WORKER = """\
import os, sys, json

# 强行关闭V1引擎并修复进程通信
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

_cache = os.path.expanduser("~/.cache/vllm_agent")
os.makedirs(_cache + "/inductor", exist_ok=True)
os.makedirs(_cache + "/triton",   exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _cache + "/inductor"
os.environ["TRITON_CACHE_DIR"]        = _cache + "/triton"
os.environ["VLLM_CACHE_ROOT"]         = _cache

def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gpu_ids    = sys.argv[1]
    model_path = sys.argv[2]
    n_gpus     = int(sys.argv[3])
    in_f, out_f = sys.argv[4], sys.argv[5]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    data       = json.load(open(in_f, encoding="utf-8"))
    system_p   = data["system_prompt"]
    messages   = data["messages"]
    max_tokens = data.get("max_tokens", 20000)

    # 用 transformers tokenizer 渲染 chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_p}] + messages,
        tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    # vLLM 离线推理（子进程结束后显存 100% 释放）
    llm = LLM(
        model=model_path,
        tensor_parallel_size=n_gpus,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.6, max_tokens=max_tokens)
    outputs = llm.generate([text], sampling_params)
    reply   = outputs[0].outputs[0].text.strip()
    json.dump({"reply": reply}, open(out_f, "w", encoding="utf-8"), ensure_ascii=False)

# ====== Python spawn 模式必须有的保护块 ======
if __name__ == "__main__":
    main()
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


# ============================================================
# 科研日志（官方格式）
# ============================================================
def _log(content: str, log_name: str, tool_calls=None):
    """写入一条官方格式 JSONL 记录。elapsed_seconds = 本次 LLM 调用耗时。"""
    entry = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(_CURRENT_LLM_ELAPSED, 3),
    }
    if content:
        entry["response"] = content
    if tool_calls:
        entry["tool_calls"] = tool_calls
    append_research_log(json.dumps(entry, ensure_ascii=False), log_name)


# ============================================================
# LLM 调用
# ============================================================

def llm_call(messages: list, model_path: str,
             max_tokens: int = 20000, task: str = "task2",
             n_llm_gpus: int = None) -> str | None:
    """
    vLLM 子进程离线推理。每轮启动子进程，推理完毕显存完全释放，
    下一步训练可以独占全部 GPU 显存。
    n_llm_gpus: LLM 占用 GPU 数（14B=1, 27B=2, 70B=4）
    """
    global _CURRENT_LLM_ELAPSED

    if n_llm_gpus is None:
        n_llm_gpus = len(LLM_GPU.split(","))

    _tmp_dir = str(LOG_DIR)  # 使用有权限的目录
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8", dir=_tmp_dir
    ) as f:
        f.write(LLM_WORKER)
        worker = f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_in.json", delete=False, encoding="utf-8", dir=_tmp_dir
    ) as f:
        json.dump({
            "system_prompt": (SYSTEM_PROMPT_TASK1 if task == "task1"
                              else SYSTEM_PROMPT_TASK2),
            "messages":   messages,
            "max_tokens": max_tokens,
        }, f, ensure_ascii=False)
        in_f = f.name

    out_f = in_f.replace("_in.json", "_out.json")

    try:
        logger.info(f"LLM 推理 GPU={LLM_GPU} n_gpus={n_llm_gpus}...")
        t0  = time.time()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = LLM_GPU

        # 清除 venv 路径，确保子进程用容器内原生的 transformers
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
        clean_env["CUDA_VISIBLE_DEVICES"] = LLM_GPU
        clean_env["PYTHONNOUSERSITE"] = "1"
        clean_env["CUTLASS_CACHE_DIR"] = str(PROJECT_DIR / "tmp" / "cutlass_cache")
        clean_env["HOME"] = str(PROJECT_DIR / "tmp")
        clean_env["TMPDIR"] = str(LOG_DIR)   # "/tmp"
        clean_env["TEMP"]   = str(LOG_DIR)
        clean_env["TMP"]    = str(LOG_DIR)

        # ======= 终极环境隔离与网络修复 =======
        clean_env["VLLM_USE_V1"] = "0"          # 彻底禁用V1引擎（OS级别注入，绝对生效）
        clean_env["VLLM_HOST_IP"] = "127.0.0.1" # 修复容器内 hostname 无法解析导致的 Socket 死锁
        clean_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        clean_env["NCCL_SOCKET_IFNAME"] = "lo"  # 强制 NCCL 走本地回环
        clean_env["NCCL_SHM_DISABLE"] = "1"     # 绕过 Apptainer 默认 64MB 的共享内存墙，强制走 P2P
        # ======================================

        # 从 PATH 里去掉 venv/bin
        clean_env["PATH"] = ":".join(
            p for p in os.environ.get("PATH", "").split(":")
            if "pde_venv" not in p
        )
        r = subprocess.run(
            ["/usr/bin/python3", worker, LLM_GPU, model_path,
             str(n_llm_gpus), in_f, out_f],
            capture_output=True, text=True,
            timeout=600, cwd=str(PROJECT_DIR), env=clean_env,
        )
        _CURRENT_LLM_ELAPSED = time.time() - t0
        logger.info(f"LLM 完成 {_CURRENT_LLM_ELAPSED:.1f}s")

        if r.returncode != 0:
            logger.error(f"LLM 失败:\n{r.stderr}")
            return None

        time.sleep(1)  # 等 CUDA context 完全释放
        return json.load(open(out_f, encoding="utf-8"))["reply"]

    except subprocess.TimeoutExpired:
        logger.error("LLM 超时（>600s）")
        return None
    except Exception as e:
        logger.error(f"LLM 异常: {e}")
        return None
    finally:
        for fp in [worker, in_f, out_f]:
            try: os.unlink(fp)
            except: pass




# ============================================================
# 解析 LLM 输出
# ============================================================
def parse_output(raw: str) -> tuple[str, list]:
    """解析 <think>...</think> 和 JSON 数组。"""
    think = ""
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        think = m.group(1).strip()

    # 去掉 think 标签，找 JSON 数组
    rest  = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    tools = []
    m2    = re.search(r"\[.*\]", rest, re.DOTALL)
    if m2:
        try:
            tools = json.loads(m2.group())
        except json.JSONDecodeError:
            # 尝试修复常见错误
            try:
                cleaned = re.sub(r",\s*]", "]", m2.group())
                tools   = json.loads(cleaned)
            except:
                pass

    return think, tools

# ============================================================
# agent_chain.md DynamicSkill-agent思维链
# ============================================================
def _append_chain(iteration: int, think: str, 
                  results: list, elapsed_min: float,
                  remaining: float):
    """每轮结束后追加一条结构化记录到 skills/agent_chain.md。"""
    chain_path = SKILLS_DIR / "agent_chain.md"
    
    lines = [f"\n## 迭代#{iteration} | 已用{elapsed_min:.1f}min | 剩余{remaining:.1f}min"]
    
    # 思维链摘要
    lines.append(f"\n**思维链**: {think[:300] if think else '（无）'}")
    
    # 工具调用与结果
    lines.append("\n**工具调用与结果**:")
    for r in results:
        tool = r.get("tool", "unknown")
        res  = r.get("result", {})
        ok   = res.get("success", False)
        icon = "success" if ok else "failed"
        
        # 根据工具类型提取关键信息
        if tool == "smoke_test":
            detail = f"val_loss={res.get('val_loss')} elapsed={res.get('elapsed')}s"
            if not ok:
                err = res.get("error", "")[:200]
                detail = f"错误: {err}"
        elif tool == "run_training":
            detail = (f"val_loss={res.get('val_loss')} "
                      f"elapsed={res.get('elapsed_seconds', 0):.0f}s")
            if not ok:
                detail = res.get("log_tail", "")[:200]
        elif tool in ("write_file", "edit_file"):
            detail = f"path={res.get('path')} lines={res.get('lines', res.get('lines_changed'))}"
            if not ok:
                detail = res.get("error", "")[:100]
        elif tool == "read_file":
            detail = f"path={res.get('path')} lines={res.get('lines')}"
        elif tool in ("full_eval", "run_evaluation"):
            detail = (f"total={res.get('total_score')} "
                      f"seg1={res.get('seg1_score')} "
                      f"seg2={res.get('seg2_score')} "
                      f"seg3={res.get('seg3_score')}")
        else:
            detail = str(res)[:100]
        
        lines.append(f"- {tool}: {icon} {detail}")
    
    # 自动结论：连续失败检测
    smoke_results = [r for r in results if r.get("tool") == "smoke_test"]
    if smoke_results and not smoke_results[-1]["result"].get("success"):
        err = smoke_results[-1]["result"].get("error", "")
        # 提取第一行错误类型
        first_err = next((l for l in err.splitlines() if "Error" in l or "Exception" in l), "")
        lines.append(f"\n**结论**: smoke_test 失败 — {first_err[:150]}")
    elif smoke_results and smoke_results[-1]["result"].get("success"):
        lines.append(f"\n**结论**: smoke_test 通过，可以正式训练")
    
    with open(chain_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _read_chain_summary(n_recent: int = 5) -> str:
    """读取最近 n 条迭代记录，注入到 observation。"""
    chain_path = SKILLS_DIR / "agent_chain.md"
    if not chain_path.exists():
        return ""
    
    content = chain_path.read_text(encoding="utf-8")
    # 按 ## 迭代# 分割
    blocks = [b for b in content.split("\n## 迭代#") if b.strip()]
    if not blocks:
        return ""
    
    recent = blocks[-n_recent:]
    summary = "\n## 迭代#".join(recent)
    return f"【历史思维链（最近{len(recent)}轮）】\n## 迭代#{summary}"

def _write_framework_files():
    import shutil as _shutil

    # 框架层文件（不提交到 code/）
    for fname in ["model.py", "dataset.py", "ddp_runner.py"]:
        src_path = AGENT_DIR / fname
        if src_path.exists():
            (WORKSPACE / fname).write_text(
                src_path.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info(f"框架层 {fname} 已写入 workspace/")

    # train.py 和 predict.py 初始模板（只在不存在时写入）
    for fname in ["train.py", "predict.py"]:
        target = WORKSPACE / fname
        if not target.exists():
            src_path = AGENT_DIR / fname
            if src_path.exists():
                content = src_path.read_text(encoding="utf-8")
                target.write_text(content, encoding="utf-8")
                code_target = SUBMISSION / "code" / fname
                code_target.parent.mkdir(parents=True, exist_ok=True)
                _shutil.copy2(target, code_target)
                logger.info(f"初始模板 {fname} 已写入 workspace/")

# ============================================================
# 构建 Observation
# ============================================================
def build_first_round_obs(task: str, budget_minutes: float) -> str:
    """第一轮专用 observation：明确的任务清单，引导 LLM 按步骤执行。"""
    task_upper = task.upper()
    if task == "task1":
        task_desc = "Task 1：固定粘性系数 Nu=0.001 的 1D Burgers 方程长时预测"
        data_note = "训练数据覆盖 t=0~1.95s，但预测需要到 t=9.96s，必须混入 val 前80条解决覆盖问题"
        budget_note = f"时间预算 {budget_minutes} 分钟（含思考时间），请控制在预算内完成"
    else:
        task_desc = "Task 2：多粘性系数 Nu=0.0001~0.01 的 1D Burgers 方程泛化预测"
        data_note = "测试时不提供 Nu 值，模型必须仅从初始条件预测，branch net 隐式学习 Nu 信息"
        budget_note = f"训练时间不计入评测，总时长控制在12小时内"

    # 直接读取 workflow.md 嵌入 observation，不让 LLM 自己去读
    try:
        workflow_content = (SKILLS_DIR / "workflow.md").read_text(encoding="utf-8")
    except Exception:
        workflow_content = "（workflow.md 读取失败）"

    return f"""迭代#1 | {task_upper} | 这是第一轮

任务：{task_desc}
{data_note}
{budget_note}

═══ 重要提示 ═══
train.py 和 predict.py 已由框架层自动写入 workspace/，你无需从头编写代码。

═══ 本轮任务 ═══
1. smoke_test("train.py")   ← 验证初始代码能正常运行（30-60秒）
2. 如果通过，下一轮直接 run_training("train.py", epochs=10, n_gpus=4)

如需调整超参数，用 edit_file 精准替换，例如：
  edit_file("train.py", "default=1e-3", "default=5e-4")
  edit_file("train.py", "default=50",   "default=100")

可调整的超参数（在 train.py 的 argparse 部分）：
  --epochs, --lr, --batch_size, --patience
  --branch_type, --latent_dim, --branch_depth, --branch_width
  --val_mix_ratio, --n_val_mix, --n_query

路径规范：
  checkpoint: /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/workspace/checkpoints/best.pt
  预测结果:   /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/repos/submission/{task}_pred.hdf5
""".strip()



def build_observation(iteration: int, last_results: list,
                      budget_minutes: float, task: str) -> str:
    elapsed_min = (time.time() - AGENT_START_TIME) / 60
    remaining   = budget_minutes - elapsed_min

    # workspace 文件列表
    files = []
    if WORKSPACE.exists():
        for f in sorted(WORKSPACE.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                files.append(f.name)

    ws_str = ", ".join(files) if files else "（空，需要创建代码文件）"

    # 上一轮结果摘要
    if last_results:
        parts = []
        for r in last_results[-6:]:
            res = r['result']
            if r['tool'] == 'read_file' and res.get('success'):
                # read_file 显示更多内容供 LLM 参考
                content_preview = res.get('content', '')[:6000]
                parts.append(f"  {r['tool']} → path={res.get('path')} lines={res.get('lines')}\n内容预览:\n{content_preview}")
            else:
                parts.append(f"  {r['tool']} → {json.dumps(res, ensure_ascii=False)[:400]}")
        results_str = "\n".join(parts)
    else:
        results_str = "（第一轮，无历史结果）"

    # 判断当前 Phase 并给出建议工具调用
    has_train_py   = (WORKSPACE / "train.py").exists()
    has_predict_py = (WORKSPACE / "predict.py").exists()
    has_ckpt       = (WORKSPACE / "checkpoints" / "best.pt").exists()
    todos_file     = WORKSPACE / "todos.json"

    # 追踪代码状态：上一轮是否修改了代码且未验证
    code_modified_no_test = False
    last_smoke_ok         = None
    last_smoke_err        = ""
    last_train_failed     = False
    last_train_error      = ""

    just_trained = False
    just_evaled  = False
    for r in (last_results or []):
        tool = r.get("tool")
        res  = r.get("result", {})
        ok   = res.get("success", False)
        if tool in ("write_file", "edit_file") and ok:
            code_modified_no_test = True
        if tool == "smoke_test":
            code_modified_no_test = False
            last_smoke_ok  = ok
            last_smoke_err =  (res.get("error", "") + "\n" + res.get("log_tail", ""))[:500]
        if tool == "run_training":
            if ok:
                code_modified_no_test = False
                just_trained = True
            else:
                last_train_failed = True
                last_train_error  = res.get("log_tail", "")[:400]
        if tool in ("full_eval", "run_evaluation") and ok:
            just_evaled = True

    if not has_train_py or not has_predict_py:
        phase   = "Phase 0: 准备代码"
        suggest = ("read_file('skills/workflow.md') -> "
                   "write_file('train.py', ...) -> write_file('predict.py', ...)")
    elif code_modified_no_test:
        phase   = "Phase 0.5: 代码已修改，必须先 smoke_test 验证"
        suggest = ("立即执行：smoke_test('train.py')  "
                   "# 单卡单epoch小batch，30-60秒，验证代码能否端到端跑通\n"
                   "通过后才能正式训练，不要跳过这一步！")
    elif last_smoke_ok is False:
        phase   = "Phase 0.5: smoke_test 失败，必须修复"
        suggest = (f"smoke_test 错误：\n{last_smoke_err[:400]}\n"
                   "修复步骤：\n"
                   "1. read_file('train.py') 读取文件\n"
                   "2. 从返回内容里逐字复制要修改的行作为 old_str（不能凭记忆写）\n"
                   "3. edit_file('train.py', old_str=..., new_str=...)\n"
                   "4. smoke_test 验证")
    elif last_train_failed:
        phase   = "Phase 1.5: 训练失败，需要诊断"
        suggest = (f"训练失败错误：\n{last_train_error[:300]}\n"
                   "先 read_file 看代码定位问题，用 edit_file 修复，"
                   "修复后用 smoke_test 验证。")
    elif not has_ckpt:
        phase   = "Phase 1-2: 正式训练"
        suggest = ("代码已通过验证，可以正式训练：\n"
                   "run_training('train.py', epochs=10, n_gpus=4)")
    else:
        phase   = "Phase 3-4: 评估 + 诊断 + 改进"
        # suggest = ("quick_eval('checkpoints/best.pt') -> "
        #            "full_eval('checkpoints/best.pt') -> 根据得分用 edit_file 改超参\n"
        #            "★ 改超参更简单：run_training('train.py', extra_args='--lr 0.0005 --latent_dim 256')\n"
        #            "★ edit_file 的 old_str 必须从 read_file 返回内容里逐字复制，不能凭记忆写")

        suggest = ("★ 每轮训练后必须 full_eval('checkpoints/task2_best.pt') 获取真实得分\n"
                   "★ Nu条件化（推荐）：--nu_dim 1 启用，--nu_dropout 0.3 控制随机遮蔽比例\n"
                   "  训练时30%概率把Nu置0，让模型同时学会有Nu和无Nu两种情况，推理时更鲁棒\n"
                   "★ 推荐命令：run_training('train.py', extra_args='--nu_dim 1 --nu_dropout 0.3 --lr 0.0005 --latent_dim 256 --epochs 100')\n"
                   "★ 不要修改 predict.py，只改 train.py 的超参\n"
                   "★ edit_file 的 old_str 必须从 read_file 返回内容里逐字复制，不能凭记忆写\n"
                   "诊断：Seg1低→加大branch_width/latent_dim  Seg3低→加大n_val_mix/val_mix_ratio")




        # 读取 todo 状态
    todo_summary = "（todos.json 不存在，建议用 todo_write 创建任务清单）"
    if todos_file.exists():
        try:
            _todos = json.loads(todos_file.read_text(encoding="utf-8"))
            _lines = []
            for t in _todos[:12]:
                _icon = {
                    "pending":     "[ ]",
                    "in_progress": "[~]",
                    "completed":   "[x]",
                    "failed":      "[!]",
                }.get(t.get("status", "pending"), "[ ]")
                _lines.append(f"  {_icon} {t.get('content', '')[:80]}")
            todo_summary = "\n".join(_lines) if _lines else "（todos 为空）"
        except Exception:
            todo_summary = "（todos.json 解析失败）"

    chain_summary = _read_chain_summary(n_recent=5)
    obs = f"""迭代#{iteration} | Task {task.upper()} | 已用{elapsed_min:.1f}分钟 | 剩余{remaining:.1f}分钟

【当前阶段】{phase}
【建议工具调用】{suggest}

【当前任务清单】
{todo_summary}

【上一轮工具执行结果】
{results_str}

workspace/ 当前文件：{ws_str}

{chain_summary}

"""
    return obs.strip()


# ============================================================
# 执行一轮 tool_calls
# ============================================================
def execute_round(tool_calls: list, budget_minutes: float,
                  task: str) -> list:
    """执行本轮所有工具调用，返回结果列表。"""
    results = []
    log_name = f"{task}_logs.log"

    for tc in tool_calls:
        tool_name = tc.get("tool")
        args      = {k: v for k, v in tc.items() if k != "tool"}

        logger.info(f"  执行工具: {tool_name}({list(args.keys())})")

        result = execute_tool(
            tool_name, args,
            start_time=AGENT_START_TIME,
            budget_minutes=budget_minutes,
        )

        results.append({"tool": tool_name, "args": args, "result": result})

        # 记录工具执行结果到日志
        _log(
            content=f"工具执行: {tool_name} → {json.dumps(result, ensure_ascii=False)[:300]}",
            log_name=log_name,
            tool_calls=json.dumps(tc, ensure_ascii=False),
        )

        # 如果是训练或评分，额外打印关键结果
        if tool_name == "run_training" and result.get("success"):
            logger.info(f"    训练完成: val_loss={result.get('val_loss')} "
                        f"耗时={result.get('elapsed_seconds'):.0f}s")
        elif tool_name == "run_evaluation" and result.get("success"):
            logger.info(f"    评分结果: 总分={result.get('total_score')} "
                        f"Seg1={result.get('seg1_score')} "
                        f"Seg2={result.get('seg2_score')} "
                        f"Seg3={result.get('seg3_score')}")

    return results

# ============================================================
# 主循环
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           default="task2",
                        choices=["task1", "task2"])
    parser.add_argument("--budget_minutes", type=float, default=600.0)
    parser.add_argument("--model_path",
                        default=str(PROJECT_DIR / "models" / "Llama"))
    parser.add_argument("--max_tokens",     type=int, default=20000)
    parser.add_argument("--reset",          action="store_true")
    args = parser.parse_args()

    task     = args.task
    log_name = f"{task}_logs.log"

    if args.reset:
        if WORKSPACE.exists():
            shutil.rmtree(WORKSPACE)
        WORKSPACE.mkdir(parents=True)
        (WORKSPACE / ".history").mkdir()
        (SUBMISSION / "code").mkdir(parents=True, exist_ok=True)
        chain_path = SKILLS_DIR / "agent_chain.md"
        if chain_path.exists():
            chain_path.unlink()
        logger.info("workspace/ 已清空")
    
    _write_framework_files()

    logger.info("=" * 65)
    logger.info(f"Agent 启动 | Task={task.upper()} | "
                f"预算={args.budget_minutes}min | "
                f"模型={Path(args.model_path).name}")
    logger.info("=" * 65)

    _log(
        content=(f"Agent 启动 | Task={task.upper()} | "
                 f"预算={args.budget_minutes}min"),
        log_name=log_name,
    )

    messages      = []   # 对话历史（保持最近N轮）
    last_results  = []   # 上一轮工具执行结果
    iteration     = 0
    best_score    = 0.0

    while True:
        iteration += 1
        elapsed_min  = (time.time() - AGENT_START_TIME) / 60
        remaining    = args.budget_minutes - elapsed_min

        if remaining < 8:
            logger.info(f"剩余时间 {remaining:.1f}min < 8min，停止")
            break

        logger.info("=" * 65)
        logger.info(f"迭代#{iteration} | 剩余={remaining:.1f}min | "
                    f"当前最优={best_score:.2f}")

        # ── 构建用户消息 ──────────────────────────────────────
        # 第一轮用专门的引导 observation，后续用标准 observation
        if iteration == 1:
            obs = build_first_round_obs(task, args.budget_minutes)
        else:
            obs = build_observation(
                iteration, last_results, args.budget_minutes, task)
        messages.append({"role": "user", "content": obs})

        # ── LLM 推理 ─────────────────────────────────────────
        raw = llm_call(messages, args.model_path,
                       args.max_tokens, task)

        if not raw:
            logger.error("LLM 返回空，跳过本轮")
            _log(f"迭代#{iteration} LLM 失败，跳过", log_name)
            messages.pop()  # 移除失败的用户消息
            continue

        think, tool_calls = parse_output(raw)

        logger.info(f"Think（前200字）: {think[:200]}...")
        logger.info(f"Tool calls: {len(tool_calls)} 个")

        # ── 记录 LLM 输出到日志 ───────────────────────────────
        _log(
            content=f"迭代#{iteration} think: {think}",
            log_name=log_name,
            tool_calls=json.dumps(tool_calls, ensure_ascii=False),
        )

        # ── 把 LLM 输出加入对话历史 ───────────────────────────
        messages.append({"role": "assistant", "content": raw})

        # 保持对话历史最近6轮（避免 context 过长）
        if len(messages) > 4:
            messages = messages[-4:]

        if not tool_calls:
            logger.warning("未解析到 tool_calls，跳过执行")
            # 把空数组问题反馈给下一轮 LLM
            last_results = [{"tool": "system", "args": {}, "result": {
                "success": False,
                "error": "上一轮输出了空的 tool_calls []，没有执行任何操作。"
                         "请确保在 </think> 后输出至少一个工具调用的 JSON 数组。"
                         "不要只在 think 里规划，必须实际调用工具。"
            }}]
            continue

        # 第一轮只允许 read_file 和 write_file，防止训练在代码写好之前执行
        if iteration == 1:
            safe_calls = [
                tc for tc in tool_calls
                if tc.get("tool") in {"read_file", "write_file",
                                      "list_files", "run_bash",
                                      "smoke_test","todo_write",
                                      "todo_read","run_training"}
            ]
            if len(safe_calls) < len(tool_calls):
                skipped = [tc["tool"] for tc in tool_calls
                           if tc not in safe_calls]
                logger.info(f"  第一轮跳过工具: {skipped}（代码写好后再训练）")
            tool_calls = safe_calls

        # ── 执行工具 ─────────────────────────────────────────
        last_results = execute_round(tool_calls, args.budget_minutes, task)

        # ── agent_chain ──────────────────────────────────────
        elapsed_min = (time.time() - AGENT_START_TIME) / 60
        remaining   = args.budget_minutes - elapsed_min
        _append_chain(iteration, think, last_results, elapsed_min, remaining)

        # ── 更新最优分数 ──────────────────────────────────────
        for r in last_results:
            if r["tool"] in ("run_evaluation", "full_eval") and r["result"].get("success"):
                score = r["result"].get("total_score", 0)
                if score > best_score:
                    best_score = score
                    logger.info(f"最优更新: {best_score:.2f}/100")

        # ── 更新 experiment_log.md ────────────────────────────
        eval_results = [r for r in last_results
                        if r["tool"] == "run_evaluation"
                        and r["result"].get("success")]
        train_results = [r for r in last_results
                         if r["tool"] == "run_training"
                         and r["result"].get("success")]

        if eval_results:
            ev = eval_results[-1]["result"]
            tr = train_results[-1]["result"] if train_results else {}
            append_experiment_log(iteration, {
                "model_desc":  think[:100] if think else "未知",
                "total_score": ev.get("total_score", 0),
                "seg1":        ev.get("seg1_score", 0),
                "seg2":        ev.get("seg2_score", 0),
                "seg3":        ev.get("seg3_score", 0),
                "train_time":  tr.get("elapsed_seconds", 0),
                "infer_time":  ev.get("infer_time", 0),
                "conclusion":  think[-200:] if think else "无",
                "next_step":   "见下一轮",
            })

        # ── 更新 task_time.csv ────────────────────────────────
        total_elapsed = time.time() - AGENT_START_TIME
        infer_t = 0.0
        if eval_results:
            infer_t = eval_results[-1]["result"].get("infer_time", 0)
        time_csv = SUBMISSION / f"{task}_time.csv"
        with open(time_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["train_time", "inference_time"])
            w.writerow([round(total_elapsed, 2), round(infer_t, 2)])


    # ── 强制最终提交（循环结束后必跑，生成提交文件并记录推理时间）──
    final_infer_time = 0.0
    ckpt_file = WORKSPACE / "checkpoints" / "best.pt"
    if ckpt_file.exists():
        logger.info("运行最终提交：full_eval(val_only=False)...")
        import gc as _gc
        _gc.collect()
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(3)  # 等待显存释放
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        from tools import full_eval as _full_eval
        _r = _full_eval(ckpt_path="checkpoints/best.pt", val_only=False, timeout=120)
        if _r.get("success"):
            final_infer_time = _r.get("infer_time", 0.0)
            s = _r.get("total_score", 0.0)
            if s > best_score:
                best_score = s
            logger.info(f"最终提交: 得分={s:.2f} 推理时间={final_infer_time:.2f}s")
        else:
            logger.error(f"最终提交失败: {_r.get('log_tail','')[-200:]}")
    else:
        logger.warning("没有 checkpoint，跳过最终提交")
    

    # ── 最终汇总 ──────────────────────────────────────────────
    total_elapsed = (time.time() - AGENT_START_TIME) / 60

    # Task2：只有精度分和推理时间分，训练时间不计分
    infer_min = final_infer_time / 60
    if infer_min <= 0:    infer_score = 40
    elif infer_min < 2:   infer_score = round(40 * (1 - infer_min / 2), 1)
    else:                 infer_score = 0

    precision_score = best_score * 0.75
    total_task2     = precision_score + infer_score

    logger.info("=" * 65)
    logger.info(f"Agent 完成 | 最优: {best_score:.2f}/100")
    logger.info(f"精度分: {precision_score:.2f}/75")
    logger.info(f"推理时间分: {infer_score}/40 ({final_infer_time:.2f}s)")
    logger.info(f"Task2 总分预测: {total_task2:.1f}/115")
    logger.info("=" * 65)

    _log(
        content=(f"Agent 完成 | 最优={best_score:.2f}/100 | "
                 f"精度分={precision_score:.2f}/75 | "
                 f"推理时间分={infer_score}/40 ({final_infer_time:.2f}s)"),
        log_name=log_name,
    )




if __name__ == "__main__":
    import shutil
    main()