"""
orchestrator.py — DeepONet Agent 主循环

核心改动（基于第一轮实验分析）：
1. 时间预算控制：用剩余时间而非迭代次数决定是否继续
   - 训练时间分：≤60min=35分，≤120min=25分
   - 默认预算 55 分钟（留 5 分钟 buffer）
2. FNO baseline 标尺：启动时用官方 FNO checkpoint 跑一次评分
   - 作为固定参考点，每轮对比
   - 确认评分没有系统性偏移
3. 每轮结束展示完整排行榜（含所有参数和得分）
4. 针对实验结果的 prompt 改进：
   - exp_001/002/003 全部 Seg1≈0，根本原因是短期数据不足
   - 引导 LLM 关注 val_mix_ratio 调整
   - 引导 LLM 加深加宽网络（d2w64 太小）
5. 正确记录推理时间到 task1_time.csv
"""

import os, re, sys, csv, json, time, logging, sqlite3, subprocess, tempfile, argparse
from pathlib import Path
from datetime import datetime, timezone

LLM_GPU    = os.environ.get("LLM_GPUS",   "0")
TRAIN_GPUS = os.environ.get("TRAIN_GPUS", "0,1,2,3")

sys.path.insert(0, str(Path(__file__).parent))
from tools import (
    init_registry, get_leaderboard, get_all_structures_tested,
    get_experiment, update_experiment, insert_experiment,
    propose_and_run, read_training_log, append_research_log,
    PROJECT_DIR, DB_PATH, SUBMISSION, CKPT_DIR,
)

MODEL_PATH       = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/models/Qwen3-14B"
FNO_CKPT_NU0001  = "/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/checkpoints/1D_Burgers/1D_Burgers_Sols_Nu0.001_FNO.pt"
LOG_PATH         = PROJECT_DIR / "logs" / "agent_orchestrator.log"
AGENT_START_TIME = time.time()
_CURRENT_LLM_ELAPSED = 0.0

LOG_PATH.parent.mkdir(exist_ok=True)


# ============================================================
# System Prompt（针对实验结果优化）
# ============================================================
DEEPONET_SYSTEM_PROMPT = """你是一个 AI 科研 Agent，目标是在 1D Burgers 方程上极致优化 DeepONet 的长时预测精度。

━━━ 科学背景 ━━━
任务: 给定初始10步 u(x, t=0.01~0.46s)，预测未来190步 u(x, t=0.46~9.96s)
模型: DeepONet 范式A（直接算子映射，无自回归误差累积）
  Branch net: 编码初始10步 → latent vector
  Trunk net:  编码查询坐标(t,x) → basis vectors（含 Fourier 特征嵌入）
  输出: dot(branch, trunk) + bias = u(x,t)

━━━ 已有实验结论（必读）━━━
第一轮实验（3个结构，均 depth=2, width=64）：
  mlp:         总分=27.72  Seg1=0.00  Seg2=0.07  Seg3=55.41
  cnn:         总分=45.72  Seg1=1.05  Seg2=29.91 Seg3=75.96  ← 最优
  fno_encoder: 总分=43.65  Seg1=0.78  Seg2=23.48 Seg3=75.16

关键诊断：Seg1≈0 说明短期预测（t=0.5~2.8s）极差，根本原因：
  - 所有模型 depth=2, width=64，模型容量严重不足
  - branch net 无法从10步初值中提取足够精细的特征
  改进方向：加深加宽网络（depth≥4, width≥128）

val_mix_ratio 的作用：
  = 1.0: 等权（当前默认，短期数据不足）
  < 1.0: 更多短期数据 → 改善 Seg1（建议先试 0.5）
  > 1.0: 更多长期数据 → 改善 Seg3（已经较好，暂不需要）
  建议：先用 val_mix_ratio=0.5 配合更大的网络

━━━ 评分公式 ━━━
Seg1(步10-57,  25%): 100·exp(-20·Rel-MSE)  ← 当前≈0，必须改善
Seg2(步57-105, 25%): 100·exp(-10·Rel-MSE)  ← cnn已有30分
Seg3(步105-200,50%): max(Lorentzian, Fréchet)  ← cnn已有76分

━━━ 参数分类 ━━━
A类（结构，改动→scratch）：
  branch_type: mlp / cnn / fno_encoder
  branch_depth, branch_width, trunk_depth, trunk_width, latent_dim
  activation: tanh / gelu / silu
  fourier_features

B类（训练，可resume）：
  lr, weight_decay, epochs(≤60), batch_size(≤64), n_query(≤4096)
  val_mix_ratio: 控制 val 前80条采样权重（建议范围 0.3~2.0）

C类（物理权重）：physics_weight

━━━ 探索策略 ━━━
探索期（已测试结构<3）：只动A类，scratch
  → 已完成：mlp/cnn/fno_encoder 各测试过一次（均 d2w64）
  → 现在进入利用期，但仍可以用更大的网络重测

利用期（已测试结构≥3）：
  1. 优先：用 cnn（最优结构）加深加宽 → depth=4, width=128
  2. 同时调低 val_mix_ratio=0.5 改善 Seg1
  3. 之后：在最优大网络上调 B 类参数

━━━ 输出格式 ━━━
<think>
[推理：当前瓶颈分析 → 假说 → 预期效果 → 风险，200-400字]
</think>
{
  "exp_id": "exp_NNN",
  "parent_id": "exp_XXX 或 scratch",
  "hypothesis": "一句话假说",
  "rationale": "详细理由",
  "changes": {"参数名": 值}
}

约束：
- changes 只写相对 parent 变化的参数
- exp_id 按序编号
- epochs ≤ 60，batch_size ≤ 64，latent_dim ≤ 256
- 失败配置禁止重复
- 只输出 <think>...</think> 和 JSON
"""

LLM_WORKER_CODE = """\
import os, sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
model_path, in_f, out_f = sys.argv[2], sys.argv[3], sys.argv[4]

data       = json.load(open(in_f, encoding="utf-8"))
prompt     = data["prompt"]
max_tokens = data.get("max_tokens", 1200)
system_p   = data["system_prompt"]

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map={"": 0}
)
model.eval()

messages = [
    {"role": "system", "content": system_p},
    {"role": "user",   "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=True,
)
inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=max_tokens,
        do_sample=True, temperature=0.6,
        pad_token_id=tokenizer.eos_token_id,
    )
reply = tokenizer.decode(
    out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
).strip()
json.dump({"reply": reply}, open(out_f, "w", encoding="utf-8"),
          ensure_ascii=False)
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
# 科研日志（官方格式）
# ============================================================
def _log(content, log_name="task1_logs.log", tool_calls=None):
    """elapsed_seconds = 本次 LLM 调用耗时（官方定义）"""
    entry = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(_CURRENT_LLM_ELAPSED, 3),
    }
    if content:
        entry["response"] = content
    if tool_calls:
        entry["tool_calls"] = tool_calls
    append_research_log(json.dumps(entry, ensure_ascii=False), log_name)


def _log_leaderboard(iteration, log_name="task1_logs.log"):
    """每轮结束写入完整排行榜到 orchestrator 日志和科研日志。"""
    board = get_leaderboard(limit=50)
    if not board:
        return

    best_score = max((r['total_score'] or 0) for r in board)

    # orchestrator 日志（终端可见）
    logger.info("=" * 65)
    logger.info(f"迭代#{iteration} 完成 | 排行榜")
    logger.info(f"{'排名':<4} {'exp_id':<10} {'总分':>6} "
                f"{'Seg1':>6} {'Seg2':>6} {'Seg3':>6} "
                f"{'结构':<22} {'lr':>7} {'vmr':>5} {'时间':>6}")
    for i, r in enumerate(board):
        s   = r['total_score'] or 0
        s1  = r['seg1_score']  or 0
        s2  = r['seg2_score']  or 0
        s3  = r['seg3_score']  or 0
        st  = f"{r['branch_type']}+d{r['branch_depth']}w{r['branch_width']}"
        vmr = r.get('val_mix_ratio') or 1.0
        tt  = r.get('train_time') or 0
        mark = " ★" if abs(s - best_score) < 0.01 else ""
        logger.info(
            f"{i+1:<4} {r['exp_id']:<10} {s:>6.2f} "
            f"{s1:>6.1f} {s2:>6.1f} {s3:>6.1f} "
            f"{st:<22} {r['lr']:>7} {vmr:>5.1f} {tt:>5.0f}s{mark}"
        )
    logger.info(f"当前最优: {best_score:.2f}/100")
    logger.info("=" * 65)

    # 科研日志
    lines = [f"=== 迭代#{iteration} 排行榜 ==="]
    for i, r in enumerate(board):
        s  = r['total_score'] or 0
        s1 = r['seg1_score']  or 0
        s2 = r['seg2_score']  or 0
        s3 = r['seg3_score']  or 0
        st = f"{r['branch_type']}+d{r['branch_depth']}w{r['branch_width']}"
        mark = " ★最高分" if abs(s - best_score) < 0.01 else ""
        lines.append(
            f"  {i+1}. {r['exp_id']}: 总分={s:.2f} "
            f"[Seg1={s1:.1f} Seg2={s2:.1f} Seg3={s3:.1f}] "
            f"{st} lr={r['lr']} vmr={r.get('val_mix_ratio',1.0):.1f}{mark}"
        )
    lines.append(f"  当前最优: {best_score:.2f}/100")
    _log("\n".join(lines), log_name)


# ============================================================
# FNO Baseline 标尺
# ============================================================
def run_fno_baseline():
    """
    用官方 FNO Nu=0.1 checkpoint 跑一次评分，作为固定标尺。
    结果写入 registry.db 的 fno_baseline 条目。
    """
    import re as _re
    logger.info("=" * 60)
    logger.info("FNO Baseline 标尺评分...")

    fno_ckpt = FNO_CKPT_NU0001
    if not Path(fno_ckpt).exists():
        logger.warning(f"FNO checkpoint 不存在: {fno_ckpt}，跳过标尺")
        return None

    venv   = str(PROJECT_DIR / "env" / "pde_venv" / "bin" / "activate")
    cmd    = (f"source {venv} && "
              f"CUDA_VISIBLE_DEVICES={TRAIN_GPUS.split(',')[0]} "
              f"python repos/agent/predict.py "
              f"--ckpt {fno_ckpt} --val_only")
    proc   = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        timeout=120, cwd=str(PROJECT_DIR), executable="/bin/bash")
    output = proc.stdout + proc.stderr

    # 解析得分
    total = 0.0
    m     = _re.search(r"预测得分:\s*([\d.]+)", output)
    if m:
        total = float(m.group(1))

    logger.info(f"FNO Nu=0.001 评分对齐校验: {total:.2f}/100")
    logger.info("用途：仅验证评分机制正确，不参与后续训练和推理")
    logger.info("后续所有推理均使用 DeepONet 模型")

    _log(
        content=(
            f"FNO Nu=0.001 评分对齐校验: 总分={total:.2f}/100"
            f" (仅用于验证评分机制，后续全部使用DeepONet推理)"
        ),
        tool_calls='predict({"ckpt":"FNO_Nu0.001","val_only":true,"purpose":"score_alignment_check"})'
    )
    return total


# ============================================================
# LLM 调用
# ============================================================
def llm_call(prompt, max_tokens=1200):
    global _CURRENT_LLM_ELAPSED

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(LLM_WORKER_CODE)
        worker = f.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_in.json", delete=False, encoding="utf-8"
    ) as f:
        json.dump({"prompt": prompt, "max_tokens": max_tokens,
                   "system_prompt": DEEPONET_SYSTEM_PROMPT},
                  f, ensure_ascii=False)
        in_f = f.name

    out_f = in_f.replace("_in.json", "_out.json")

    try:
        logger.info(f"LLM子进程 GPU={LLM_GPU}...")
        t0  = time.time()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = LLM_GPU
        r   = subprocess.run(
            [sys.executable, worker, LLM_GPU, MODEL_PATH, in_f, out_f],
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_DIR), env=env)
        _CURRENT_LLM_ELAPSED = time.time() - t0
        logger.info(f"LLM完成 {_CURRENT_LLM_ELAPSED:.1f}s")
        if r.returncode != 0:
            logger.error(f"LLM失败:\n{r.stderr[-300:]}")
            return None
        time.sleep(1)
        return json.load(open(out_f, encoding="utf-8"))["reply"]
    except subprocess.TimeoutExpired:
        logger.error("LLM超时")
        return None
    finally:
        for fp in [worker, in_f, out_f]:
            try: os.unlink(fp)
            except: pass


# ============================================================
# Observation 构建
# ============================================================
def build_observation(iteration, fno_baseline, budget_remaining_min):
    board    = get_leaderboard(limit=15)
    structs  = get_all_structures_tested()
    n_structs = len(structs)
    phase    = "探索期" if n_structs < 3 else "利用期"

    if board:
        best       = board[0]
        best_id    = best['exp_id']
        best_score = best['total_score'] or 0

        board_lines = [
            f"{'排名':<4} {'exp_id':<10} {'总分':>6} "
            f"{'Seg1':>6} {'Seg2':>6} {'Seg3':>6} "
            f"{'结构':<22} {'vmr':>5}"
        ]
        for i, r in enumerate(board):
            s   = r['total_score'] or 0
            s1  = r['seg1_score']  or 0
            s2  = r['seg2_score']  or 0
            s3  = r['seg3_score']  or 0
            st  = (f"{r['branch_type']}+"
                   f"d{r['branch_depth']}w{r['branch_width']}")
            vmr = r.get('val_mix_ratio') or 1.0
            mark = " ★" if i == 0 else ""
            board_lines.append(
                f"{i+1:<4} {r['exp_id']:<10} {s:>6.2f} "
                f"{s1:>6.1f} {s2:>6.1f} {s3:>6.1f} "
                f"{st:<22} {vmr:>5.1f}{mark}"
            )
        board_str = "\n".join(board_lines)

        best_cfg = (
            f"branch_type={best.get('branch_type')}  "
            f"depth={best.get('branch_depth')}  "
            f"width={best.get('branch_width')}\n"
            f"trunk_depth={best.get('trunk_depth')}  "
            f"trunk_width={best.get('trunk_width')}  "
            f"latent_dim={best.get('latent_dim')}\n"
            f"lr={best.get('lr')}  epochs={best.get('epochs')}  "
            f"val_mix_ratio={best.get('val_mix_ratio', 1.0)}"
        )
        infl  = best.get('inflection_step', -1)
        ratio = best.get('extrap_ratio', -1)
        diag  = (f"误差拐点: 步{infl}  外推/覆盖比: {ratio:.1f}x"
                 if infl and infl > 0 else "暂无诊断")
    else:
        board_str  = "（无实验记录）"
        best_id    = "scratch"
        best_score = 0.0
        best_cfg   = "无"
        diag       = "无"

    # 下一个 exp_id
    n_exps  = len(sqlite3.connect(str(DB_PATH)).execute(
        "SELECT exp_id FROM experiments").fetchall()) if DB_PATH.exists() else 0
    next_id = f"exp_{n_exps+1:03d}"

    fno_line = (f"FNO Baseline (固定标尺): {fno_baseline:.2f}/100"
                if fno_baseline is not None
                else "FNO Baseline: 未运行")

    obs = f"""=== 迭代#{iteration} ===
剩余时间预算: {budget_remaining_min:.1f} 分钟
{fno_line}

=== 排行榜 ===
{board_str}

=== 当前最优 ({best_id}) ===
{best_cfg}

=== 误差诊断 ===
{diag}

=== 探索状态 ===
已测试结构数: {n_structs}  阶段: {phase}
{'→ 建议：加深加宽 cnn（depth=4,width=128）+ val_mix_ratio=0.5 改善 Seg1' if phase == '利用期' else '→ 继续探索新结构'}

下一个 exp_id: {next_id}
输出 <think>...</think> 和 JSON。"""
    return obs.strip()


# ============================================================
# 解析 LLM 输出
# ============================================================
def parse_llm_output(raw):
    think = ""
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        think = m.group(1).strip()
    rest  = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m2    = re.search(r"\{.*\}", rest, re.DOTALL)
    plan  = {}
    if m2:
        try:
            plan = json.loads(m2.group())
        except json.JSONDecodeError:
            pass
    return think, plan


# ============================================================
# 单轮迭代
# ============================================================
def run_iteration(iteration, fno_baseline, budget_remaining_min):
    logger.info("=" * 60)
    logger.info(f"迭代#{iteration} | 剩余时间: {budget_remaining_min:.1f}min")

    obs = build_observation(iteration, fno_baseline, budget_remaining_min)
    raw = llm_call(obs, max_tokens=1200)
    if not raw:
        logger.error("LLM返回空，跳过本轮")
        _log(f"迭代#{iteration} LLM失败")
        return None

    think, plan = parse_llm_output(raw)
    if not plan:
        logger.warning("JSON解析失败，跳过")
        _log(f"迭代#{iteration} JSON解析失败: {raw[:200]}")
        return None

    exp_id    = plan.get("exp_id",    f"exp_{iteration:03d}")
    parent_id = plan.get("parent_id", "scratch")
    changes   = plan.get("changes",   {})
    rationale = plan.get("rationale", "")
    hypothesis = plan.get("hypothesis", "")

    logger.info(f"Plan: {exp_id} parent={parent_id} "
                f"changes={list(changes.keys())}")

    _log(
        content=(f"迭代#{iteration}[{exp_id}] "
                 f"think: {think[:500]}" if think else
                 f"迭代#{iteration}[{exp_id}] hypothesis: {hypothesis}"),
        tool_calls=f'propose_and_run({json.dumps({"exp_id":exp_id,"parent_id":parent_id,"changes":changes}, ensure_ascii=False)})'
    )

    result = propose_and_run(
        exp_id=exp_id, parent_id=parent_id, changes=changes,
        rationale=rationale, hypothesis=hypothesis, gpu_ids=TRAIN_GPUS)

    if result["success"]:
        s  = result.get("total_score", 0)
        s1 = result.get("seg1_score",  0)
        s2 = result.get("seg2_score",  0)
        s3 = result.get("seg3_score",  0)
        it = result.get("infer_time",  0)
        _log(content=(
            f"迭代#{iteration}[{exp_id}] 完成 "
            f"总分={s:.2f} Seg1={s1:.1f} Seg2={s2:.1f} Seg3={s3:.1f} "
            f"推理={it:.1f}s"
        ))
        update_experiment(exp_id, {
            "conclusion": (f"总分={s:.2f} Seg1={s1:.1f} "
                           f"Seg2={s2:.1f} Seg3={s3:.1f} 推理={it:.1f}s")
        })
    else:
        _log(content=f"迭代#{iteration}[{exp_id}] 失败: "
                     f"{result.get('result','')[:200]}")

    _log_leaderboard(iteration)

    # 更新 task1_time.csv（train_time = agent 总运行时间）
    total_elapsed = time.time() - AGENT_START_TIME
    time_csv = SUBMISSION / "task1_time.csv"
    infer_t  = result.get("infer_time", 0) if result.get("success") else 0
    with open(time_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['train_time', 'inference_time'])
        w.writerow([total_elapsed, infer_t])

    return result


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_minutes", type=float, default=55.0,
                        help="总时间预算（分钟），默认55分钟（留5分钟buffer保证≤60min得35分）")
    parser.add_argument("--target_score",   type=float, default=80.0)
    parser.add_argument("--reset",          action="store_true")
    args = parser.parse_args()

    if args.reset:
        if DB_PATH.exists():
            DB_PATH.unlink()
            logger.info("registry.db 已清除")

    init_registry()
    logger.info("=" * 60)
    logger.info(f"DeepONet Agent 启动 | 目标:{args.target_score} "
                f"| 预算:{args.budget_minutes}min")
    logger.info("=" * 60)
    _log(f"Agent启动 | 目标:{args.target_score}/100 | 预算:{args.budget_minutes}min")

    # FNO Baseline 标尺
    fno_baseline = run_fno_baseline()

    iteration = 0
    while True:
        iteration += 1
        elapsed_min    = (time.time() - AGENT_START_TIME) / 60
        remaining_min  = args.budget_minutes - elapsed_min

        # 时间检查：每轮训练约需 5-10 分钟，留足时间
        if remaining_min < 8:
            logger.info(f"剩余时间 {remaining_min:.1f}min < 8min，停止迭代")
            break

        result = run_iteration(iteration, fno_baseline, remaining_min)

        board = get_leaderboard(limit=1)
        best  = board[0]['total_score'] if board else 0.0

        if best >= args.target_score:
            logger.info(f"已达目标 {args.target_score}，停止")
            break

    # 最终汇总
    board = get_leaderboard(limit=1)
    final = board[0]['total_score'] if board else 0.0
    total_elapsed = (time.time() - AGENT_START_TIME) / 60

    logger.info("=" * 60)
    logger.info(f"Agent完成 | 分段总分:{final:.2f}/100 | 总耗时:{total_elapsed:.1f}min")

    # 训练时间分
    if total_elapsed <= 60:
        time_score = 35
    elif total_elapsed <= 120:
        time_score = 25
    elif total_elapsed <= 300:
        time_score = 20
    elif total_elapsed <= 500:
        time_score = 10
    else:
        time_score = 0

    # 精度分（×0.75）
    precision_score = final * 0.75

    # 推理时间分（从最后一次实验取）
    infer_t = 0.0
    time_csv = SUBMISSION / "task1_time.csv"
    if time_csv.exists():
        with open(time_csv, 'r') as f:
            for row in csv.DictReader(f):
                infer_t = float(row.get('inference_time', 0))
    infer_min = infer_t / 60
    if infer_min <= 0:
        infer_score = 40
    elif infer_min < 2:
        infer_score = 40 * (1 - infer_min / 2)
    else:
        infer_score = 0

    total_task1 = precision_score + time_score + infer_score

    logger.info(f"=" * 60)
    logger.info(f"Task1 总分预测: {total_task1:.1f} / 150")
    logger.info(f"  精度分:      {precision_score:.2f} / 75  "
                f"(分段总分{final:.2f} × 0.75)")
    logger.info(f"  训练时间分:  {time_score} / 35  "
                f"({total_elapsed:.1f}min)")
    logger.info(f"  推理时间分:  {infer_score:.1f} / 40  "
                f"({infer_t:.1f}s = {infer_min:.2f}min)")
    logger.info(f"=" * 60)

    _log(
        content=(
            f"Agent完成 | 分段总分:{final:.2f}/100 | "
            f"精度分:{precision_score:.2f}/75 | "
            f"训练时间分:{time_score}/35 ({total_elapsed:.1f}min) | "
            f"推理时间分:{infer_score:.1f}/40 ({infer_t:.1f}s) | "
            f"Task1总分预测:{total_task1:.1f}/150"
        )
    )

    # 最终更新 task1_time.csv
    time_csv = SUBMISSION / "task1_time.csv"
    infer_t  = 0.0
    if time_csv.exists():
        with open(time_csv, 'r') as f:
            for row in csv.DictReader(f):
                infer_t = float(row.get('inference_time', 0))
    with open(time_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['train_time', 'inference_time'])
        w.writerow([(time.time() - AGENT_START_TIME), infer_t])


if __name__ == "__main__":
    main()