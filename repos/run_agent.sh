#!/bin/bash
# ============================================================
# run_agent.sh — 启动 PDE Agent
#
# 用法：
#   bash run_agent.sh                          # 默认：task1, Qwen3.5-27B, 55min
#   bash run_agent.sh --task task2             # Task 2（自动设 budget=600）
#   bash run_agent.sh --model 14B              # 使用 Qwen3-14B
#   bash run_agent.sh --model 70B              # 使用 Llama-70B
#   bash run_agent.sh --reset                  # 清空 workspace 重新开始
#   bash run_agent.sh --task task2 --model 70B --reset
# ============================================================

PROJECT="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent"
SIF="/apps/containers/vLLM/vllm-0.19.1.sif"
VENV="${PROJECT}/env/pde_venv/bin/activate"

# ── 默认参数 ─────────────────────────────────────────────────
TASK="task1"
MODEL_SIZE="70B"
BUDGET=""        # 空则根据 task 自动设置
RESET=""
MAX_TOKENS=10000

# ── 解析命令行参数 ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)    TASK="$2";       shift 2 ;;
        --model)   MODEL_SIZE="$2"; shift 2 ;;
        --budget)  BUDGET="$2";     shift 2 ;;
        --reset)   RESET="--reset"; shift   ;;
        --tokens)  MAX_TOKENS="$2"; shift 2 ;;
        *)
            echo "未知参数: $1"
            echo "用法: bash run_agent.sh [--task task1|task2] [--model 14B|27B|70B] [--budget N] [--reset]"
            exit 1
            ;;
    esac
done

# ── 根据模型选择路径和 GPU 数量 ───────────────────────────────
case "$MODEL_SIZE" in
    14B)
        MODEL_PATH="${PROJECT}/models/Qwen3-14B"
        LLM_GPUS="0"          # 单卡推理
        ;;
    27B)
        MODEL_PATH="${PROJECT}/models/Qwen3.5-27B"
        LLM_GPUS="0,1"        # 双卡推理
        ;;
    70B)
        MODEL_PATH="${PROJECT}/models/Llama"
        LLM_GPUS="0,1,2,3"   # 四卡推理
        ;;
    *)
        echo "错误：未知模型 ${MODEL_SIZE}，支持 14B / 27B / 70B"
        exit 1
        ;;
esac

# 训练始终用全部4张卡
TRAIN_GPUS="0,1,2,3"

# ── 根据 task 自动设置 budget ────────────────────────────────
if [[ -z "$BUDGET" ]]; then
    if [[ "$TASK" == "task2" ]]; then
        BUDGET=600    # Task 2：训练时间不计入评测，给足10小时
    else
        BUDGET=55     # Task 1：含思考时间，55分钟保证≤60min得35分
    fi
fi

# ── 检查模型路径 ─────────────────────────────────────────────
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "错误：模型路径不存在: ${MODEL_PATH}"
    exit 1
fi

# ── 清理旧进程 ───────────────────────────────────────────────
echo "清理旧进程..."
pkill -f "orchestrator.py" 2>/dev/null
pkill -f "torchrun"         2>/dev/null
sleep 2

# ── 清理日志与历史文件（--reset 时） ─────────────────────────
if [[ -n "$RESET" ]]; then
    echo "清理历史文件..."
    > "${PROJECT}/logs/${TASK}/agent_orchestrator.log"
    > "${PROJECT}/logs/${TASK}/train_live.log"
    > "${PROJECT}/repos/submission/${TASK}_logs.log"
    rm -f "${PROJECT}/repos/submission/${TASK}_pred.hdf5"
    echo "清理完成"
fi

# ── 确保目录结构 ─────────────────────────────────────────────
mkdir -p "${PROJECT}/repos/workspace"
mkdir -p "${PROJECT}/repos/workspace/.history"
mkdir -p "${PROJECT}/repos/submission/code"
mkdir -p "${PROJECT}/logs"

echo "=============================="
echo "Agent 启动"
echo "  Task:       ${TASK}"
echo "  Model:      Qwen3.5/${MODEL_SIZE} (${MODEL_PATH})"
echo "  Budget:     ${BUDGET} 分钟"
echo "  LLM GPUs:   ${LLM_GPUS}"
echo "  Train GPUs: ${TRAIN_GPUS}"
echo "  Reset:      ${RESET:-否}"
echo "  时间:       $(date)"
echo "=============================="

# ── 启动 Agent 主循环 ────────────────────────────────────────
apptainer exec --nv --bind /mimer:/mimer "${SIF}" bash -c "
    source ${VENV}
    cd ${PROJECT}
    export LLM_GPUS=${LLM_GPUS}
    export TRAIN_GPUS=${TRAIN_GPUS}
    export PDE_TASK=${TASK}
    # export NCCL_P2P_DISABLE=1
    # export NCCL_IB_DISABLE=1
    export TORCH_NCCL_BLOCKING_WAIT=1
    # 选择是否开启调试模式
    export NCCL_DEBUG=INFO

    python repos/${TASK}/orchestrator.py \
        --task ${TASK} \
        --budget_minutes ${BUDGET} \
        --model_path ${MODEL_PATH} \
        --max_tokens ${MAX_TOKENS} \
        ${RESET}
"

EXIT_CODE=$?
echo "=============================="
echo "Agent 完成: $(date)"
echo "退出码: ${EXIT_CODE}"
echo "=============================="
exit ${EXIT_CODE}