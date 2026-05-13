#!/bin/bash
# ============================================================
# PDE Agent 启动脚本（sbatch 批处理版）
#
# 用法：
#   bash start_agent_session.sh                        # 提交 job（不清理历史）
#   bash start_agent_session.sh --reset                # 清理历史后提交
#   bash start_agent_session.sh --reset --watch        # 清理+提交+实时监控
#   bash start_agent_session.sh --watch                # 提交+实时监控
#
# GPU 分配：
#   LLM 推理  → GPU 0（单卡，Qwen3-14B float16 ~28GB）
#   DDP 训练  → GPU 0,1,2,3（4卡，训练完毕后释放）
#   两者串行，不重叠，不会 OOM
# ============================================================

PROJECT_DIR="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent"
SIF="${PROJECT_DIR}/env/pytorch.sif"
JOB_SCRIPT="${PROJECT_DIR}/env/agent_job.sh"
AGENT_LOG="${PROJECT_DIR}/logs/slurm_agent.log"

# ── 解析参数 ────────────────────────────────────────────────
DO_RESET=false
DO_WATCH=false
for arg in "$@"; do
    case $arg in
        --reset) DO_RESET=true ;;
        --watch) DO_WATCH=true ;;
    esac
done

# ── 第一步：生成 sbatch job 脚本 ────────────────────────────
cat > "${JOB_SCRIPT}" << 'JOBEOF'
#!/bin/bash
#SBATCH -A NAISS2026-4-674
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:4
#SBATCH --time=12:00:00
#SBATCH --job-name=pde_agent
#SBATCH --output=/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/logs/slurm_agent.log
#SBATCH --error=/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/logs/slurm_agent.log

PROJECT_DIR="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent"
SIF="${PROJECT_DIR}/env/pytorch.sif"

# ── GPU 分配 ────────────────────────────────────────────────
# LLM 推理使用 GPU 0（orchestrator 子进程自行设置 CUDA_VISIBLE_DEVICES=0）
# DDP 训练使用全部 4 卡（tools.py 子进程自行设置 CUDA_VISIBLE_DEVICES=0,1,2,3）
# orchestrator 主进程本身不占 GPU
export LLM_GPUS="0"
export TRAIN_GPUS="0,1,2,3"

# ── A40 NCCL 稳定性设置 ─────────────────────────────────────
# 子进程会继承这些变量，torchrun 启动 DDP 时生效
export NCCL_P2P_DISABLE=1           # 禁用 P2P，A40 驱动兼容性更好
export NCCL_IB_DISABLE=1            # 单节点不需要 InfiniBand
export TORCH_NCCL_BLOCKING_WAIT=1   # 超时时报错而不是卡死
export NCCL_TIMEOUT=1800            # NCCL 超时 30 分钟

echo "=============================="
echo "Job 启动: $(date)"
echo "节点: $SLURMD_NODENAME"
echo "GPU: $SLURM_JOB_GPUS"
echo "LLM_GPUS=${LLM_GPUS}  TRAIN_GPUS=${TRAIN_GPUS}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "=============================="

apptainer exec --nv --bind /mimer:/mimer "${SIF}" bash -c "
    source ${PROJECT_DIR}/env/pde_venv/bin/activate
    cd ${PROJECT_DIR}

    # 导出到容器内部（apptainer 不一定继承外部 export）
    export LLM_GPUS=${LLM_GPUS}
    export TRAIN_GPUS=${TRAIN_GPUS}
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export TORCH_NCCL_BLOCKING_WAIT=1
    export NCCL_TIMEOUT=1800

    # orchestrator 主进程不设 CUDA_VISIBLE_DEVICES
    # 让 LLM 子进程和训练子进程各自管理自己的 GPU 可见性
    python repos/agent/orchestrator.py --max_iterations 5
"

echo "=============================="
echo "Job 完成: $(date)"
echo "=============================="
JOBEOF

chmod +x "${JOB_SCRIPT}"
echo "✅ job 脚本已写入: ${JOB_SCRIPT}"

# ── 第二步：按需清理历史 ─────────────────────────────────────
if [ "${DO_RESET}" = true ]; then
    echo ""
    echo "🧹 清理历史文件（--reset 模式）..."
    BASE="${PROJECT_DIR}"
    rm -f  "${BASE}/checkpoints/task1_trained/best_model.pt"
    rm -f  "${BASE}/checkpoints/task1_trained/best_model_backup.pt"
    rm -f  "${BASE}/checkpoints/task1_trained/best_model_candidate.pt"
    rm -rf "${BASE}/checkpoints/task1_trained/code_backup/"
    rm -f  "${BASE}/logs/conv_history.json"
    rm -f  "${BASE}/logs/experiment_history.json"
    > "${BASE}/logs/task1_train.log"
    > "${BASE}/logs/task1_predict.log"
    > "${BASE}/logs/agent_orchestrator.log"
    > "${BASE}/repos/submission/task1_logs.log"
    rm -f  "${BASE}/repos/submission/task1_pred.hdf5"
    > "${BASE}/repos/submission/task1_time.csv"
    echo "✅ 清理完成"
else
    echo ""
    echo "ℹ️  保留历史记录（如需清理请加 --reset 参数）"
fi

# ── 第三步：确保日志目录存在 ────────────────────────────────
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/repos/submission"
mkdir -p "${PROJECT_DIR}/checkpoints/task1_trained"

# ── 第四步：提交 job ─────────────────────────────────────────
echo ""
JOB_ID=$(sbatch "${JOB_SCRIPT}" | awk '{print $4}')

if [ -z "${JOB_ID}" ]; then
    echo "❌ 提交失败，请检查 sbatch 输出"
    exit 1
fi

echo "✅ Job 已提交: ${JOB_ID}"
echo ""
echo "常用命令:"
echo "  squeue -j ${JOB_ID}              # 查看 job 状态"
echo "  tail -f ${AGENT_LOG}             # 实时查看日志"
echo "  scancel ${JOB_ID}               # 取消 job"
echo "  tail -f ${PROJECT_DIR}/repos/submission/task1_logs.log  # 查看科研日志"
echo ""

# ── 第五步：按需监控日志 ─────────────────────────────────────
if [ "${DO_WATCH}" = true ]; then
    echo "⏳ 等待 job 启动..."
    for i in $(seq 1 60); do
        sleep 5
        STATE=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null)
        if [ "${STATE}" = "RUNNING" ]; then
            echo "🚀 Job 开始运行，监控日志..."
            echo ""
            tail -f "${AGENT_LOG}"
            break
        elif [ -z "${STATE}" ]; then
            echo "✅ Job 已完成"
            break
        fi
        echo "  等待中... (${i}/60) 状态: ${STATE}"
    done
fi