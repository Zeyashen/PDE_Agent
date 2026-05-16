#!/bin/bash

PROJECT_DIR="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent"
OUTPUT_DIR="${PROJECT_DIR}/logs"
mkdir -p "$OUTPUT_DIR"

JOB_SCRIPT="task1_full_job.sh"

cat > "$JOB_SCRIPT" << 'SBATCH_SCRIPT'
#!/bin/bash
#SBATCH -A NAISS2026-4-674
#SBATCH -p alvis
#SBATCH --job-name=task1_full
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -t 01:30:00
#SBATCH --output=OUTPUT_DIR_PLACEHOLDER/task1_full_%j.out
#SBATCH --error=OUTPUT_DIR_PLACEHOLDER/task1_full_%j.error

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset PKG_CONFIG_PATH
module purge >/dev/null 2>&1 || true

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cd PROJECT_DIR_PLACEHOLDER || exit 1

SIF="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/env/pytorch.sif"
VENV="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/env/pde_venv/bin/activate"

log "=== [1/2] Task1 训练 ==="
apptainer exec --nv --bind /mimer:/mimer "$SIF" bash -c "
    source $VENV &&
    python repos/agent/train.py \
        --epochs 50 \
        --batch_size 64 \
        --modes 16 \
        --width 64
"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    log "❌ 训练失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi

log "=== [2/2] Task1 推理 ==="
apptainer exec --nv --bind /mimer:/mimer "$SIF" bash -c "
    source $VENV &&
    python repos/agent/predict.py
"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    log "❌ 推理失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi

log "✅ Task1 全流程完成"
SBATCH_SCRIPT

sed -i "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" "$JOB_SCRIPT"
sed -i "s|PROJECT_DIR_PLACEHOLDER|${PROJECT_DIR}|g" "$JOB_SCRIPT"

sbatch "$JOB_SCRIPT"
echo "✅ 已提交！"
echo "查看状态 : squeue -u \$USER"
echo "查看输出 : tail -f ${OUTPUT_DIR}/task1_full_*.out"