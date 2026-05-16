#!/bin/bash
#SBATCH -A NAISS2026-4-674
#SBATCH -p alvis
#SBATCH --job-name=train_smoke
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -t 00:10:00
#SBATCH --output=/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/logs/train_smoke_%j.out
#SBATCH --error=/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/logs/train_smoke_%j.error

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
module purge >/dev/null 2>&1 || true

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cd /mimer/NOBACKUP/groups/phy_geo/PDE_Agent || exit 1

log "=== 冒烟测试：100样本 × 2 epochs ==="

SIF_PATH="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/env/pytorch.sif"
VENV="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/env/pde_venv/bin/activate"

apptainer exec --nv --bind /mimer:/mimer "$SIF_PATH" bash -c '
    source /mimer/NOBACKUP/groups/phy_geo/PDE_Agent/env/pde_venv/bin/activate &&
    python repos/agent/train.py --n_samples 100 --epochs 2 --batch_size 32
'

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    log "❌ 训练失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi

log "✅ 冒烟测试通过"
