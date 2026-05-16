#!/bin/bash
#SBATCH -A NAISS2026-4-674
#SBATCH -p alvis
#SBATCH --job-name=fno_zeroshot
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -t 00:15:00
#SBATCH --output=/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/logs/fno_zeroshot_%j.out
#SBATCH --error=/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/logs/fno_zeroshot_%j.error

# 清理宿主机 Python/module 环境污染
unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset PKG_CONFIG_PATH
module purge >/dev/null 2>&1 || true

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd /mimer/NOBACKUP/groups/phy_geo/PDE_Agent || exit 1

log "=== [阶段 1/2] 开始执行 FNO 模型推理与插值拉伸 ==="
# 直接调用我们之前写好的沙盒封装器！它会自动处理环境挂载。
./run_sandbox.sh repos/submission/code/train.py

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    log "❌ 推理脚本执行失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi

log "=== [阶段 2/2] 开始执行裁判长算分 ==="
# 同样使用沙盒运行 evaluator.py，确保能调用到 h5py 和 numpy 等依赖
./run_sandbox.sh evaluator.py \
    --pred repos/submission/task1_pred.hdf5 \
    --gt data_and_sample_submission/train_val_test_init/task1_val.hdf5

log "=== 作业全部完成 ==="
