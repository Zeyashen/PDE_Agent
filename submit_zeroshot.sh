#!/bin/bash

# ============================================
# 1. 路径与变量
PROJECT_DIR="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent"
OUTPUT_DIR="${PROJECT_DIR}/logs"

# 创建日志目录
mkdir -p "$OUTPUT_DIR"

JOB_NAME="fno_zeroshot"
JOB_SCRIPT="${JOB_NAME}_job.sh"

echo "=> 准备生成 PDEBench 零样本推理作业: $JOB_NAME"

# ============================================
# 2. 动态生成 SLURM 脚本
# ============================================
cat > "$JOB_SCRIPT" << 'SBATCH_SCRIPT'
#!/bin/bash
#SBATCH -A NAISS2026-4-674
#SBATCH -p alvis
#SBATCH --job-name=JOBNAME_PLACEHOLDER
#SBATCH --gpus-per-node=A100fat:1
#SBATCH -t 00:15:00
#SBATCH --output=OUTPUT_DIR_PLACEHOLDER/JOBNAME_PLACEHOLDER_%j.out
#SBATCH --error=OUTPUT_DIR_PLACEHOLDER/JOBNAME_PLACEHOLDER_%j.error

# 清理宿主机 Python/module 环境污染
unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset PKG_CONFIG_PATH
module purge >/dev/null 2>&1 || true

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd PROJECT_DIR_PLACEHOLDER || exit 1

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
SBATCH_SCRIPT

# ============================================
# 3. 替换占位符
# ============================================
sed -i "s|JOBNAME_PLACEHOLDER|${JOB_NAME}|g" "$JOB_SCRIPT"
sed -i "s|OUTPUT_DIR_PLACEHOLDER|${OUTPUT_DIR}|g" "$JOB_SCRIPT"
sed -i "s|PROJECT_DIR_PLACEHOLDER|${PROJECT_DIR}|g" "$JOB_SCRIPT"

# ============================================
# 4. 提交作业到队列
# ============================================
echo "Submitting: $JOB_SCRIPT"
sbatch "$JOB_SCRIPT"

echo "✅ 提交成功！"
echo "你可以使用 'squeue -u $USER' 查看排队状态。"
echo "运行结束后，请去 ${OUTPUT_DIR} 目录下查看 .out 文件获取最终得分。"