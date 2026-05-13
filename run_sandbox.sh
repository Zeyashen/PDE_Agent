#!/bin/bash

if [ -z "$1" ]; then
    echo "❌ 错误: 请指定要运行的 Python 脚本！"
    echo "💡 用法: ./run_sandbox.sh <你的python脚本路径> [其他参数]"
    exit 1
fi

TARGET_SCRIPT=$1
shift

SIF_PATH="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/env/pytorch.sif"
VENV_ACTIVATE="/mimer/NOBACKUP/groups/phy_geo/PDE_Agent/env/pde_venv/bin/activate"

apptainer exec --nv --bind /mimer:/mimer "$SIF_PATH" bash -c "
    source \"$VENV_ACTIVATE\" && 
    python \"$TARGET_SCRIPT\" $@
"
