#!/bin/bash
PROJECT=/mimer/NOBACKUP/groups/phy_geo/PDE_Agent
SIF=${PROJECT}/env/pytorch.sif
VENV=${PROJECT}/env/pde_venv/bin/activate

export LLM_GPUS=0
export TRAIN_GPUS=0,1,2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1

apptainer exec --nv --bind /mimer:/mimer ${SIF} bash -c "
    source ${VENV}
    cd ${PROJECT}
    export LLM_GPUS=${LLM_GPUS}
    export TRAIN_GPUS=${TRAIN_GPUS}
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export TORCH_NCCL_BLOCKING_WAIT=1
    python repos/agent/orchestrator.py
"
