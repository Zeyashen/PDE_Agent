#!/bin/bash
PROJECT=/mimer/NOBACKUP/groups/phy_geo/PDE_Agent
LOG=$PROJECT/guandi_agent/logs/agent_test.log

echo "启动Agent..."
echo "日志: $LOG"
echo ""

apptainer exec --nv --bind /mimer:/mimer \
    $PROJECT/env/pytorch.sif bash -c "
    source $PROJECT/env/pde_venv/bin/activate
    export PYTHONPATH=$PROJECT/guandi_agent/src
    cd $PROJECT/guandi_agent
    python src/agent/orchestrator.py
" 2>&1 | tee $LOG
